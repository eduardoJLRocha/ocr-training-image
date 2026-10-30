import os
from io import BytesIO
from time import perf_counter
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from .ocr_service import FINETUNED_MODEL_PATH, recognize

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
SUPPORTED_MIME_TYPES = {
    "image/jpeg": {"JPEG"},
    "image/png": {"PNG"},
    "image/webp": {"WEBP"},
}


class OcrLine(BaseModel):
    text: str
    confidence: float


class OcrResponse(BaseModel):
    text: str
    lines: list[OcrLine]
    engine: str
    model: str
    processing_time_ms: float


app = FastAPI(title="Image OCR API", version="1.0.0")
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ocr", response_model=OcrResponse)
async def extract_text(
    image: UploadFile = File(...),
    model: Literal["original", "finetuned"] = Query("original"),
) -> OcrResponse:
    accepted_formats = SUPPORTED_MIME_TYPES.get(image.content_type or "")
    if accepted_formats is None:
        raise HTTPException(
            status_code=415, detail="Use uma imagem JPEG, PNG ou WEBP."
        )

    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail="A imagem deve ter no máximo 10 MB."
        )
    if not content:
        raise HTTPException(status_code=400, detail="O ficheiro está vazio.")

    try:
        with Image.open(BytesIO(content)) as source:
            if source.format not in accepted_formats:
                raise HTTPException(
                    status_code=415,
                    detail="O conteúdo do ficheiro não corresponde ao formato indicado.",
                )
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="A imagem excede o limite de 25 megapíxeis.",
                )
            source.verify()
        with Image.open(BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(status_code=400, detail="O ficheiro não é uma imagem válida.") from error

    started = perf_counter()
    image_array = np.asarray(image)[:, :, ::-1].copy()
    if model == "finetuned" and not FINETUNED_MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="O modelo fine-tuned ainda não está disponível. Executa o treino primeiro.",
        )
    text, lines, engine_time_ms = recognize(image_array, model)
    elapsed_ms = round((perf_counter() - started) * 1000, 2)

    return OcrResponse(
        text=text,
        lines=[OcrLine(**line) for line in lines],
        engine="RapidOCR (PP-OCRv4)",
        model="PP-OCRv4 fine-tuned pilot" if model == "finetuned" else "PP-OCRv4 original",
        processing_time_ms=engine_time_ms or elapsed_ms,
    )
