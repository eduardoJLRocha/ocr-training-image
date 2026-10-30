from functools import lru_cache
import os
from pathlib import Path

import numpy as np
from rapidocr_onnxruntime import RapidOCR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINETUNED_MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "ocr_finetune" / "recognizer_finetuned.onnx"
)

@lru_cache(maxsize=1)
def get_engine(variant: str = "original") -> RapidOCR:
    if variant == "original":
        return RapidOCR()
    if variant != "finetuned":
        raise ValueError(f"Unknown OCR model variant: {variant}")
    if not FINETUNED_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Fine-tuned OCR model not found: {FINETUNED_MODEL_PATH}")
    return RapidOCR(rec_model_path=str(FINETUNED_MODEL_PATH), rec_batch_num=1)


def recognize(
    image: np.ndarray, variant: str = "original"
) -> tuple[str, list[dict[str, float | str]], float]:
    result, elapsed = get_engine(variant)(image)
    lines: list[dict[str, float | str]] = []

    for item in result or []:
        text = str(item[1]).strip()
        if text:
            lines.append({"text": text, "confidence": float(item[2])})

    return (
        "\n".join(str(line["text"]) for line in lines),
        lines,
        round(sum(elapsed or []) * 1000, 2),
    )
