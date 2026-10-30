from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "ocr" / "manifest.csv"
TEST_DIR = ROOT / "data" / "ocr" / "test"
OUTPUT_DIR = ROOT / "artifacts" / "ocr_finetune"
BASE_MODEL_PATH = (
    Path(__import__("rapidocr_onnxruntime").__file__).resolve().parent
    / "models"
    / "ch_PP-OCRv4_rec_infer.onnx"
)


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def edit_distance(reference: list[str], prediction: list[str]) -> int:
    previous = list(range(len(prediction) + 1))
    for row_index, reference_item in enumerate(reference, start=1):
        current = [row_index]
        for column_index, prediction_item in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1]
                    + (reference_item != prediction_item),
                )
            )
        previous = current
    return previous[-1]


def read_holdout() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["split"] == "test" and row["reference_verified"] == "true"
        ]
    if not rows:
        raise ValueError("No verified, held-out test labels exist in the manifest.")
    if len({row["file"] for row in rows}) != len(rows):
        raise ValueError("Duplicate test manifest entries.")
    for row in rows:
        if not (TEST_DIR / row["file"]).is_file():
            raise FileNotFoundError(f"Missing held-out test image: {row['file']}")
    return rows


def run_evaluation(variant: str) -> dict[str, Any]:
    if variant == "original":
        model_path = BASE_MODEL_PATH
        engine = RapidOCR()
    else:
        model_path = OUTPUT_DIR / "recognizer_finetuned.onnx"
        if not model_path.is_file():
            raise FileNotFoundError(f"Fine-tuned model not found: {model_path}")
        engine = RapidOCR(rec_model_path=str(model_path), rec_batch_num=1)
    if not model_path.is_file():
        raise FileNotFoundError(f"Pretrained recognizer not found: {model_path}")

    results = []
    for row in read_holdout():
        image_path = TEST_DIR / row["file"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode held-out test image: {image_path}")
        started = time.perf_counter()
        predictions, _ = engine(image)
        predicted_text = "\n".join(
            str(item[1]).strip() for item in (predictions or []) if str(item[1]).strip()
        )
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        reference = normalize(row["reference_text"].replace("/", " "))
        prediction = normalize(predicted_text)
        reference_chars = list(reference)
        prediction_chars = list(prediction)
        reference_words = reference.split()
        prediction_words = prediction.split()
        results.append(
            {
                "file": row["file"],
                "reference": reference,
                "prediction": prediction,
                "character_errors": edit_distance(reference_chars, prediction_chars),
                "reference_characters": len(reference_chars),
                "word_errors": edit_distance(reference_words, prediction_words),
                "reference_words": len(reference_words),
                "processing_time_ms": duration_ms,
            }
        )

    character_errors = sum(result["character_errors"] for result in results)
    reference_characters = sum(result["reference_characters"] for result in results)
    word_errors = sum(result["word_errors"] for result in results)
    reference_words = sum(result["reference_words"] for result in results)
    return {
        "status": "evaluation_completed",
        "model": variant,
        "model_path": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "test_images": len(results),
        "cer": character_errors / reference_characters if reference_characters else 0,
        "wer": word_errors / reference_words if reference_words else 0,
        "character_errors": character_errors,
        "reference_characters": reference_characters,
        "word_errors": word_errors,
        "reference_words": reference_words,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR on the untouched test split.")
    parser.add_argument("--model", choices=("original", "finetuned"), required=True)
    args = parser.parse_args()
    report = run_evaluation(args.model)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"evaluation-{args.model}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    main()
