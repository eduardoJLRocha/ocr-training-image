import csv
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_IMAGES = PROJECT_ROOT / "data" / "ocr" / "test"


def test_health_endpoint() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_rejects_non_image_upload() -> None:
    response = client.post(
        "/api/ocr",
        files={"image": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"]


def test_rejects_invalid_image_data() -> None:
    response = client.post(
        "/api/ocr",
        files={"image": ("broken.png", b"not a png", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]


def test_rejects_mismatched_image_content_type() -> None:
    image_path = TEST_IMAGES / "malaysia-stop-sign.png"
    response = client.post(
        "/api/ocr",
        files={"image": (image_path.name, image_path.read_bytes(), "image/jpeg")},
    )

    assert response.status_code == 415


def test_finetuned_model_is_explicitly_selectable() -> None:
    image_path = TEST_IMAGES / "malaysia-stop-sign.png"
    response = client.post(
        "/api/ocr?model=finetuned",
        files={"image": (image_path.name, image_path.read_bytes(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["text"].strip() == "BERHENTI"
    assert response.json()["model"] == "PP-OCRv4 fine-tuned pilot"


def test_unknown_model_variant_is_rejected() -> None:
    image_path = TEST_IMAGES / "malaysia-stop-sign.png"
    response = client.post(
        "/api/ocr?model=unknown",
        files={"image": (image_path.name, image_path.read_bytes(), "image/png")},
    )
    assert response.status_code == 422


def test_rejects_oversized_upload() -> None:
    response = client.post(
        "/api/ocr",
        files={"image": ("large.jpg", b"x" * (10 * 1024 * 1024 + 1), "image/jpeg")},
    )

    assert response.status_code == 413


def test_ocr_on_every_held_out_image() -> None:
    image_paths = sorted(
        path
        for path in TEST_IMAGES.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    assert image_paths, "Add at least one licensed image to data/ocr/test."
    with (PROJECT_ROOT / "data" / "ocr" / "manifest.csv").open(
        encoding="utf-8", newline=""
    ) as manifest_file:
        expected_text = {
            entry["file"]: entry["reference_text"]
            for entry in csv.DictReader(manifest_file)
            if entry["split"] == "test" and entry["reference_verified"] == "true"
        }

    for image_path in image_paths:
        response = client.post(
            "/api/ocr",
            files={
                "image": (
                    image_path.name,
                    image_path.read_bytes(),
                    "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg",
                )
            },
        )

        assert response.status_code == 200, image_path.name
        payload = response.json()
        assert isinstance(payload["text"], str), image_path.name
        assert payload["lines"], image_path.name
        assert all(
            0 <= line["confidence"] <= 1 for line in payload["lines"]
        ), image_path.name
        assert payload["processing_time_ms"] >= 0, image_path.name
        reference = expected_text.get(image_path.name)
        if reference:
            normalized_result = " ".join(payload["text"].casefold().split())
            normalized_reference = " ".join(reference.casefold().split())
            assert normalized_reference in normalized_result, image_path.name
