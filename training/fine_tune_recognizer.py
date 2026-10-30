from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper
from onnx2torch import convert
from onnx2torch.node_converters.batch_norm import OnnxBatchNorm
from onnx2torch.utils.common import OperationConverterResult, onnx_mapping_from_node
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_PATH = ROOT / "data" / "ocr" / "annotations.csv"
OUTPUT_DIR = ROOT / "artifacts" / "ocr_finetune"
CONFIG_PATH = Path(__file__).with_name("config.json")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
TRAIN_CROPS = ROOT / "data" / "ocr" / "train" / "recognition"
VALIDATION_CROPS = ROOT / "data" / "ocr" / "validation"
LOGGER = logging.getLogger("ocr_finetune")


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def base_model_path() -> Path:
    package_dir = Path(__import__("rapidocr_onnxruntime").__file__).resolve().parent
    path = package_dir / "models" / "ch_PP-OCRv4_rec_infer.onnx"
    if not path.is_file():
        raise FileNotFoundError(f"RapidOCR base recognizer not found: {path}")
    return path


def read_annotations() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with ANNOTATIONS_PATH.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row["annotation_method"] != "manual-visual" for row in rows):
        raise ValueError("Expected the checked-in, manually reviewed line annotations.")

    image_splits: dict[str, set[str]] = {}
    for row in rows:
        image_splits.setdefault(row["image"], set()).add(row["split"])
    if any(len(splits) > 1 for splits in image_splits.values()):
        raise ValueError("A source image cannot occur in both training and validation.")

    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if not train or not validation:
        raise ValueError("Both training and validation annotations are required.")
    if any(row["split"] not in {"train", "validation"} for row in rows):
        raise ValueError("Training must never read the held-out test split.")
    return train, validation


def save_crops(rows: list[dict[str, str]]) -> list[tuple[Path, str]]:
    crops: list[tuple[Path, str]] = []
    for row in rows:
        image_path = ROOT / "data" / "ocr" / row["split"] / row["image"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Annotated training-source image is missing: {image_path}")
        with Image.open(image_path) as source:
            left, top, right, bottom = (
                int(row[key]) for key in ("left", "top", "right", "bottom")
            )
            if not (0 <= left < right <= source.width and 0 <= top < bottom <= source.height):
                raise ValueError(f"Invalid text box for {row['image']}: {row}")
            crop = source.convert("RGB").crop((left, top, right, bottom))
            destination_dir = TRAIN_CROPS if row["split"] == "train" else VALIDATION_CROPS
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / (
                f"{Path(row['image']).stem}-{row['text'].lower().replace(' ', '-')}.png"
            )
            crop.save(destination, format="PNG")
            crops.append((destination, row["text"]))
    return crops


def convert_pretrained_graph(model_path: Path) -> tuple[torch.nn.Module, list[str]]:
    model = onnx.load(str(model_path))
    model_input = model.graph.input[0]
    for dim, value in zip(model_input.type.tensor_type.shape.dim, (None, 3, 48, None)):
        dim.ClearField("dim_value")
        dim.ClearField("dim_param")
        if value is None:
            dim.dim_param = "dynamic"
        else:
            dim.dim_value = value

    graph_nodes = []
    for node in model.graph.node:
        if node.op_type == "Constant" and len(node.output) == 1:
            value = next((attr for attr in node.attribute if attr.name == "value"), None)
            if value is not None:
                model.graph.initializer.append(helper.get_attribute_value(value))
                continue
        graph_nodes.append(node)
    model.graph.ClearField("node")
    model.graph.node.extend(graph_nodes)

    import onnx2torch.converter as converter_module

    original_get_converter = converter_module.get_converter

    def get_converter(**kwargs: Any):
        original = original_get_converter(**kwargs)
        if kwargs.get("operation_type") != "BatchNormalization":
            return original

        def dynamic_batch_norm(node: Any, graph: Any) -> OperationConverterResult:
            attributes = node.attributes
            module = OnnxBatchNorm(
                momentum=1 - attributes.get("momentum", 0.9),
                epsilon=attributes.get("epsilon", 1e-5),
            )
            return OperationConverterResult(module, onnx_mapping_from_node(node))

        return dynamic_batch_norm

    converter_module.get_converter = get_converter
    try:
        model = convert(model)
    finally:
        converter_module.get_converter = original_get_converter

    metadata = {item.key: item.value for item in onnx.load(str(model_path)).metadata_props}
    character_list = metadata["character"].splitlines()
    character_list.append(" ")
    character_list.insert(0, "blank")
    if len(character_list) != 6625:
        raise ValueError(f"Unexpected pretrained CTC dictionary size: {len(character_list)}")
    return model, character_list


def verify_conversion(model: torch.nn.Module, model_path: Path) -> None:
    model.eval()
    probe = np.random.default_rng(20261029).standard_normal((1, 3, 48, 320)).astype(
        np.float32
    )
    expected = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    ).run(None, {"x": probe})[0]
    with torch.no_grad():
        converted = model(torch.from_numpy(probe)).cpu().numpy()
    max_error = float(np.max(np.abs(expected - converted)))
    if not np.allclose(expected, converted, rtol=1e-4, atol=1e-5):
        raise ValueError(f"Converted pretrained model does not match ONNX (max error {max_error}).")
    LOGGER.info("Converted pretrained model verified against ONNX (max_abs_error=%.8g)", max_error)


def model_input(image_path: Path, height: int, width: int) -> torch.Tensor:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode annotated crop: {image_path}")
    source_height, source_width = image.shape[:2]
    resized_width = min(width, int(np.ceil(height * source_width / source_height)))
    resized = cv2.resize(image, (resized_width, height)).astype(np.float32) / 255.0
    resized = (resized - 0.5) / 0.5
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    canvas[:, :resized_width] = resized
    return torch.from_numpy(canvas.transpose(2, 0, 1)).unsqueeze(0)


def augment(image: torch.Tensor, rng: random.Random) -> torch.Tensor:
    brightness = rng.uniform(0.9, 1.1)
    contrast = rng.uniform(0.9, 1.1)
    return (image * brightness * contrast).clamp(-1, 1)


def target_ids(text: str, characters: list[str]) -> torch.Tensor:
    indices = {character: index for index, character in enumerate(characters)}
    unknown = sorted(set(text) - indices.keys())
    if unknown:
        raise ValueError(f"Training label contains characters absent from base model: {unknown}")
    return torch.tensor([indices[character] for character in text], dtype=torch.long)


def validate(
    model: torch.nn.Module,
    crops: list[tuple[Path, str]],
    characters: list[str],
    height: int,
    width: int,
) -> float:
    model.eval()
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    total_loss = 0.0
    with torch.no_grad():
        for image_path, text in crops:
            image = model_input(image_path, height, width)
            probabilities = model(image).clamp_min(1e-7)
            labels = target_ids(text, characters)
            log_probs = probabilities.log().transpose(0, 1)
            loss = loss_fn(
                log_probs,
                labels,
                torch.tensor([log_probs.shape[0]], dtype=torch.long),
                torch.tensor([len(labels)], dtype=torch.long),
            )
            total_loss += float(loss.item())
    return total_loss / len(crops)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune PP-OCRv4's CTC recognizer head.")
    parser.add_argument("--epochs", type=int, help="Override the reproducible config epoch count.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "training.log", mode="w", encoding="utf-8"),
        ],
    )
    config = load_config()
    epochs = args.epochs or int(config["epochs"])
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    train_rows, validation_rows = read_annotations()
    if {row["image"] for row in train_rows} & {row["image"] for row in validation_rows}:
        raise ValueError("Source-image leakage between training and validation.")
    train_crops = save_crops(train_rows)
    validation_crops = save_crops(validation_rows)
    base_path = base_model_path()
    model, characters = convert_pretrained_graph(base_path)
    verify_conversion(model, base_path)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    class_weights = [
        (name, buffer)
        for name, buffer in model.named_buffers()
        if buffer.ndim == 2 and buffer.shape[-1] == len(characters)
    ]
    if len(class_weights) != 1:
        raise ValueError(f"Expected one pretrained CTC projection, found {len(class_weights)}")
    parameter_name, pretrained_projection = class_weights[0]
    module_path, parameter_leaf = parameter_name.rsplit(".", 1)
    target_module = model.get_submodule(module_path)
    target_module._buffers.pop(parameter_leaf)
    target_module.register_parameter(
        parameter_leaf,
        torch.nn.Parameter(pretrained_projection.detach().clone(), requires_grad=True),
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(trainable_parameters) != 1:
        raise ValueError("The pilot fine-tune must update only the pretrained CTC projection.")

    initial_validation_loss = validate(
        model, validation_crops, characters, config["image_height"], config["image_width"]
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    rng = random.Random(seed)
    best_validation_loss = float("inf")
    best_checkpoint = CHECKPOINT_DIR / "best.pt"
    history: list[dict[str, float | int]] = []
    LOGGER.info(
        "Starting real fine-tuning: base=%s train_lines=%d validation_lines=%d epochs=%d trainable=%d/%d",
        base_path,
        len(train_crops),
        len(validation_crops),
        epochs,
        sum(parameter.numel() for parameter in trainable_parameters),
        sum(parameter.numel() for parameter in model.parameters()),
    )
    LOGGER.info("Initial validation CTC loss: %.6f", initial_validation_loss)

    for epoch in range(1, epochs + 1):
        model.train()
        for module in model.modules():
            if isinstance(module, OnnxBatchNorm):
                module.eval()
        epoch_loss = 0.0
        sample_order = list(range(len(train_crops)))
        rng.shuffle(sample_order)
        for sample_index in sample_order:
            image_path, text = train_crops[sample_index]
            image = augment(
                model_input(image_path, config["image_height"], config["image_width"]), rng
            )
            probabilities = model(image).clamp_min(1e-7)
            log_probs = probabilities.log().transpose(0, 1)
            labels = target_ids(text, characters)
            loss = loss_fn(
                log_probs,
                labels,
                torch.tensor([log_probs.shape[0]], dtype=torch.long),
                torch.tensor([len(labels)], dtype=torch.long),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item())

        validation_loss = validate(
            model, validation_crops, characters, config["image_height"], config["image_width"]
        )
        record = {
            "epoch": epoch,
            "train_ctc_loss": epoch_loss / len(train_crops),
            "validation_ctc_loss": validation_loss,
        }
        history.append(record)
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": config},
            CHECKPOINT_DIR / f"epoch-{epoch:03d}.pt",
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "config": config},
                best_checkpoint,
            )
        LOGGER.info(
            "epoch=%d/%d train_ctc_loss=%.6f validation_ctc_loss=%.6f best_validation_ctc_loss=%.6f",
            epoch,
            epochs,
            record["train_ctc_loss"],
            validation_loss,
            best_validation_loss,
        )

    best = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, config["image_height"], config["image_width"]),
        OUTPUT_DIR / "recognizer_finetuned.onnx",
        input_names=["x"],
        output_names=["softmax_11.tmp_0"],
        dynamic_axes={
            "x": {3: "width"},
            "softmax_11.tmp_0": {1: "sequence"},
        },
        opset_version=17,
        dynamo=False,
    )
    exported = onnx.load(str(OUTPUT_DIR / "recognizer_finetuned.onnx"))
    onnx.checker.check_model(exported)
    for metadata in onnx.load(str(base_path)).metadata_props:
        exported.metadata_props.add(key=metadata.key, value=metadata.value)
    onnx.save(exported, OUTPUT_DIR / "recognizer_finetuned.onnx")
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "status": "training_completed",
                "base_model": str(base_path),
                "base_sha256": __import__("hashlib").sha256(base_path.read_bytes()).hexdigest(),
                "seed": seed,
                "epochs_executed": epochs,
                "best_epoch": best["epoch"],
                "train_line_crops": len(train_crops),
                "validation_line_crops": len(validation_crops),
                "held_out_test_images": 0,
                "initial_validation_ctc_loss": initial_validation_loss,
                "best_validation_ctc_loss": best_validation_loss,
                "history": history,
                "trainable_parameter": parameter_name,
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in trainable_parameters
                ),
                "weights_exported": str(OUTPUT_DIR / "recognizer_finetuned.onnx"),
            },
            stream,
            indent=2,
        )
    LOGGER.info(
        "Training completed. Exported %s; best epoch %d; validation CTC loss %.6f -> %.6f",
        OUTPUT_DIR / "recognizer_finetuned.onnx",
        best["epoch"],
        initial_validation_loss,
        best_validation_loss,
    )


if __name__ == "__main__":
    main()
