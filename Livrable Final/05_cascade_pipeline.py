"""
05 : cascade inference pipeline for real-world image sorting.

This script applies the full trained classification system on a custom folder of images.
It is designed as a practical deployment tool, independent from training and benchmarking.

The pipeline works as follows:
    1. Load a folder of unseen images provided via CLI argument (--input_dir).
    2. Run a 5-class CNN (Painting, Photo, Schematics, Sketch, Text).
    3. If the predicted class is Painting or Photo, the image is passed through a
       specialized binary classifier (Painting vs Photo) for refinement.
    4. Final prediction is used to sort images into class-specific folders.

All inputs and outputs are configurable:
    - input_dir: folder containing images to classify
    - output_dir: destination folder for sorted images (default: Filtered_images)
    - config: JSON configuration file containing model paths (default: config_pipeline.json)

Output structure:
    output_dir/
        Photo/
        Painting/
        Schematics/
        Sketch/
        Text/

Usage:
    python3 05_cascade_pipeline.py --input_dir <path_to_images> --output_dir <output_folder>

Typical use case:
    - Test the model on unseen real-world images
    - Quickly filter and organize new datasets
    - Validate end-to-end cascade pipeline outside evaluation benchmarks
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "03_binary_model"))

from binary_model import _preprocess_for, BINARY_CLASSES
from utils import CLASS_NAMES, IMG_SIZE

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def decode_image(path: Path) -> tf.Tensor:
    img_h, img_w = IMG_SIZE
    raw = tf.io.read_file(str(path))
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_h, img_w])
    return img


def ensure_dir(path: Path) -> None: path.mkdir(parents=True, exist_ok=True)


def save_image(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


# Main cascade logic

def run_cascade(input_dir: Path, output_dir: Path, cfg: dict) -> None:
    print("Loading models...")
    multi_model = tf.keras.models.load_model(cfg["multiclass_model"])
    bin_model = tf.keras.models.load_model(cfg["binary_model"])
    binary_backbone = bin_model.name.replace("_binary", "")

    print(f"Classes : {CLASS_NAMES}  |  Backbone binaire : {binary_backbone}")

    idx_painting = CLASS_NAMES.index(BINARY_CLASSES[0])
    idx_photo    = CLASS_NAMES.index(BINARY_CLASSES[1])

    files = list_images(input_dir)

    if not files:
        print("No images found.")
        return

    print(f"Processing {len(files)} images...")

    for c in CLASS_NAMES: ensure_dir(output_dir / c)

    preprocess = _preprocess_for({"backbone": binary_backbone})

    for path in files:
        img = decode_image(path)
        img_input = tf.cast(img, tf.float32) / 255.0
        img_input = tf.expand_dims(img_input, axis=0)

        # Multi-class prediction
        probs_multi = multi_model(img_input, training=False).numpy()[0]
        pred = int(np.argmax(probs_multi))

        # Cascade refinement
        if pred in [idx_painting, idx_photo]:
            bin_input = preprocess(img)
            bin_input = tf.expand_dims(bin_input, axis=0)
            prob_bin = bin_model(bin_input, training=False).numpy().flatten()[0]
            pred = idx_photo if prob_bin >= 0.5 else idx_painting

        dst = output_dir / CLASS_NAMES[pred] / path.name
        save_image(path, dst)

    print(f"Done. Results saved in: {output_dir}")


# CLI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="Filtered_images")
    parser.add_argument("--config", type=str, default="config_pipeline.json")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    run_cascade(Path(args.input_dir), Path(args.output_dir), cfg)


if __name__ == "__main__":
    main()
