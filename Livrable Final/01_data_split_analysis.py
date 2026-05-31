"""
01 : Dataset split + resize + exploratory analysis.

Usage:
    python3 01_data_split_analysis.py

Return:
    - data_split/{train,val,test}/{classe}/
    - all the resized images to 256x256 RGB
    - figures/01_*.png
"""

from __future__ import annotations

import hashlib
import random
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image, ImageOps

from utils import IMG_SIZE

# Configuration

SEED = 42

DATASET_DIR = Path("Dataset/Dataset")
SPLIT_DIR = Path("data_split")
FIGURES_DIR = Path("figures")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

TARGET_SIZE = IMG_SIZE

IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tiff",
    ".webp",
}

# Helpers

def list_images(class_dir: Path) -> list[Path]:
    return [
        p
        for p in class_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    ]


def filter_valid_images(images: list[Path]) -> tuple[list[Path], list[Path]]:
    """Return (good, corrupted) by testing the opening of each image."""
    valid, corrupted = [], []
    for p in images:
        try:
            with Image.open(p) as img:
                img.verify()
            valid.append(p)
        except Exception:
            corrupted.append(p)
    return valid, corrupted


def find_duplicates(images_by_class: dict[str, list[Path]]) -> dict[str, list[Path]]:
    """Detect duplicates across and within classes using MD5 hash.
    Returns a dictionary mapping hashes to lists of image paths for duplicated images.
    """
    hash_map: dict[str, list[Path]] = {}
    for paths in images_by_class.values():
        for p in paths:
            try:
                h = hashlib.md5(p.read_bytes()).hexdigest()
                hash_map.setdefault(h, []).append(p)
            except Exception:
                pass
    return {h: paths for h, paths in hash_map.items() if len(paths) > 1}


def split_class(
    images: list[Path],
    seed: int = SEED,
) -> tuple[list[Path], list[Path], list[Path]]:

    imgs = images.copy()

    random.Random(seed).shuffle(imgs)

    n = len(imgs)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = imgs[:n_train]
    val = imgs[n_train:n_train + n_val]
    test = imgs[n_train + n_val:]

    return train, val, test


def resize_and_save_image(src_path: Path, dst_path: Path) -> bool:
    """
    Resize an image while preserving the aspect ratio with a centered crop.
    Save as RGB JPEG.
    """

    try:
        with Image.open(src_path) as img:

            # conversion to RGB
            img = img.convert("RGB")

            # resize + center crop
            img = ImageOps.fit(
                img,
                TARGET_SIZE,
                method=Image.Resampling.LANCZOS,
            )

            # clean jpg
            dst_path = dst_path.with_suffix(".jpg")

            img.save(
                dst_path,
                format="JPEG",
                quality=95,
                optimize=True,
            )

        return True

    except Exception as e:
        print(f"Erreur image {src_path} : {e}")
        return False

# Main

def main() -> None:

    t0 = time.time()

    random.seed(SEED)
    np.random.seed(SEED)

    FIGURES_DIR.mkdir(exist_ok=True)

    # 1. Scan dataset

    print("=" * 70)
    print("1. Dataset scan:")
    print("=" * 70)

    print(f"Dataset dir : {DATASET_DIR.resolve()}")

    if not DATASET_DIR.exists():
        raise SystemExit(
            f"ERREUR : {DATASET_DIR} not found."
        )

    EXPECTED_CLASSES = {
        "Painting",
        "Photo",
        "Schematics",
        "Sketch",
        "Text",
    }

    classes = sorted([
        d.name
        for d in DATASET_DIR.iterdir()
        if d.is_dir() and d.name in EXPECTED_CLASSES
    ])

    print(f"Class found : {classes}")

    images_by_class_raw = {
        c: list_images(DATASET_DIR / c)
        for c in classes
    }

    for c, imgs in images_by_class_raw.items():
        print(f"  {c:12s} : {len(imgs):6d} images trouvées")

    # Filtering of corrupted images + reporting
    print("\n--- Vérification des images corrompues ---")
    images_by_class: dict[str, list[Path]] = {}
    total_corrupted = 0
    for c, imgs in images_by_class_raw.items():
        valid, corrupted = filter_valid_images(imgs)
        images_by_class[c] = valid
        total_corrupted += len(corrupted)
        if corrupted:
            print(f"  {c:12s} : {len(corrupted)} image(s) corrupted deleted")
            for p in corrupted:
                print(f"    - {p}")
        else:
            print(f"  {c:12s} : OK ({len(valid)} images valides)")
    print(f"Total corrupted: {total_corrupted}")

    # Duplicate detection + reporting
    print("\n Duplicate detection")
    duplicates = find_duplicates(images_by_class)
    if duplicates:
        print(f"  {len(duplicates)} groupe(s) de doublons trouvé(s) :")
        dup_paths: set[Path] = set()
        for paths in duplicates.values():
            print(f"  Doublon ({len(paths)} fichiers) :")
            for p in paths:
                print(f"    - {p}")
            # Keep only one image per group of duplicates, remove the others
            for p in paths[1:]:
                dup_paths.add(p)
        for c in classes:
            images_by_class[c] = [p for p in images_by_class[c] if p not in dup_paths]
        print(f"  {len(dup_paths)} duplicate(s) removed from the dataset.")
    else:
        print("  No duplicates found.")

    for c, imgs in images_by_class.items():
        print(f"  {c:12s} : {len(imgs):6d} image final after filtering")

    # 2. Class balance analysis

    print("\n" + "=" * 70)
    print("2. Class balance analysis")
    print("=" * 70)

    counts = {
        c: len(imgs)
        for c, imgs in images_by_class.items()
    }

    df_counts = pd.DataFrame({
        "class": list(counts.keys()),
        "n_images": list(counts.values()),
    })

    df_counts["ratio"] = (
        df_counts["n_images"] / df_counts["n_images"].sum()
    )

    print(df_counts.to_string(index=False))

    n_max = df_counts["n_images"].max()
    n_min = df_counts["n_images"].min()

    imbalance = n_max / max(n_min, 1)

    print(f"\n Imbalance ratio (max/min) : {imbalance:.2f}")

    if imbalance > 2:
        print("--> Imbalance dataset")
    else:
        print("--> Dataset effectively balanced")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        df_counts["class"],
        df_counts["n_images"],
        color="steelblue",
    )

    ax.set_ylabel("Image number")
    ax.set_title("Distribution of images by class")

    for i, v in enumerate(df_counts["n_images"]):
        ax.text(i, v, str(v), ha="center", va="bottom")

    plt.tight_layout()

    fig_path = FIGURES_DIR / "01_class_distribution.png"

    plt.savefig(fig_path, dpi=120)
    plt.close()

    print(f"Figure saved: {fig_path}")

    # 3. Image size distribution

    print("\n" + "=" * 70)
    print("3. Image size distribution")
    print("=" * 70)

    rows = []

    for c, imgs in images_by_class.items():

        sample = random.sample(
            imgs,
            min(200, len(imgs)),
        )

        for p in sample:

            try:
                with Image.open(p) as im:

                    rows.append({
                        "class": c,
                        "width": im.size[0],
                        "height": im.size[1],
                        "mode": im.mode,
                    })

            except Exception as e:
                print(f"Image illisible : {p} : {e}")

    df_sizes = pd.DataFrame(rows)

    print(df_sizes.describe().to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for c in classes:

        sub = df_sizes[df_sizes["class"] == c]

        axes[0].scatter(
            sub["width"],
            sub["height"],
            alpha=0.4,
            label=c,
            s=10,
        )

    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Height")
    axes[0].set_title("Dimensions of the images")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    df_sizes.boxplot(
        column="width",
        by="class",
        ax=axes[1],
    )

    axes[1].set_title("Width by class")

    plt.suptitle("")
    plt.tight_layout()

    fig_path = FIGURES_DIR / "01_image_sizes.png"

    plt.savefig(fig_path, dpi=120)
    plt.close()

    print(f"Figure saved: {fig_path}")

    # 4. Visual examples

    fig, axes = plt.subplots(
        len(classes),
        3,
        figsize=(10, 3 * len(classes)),
    )

    for i, c in enumerate(classes):

        sample = random.sample(
            images_by_class[c],
            min(3, len(images_by_class[c])),
        )

        for j, p in enumerate(sample):

            try:
                img = Image.open(p)

                axes[i, j].imshow(img)

                axes[i, j].set_title(
                    f"{c}\n{img.size}",
                    fontsize=9,
                )

            except Exception:
                axes[i, j].set_title(f"{c} : erreur")

            axes[i, j].axis("off")

    plt.tight_layout()

    fig_path = FIGURES_DIR / "01_samples_per_class.png"

    plt.savefig(fig_path, dpi=120)
    plt.close()

    print(f"Figure saved: {fig_path}")

    # 5. Split + resize

    print("\n" + "=" * 70)
    print("5. Split + resize of images")
    print("=" * 70)

    if SPLIT_DIR.exists():
        print(f"Deleting old directory {SPLIT_DIR}")
        shutil.rmtree(SPLIT_DIR)

    total_processed = 0
    split_summary = []

    for c in classes:

        train, val, test = split_class(images_by_class[c])

        for subset_name, subset in [
            ("train", train),
            ("val", val),
            ("test", test),
        ]:

            dst_dir = SPLIT_DIR / subset_name / c
            dst_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[{subset_name.upper()}] {c} : {len(subset)} images")

            for p in subset:

                dst_path = dst_dir / p.stem

                ok = resize_and_save_image(
                    src_path=p,
                    dst_path=dst_path,
                )

                if ok:
                    total_processed += 1

        split_summary.append({
            "class": c,
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "total": len(train) + len(val) + len(test),
        })

    df_split = pd.DataFrame(split_summary)

    df_split.loc["TOTAL"] = (
        df_split[["train", "val", "test", "total"]]
        .sum()
    )

    df_split.loc["TOTAL", "class"] = "TOTAL"

    print("\n")
    print(df_split.to_string(index=False))

    # 6. Disk check

    print("\n" + "=" * 70)
    print("6. Disk check")
    print("=" * 70)

    for subset in ["train", "val", "test"]:

        print(f"\n{subset.upper()} :")

        for c in classes:

            n = len(list(
                (SPLIT_DIR / subset / c).iterdir()
            ))

            print(f"  {c:12s} : {n:6d}")

    # End

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("Finish")
    print("=" * 70)

    print(f"Images processed: {total_processed}")

    print(f"\nTTotal time: {elapsed:.1f} s")

    print(f"\nDataset final : {SPLIT_DIR}")
    print(f"Resize final    : {TARGET_SIZE}")
    print(f"Plots           : {FIGURES_DIR}")

    print("=" * 70)

if __name__ == "__main__":
    main()