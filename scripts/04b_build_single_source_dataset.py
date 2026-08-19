"""
04b_build_single_source_dataset.py
==================================
Builds isolated YOLO11m Person Detection dataset directories for single-source
training (e.g., just CrowdHuman, or just COCO) or combined.

Usage:
    python scripts/04b_build_single_source_dataset.py --source crowdhuman
    python scripts/04b_build_single_source_dataset.py --source coco
    python scripts/04b_build_single_source_dataset.py --source combined
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
CONFIG_DIR = PROJECT_ROOT / "config"

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# Source definitions
SOURCES = {
    "coco_train": {
        "prefix": "coco_",
        "images": RAW_DIR / "coco" / "images" / "train2017",
        "labels": RAW_DIR / "coco" / "labels_yolo" / "train",
    },
    "coco_val": {
        "prefix": "coco_",
        "images": RAW_DIR / "coco" / "images" / "val2017",
        "labels": RAW_DIR / "coco" / "labels_yolo" / "val",
    },
    "crowdhuman_train": {
        "prefix": "ch_",
        "images": RAW_DIR / "crowdhuman" / "images",
        "labels": RAW_DIR / "crowdhuman" / "labels_yolo" / "train",
    },
    "crowdhuman_val": {
        "prefix": "ch_",
        "images": RAW_DIR / "crowdhuman" / "images",
        "labels": RAW_DIR / "crowdhuman" / "labels_yolo" / "val",
    },
}

def print_header(msg: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {msg}")
    print(f"{'=' * width}")

def find_paired_samples(images_dir: Path, labels_dir: Path) -> list:
    paired = []
    if not labels_dir.exists():
        return paired

    label_files = {f.stem: f for f in labels_dir.glob("*.txt")}
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for img_path in images_dir.glob(ext):
            if img_path.stem in label_files:
                paired.append((img_path, label_files[img_path.stem]))
    return paired

def split_data(paired: list, seed: int = RANDOM_SEED) -> dict:
    if len(paired) == 0:
        return {"train": [], "val": [], "test": []}
    train, temp = train_test_split(paired, train_size=TRAIN_RATIO, random_state=seed)
    val_ratio_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val, test = train_test_split(temp, train_size=val_ratio_of_temp, random_state=seed)
    return {"train": train, "val": val, "test": test}

def copy_files(pairs: list, prefix: str, split_name: str, dataset_dir: Path, stats: dict) -> None:
    img_dest = dataset_dir / "images" / split_name
    lbl_dest = dataset_dir / "labels" / split_name
    img_dest.mkdir(parents=True, exist_ok=True)
    lbl_dest.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in pairs:
        new_img_name = f"{prefix}{img_path.name}"
        new_lbl_name = f"{prefix}{lbl_path.name}"
        img_target = img_dest / new_img_name
        lbl_target = lbl_dest / new_lbl_name

        if not img_target.exists():
            shutil.copy2(str(img_path), str(img_target))
        if not lbl_target.exists():
            shutil.copy2(str(lbl_path), str(lbl_target))

        stats[split_name] = stats.get(split_name, 0) + 1

def generate_data_yaml(source: str, dataset_dir: Path) -> None:
    yaml_path = CONFIG_DIR / f"data_{source}.yaml"
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path_str = str(dataset_dir).replace("\\", "/")
    
    yaml_content = f"""# YOLO11m Person Detection — Dataset Configuration
# Source: {source}

path: {dataset_path_str}
train: images/train
val: images/val
test: images/test

names:
  0: person
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"  ✓ Written config to: {yaml_path}")

def count_dir_files(dir_path: Path, pattern: str = "*.*") -> int:
    if not dir_path.exists():
        return 0
    return len(list(dir_path.glob(pattern)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, choices=["crowdhuman", "coco", "combined"])
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / f"dataset_{args.source}"
    print_header(f"Building Dataset for Source: {args.source}")
    print(f"  Output directory: {dataset_dir}")

    # Determine which sources to process
    keys_to_process = []
    if args.source in ["coco", "combined"]:
        keys_to_process.extend(["coco_train", "coco_val"])
    if args.source in ["crowdhuman", "combined"]:
        keys_to_process.extend(["crowdhuman_train", "crowdhuman_val"])

    source_groups = {
        "COCO": [],
        "CrowdHuman": [],
    }

    for k in keys_to_process:
        cfg = SOURCES[k]
        print(f"\n  → Scanning {k}...")
        paired = find_paired_samples(cfg["images"], cfg["labels"])
        print(f"    Found {len(paired):,} paired samples")
        
        if "coco" in k:
            source_groups["COCO"].extend([(p, cfg["prefix"]) for p in paired])
        else:
            source_groups["CrowdHuman"].extend([(p, cfg["prefix"]) for p in paired])

    overall_stats = {"train": 0, "val": 0, "test": 0}

    for group_name, items in source_groups.items():
        if not items:
            continue
        
        paired_list = [item[0] for item in items]
        prefix = items[0][1]
        
        print(f"\n  Splitting {group_name} ({len(paired_list):,} total samples)")
        splits = split_data(paired_list, seed=RANDOM_SEED)
        
        for split_name, pairs in splits.items():
            print(f"    {split_name:5s}: {len(pairs):,} samples")
            group_stats = {}
            copy_files(pairs, prefix, split_name, dataset_dir, group_stats)
            overall_stats[split_name] += group_stats.get(split_name, 0)

    generate_data_yaml(args.source, dataset_dir)

    print_header("Summary")
    total = sum(overall_stats.values())
    if total == 0:
        print("  ⚠ No samples processed!")
    else:
        for split_name in ["train", "val", "test"]:
            count = overall_stats[split_name]
            pct = (count / total * 100)
            img_count = count_dir_files(dataset_dir / "images" / split_name, "*.jpg") + count_dir_files(dataset_dir / "images" / split_name, "*.png")
            lbl_count = count_dir_files(dataset_dir / "labels" / split_name, "*.txt")
            print(f"  {split_name:5s}: {count:>7,} samples ({pct:5.1f}%) | images: {img_count:,} labels: {lbl_count:,}")
        print(f"\n  Total dataset size: {total:,} samples")
