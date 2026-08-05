"""
02_convert_coco.py
==================
YOLO11m Person Detection Pipeline — Step 2: Convert COCO Annotations

Loads COCO annotations via pycocotools, filters to person class only
(category_id=1), converts to YOLO normalized format, and writes one
.txt label file per image.

Key behaviors:
    - Filters to category_id == 1 (person) only — drops all 79 other categories
    - Skips annotations with iscrowd=1 (crowd regions)
    - Converts [x_min, y_min, width, height] → YOLO [x_center, y_center, w, h] (0-1)
    - Clamps all coordinates to [0.0, 1.0]
    - Writes labels to raw/coco/labels_yolo/ staging area

Usage:
    python scripts/02_convert_coco.py
"""

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
COCO_DIR = PROJECT_ROOT / "raw" / "coco"
LABELS_OUT = COCO_DIR / "labels_yolo"

# COCO person category ID
PERSON_CATEGORY_ID = 1
# YOLO class index for person
YOLO_CLASS_ID = 0

# Annotation files to process
ANNOTATION_FILES = {
    "train": COCO_DIR / "annotations" / "instances_train2017.json",
    "val": COCO_DIR / "annotations" / "instances_val2017.json",
}


def print_header(msg: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {msg}")
    print(f"{'=' * width}")


def clamp(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a value to [min_val, max_val]."""
    return max(min_val, min(val, max_val))


def convert_coco_split(ann_file: Path, split_name: str) -> dict:
    """
    Convert a single COCO annotation file to YOLO format.

    Returns stats dict with counts.
    """
    print(f"\n  Processing: {ann_file.name} ({split_name})")

    if not ann_file.exists():
        print(f"  ✗ Annotation file not found: {ann_file}")
        return {"images": 0, "boxes": 0, "skipped_crowd": 0, "skipped_other": 0}

    # Load COCO annotations
    print(f"  → Loading annotations (this may take a moment for large files)...")
    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    # Build image ID → image info lookup
    image_info = {}
    for img in coco_data["images"]:
        image_info[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        }

    # Group person annotations by image
    print(f"  → Filtering person annotations...")
    person_boxes_by_image = {}
    stats = {
        "total_annotations": 0,
        "person_boxes": 0,
        "skipped_crowd": 0,
        "skipped_other_class": 0,
        "skipped_invalid": 0,
    }

    for ann in coco_data["annotations"]:
        stats["total_annotations"] += 1

        # Skip non-person categories
        if ann["category_id"] != PERSON_CATEGORY_ID:
            stats["skipped_other_class"] += 1
            continue

        # Skip crowd annotations (iscrowd=1)
        if ann.get("iscrowd", 0) == 1:
            stats["skipped_crowd"] += 1
            continue

        img_id = ann["image_id"]
        if img_id not in image_info:
            stats["skipped_invalid"] += 1
            continue

        bbox = ann["bbox"]  # [x_min, y_min, width, height] in pixels
        if bbox[2] <= 0 or bbox[3] <= 0:
            stats["skipped_invalid"] += 1
            continue

        if img_id not in person_boxes_by_image:
            person_boxes_by_image[img_id] = []
        person_boxes_by_image[img_id].append(bbox)
        stats["person_boxes"] += 1

    # Convert to YOLO format and write label files
    print(f"  → Converting to YOLO format and writing label files...")
    output_dir = LABELS_OUT / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    images_with_labels = 0
    for img_id, boxes in person_boxes_by_image.items():
        img = image_info[img_id]
        img_w = img["width"]
        img_h = img["height"]
        file_stem = Path(img["file_name"]).stem

        label_lines = []
        for bbox in boxes:
            x_min, y_min, w, h = bbox

            # Convert to YOLO normalized center format
            x_center = (x_min + w / 2.0) / img_w
            y_center = (y_min + h / 2.0) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Clamp to [0, 1]
            x_center = clamp(x_center)
            y_center = clamp(y_center)
            w_norm = clamp(w_norm)
            h_norm = clamp(h_norm)

            # Skip degenerate boxes after clamping
            if w_norm <= 0.001 or h_norm <= 0.001:
                continue

            label_lines.append(f"{YOLO_CLASS_ID} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        if label_lines:
            label_file = output_dir / f"{file_stem}.txt"
            with open(label_file, "w") as f:
                f.write("\n".join(label_lines) + "\n")
            images_with_labels += 1

    stats["images_with_labels"] = images_with_labels
    return stats


def print_stats(split_name: str, stats: dict) -> None:
    """Print conversion statistics for a split."""
    print(f"\n  --- {split_name} Conversion Stats ---")
    print(f"  Total annotations in file : {stats.get('total_annotations', 0):,}")
    print(f"  Person boxes kept         : {stats.get('person_boxes', 0):,}")
    print(f"  Skipped (other classes)   : {stats.get('skipped_other_class', 0):,}")
    print(f"  Skipped (iscrowd=1)       : {stats.get('skipped_crowd', 0):,}")
    print(f"  Skipped (invalid)         : {stats.get('skipped_invalid', 0):,}")
    print(f"  Images with labels        : {stats.get('images_with_labels', 0):,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print_header("YOLO11m Person Detection — COCO Conversion")
    print(f"  Source: {COCO_DIR / 'annotations'}")
    print(f"  Output: {LABELS_OUT}")

    total_stats = {
        "total_annotations": 0,
        "person_boxes": 0,
        "skipped_crowd": 0,
        "skipped_other_class": 0,
        "skipped_invalid": 0,
        "images_with_labels": 0,
    }

    for split_name, ann_file in ANNOTATION_FILES.items():
        stats = convert_coco_split(ann_file, split_name)
        print_stats(split_name, stats)
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)

    print_header("COCO Conversion Complete — Totals")
    print(f"  Total person boxes   : {total_stats['person_boxes']:,}")
    print(f"  Total labeled images : {total_stats['images_with_labels']:,}")
    print(f"  Labels written to    : {LABELS_OUT}")
    print()
    print("  Next step: python scripts/03_convert_crowdhuman.py")
    print()
