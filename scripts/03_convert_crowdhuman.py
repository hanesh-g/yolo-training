"""
03_convert_crowdhuman.py
========================
YOLO11m Person Detection Pipeline — Step 3: Convert CrowdHuman Annotations

Parses CrowdHuman .odgt annotation files (JSON-lines format), extracts
fbox (full-body box) ONLY, converts to YOLO normalized format, and writes
one .txt label file per image.

Key behaviors:
    - Extracts fbox only — ignores vbox and hbox entirely
    - Skips annotations tagged with "extra": {"ignore": 1}
    - Converts [x_min, y_min, width, height] → YOLO [x_center, y_center, w, h] (0-1)
    - Clamps all coordinates to [0.0, 1.0]
    - Needs image dimensions — reads from actual image files
    - Writes labels to raw/crowdhuman/labels_yolo/ staging area

Usage:
    python scripts/03_convert_crowdhuman.py
"""

import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CH_DIR = PROJECT_ROOT / "raw" / "crowdhuman"
LABELS_OUT = CH_DIR / "labels_yolo"
IMAGES_DIR = CH_DIR / "images"

# YOLO class index for person
YOLO_CLASS_ID = 0

# Annotation files
ANNOTATION_FILES = {
    "train": CH_DIR / "annotations" / "annotation_train.odgt",
    "val": CH_DIR / "annotations" / "annotation_val.odgt",
}


def print_header(msg: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {msg}")
    print(f"{'=' * width}")


def clamp(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a value to [min_val, max_val]."""
    return max(min_val, min(val, max_val))


def get_image_dimensions(image_id: str) -> tuple:
    """
    Get image dimensions by reading the actual image file.
    CrowdHuman images are named by their ID with .jpg extension.
    Returns (width, height) or None if not found.
    """
    for ext in [".jpg", ".jpeg", ".png"]:
        img_path = IMAGES_DIR / f"{image_id}{ext}"
        if img_path.exists():
            with Image.open(img_path) as img:
                return img.size  # (width, height)
    return None


def parse_odgt_file(odgt_path: Path) -> list:
    """
    Parse a CrowdHuman .odgt file (JSON-lines format).
    Returns a list of dicts, one per image.
    """
    entries = []
    with open(odgt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def convert_crowdhuman_split(ann_file: Path, split_name: str) -> dict:
    """
    Convert a single CrowdHuman .odgt file to YOLO format.
    Extracts fbox only.

    Returns stats dict.
    """
    print(f"\n  Processing: {ann_file.name} ({split_name})")

    if not ann_file.exists():
        print(f"  ✗ Annotation file not found: {ann_file}")
        return {"images": 0, "fbox_count": 0, "skipped_ignore": 0, "skipped_no_image": 0}

    # Parse .odgt file
    print(f"  → Parsing .odgt file...")
    entries = parse_odgt_file(ann_file)
    print(f"  → Found {len(entries):,} image entries")

    # Output directory
    output_dir = LABELS_OUT / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_entries": len(entries),
        "total_instances": 0,
        "fbox_count": 0,
        "skipped_ignore": 0,
        "skipped_no_fbox": 0,
        "skipped_no_image": 0,
        "skipped_invalid": 0,
        "images_with_labels": 0,
    }

    # Cache for image dimensions to avoid re-reading
    print(f"  → Converting fbox annotations to YOLO format...")

    for entry in tqdm(entries, desc=f"  {split_name}", unit="img"):
        image_id = entry["ID"]
        gt_boxes = entry.get("gtboxes", [])

        # Get image dimensions
        dims = get_image_dimensions(image_id)
        if dims is None:
            stats["skipped_no_image"] += 1
            continue

        img_w, img_h = dims
        label_lines = []

        for gt in gt_boxes:
            stats["total_instances"] += 1

            # Skip ignore regions
            extra = gt.get("extra", {})
            if extra.get("ignore", 0) == 1:
                stats["skipped_ignore"] += 1
                continue

            # Only process "person" tag (skip "mask" or other tags if present)
            if gt.get("tag", "person") != "person":
                stats["skipped_ignore"] += 1
                continue

            # Extract fbox — [x_min, y_min, width, height] in pixels
            fbox = gt.get("fbox")
            if fbox is None:
                stats["skipped_no_fbox"] += 1
                continue

            x_min, y_min, w, h = fbox

            # Skip degenerate boxes
            if w <= 0 or h <= 0:
                stats["skipped_invalid"] += 1
                continue

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
                stats["skipped_invalid"] += 1
                continue

            label_lines.append(f"{YOLO_CLASS_ID} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            stats["fbox_count"] += 1

        if label_lines:
            label_file = output_dir / f"{image_id}.txt"
            with open(label_file, "w") as f:
                f.write("\n".join(label_lines) + "\n")
            stats["images_with_labels"] += 1

    return stats


def print_stats(split_name: str, stats: dict) -> None:
    """Print conversion statistics for a split."""
    print(f"\n  --- {split_name} Conversion Stats ---")
    print(f"  Image entries in .odgt    : {stats.get('total_entries', 0):,}")
    print(f"  Total person instances    : {stats.get('total_instances', 0):,}")
    print(f"  fbox annotations kept     : {stats.get('fbox_count', 0):,}")
    print(f"  Skipped (ignore regions)  : {stats.get('skipped_ignore', 0):,}")
    print(f"  Skipped (no fbox field)   : {stats.get('skipped_no_fbox', 0):,}")
    print(f"  Skipped (image not found) : {stats.get('skipped_no_image', 0):,}")
    print(f"  Skipped (invalid/degen.)  : {stats.get('skipped_invalid', 0):,}")
    print(f"  Images with labels        : {stats.get('images_with_labels', 0):,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print_header("YOLO11m Person Detection — CrowdHuman Conversion (fbox only)")
    print(f"  Source annotations : {CH_DIR / 'annotations'}")
    print(f"  Source images      : {IMAGES_DIR}")
    print(f"  Output labels      : {LABELS_OUT}")

    # Verify images directory exists
    if not IMAGES_DIR.exists() or not any(IMAGES_DIR.iterdir()):
        print("\n  ✗ No images found in raw/crowdhuman/images/")
        print("    Run 01_download_datasets.py first, or place images manually.")
        exit(1)

    total_stats = {
        "total_entries": 0,
        "total_instances": 0,
        "fbox_count": 0,
        "skipped_ignore": 0,
        "skipped_no_fbox": 0,
        "skipped_no_image": 0,
        "skipped_invalid": 0,
        "images_with_labels": 0,
    }

    for split_name, ann_file in ANNOTATION_FILES.items():
        stats = convert_crowdhuman_split(ann_file, split_name)
        print_stats(split_name, stats)
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)

    print_header("CrowdHuman Conversion Complete — Totals")
    print(f"  Total fbox annotations : {total_stats['fbox_count']:,}")
    print(f"  Total labeled images   : {total_stats['images_with_labels']:,}")
    print(f"  Labels written to      : {LABELS_OUT}")
    print()
    print("  Next step: python scripts/04_split_and_organize.py")
    print()
