"""
05_sanity_check.py
==================
YOLO11m Person Detection Pipeline — Step 5: Sanity Check Visualization

Randomly samples ~30 images per source dataset from the train split,
reads the YOLO label files, de-normalizes boxes back to pixel coordinates,
draws bounding boxes on images, and saves annotated images to sanity_checks/.

This is a NON-NEGOTIABLE step to catch conversion bugs before training.

Usage:
    python scripts/05_sanity_check.py [--samples 30] [--split train]
"""

import argparse
import random
import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "sanity_checks"

RANDOM_SEED = 42

# Color scheme per source (BGR format for OpenCV)
SOURCE_COLORS = {
    "coco_": (0, 200, 80),       # Green for COCO
    "ch_": (255, 100, 0),        # Blue-ish for CrowdHuman
    "unknown": (0, 0, 255),      # Red for unknown
}

# Box drawing settings
BOX_THICKNESS = 2
FONT_SCALE = 0.5
FONT_THICKNESS = 1


def print_header(msg: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {msg}")
    print(f"{'=' * width}")


def get_source_prefix(filename: str) -> str:
    """Determine the source prefix from a filename."""
    for prefix in ["coco_", "ch_"]:
        if filename.startswith(prefix):
            return prefix
    return "unknown"


def get_source_name(prefix: str) -> str:
    """Get human-readable source name from prefix."""
    names = {
        "coco_": "COCO",
        "ch_": "CrowdHuman",
        "unknown": "Unknown",
    }
    return names.get(prefix, "Unknown")


def read_yolo_labels(label_path: Path) -> list:
    """
    Read a YOLO label file.
    Returns list of (class_id, x_center, y_center, width, height) tuples.
    """
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:])
                boxes.append((cls_id, x_c, y_c, w, h))
    return boxes


def draw_boxes(image: np.ndarray, boxes: list, source_prefix: str) -> np.ndarray:
    """
    Draw YOLO boxes on an image.
    De-normalizes from YOLO format to pixel coordinates.
    """
    img_h, img_w = image.shape[:2]
    color = SOURCE_COLORS.get(source_prefix, SOURCE_COLORS["unknown"])
    source_name = get_source_name(source_prefix)

    annotated = image.copy()

    for cls_id, x_c, y_c, w, h in boxes:
        # De-normalize to pixel coordinates
        x_center_px = x_c * img_w
        y_center_px = y_c * img_h
        w_px = w * img_w
        h_px = h * img_h

        # Convert center format to corner format
        x1 = int(x_center_px - w_px / 2)
        y1 = int(y_center_px - h_px / 2)
        x2 = int(x_center_px + w_px / 2)
        y2 = int(y_center_px + h_px / 2)

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Draw label
        label = f"person ({source_name})"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
        )
        cv2.rectangle(annotated, (x1, y1 - label_h - baseline - 4), (x1 + label_w, y1), color, -1)
        cv2.putText(
            annotated, label, (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS
        )

    # Add summary text in top-left corner
    summary = f"Source: {source_name} | Boxes: {len(boxes)} | Size: {img_w}x{img_h}"
    cv2.putText(annotated, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return annotated


def run_sanity_check(split: str = "train", samples_per_source: int = 30) -> None:
    """Run the sanity check visualization."""
    images_dir = DATASET_DIR / "images" / split
    labels_dir = DATASET_DIR / "labels" / split

    if not images_dir.exists():
        print(f"  ✗ Images directory not found: {images_dir}")
        return

    # Group images by source
    images_by_source = {}
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            prefix = get_source_prefix(img_path.name)
            if prefix not in images_by_source:
                images_by_source[prefix] = []
            images_by_source[prefix].append(img_path)

    if not images_by_source:
        print(f"  ✗ No images found in {images_dir}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_visualized = 0
    total_boxes = 0

    for prefix, img_paths in images_by_source.items():
        source_name = get_source_name(prefix)
        print(f"\n  --- {source_name} ({len(img_paths):,} images available) ---")

        # Random sample
        random.seed(RANDOM_SEED)
        sample_count = min(samples_per_source, len(img_paths))
        sampled = random.sample(img_paths, sample_count)

        source_output = OUTPUT_DIR / source_name.lower()
        source_output.mkdir(parents=True, exist_ok=True)

        for img_path in sampled:
            # Find corresponding label file
            label_stem = img_path.stem
            label_path = labels_dir / f"{label_stem}.txt"

            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"    ⚠ Could not read: {img_path.name}")
                continue

            # Read labels
            boxes = read_yolo_labels(label_path)

            # Draw boxes
            annotated = draw_boxes(image, boxes, prefix)

            # Save
            out_path = source_output / f"check_{img_path.name}"
            cv2.imwrite(str(out_path), annotated)
            total_visualized += 1
            total_boxes += len(boxes)

        print(f"  ✓ {sample_count} images visualized → {source_output}")

    return total_visualized, total_boxes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check visualization for YOLO labels")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples per source (default: 30)")
    parser.add_argument("--split", type=str, default="train", help="Which split to visualize (default: train)")
    args = parser.parse_args()

    print_header("YOLO11m Person Detection — Sanity Check")
    print(f"  Split: {args.split}")
    print(f"  Samples per source: {args.samples}")
    print(f"  Output: {OUTPUT_DIR}")

    result = run_sanity_check(split=args.split, samples_per_source=args.samples)

    if result:
        total_vis, total_boxes = result
        print_header("Sanity Check Complete")
        print(f"  Total images visualized : {total_vis}")
        print(f"  Total boxes drawn       : {total_boxes}")
        print(f"  Output directory        : {OUTPUT_DIR}")
        print()
        print("  ⚠ REVIEW these images manually before proceeding to training!")
        print("    Check that boxes are:")
        print("    • Correctly placed on people (not offset or inverted)")
        print("    • Properly scaled (not too large or too small)")
        print("    • Present for each visible person in the image")
        print()
        print("  Next step: python scripts/06_train.py")
        print()
