"""
01_download_datasets.py
=======================
YOLO11m Person Detection Pipeline — Step 1: Download Datasets

Downloads COCO (auto-download) and CrowdHuman (manual-first, gdown fallback)
datasets into the raw/ directory.

COCO:
    - Auto-downloads train2017.zip, val2017.zip, and annotation JSONs
      from the official COCO mirror.

CrowdHuman:
    - Expects manual placement of files in raw/crowdhuman/ FIRST.
    - Falls back to gdown (Google Drive) if files are missing.
    - Required files:
        * CrowdHuman_train01.zip, CrowdHuman_train02.zip, CrowdHuman_train03.zip
        * CrowdHuman_val.zip
        * annotation_train.odgt, annotation_val.odgt

All downloads are idempotent — existing files are detected and skipped.

Usage:
    python scripts/01_download_datasets.py
"""

import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
COCO_DIR = RAW_DIR / "coco"
CH_DIR = RAW_DIR / "crowdhuman"

# COCO download URLs (official mirror)
COCO_URLS = {
    "images/train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "images/val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations/annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

# CrowdHuman Google Drive file IDs (for gdown fallback)
# These are the official CrowdHuman download IDs
CROWDHUMAN_GDRIVE = {
    "CrowdHuman_train01.zip": "134QOvaatwKdy0iIeNqA_p-xkAhkV4F8Y",
    "CrowdHuman_train02.zip": "17evzPh7gc1JBNvnW1ENXLy2izQ1Byebh",
    "CrowdHuman_train03.zip": "1tdp0UCgxrqy1B6p8LkR-4wFP-cKFQUpc",
    "CrowdHuman_val.zip": "18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO",
    "annotation_train.odgt": "1UUTea5mYqvlUObsC1Z8CFldHJAtLtMX3",
    "annotation_val.odgt": "10WIRwu8ju8GRLuCkZ_vT6hnNxs5ptwoL",
}


class DownloadProgressBar:
    """Simple progress indicator for urlretrieve."""

    def __init__(self, filename: str):
        self.filename = filename
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if total_size > 0:
            percent = int(block_num * block_size * 100 / total_size)
            percent = min(percent, 100)
            if percent != self.last_percent and percent % 5 == 0:
                size_gb = total_size / (1024**3)
                print(f"    {self.filename}: {percent}% of {size_gb:.1f} GB", end="\r")
                self.last_percent = percent
        else:
            downloaded_mb = block_num * block_size / (1024**2)
            if block_num % 100 == 0:
                print(f"    {self.filename}: {downloaded_mb:.0f} MB downloaded", end="\r")


def print_header(msg: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {msg}")
    print(f"{'=' * width}")


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file from URL if it doesn't exist. Returns True if downloaded."""
    if dest.exists():
        print(f"  ✓ Already exists: {dest.name}")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    display_name = desc or dest.name
    print(f"  → Downloading {display_name}...")
    try:
        progress = DownloadProgressBar(dest.name)
        urlretrieve(url, str(dest), reporthook=progress)
        print(f"\n  ✓ Downloaded: {dest.name}")
        return True
    except Exception as e:
        print(f"\n  ✗ Failed to download {display_name}: {e}")
        if dest.exists():
            dest.unlink()  # Clean up partial download
        return False


def extract_zip(zip_path: Path, extract_to: Path, desc: str = "") -> None:
    """Extract a zip file if the target directory suggests it hasn't been extracted."""
    display_name = desc or zip_path.name
    print(f"  → Extracting {display_name}...")
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(extract_to))
        print(f"  ✓ Extracted: {display_name}")
    except zipfile.BadZipFile:
        print(f"  ✗ Corrupt zip file: {zip_path}")
        print(f"    Delete it and re-run this script to re-download.")


def download_coco() -> None:
    """Download COCO dataset (auto-download with skip-if-exists)."""
    print_header("Downloading COCO Dataset")

    # Download images and annotations
    for rel_path, url in COCO_URLS.items():
        dest = COCO_DIR / rel_path
        download_file(url, dest, rel_path)

    # Extract images if not already extracted
    for split in ["train2017", "val2017"]:
        zip_path = COCO_DIR / f"images/{split}.zip"
        extracted_dir = COCO_DIR / f"images/{split}"
        if zip_path.exists() and not extracted_dir.exists():
            extract_zip(zip_path, COCO_DIR / "images", split)
        elif extracted_dir.exists():
            num_images = len(list(extracted_dir.glob("*.jpg")))
            print(f"  ✓ {split}/ already extracted ({num_images:,} images)")

    # Extract annotations
    ann_zip = COCO_DIR / "annotations/annotations_trainval2017.zip"
    ann_check = COCO_DIR / "annotations/instances_train2017.json"
    if ann_zip.exists() and not ann_check.exists():
        extract_zip(ann_zip, COCO_DIR, "annotations")
        # The COCO zip extracts into an 'annotations/' subfolder inside the target.
        # Move files if they ended up nested
        nested_dir = COCO_DIR / "annotations" / "annotations"
        if nested_dir.exists():
            import shutil
            for f in nested_dir.iterdir():
                target = COCO_DIR / "annotations" / f.name
                if not target.exists():
                    shutil.move(str(f), str(target))
            nested_dir.rmdir()
    elif ann_check.exists():
        print(f"  ✓ Annotations already extracted")

    # Final verification
    print("\n  --- COCO Download Summary ---")
    for split in ["train2017", "val2017"]:
        img_dir = COCO_DIR / f"images/{split}"
        if img_dir.exists():
            count = len(list(img_dir.glob("*.jpg")))
            print(f"  {split}: {count:,} images")
        else:
            print(f"  ⚠ {split}: NOT FOUND")

    for ann_file in ["instances_train2017.json", "instances_val2017.json"]:
        ann_path = COCO_DIR / f"annotations/{ann_file}"
        status = "✓ found" if ann_path.exists() else "✗ MISSING"
        print(f"  {ann_file}: {status}")


def download_crowdhuman() -> None:
    """Download CrowdHuman (manual-first, gdown fallback, skip-if-exists)."""
    print_header("CrowdHuman Dataset")

    # Check for manually placed files first
    all_present = True
    missing_files = []
    for filename in CROWDHUMAN_GDRIVE.keys():
        filepath = CH_DIR / filename
        # Annotations go into annotations/ subfolder
        if filename.endswith(".odgt"):
            filepath = CH_DIR / "annotations" / filename
        if filepath.exists():
            print(f"  ✓ Found (manual): {filename}")
        else:
            all_present = False
            missing_files.append(filename)

    if all_present:
        print("\n  ✓ All CrowdHuman files found (manually placed). Skipping download.")
    else:
        print(f"\n  ⚠ {len(missing_files)} file(s) missing — attempting gdown fallback:")
        for f in missing_files:
            print(f"    • {f}")

        try:
            import gdown
        except ImportError:
            print("\n  ✗ gdown not installed. Install it with: pip install gdown")
            print("    Or manually download and place files in:")
            print(f"    {CH_DIR}/")
            return

        for filename in missing_files:
            file_id = CROWDHUMAN_GDRIVE[filename]
            if filename.endswith(".odgt"):
                dest = CH_DIR / "annotations" / filename
            else:
                dest = CH_DIR / filename
            dest.parent.mkdir(parents=True, exist_ok=True)

            if dest.exists():
                print(f"  ✓ Already exists: {filename}")
                continue

            print(f"  → Downloading {filename} via gdown...")
            try:
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, str(dest), quiet=False)
                if dest.exists():
                    print(f"  ✓ Downloaded: {filename}")
                else:
                    print(f"  ✗ Download failed for {filename}")
            except Exception as e:
                print(f"  ✗ gdown failed for {filename}: {e}")
                print(f"    Please download manually and place in: {dest}")

    # Extract CrowdHuman zips
    print_header("Extracting CrowdHuman Images")
    images_dir = CH_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    ch_zips = [
        "CrowdHuman_train01.zip",
        "CrowdHuman_train02.zip",
        "CrowdHuman_train03.zip",
        "CrowdHuman_val.zip",
    ]

    for zip_name in ch_zips:
        zip_path = CH_DIR / zip_name
        if not zip_path.exists():
            print(f"  ⚠ {zip_name} not found — skipping extraction")
            continue

        # Use a marker file to track whether this specific zip was already extracted
        marker = CH_DIR / f".extracted_{zip_name}"
        if marker.exists():
            print(f"  ✓ {zip_name} already extracted — skipping")
            continue

        print(f"  → Extracting {zip_name}...")
        try:
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(CH_DIR))
            marker.touch()  # Mark this zip as extracted
            print(f"  ✓ Extracted: {zip_name}")
        except zipfile.BadZipFile:
            print(f"  ✗ Corrupt zip: {zip_name} — delete and re-download")

    # CrowdHuman zips extract into an "Images/" folder — move to images/
    nested_images = CH_DIR / "Images"
    if nested_images.exists() and nested_images != images_dir:
        import shutil
        print("  → Moving extracted images to images/ ...")
        for img_file in nested_images.iterdir():
            target = images_dir / img_file.name
            if not target.exists():
                shutil.move(str(img_file), str(target))
        if not any(nested_images.iterdir()):
            nested_images.rmdir()
        print("  ✓ Images moved to images/")

    # Final verification
    print("\n  --- CrowdHuman Download Summary ---")
    if images_dir.exists():
        jpg_count = len(list(images_dir.glob("*.jpg")))
        print(f"  Images: {jpg_count:,} .jpg files")
    else:
        print("  ⚠ Images directory not found")

    for ann_file in ["annotation_train.odgt", "annotation_val.odgt"]:
        ann_path = CH_DIR / "annotations" / ann_file
        status = "✓ found" if ann_path.exists() else "✗ MISSING"
        print(f"  {ann_file}: {status}")


def print_summary() -> None:
    """Print final summary."""
    print_header("Download Step Complete")
    print(f"  COCO directory      : {COCO_DIR}")
    print(f"  CrowdHuman directory: {CH_DIR}")
    print()
    print("  Next step: python scripts/02_convert_coco.py")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print_header("YOLO11m Person Detection — Dataset Download")

    download_coco()
    download_crowdhuman()
    print_summary()
