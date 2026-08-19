"""
10_video_latency_benchmark.py
=============================
YOLO11m Person Detection — Video Latency & Accuracy Benchmark

Processes 4 real-world videos frame-by-frame and measures:
  1. Latency: Decode, Pre-process, Inference, Post-process (NMS) per frame
  2. Accuracy: mAP, Precision, Recall on manually annotated frames
  3. Throughput: Processing FPS vs. Video Native FPS (real-time margin)

Usage:
    python scripts/10_video_latency_benchmark.py
    python scripts/10_video_latency_benchmark.py --model path/to/best.pt
"""

import argparse
import json
import time
import cv2
import numpy as np
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_ROOT / "videos"
ANNOTATIONS_DIR = VIDEOS_DIR / "annotations"
OUTPUT_DIR = PROJECT_ROOT / "config"

DEFAULT_MODEL = PROJECT_ROOT / "innovision_person_detection" / "yolo11m_combined_v1" / "weights" / "best.pt"

# Video → Annotation folder mapping
VIDEO_MAP = {
    "WhatsApp Video 2026-08-19 at 12.30.47 PM.mp4": "factory",
    "WhatsApp Video 2026-08-19 at 12.30.48 PM.mp4": "night",
    "WhatsApp Video 2026-08-19 at 12.30.48 PM(1).mp4": "office",
    "WhatsApp Video 2026-08-19 at 12.30.49 PM.mp4": "pedestrian",
}

# IoU threshold for matching predictions to ground truth
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25


def print_header(msg: str) -> None:
    width = 65
    print(f"\n{'=' * width}")
    print(f"  {msg}")
    print(f"{'=' * width}")


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Convert YOLO normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def load_gt_labels(label_path, img_w, img_h):
    """Load YOLO-format ground truth labels and convert to [x1, y1, x2, y2]."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                if cls_id == 0:  # person only
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    boxes.append(yolo_to_xyxy(cx, cy, w, h, img_w, img_h))
    return boxes


def compute_frame_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Compute TP, FP, FN for a single frame.
    pred_boxes and gt_boxes are lists of [x1, y1, x2, y2].
    """
    matched_gt = set()
    tp = 0
    fp = 0

    for pred in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def benchmark_video(model, video_path: Path, scene_name: str, imgsz: int = 960) -> dict:
    """
    Process every frame of a video. Measure latency for all frames,
    and accuracy only for frames that have ground truth annotations.
    """
    ann_dir = ANNOTATIONS_DIR / scene_name / "labels" / "train"

    # Get list of annotated frame indices
    annotated_frames = set()
    if ann_dir.exists():
        for f in ann_dir.iterdir():
            if f.suffix == ".txt":
                # Extract frame number from filename like "frame_0026.txt"
                try:
                    frame_num = int(f.stem.split("_")[1])
                    annotated_frames.add(frame_num)
                except (IndexError, ValueError):
                    pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ✗ Cannot open video: {video_path}")
        return {"error": "Cannot open video"}

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video     : {video_path.name}")
    print(f"  Scene     : {scene_name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Native FPS: {native_fps:.1f}")
    print(f"  Frames    : {total_frames}")
    print(f"  Annotated : {len(annotated_frames)} frames")
    print(f"  Processing...")

    # Latency tracking
    decode_times = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    total_times = []

    # Accuracy tracking
    total_tp = 0
    total_fp = 0
    total_fn = 0
    evaluated_frames = 0

    frame_idx = 0
    while True:
        # --- Decode ---
        t_decode_start = time.perf_counter()
        ret, frame = cap.read()
        t_decode_end = time.perf_counter()

        if not ret:
            break

        decode_ms = (t_decode_end - t_decode_start) * 1000
        decode_times.append(decode_ms)

        # --- Inference (Pre + Model + Post) ---
        results = model.predict(frame, imgsz=imgsz, conf=CONF_THRESHOLD, verbose=False, device=0)

        if results and hasattr(results[0], 'speed'):
            speed = results[0].speed
            pre_ms = speed.get('preprocess', 0.0)
            inf_ms = speed.get('inference', 0.0)
            post_ms = speed.get('postprocess', 0.0)
        else:
            pre_ms, inf_ms, post_ms = 0.0, 0.0, 0.0

        preprocess_times.append(pre_ms)
        inference_times.append(inf_ms)
        postprocess_times.append(post_ms)
        total_times.append(decode_ms + pre_ms + inf_ms + post_ms)

        # --- Accuracy (only for annotated frames) ---
        if frame_idx in annotated_frames:
            label_path = ann_dir / f"frame_{frame_idx:04d}.txt"
            gt_boxes = load_gt_labels(label_path, width, height)

            # Extract prediction boxes
            pred_boxes = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    pred_boxes.append(xyxy)

            tp, fp, fn = compute_frame_metrics(pred_boxes, gt_boxes, IOU_THRESHOLD)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            evaluated_frames += 1

        frame_idx += 1

    cap.release()

    # --- Compute statistics ---
    total_arr = np.array(total_times)
    decode_arr = np.array(decode_times)
    pre_arr = np.array(preprocess_times)
    inf_arr = np.array(inference_times)
    post_arr = np.array(postprocess_times)

    mean_total = float(np.mean(total_arr))
    processing_fps = 1000.0 / mean_total if mean_total > 0 else 0
    realtime_margin = processing_fps / native_fps if native_fps > 0 else 0

    # Accuracy
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "scene": scene_name,
        "video": video_path.name,
        "resolution": f"{width}x{height}",
        "native_fps": float(native_fps),
        "total_frames": total_frames,
        "annotated_frames": evaluated_frames,
        "latency": {
            "decode_mean_ms": float(np.mean(decode_arr)),
            "preprocess_mean_ms": float(np.mean(pre_arr)),
            "inference_mean_ms": float(np.mean(inf_arr)),
            "postprocess_mean_ms": float(np.mean(post_arr)),
            "total_mean_ms": mean_total,
            "total_std_ms": float(np.std(total_arr)),
            "total_min_ms": float(np.min(total_arr)),
            "total_max_ms": float(np.max(total_arr)),
            "total_p95_ms": float(np.percentile(total_arr, 95)),
            "processing_fps": processing_fps,
            "realtime_margin": f"{realtime_margin:.1f}x",
        },
        "accuracy": {
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        },
    }

    # Print results
    lat = metrics["latency"]
    acc = metrics["accuracy"]
    print(f"\n  --- Latency ---")
    print(f"  Decode      : {lat['decode_mean_ms']:.2f} ms")
    print(f"  Pre-process : {lat['preprocess_mean_ms']:.2f} ms")
    print(f"  Inference   : {lat['inference_mean_ms']:.2f} ms")
    print(f"  Post-process: {lat['postprocess_mean_ms']:.2f} ms")
    print(f"  Total       : {lat['total_mean_ms']:.2f} ms ± {lat['total_std_ms']:.2f} ms")
    print(f"  P95 Latency : {lat['total_p95_ms']:.2f} ms")
    print(f"  Processing  : {lat['processing_fps']:.1f} FPS")
    print(f"  Real-time   : {lat['realtime_margin']} faster than native")

    print(f"\n  --- Accuracy (on {evaluated_frames} annotated frames) ---")
    print(f"  True Positives  : {acc['TP']}")
    print(f"  False Positives : {acc['FP']}")
    print(f"  False Negatives : {acc['FN']}")
    print(f"  Precision       : {acc['precision']:.4f}")
    print(f"  Recall          : {acc['recall']:.4f}")
    print(f"  F1 Score        : {acc['f1_score']:.4f}")

    return metrics


def print_summary_tables(all_results: dict) -> None:
    """Print final comparison tables."""
    print_header("Latency Summary (All Videos)")
    print(f"  {'Scene':<14} {'Resolution':<12} {'Decode':>8} {'Pre':>8} {'Infer':>8} {'NMS':>8} {'Total':>8} {'FPS':>8} {'Margin':>8}")
    print(f"  {'-'*90}")
    for name, res in all_results.items():
        if "error" in res:
            continue
        lat = res["latency"]
        print(f"  {res['scene']:<14} {res['resolution']:<12} "
              f"{lat['decode_mean_ms']:>8.2f} {lat['preprocess_mean_ms']:>8.2f} "
              f"{lat['inference_mean_ms']:>8.2f} {lat['postprocess_mean_ms']:>8.2f} "
              f"{lat['total_mean_ms']:>8.2f} {lat['processing_fps']:>8.1f} {lat['realtime_margin']:>8}")

    print_header("Accuracy Summary (Annotated Frames Only)")
    print(f"  {'Scene':<14} {'Frames':>8} {'TP':>8} {'FP':>8} {'FN':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*82}")
    for name, res in all_results.items():
        if "error" in res:
            continue
        acc = res["accuracy"]
        print(f"  {res['scene']:<14} {res['annotated_frames']:>8} "
              f"{acc['TP']:>8} {acc['FP']:>8} {acc['FN']:>8} "
              f"{acc['precision']:>10.4f} {acc['recall']:>10.4f} {acc['f1_score']:>10.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11m Video Latency & Accuracy Benchmark")
    parser.add_argument("--model", type=str, default=None, help=f"Path to model (default: combined best.pt)")
    parser.add_argument("--imgsz", type=int, default=960, help="Image size (default: 960)")
    args = parser.parse_args()

    from ultralytics import YOLO

    model_path = Path(args.model) if args.model else DEFAULT_MODEL
    if not model_path.exists():
        print(f"  ✗ Model not found: {model_path}")
        exit(1)

    print_header("YOLO11m Video Latency & Accuracy Benchmark")
    print(f"  Model : {model_path}")
    print(f"  ImgSz : {args.imgsz}")

    # Load model once
    print(f"\n  → Loading model...")
    model = YOLO(str(model_path))

    # Warmup on first frame
    print(f"  → Warmup (50 iterations on dummy input)...")
    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    for _ in range(50):
        model.predict(dummy, imgsz=args.imgsz, verbose=False, device=0)
    torch.cuda.synchronize()
    print(f"  ✓ Warmup complete")

    all_results = {}
    for video_name, scene_name in VIDEO_MAP.items():
        video_path = VIDEOS_DIR / video_name
        if not video_path.exists():
            print(f"\n  ⚠ Video not found: {video_path}")
            continue
        print_header(f"Processing: {scene_name.upper()}")
        all_results[scene_name] = benchmark_video(model, video_path, scene_name, imgsz=args.imgsz)

    # Summary tables
    print_summary_tables(all_results)

    # Save JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "video_latency_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  ✓ Results saved to: {json_path}\n")
