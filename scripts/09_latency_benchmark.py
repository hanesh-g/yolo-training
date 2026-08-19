"""
09_latency_benchmark.py
=======================
YOLO11m Person Detection Pipeline — Step 9: Rigorous GPU Latency Benchmark

Measures accurate latency breakdown from First Principles:
    1. Warmup runs (100 iterations) to stabilize GPU P-state and CUDA kernels
    2. CUDA Event timing (torch.cuda.Event) for millisecond-level precision
    3. Latency stage breakdown: Pre-process, Model Forward Pass, Post-process (NMS)
    4. Statistical metrics: Mean, Std Dev, Min, Max, 95th Percentile (P95), FPS

Usage:
    python scripts/09_latency_benchmark.py
    python scripts/09_latency_benchmark.py --imgsz 960 --runs 500
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "config"

MODELS_TO_BENCHMARK = {
    "CrowdHuman Model": PROJECT_ROOT / "innovision_person_detection" / "yolo11m_crowdhuman_v1" / "weights" / "best.pt",
    "COCO Model": PROJECT_ROOT / "innovision_person_detection" / "yolo11m_coco_v1-4" / "weights" / "best.pt",
    "Combined Model": PROJECT_ROOT / "innovision_person_detection" / "yolo11m_combined_v1" / "weights" / "best.pt",
}

def print_header(msg: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {msg}")
    print(f"{'=' * width}")

def benchmark_model(model_path: Path, imgsz: int = 960, warmup: int = 100, num_runs: int = 500) -> dict:
    """
    Measures GPU latency of a YOLO model using CUDA Events for millisecond precision.
    """
    from ultralytics import YOLO

    if not model_path.exists():
        print(f"  ✗ Model weights not found: {model_path}")
        return {"error": f"File not found: {model_path}"}

    print(f"\n  → Loading model: {model_path.name}")
    model = YOLO(str(model_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  → Target Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Generate dummy input tensor (1, 3, imgsz, imgsz)
    dummy_input = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32, device=device)

    # -----------------------------------------------------------------------
    # 1. Warmup Phase
    # -----------------------------------------------------------------------
    print(f"  → Running {warmup} warmup iterations (stabilizing GPU clocks & CUDA kernels)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.predict(dummy_input, verbose=False, device=0)
    torch.cuda.synchronize()

    # -----------------------------------------------------------------------
    # 2. Timing Benchmark using CUDA Events
    # -----------------------------------------------------------------------
    print(f"  → Benchmarking across {num_runs} iterations...")

    preprocess_times = []
    inference_times = []
    postprocess_times = []
    total_times = []

    # Using dummy image array for full pipeline test (Pre-process + Inference + NMS)
    dummy_img_np = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    with torch.no_grad():
        for _ in range(num_runs):
            # CUDA Events for precise hardware timing
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)

            start_evt.record()
            results = model.predict(dummy_img_np, imgsz=imgsz, verbose=False, device=0)
            end_evt.record()

            torch.cuda.synchronize()
            total_elapsed_ms = start_evt.elapsed_time(end_evt)

            # Extract detailed stage breakdown from Ultralytics result object
            if results and hasattr(results[0], 'speed'):
                speed = results[0].speed
                pre = speed.get('preprocess', 0.0)
                inf = speed.get('inference', 0.0)
                post = speed.get('postprocess', 0.0)
                preprocess_times.append(pre)
                inference_times.append(inf)
                postprocess_times.append(post)
                total_times.append(pre + inf + post)
            else:
                total_times.append(total_elapsed_ms)

    # Convert to numpy arrays for statistics
    total_arr = np.array(total_times)
    inf_arr = np.array(inference_times) if inference_times else total_arr
    pre_arr = np.array(preprocess_times) if preprocess_times else np.zeros_like(total_arr)
    post_arr = np.array(postprocess_times) if postprocess_times else np.zeros_like(total_arr)

    mean_total = float(np.mean(total_arr))
    p95_total = float(np.percentile(total_arr, 95))
    fps = 1000.0 / mean_total if mean_total > 0 else 0.0

    metrics = {
        "model_name": model_path.name,
        "imgsz": imgsz,
        "runs": num_runs,
        "preprocess_mean_ms": float(np.mean(pre_arr)),
        "inference_mean_ms": float(np.mean(inf_arr)),
        "postprocess_mean_ms": float(np.mean(post_arr)),
        "total_mean_ms": mean_total,
        "total_std_ms": float(np.std(total_arr)),
        "total_min_ms": float(np.min(total_arr)),
        "total_max_ms": float(np.max(total_arr)),
        "total_p95_ms": p95_total,
        "fps": float(fps),
    }

    print(f"  ✓ Pre-process : {metrics['preprocess_mean_ms']:.2f} ms")
    print(f"  ✓ Inference   : {metrics['inference_mean_ms']:.2f} ms")
    print(f"  ✓ Post-process: {metrics['postprocess_mean_ms']:.2f} ms")
    print(f"  ✓ Total       : {metrics['total_mean_ms']:.2f} ms ± {metrics['total_std_ms']:.2f} ms")
    print(f"  ✓ P95 Latency : {metrics['total_p95_ms']:.2f} ms")
    print(f"  ✓ Throughput  : {metrics['fps']:.1f} FPS")

    return metrics

def print_latency_table(results_dict: dict) -> None:
    print_header("Latency & Throughput Comparison Table")
    print(f"  {'Model':<20} {'Pre (ms)':>10} {'Infer (ms)':>12} {'Post (ms)':>10} {'Total (ms)':>12} {'FPS':>10}")
    print(f"  {'-'*78}")

    for name, res in results_dict.items():
        if "error" in res:
            print(f"  {name:<20} {'N/A':>10} {'N/A':>12} {'N/A':>10} {'N/A':>12} {'N/A':>10}")
        else:
            print(f"  {name:<20} "
                  f"{res['preprocess_mean_ms']:>10.2f} "
                  f"{res['inference_mean_ms']:>12.2f} "
                  f"{res['postprocess_mean_ms']:>10.2f} "
                  f"{res['total_mean_ms']:>12.2f} "
                  f"{res['fps']:>10.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11m Latency Benchmark")
    parser.add_argument("--imgsz", type=int, default=960, help="Image resolution (default: 960)")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations (default: 100)")
    parser.add_argument("--runs", type=int, default=500, help="Benchmark iterations (default: 500)")
    args = parser.parse_args()

    print_header("YOLO11m Latency Benchmark Pipeline")
    print(f"  Resolution : {args.imgsz}x{args.imgsz}")
    print(f"  Warmup     : {args.warmup} runs")
    print(f"  Benchmark  : {args.runs} runs")

    all_results = {}
    for name, path in MODELS_TO_BENCHMARK.items():
        print_header(f"Benchmarking: {name}")
        all_results[name] = benchmark_model(path, imgsz=args.imgsz, warmup=args.warmup, num_runs=args.runs)

    print_latency_table(all_results)

    # Save to JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "latency_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  ✓ Latency benchmark results saved to: {json_path}\n")
