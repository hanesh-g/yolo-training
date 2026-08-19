# Video Latency & Accuracy Evaluation Report

**Project:** Innovision Multi-Analytics Security Platform — Person Detection Module  
**Hardware:** NVIDIA GeForce RTX 5080 (16 GB VRAM), Linux  
**Date:** August 19, 2026  

---

## 1. Objective

The goal of this phase was to rigorously evaluate our custom-trained model (`yolo11m_combined_v1`) against the off-the-shelf, untrained base model (`yolo11m.pt`). 

We needed to answer two critical questions:
1. **Speed:** Does the model process video frames fast enough for real-time security deployment (>30 FPS)?
2. **Accuracy:** Did our custom training actually improve real-world performance on deployment footage compared to just using the base model?

To answer this, we evaluated both models on four real-world video streams (Factory, Night, Office, and Pedestrian scenes) containing manual ground-truth annotations.

---

## 2. Methodology: How the Tests Were Performed

Benchmarking object detection models requires a rigorous approach. Simply using Python's `time.time()` around a prediction call is inaccurate because GPU execution is asynchronous.

Here is the "First Principles" procedure we followed using our custom `10_video_latency_benchmark.py` script:

### A. Latency Measurement
1. **Warmup Phase:** Before measuring, we run 50 dummy inferences. This forces the NVIDIA RTX 5080 out of its low-power idle state into its maximum performance state (P0) and ensures CUDA kernels are fully compiled.
2. **Pipeline Breakdown:** We break down the processing of every single frame into four stages:
    *   **Decode Time:** Reading the compressed video packet (H.264/H.265) and converting it to raw pixels.
    *   **Pre-process Time:** Resizing the frame to 960x960, normalizing, and copying it from CPU RAM to GPU VRAM.
    *   **Inference Time:** The actual neural network math running on the GPU Tensor Cores.
    *   **Post-process (NMS) Time:** Filtering out overlapping bounding boxes (Non-Maximum Suppression).
3. **Statistical Aggregation:** We compute the Mean, Standard Deviation, and 95th Percentile (P95) latency across all frames to ensure stable performance without jitter.

### B. Accuracy Measurement
For specific frames in the videos, we had manual ground-truth annotations (bounding boxes drawn by humans).
1. We run the model on an annotated frame.
2. We compare the model's predicted boxes against the human-drawn boxes using **IoU (Intersection over Union)**.
3. If IoU > 0.5, it's a **True Positive (TP)**. If the model predicts a box where no human drew one, it's a **False Positive (FP)**. If the model misses a human, it's a **False Negative (FN)**.
4. We calculate **Precision**, **Recall**, and **F1-Score** for both the untrained base model and our trained model.

---

## 3. Results Comparison: Trained vs. Untrained Model

### A. Latency & Throughput (Speed)

Both models are based on the YOLO11m architecture, so their raw inference speeds are nearly identical, which is exactly what we expect. 

| Scene | Resolution | Native FPS | Total Latency (ms) | Processing FPS | Real-time Margin |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Factory** | 1920x1080 | 6.0 | ~7.76 ms | **~129 FPS** | 21.5x faster |
| **Night** | 1280x720 | 15.0 | ~6.66 ms | **~150 FPS** | 10.0x faster |
| **Office** | 1280x720 | 25.0 | ~6.73 ms | **~148 FPS** | 5.9x faster |
| **Pedestrian** | 640x360 | 30.0 | ~6.16 ms | **~162 FPS** | 5.4x faster |

**Conclusion on Speed:** The model is blazing fast. On the RTX 5080, it processes frames between **129 and 162 FPS**. It easily clears the 30 FPS requirement, operating at a 5x to 21x real-time safety margin depending on the video resolution.

### B. Accuracy (The Impact of Fine-Tuning)

This is where the custom training proves its immense value. The base `yolo11m.pt` model is trained on 80 classes. By fine-tuning it specifically for person detection on CrowdHuman and COCO, we drastically altered its behavior.

| Scene | Metric | Untrained Base Model | **Our Trained Model** | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Factory** | False Positives | 291 | **132** | 📉 -54% fewer false alarms |
| | F1 Score | 0.7039 | **0.8061** | 📈 +14.5% overall accuracy |
| **Night** | False Positives | 75 | **5** | 📉 -93% fewer false alarms |
| | F1 Score | 0.6051 | **0.9355** | 📈 +54.6% overall accuracy |
| **Office** | False Positives | 821 | **132** | 📉 -84% fewer false alarms |
| | F1 Score | 0.4196 | **0.6792** | 📈 +61.8% overall accuracy |
| **Pedestrian** | False Positives | 709 | **110** | 📉 -84% fewer false alarms |
| | F1 Score | 0.5023 | **0.7710** | 📈 +53.5% overall accuracy |

### Why is the Trained Model So Much Better?

1.  **Eradication of False Positives:** The untrained model generates hundreds of False Positives (FP) because it is a generalist. It often hallucinates people in shadows, reflections, or objects shaped vaguely like humans (chairs, coats). Our trained model is highly specialized, reducing false alarms by up to 93% (Night scene) and 84% (Office scene). In a security context, false alarms are costly; our model solves this.
2.  **Night-Time Robustness:** The trained model's performance in the "Night" scene jumped from an F1 of 0.60 to a stellar **0.93**. It learned to detect humans in low-light conditions much better than the base model.
3.  **Crowd Handling:** The inclusion of the `CrowdHuman` dataset during training taught our model to handle occlusions (people blocking other people), which is why it performs significantly better in the "Pedestrian" and "Factory" scenes.

---

## 4. Future Improvements: The Path to Perfection

While our model is vastly superior to the base model, the Office (F1: 0.68) and Pedestrian (F1: 0.77) scenes show room for growth. Here is how we can improve the model further:

### A. Adding the WiderPerson Dataset
**Recommendation: HIGHLY RECOMMENDED**
*   **Why:** Our current model was trained on COCO (diverse scenes) and CrowdHuman (dense crowds). However, security cameras are often mounted high up, looking down at a steep angle. The **WiderPerson** dataset contains thousands of images taken from surveillance-style angles with heavy occlusion.
*   **Impact:** Adding WiderPerson to our training mix would directly boost accuracy in scenes like "Pedestrian" and "Office" where people are viewed from challenging top-down angles or are partially hidden behind desks.

### B. Domain Adaptation (Fine-tuning on Target Data)
*   **Why:** The model struggles slightly in the Office scene. This could be due to specific lighting, glare, or furniture (like office chairs) that it hasn't seen enough of.
*   **Impact:** If we extract just 500-1000 frames from the actual cameras where this model will be deployed, annotate them, and add them to the training set, the model will adapt to the exact environment, likely pushing the F1 score > 0.90 across the board.

### C. Resolution Tuning & Tiling (SAHI)
*   **Why:** In the Pedestrian video (640x360), people far away are very tiny (only a few pixels wide). Compressing them further during pre-processing makes them invisible.
*   **Impact:** Implementing Slicing Aided Hyper Inference (SAHI) — where we slice the image into grids, run inference on the grids, and merge the results — can dramatically improve the detection of extremely small, distant subjects in surveillance feeds without needing to train a larger model.

---

## 5. Summary

The custom-trained YOLO11m Person Detection model is a massive success. It maintains the blazing **~140 FPS** real-time speed of the base architecture while cutting false alarms by up to **93%** and boosting overall accuracy by over **50%** in challenging scenarios. 

To push the model to enterprise-grade perfection, the next logical step is to integrate the **WiderPerson** dataset to improve performance on surveillance-angle footage.
