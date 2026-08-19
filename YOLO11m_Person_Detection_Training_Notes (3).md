# YOLO11m Person Detection — Training Notes
**Project:** Innovision Multi-Analytics Security Platform — Person Detection Module
**Datasets (Phase 1):** COCO (Person Class) + CrowdHuman — **WiderPerson deferred to a later phase (team lead's decision)**
**Split:** 70/20/10 (Train/Val/Test)
**Target model:** YOLO11m (fine-tuned from pretrained checkpoint)
**Export format:** `.pt` (confirmed)
**CrowdHuman approach:** `fbox` only for this round — single training run. `vbox` comparison deferred (can be revisited later if occlusion-scene results need investigation).

---

## 0. Decisions Confirmed with Team Lead / Final Scope

1. **WiderPerson excluded from Phase 1** — will be added in a later phase. Current training uses **COCO + CrowdHuman only**.
2. **Export format is `.pt`** (native PyTorch/Ultralytics checkpoint) — not `.onnx`/`.engine` for now.
3. **CrowdHuman fbox/vbox** — decided to train **fbox only** for this round, to keep this to a single training run and avoid doubling GPU time. Reasoning: `fbox` gives complete, consistent boxes for partially-occluded people, which better serves headcount, crowd density, and downstream tracking/re-ID — all of which depend on stable full-person boxes rather than fragments of visible area. The tradeoff is that `fbox` ground truth for extremely occluded people is an annotator *estimate* rather than a precise measurement — an acceptable cost for the consistency gained. `vbox` is deferred and can be trained/compared later if occlusion-scene performance specifically needs investigation.

---

## 1. Project Context

This detection module feeds a multi-analytics security platform with 8 downstream capabilities: face recognition + dedup, face tokenization, restricted entry detection, headcount monitoring, crowd density management, zone-based monitoring, intruder detection, and pedestrian detection.

**Important framing point:** better detection accuracy is foundational and will improve everything downstream — but it is **not sufficient on its own**. Each dependent module has its own additional failure modes:
- Face recognition depends on face crop quality/resolution, not just correct person boxes.
- Tracking/re-identification depends on box **consistency** across frames — an occluded person's box jumping around confuses trackers.
- Crowd density depends specifically on recall in dense, occluded scenes — a harder problem than general detection.

Worth keeping in mind when reporting results to the team lead: "detection accuracy improved" should be broken down by these downstream-relevant dimensions, not just a single overall mAP number.

---

## 2. Dataset Roles (per the table provided)

| Dataset | Role | Key characteristic | Phase |
|---|---|---|---|
| COCO Person Class | General person detection | Everyday scenes, wide variety of poses/contexts | **Phase 1 (now)** |
| CrowdHuman | Crowded scenes | Very high person density (~22 people/image avg), heavy occlusion | **Phase 1 (now)** |
| WiderPerson | Surveillance views | Wide-angle, pedestrian/surveillance-camera perspective | **Phase 2 (deferred)** |

For Phase 1, COCO + CrowdHuman are combined into **one single-class dataset**: `0: person`.

**Note for later:** since WiderPerson was the surveillance-angle/wide-view proxy, the Phase 1 model will likely be comparatively weaker on that specific camera perspective until WiderPerson is folded in during Phase 2. Worth keeping in mind if benchmarking against real deployment footage before then — not a defect, just a known current scope gap.

---

## 3. Environment Setup

```bash
pip install ultralytics --upgrade
```

Verify GPU access before doing anything else:
```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

**Note:** `pycocotools` (for COCO parsing) and Ultralytics' converter utilities are separate tools from the pretrained model itself — `yolo11m.pt` is just trained weights; it has no built-in relationship to annotation conversion tooling. Both come from `pip install ultralytics` / `pip install pycocotools`, but conceptually keep "model weights" and "data tooling" separate.

---

## 4. Dataset Preparation

### 4.1 Understand each source's native annotation format first (Phase 1: COCO + CrowdHuman)

| Dataset | Native format | Coordinate system | Class scheme |
|---|---|---|---|
| COCO | Single large JSON | `[x_min, y_min, width, height]`, pixel-based | 80 categories (person = id `1`) |
| CrowdHuman | `.odgt` (JSON-lines — one JSON object per line/image) | `fbox`/`vbox`/`hbox` fields, pixel-based | Person only, but 3 box types per instance |

> **WiderPerson (deferred to Phase 2)** — native format is a custom `.txt` (NOT YOLO format despite the same extension): `[x_min, y_min, x_max, y_max]` corner format, pixel-based, first line = box count, 5 classes (1=pedestrian, 2=rider, 3=partially-visible, 4=ignore, 5=crowd). Kept here for reference so the conversion script can be extended later without re-deriving this. **Important point to remember for Phase 2:** "same file extension" does not mean "same schema" — WiderPerson's `.txt` structurally differs from YOLO's `.txt` in coordinate system, scale, and class numbering.

### 4.2 CrowdHuman — fbox only (confirmed final decision for this round)

- **`vbox`** = box around only the visible/unoccluded portion of the person.
- **`fbox`** = box estimating the full body extent, including occluded parts.
- **`hbox`** = head-only box (not relevant here).

**Decision: train on `fbox` only**, single run — to avoid doubling GPU time on two full training experiments. `fbox` gives one complete, consistent box per person even when partially blocked, which is what headcount, crowd density, and downstream tracking/re-ID depend on. `vbox` is deferred; can be revisited later if occlusion-scene results specifically warrant a comparison.

**Practically, this means:**
- Only one CrowdHuman label set is needed: extract `fbox` during conversion, ignore `vbox`/`hbox` fields entirely.
- Only one folder structure, one `data.yaml`, one training run — same as the original single-model plan.

### 4.3 Class mapping (collapsing everything to single class) — Phase 1

| Source | Original class | Maps to |
|---|---|---|
| COCO | category id `1` (person) | class `0` |
| CrowdHuman | person (`fbox`) | class `0` |

All non-person COCO categories (79 of the 80 classes) are dropped entirely. CrowdHuman's `vbox`/`hbox` fields are ignored for this round.

### 4.4 WiderPerson class 5 ("crowd") — deferred note for Phase 2

Not applicable to Phase 1 (WiderPerson excluded from current scope), but keeping this note for when it's added later:

Class 5 is not "many individually boxed people" — it's a **single box drawn around a region the annotators judged too dense to box individually**. Mapping this to `person` would teach the model "one giant box = one person," which corrupts box-regression scale and damages headcount accuracy in dense scenes. When WiderPerson is added in Phase 2, either exclude class 5 entirely (simpler, recommended starting point) or mark it as an "ignore region" so loss isn't penalized for detections there (more advanced).

### 4.5 Conversion & normalization — step by step (Phase 1: COCO + CrowdHuman, fbox only)

1. **Parse native format** — COCO via `pycocotools`/JSON; CrowdHuman via `.odgt` JSON-lines parsing, extracting the `fbox` field only.
2. **Extract raw pixel-coordinate boxes** — confirm which corner/format each source actually uses before writing conversion code.
3. **Remap classes** — every valid person-type box → class `0`; drop all non-person COCO categories.
4. **Normalize to YOLO format**: `class x_center y_center width height`, all values 0–1 relative to image dimensions:
   ```
   x_center = (x_min + width/2) / image_width
   y_center = (y_min + height/2) / image_height
   width_norm  = width  / image_width
   height_norm = height / image_height
   ```
   *(This step is the most common source of bugs — mixing up `x_max` vs `width`, or forgetting to normalize.)*
5. **Write one `.txt` label file per image** — same filename as the image, one line per person box (COCO person boxes + CrowdHuman `fbox` boxes).
6. **Organize into folder structure:**
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── labels/
       ├── train/
       ├── val/
       └── test/
   ```

### 4.6 Splitting — separate concept from class exclusion

Splitting is a decision about **which images** go into train/val/test, separate from which annotations are valid.

- Apply the 70/20/10 split **per source dataset** (COCO, CrowdHuman), not just on the combined pool — otherwise val/test could end up skewed toward one domain.
- All splits use the **same single class** (`0: person`).

### 4.7 data.yaml

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: person
```

### 4.8 Sanity-check before training (non-negotiable)

Draw the converted YOLO boxes back onto ~20–30 sample images per source dataset using OpenCV/matplotlib, and visually confirm boxes are correctly placed and scaled.

**This step exists purely to catch conversion bugs** (wrong axis, bad scaling, coordinate mix-ups) — it is *not* manual annotation and does not create new training data. Catching a bug here saves a wasted multi-hour/day training run later.

---

## 5. Training — single run (fbox)

Fine-tune from the **pretrained** `yolo11m.pt` checkpoint (not from scratch) to inherit COCO's general object priors.

```python
from ultralytics import YOLO

model = YOLO("yolo11m.pt")

results = model.train(
    data="data.yaml",
    epochs=150,
    imgsz=960,          # see note below — NOT about "low resolution," about small/distant objects
    batch=16,
    patience=30,
    device=0,
    workers=8,
    optimizer="auto",
    cos_lr=True,
    close_mosaic=10,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,      # boosts effective crowd density during training
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    project="innovision_person_detection",
    name="yolo11m_fbox_v1"
)
```

### 5.1 Clarification on `imgsz` — common misunderstanding

`imgsz` is **not** about training the model to handle low-quality/low-resolution camera footage. It's the resolution images are resized to before entering the network.

- Surveillance cameras are wide-angle and far from subjects, so a person can occupy a tiny pixel region in the original frame.
- A **lower** `imgsz` downscales more aggressively, shrinking that already-small person further — making detection *harder*, not easier.
- A **higher** `imgsz` (960 or more) preserves more detail on small/distant people — this is the correct lever for "detect people who appear small in wide surveillance frames."

If the actual concern is **genuinely low-quality/noisy camera source** (old analog cameras, heavy compression) — that's a different problem, solved via **data augmentation** (blur, compression artifacts, noise added during training) or a denoising pre-processing stage (see Section 8), not by lowering `imgsz`.

### 5.2 Why specific augmentation choices matter here

- `copy_paste`: synthetically increases person density in training images — directly relevant since crowd density is a named module and CrowdHuman is your dense-scene proxy.
- `close_mosaic=10`: disables mosaic augmentation for the final 10 epochs so the model converges on more realistic, non-composited images at the end of training.

---

## 6. Validation

```python
metrics = model.val(data="data.yaml", split="test")
```

Don't stop at overall mAP50-95 — break it down:
- **mAP per source dataset** (run val separately on COCO-only / CrowdHuman-only subsets) — catches whether one domain is dragging the average down.
- **Recall by object size** (small/medium/large, reported natively by Ultralytics) — maps directly to the surveillance small-object problem.
- **Precision on high-occlusion (CrowdHuman) subset specifically** — this is where the fbox choice should show its benefit (consistent full-person boxes even under partial occlusion).

### If crowd/occlusion performance lags:
Increase `copy_paste`/mosaic weight, or use a curriculum: pretrain on COCO first (cleaner boxes), then fine-tune further on CrowdHuman (denser, harder). If results are still weak specifically on heavily-occluded people, that's the signal to revisit the deferred `vbox` comparison (Section 0).

### If small-object recall lags:
Push `imgsz` higher (1280) if compute allows, or use tiled inference at deployment time.

### If crowd/occlusion performance lags:
Increase `copy_paste`/mosaic weight, or use a curriculum: pretrain on COCO+WiderPerson first (cleaner boxes), then fine-tune further on CrowdHuman (denser, harder).

### If small-object recall lags:
Push `imgsz` higher (1280) if compute allows, or use tiled inference at deployment time.

---

## 7. Export — `.pt` (confirmed)

Team lead confirmed the output format is `.pt` — the native PyTorch/Ultralytics checkpoint. No additional export step is needed beyond training itself: `best.pt` and `last.pt` are saved automatically in the run's output directory (`innovision_person_detection/yolo11m_fbox/weights/` and `.../yolo11m_vbox/weights/`).

- `best.pt` = checkpoint from the epoch with the best validation performance — this is the one to hand off/deploy.
- `last.pt` = checkpoint from the final epoch — useful mainly if you want to resume training later.

**For reference (not needed now, but useful if deployment format changes later):** `.pt` can still be exported to `.onnx` (cross-platform) or `.engine`/TensorRT (NVIDIA-GPU-optimized, fastest for real-time multi-camera inference) at any point in the future from the same checkpoint — nothing about training changes if the export target changes down the line.

---

## 8. Denoising / Upscaling Pre-Processing Layer (separate POC question)

Raised as a possible layer between camera and detection, to upscale/clean frames before detection.

**Where it could help:** genuinely low-quality camera sources (analog, heavy compression, low light) — a documented technique in production CV pipelines; super-resolution for small/distant persons is also established.

**Where it adds real cost:**
- **Latency** — every frame would pass through two neural networks instead of one, before tracking/recognition even starts; could become the actual throughput bottleneck across many concurrent camera streams.
- **Compounding errors** — super-resolution models can hallucinate plausible-but-incorrect detail, potentially feeding the detector clean-looking but wrong input.
- **Possible redundancy** — if the real issue is "small/distant people in high-res wide frames" (not genuine noise), training-side fixes (`imgsz`, `copy_paste`, scale augmentation) already address it without adding a second live-inference model.

**Recommended approach — validate before building:**
1. Train the detector first, then evaluate failure cases specifically on **real deployment camera footage**, not just benchmark test sets.
2. If failures cluster around noise/low-light/compression specifically (not just small/distant objects) → denoising layer is justified, with concrete evidence for the added latency cost.
3. If built, benchmark a **lightweight** denoising model first rather than a heavy super-resolution GAN — likely sufficient and much cheaper on latency.

---

## 9. Timeline — 4-Day Plan (Phase 1: COCO + CrowdHuman, single fbox run)

**Reality check:** data preparation is likely to consume more time than the actual training run — budget accordingly.

Combined dataset scale for Phase 1: ~79,000 training images (COCO ~64k person-filtered, CrowdHuman ~15k images but ~470k person instances) — WiderPerson (~8k images) deferred to Phase 2.

| Day | Focus |
|---|---|
| **Day 1** | Download COCO + CrowdHuman; write/adapt conversion scripts (COCO filter+remap, CrowdHuman `.odgt`→YOLO extracting `fbox` only); run conversions; organize folder structure with stratified 70/20/10 split; **run sanity-check visualization** |
| **Day 2** | Set up GPU environment, verify CUDA/GPU access; launch training early in the day so it can run overnight |
| **Day 3** | Training likely completes (~20–30 hrs on A100 fits this window); run `model.val()` on held-out test set; break down metrics per source dataset |
| **Day 4** | Review per-source and per-object-size metrics in detail; confirm `best.pt` as the deliverable; buffer time for one quick re-training pass if a fixable issue surfaces |

Single-run scope keeps this comfortably within the 4-day window — no timeline extension needed with the fbox-only decision.

**⚠️ Action item:** confirm GPU hardware access is locked in *before* the 4-day clock starts — cloud GPU queueing/provisioning delays are the most common real-world cause of timeline slippage, more so than training compute itself.

---

## 10. Hardware Recommendations

| Tier | GPU | VRAM | Approx. epoch time (imgsz=960, batch=16, ~65k images) |
|---|---|---|---|
| Ideal | NVIDIA A100 (40/80GB) | 40–80GB | ~8–12 min/epoch |
| Good | NVIDIA RTX 4090 | 24GB | ~15–20 min/epoch |
| Workable | NVIDIA RTX 3090 / A6000 | 24–48GB | ~20–25 min/epoch |
| Tight | RTX 4080 / T4 | 12–16GB | ~30–40 min/epoch (may need smaller batch) |

At 150 epochs on an A100: roughly **20–30 hours** total training time.

- **CPU:** 8–16 cores sufficient (mainly for `workers` dataloader parallelism); secondary to GPU choice.
- **RAM:** 32GB minimum, 64GB comfortable (CrowdHuman's high instance density + mosaic/copy_paste augmentation hold multiple images in memory at once).
- **Storage:** Fast NVMe SSD, not spinning disk — ~50–100GB budget; disk I/O can bottleneck epoch time if slow.
- **Single GPU is sufficient** for a YOLO11m run at this dataset scale — no need for multi-GPU.

---

## 11. Open Items to Confirm with Team Lead

### Resolved this round
- ~~fbox vs vbox for CrowdHuman~~ → **confirmed: fbox only, single training run. vbox deferred, can revisit later if occlusion-scene results warrant it.**
- ~~WiderPerson scope~~ → **confirmed: excluded from Phase 1, added later.**
- ~~Export format~~ → **confirmed: `.pt`.**

### Still open
1. **Denoising/upscaling layer** — recommend validating against real camera footage failure modes *before* committing to build it, given latency and error-compounding risks. Unrelated to the fbox/WiderPerson decisions, can be discussed separately.
2. **Detection-accuracy vs. downstream-module framing** — recommend reporting results broken down by dimensions relevant to face recognition, tracking, and crowd density separately, not just a single aggregate detection metric.how to switch on hotspot in ubuntu when connected to ethernet
3. **WiderPerson Phase 2 timing** — worth asking roughly when Phase 2 is expected, so the Phase 1 pipeline/scripts can be built in a way that's easy to extend later (e.g., conversion scripts structured to add a third source without a rewrite).
4. **GPU access/provisioning** — confirm before the 4-day timeline starts, to avoid queueing delays eating into the schedule.

---

## 12. Summary Checklist — Phase 1 (COCO + CrowdHuman, fbox only)

- [ ] Environment set up (`ultralytics`, `pycocotools`, GPU verified)
- [ ] COCO + CrowdHuman datasets downloaded (WiderPerson deferred)
- [ ] COCO filtered to person class, remapped to class `0`
- [ ] CrowdHuman `.odgt` parsed, `fbox` extracted, remapped to class `0`
- [ ] All boxes normalized to YOLO format (center + width/height, 0–1 scale)
- [ ] Folder structure built (`images/`, `labels/` × train/val/test)
- [ ] Split stratified 70/20/10 **per source dataset**
- [ ] `data.yaml` created
- [ ] Sanity-check visualization run on samples from both sources
- [ ] Training launched from pretrained `yolo11m.pt` — single fbox run
- [ ] Validation run with per-source and per-object-size breakdown
- [ ] `best.pt` confirmed as the deliverable
- [ ] Open items (Section 11) confirmed with team lead
- [ ] Pipeline noted as extensible for WiderPerson (Phase 2) and vbox comparison (if revisited later)
