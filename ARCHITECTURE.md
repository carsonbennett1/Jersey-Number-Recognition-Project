# Jersey Number Recognition — Architecture & System Flow

## What This Project Does

This project implements the pipeline from the CVPR 2024 paper *"A General Framework for Jersey Number Recognition in Sports Video"* (Koshkina & Elder). Given raw video frames of sports players organized into **tracklets** (sequences of images tracking a single person), the system identifies what jersey number each player is wearing.

It supports two datasets: **SoccerNet** (full 9-stage pipeline) and **Hockey** (simplified 2-stage pipeline).

---

## High-Level Pipeline

### SoccerNet (Full Pipeline)

```
Raw Tracklet Images
        │
        ▼
┌──────────────────────┐
│ 1. Soccer Ball Filter│  Removes non-player tracklets (balls) by image size
└────────┬─────────────┘
         ▼
┌──────────────────────┐
│ 2. ReID Features     │  Extracts 128-dim identity features per image (ResNet50)
└────────┬─────────────┘
         ▼
┌──────────────────────┐
│ 3. Gaussian Outliers │  Removes background/occluded frames via statistical filtering
└────────┬─────────────┘
         ▼
┌──────────────────────┐
│ 4. Legibility Filter │  Binary classifier: is the jersey number readable? (ResNet34)
└────────┬─────────────┘
         ▼
┌──────────────────────┐
│ 5. Pose Estimation   │  Detects body keypoints — shoulders & hips (ViTPose-Huge)
└────────┬─────────────┘
         ▼
┌──────────────────────┐
│ 6. Torso Cropping    │  Crops the torso region using pose keypoints
└────────┬─────────────┘
         ▼
┌──────────────────────┐
│ 7. PARSeq STR        │  Reads the jersey number text from torso crops
└────────┬─────────────┘
         ▼
┌──────────────────────┐
│ 8. Consolidation     │  Bayesian inference: combine per-image predictions into one number
└────────┬─────────────┘
         ▼
    Final Jersey Number
    per Tracklet
```

### Hockey (Simplified Pipeline)

```
Raw Tracklet Images → Legibility Filter → PARSeq STR → Final Prediction
```

Hockey skips ReID, outlier removal, and pose estimation because the dataset provides cleaner, pre-cropped images.

---

## Pipeline Stages in Detail

### Stage 1: Soccer Ball Filter

**File:** `helpers.py` → `identify_soccer_balls()`

**Problem:** Some tracklets track the soccer ball instead of a player. These need to be excluded.

**How it works:**
- Scans each tracklet's images
- If all images are smaller than 35px tall and 30px wide, the tracklet is flagged as a soccer ball
- Produces `ball_tracks.json` listing excluded tracklets

**Output:** `ball_tracks.json`

---

### Stage 2: ReID Feature Extraction

**File:** `centroid_reid.py` → `generate_features()`
**Conda env:** `centroids` (Python 3.8, PyTorch 1.7.1)

**Problem:** A tracklet may contain frames where the tracked person is occluded, swapped with another player, or the camera shows a different view. We need a way to identify which frames actually show the target person.

**How it works:**
1. Loads a pre-trained Centroid-ReID model (ResNet50 backbone)
2. For each tracklet, processes every image through the model
3. Extracts a 128-dimensional feature vector per image
4. Saves as NumPy arrays: `{tracklet_name}_features.npy`

**Why a separate environment:** Centroid-ReID depends on an older PyTorch (1.7.1) and pytorch-lightning (1.1.4) that conflicts with newer packages.

**Output:** `.npy` feature files per tracklet

---

### Stage 3: Gaussian Outlier Removal

**File:** `gaussian_outliers.py` → `get_main_subject()`
**Conda env:** `jersey`

**Problem:** Even within a tracklet, some frames show the wrong person or heavy occlusion. These frames will confuse the jersey number reader.

**How it works:**
1. Loads the feature vectors from Stage 2
2. Computes the mean feature vector for the tracklet
3. Calculates each image's Euclidean distance from the mean
4. Removes images that are more than `3.5 × standard_deviation` away from the mean
5. Repeats this pruning for 3 rounds, each time recalculating the mean from the remaining images

This progressively removes outlier frames, keeping only images that look like the "main subject."

**Output:** `main_subject_gauss_th=3.5_r={1,2,3}.json` — filtered image lists per round

---

### Stage 4: Legibility Classification

**File:** `legibility_classifier.py` → `run()`
**Conda env:** `jersey` (Python 3.14)

**Problem:** Many frames show a player from behind, from far away, or with motion blur — the jersey number simply isn't readable. Sending these to the text recognizer would introduce noise.

**How it works:**
1. Loads a ResNet34 model fine-tuned for binary classification (legible vs illegible)
2. Processes each surviving image from Stage 3
3. Applies sigmoid activation → threshold at 0.5
4. Splits tracklets into `legible.json` and `illegible.json`

**Model details:**
- Architecture: ResNet34 backbone → single sigmoid output (defined in `networks.py` as `LegibilityClassifier34`)
- Input: 256×256 images, ImageNet-normalized
- Training: Uses SAM (Sharpness-Aware Minimization) optimizer for better generalization
- Pre-trained on Hockey data, then fine-tuned on SoccerNet

**Output:** `legible.json` (tracklets with readable jerseys) and `illegible.json`

---

### Stage 5: Pose Estimation

**File:** `pose.py`
**Conda env:** `vitpose2` (Python 3.8)

**Problem:** Jersey numbers appear on the torso. To crop just the torso, we need to know where the body is.

**How it works:**
1. Takes a COCO-format JSON with image paths and bounding boxes (generated by `helpers.generate_json()`)
2. Runs ViTPose-Huge — a Vision Transformer model for human pose estimation
3. Detects 17 COCO keypoints per person, each with (x, y, confidence)
4. The pipeline specifically uses 4 keypoints for torso localization:
   - Index 5: Left shoulder
   - Index 6: Right shoulder
   - Index 11: Left hip
   - Index 12: Right hip

**Why a separate environment:** ViTPose requires mmcv-full 1.4.8 and mmpose, which are pinned to Python 3.8 and have strict dependency chains.

**Output:** `pose_results.json` with per-image keypoint coordinates

---

### Stage 6: Torso Cropping

**File:** `helpers.py` → `generate_crops()`
**Conda env:** `jersey`

**Problem:** The full player image contains legs, arms, background — all distracting for text recognition. We want just the torso.

**How it works:**
1. For each legible image, loads its pose keypoints from Stage 5
2. Checks that all 4 torso keypoints (shoulders + hips) have confidence > 0.4
3. Computes a bounding box from the min/max of those 4 keypoints
4. Adds 5px padding on all sides, clipped to image bounds
5. Crops and saves the torso region

Images where keypoints are low-confidence are skipped (pose was unreliable).

**Output:** Cropped torso images in `crops/imgs/` directory

---

### Stage 7: PARSeq Scene Text Recognition

**File:** `str.py` → `run_inference()`
**Conda env:** `parseq2` (Python 3.9)

**Problem:** Now we have clean torso crops — we need to actually read the number.

**How it works:**
1. Loads a PARSeq model fine-tuned on jersey numbers
2. PARSeq (Permuted Autoregressive Sequence-to-Sequence) is a state-of-the-art scene text recognition model
3. Processes each torso crop and predicts up to 3 characters
4. Character set: digits 0-9 plus END token (11 classes total)
5. Extracts both the predicted label and per-character logits (confidence scores)

**Optional post-processing:**
- Temperature scaling (factor 2.367) for confidence calibration
- Confidence threshold filtering (0.2 minimum)
- Bias toward 2-digit numbers (jersey numbers are usually 1-99)

**Why a separate environment:** PARSeq depends on pytorch-lightning 1.9.5 and a specific timm version.

**Output:** `jersey_id_results.json` with `{filename: {label, confidence, raw, logits}}`

---

### Stage 8: Result Consolidation (Bayesian Inference)

**File:** `helpers.py` → `process_jersey_id_predictions()` and `predict_jersey_number()`
**Conda env:** `jersey`

**Problem:** Each tracklet has multiple frames, each producing a prediction. These predictions may disagree. We need one final answer.

**How it works:**
1. Groups predictions by tracklet
2. For each tracklet, applies **Bayesian log-likelihood aggregation**:
   - Converts PARSeq logits to log-probabilities via log-softmax
   - Sums log-probabilities across all images in the tracklet
   - Optionally adds a prior bias favoring 2-digit numbers
3. Selects the jersey number with the highest combined score
4. Maps predictions to the 0-99 range

**Output:** `final_results.json` — one jersey number per tracklet

---

### Stage 9: Evaluation

**File:** `helpers.py` → `evaluate_results()`

Compares final predictions against ground truth annotations and reports accuracy.

---

## Multi-Environment Architecture

The project uses **4 separate conda environments** because its components have incompatible dependency chains:

```
┌─────────────────────────────────────────────────────────┐
│                    jersey (Python 3.14)                  │
│  Main pipeline orchestration, legibility classifier,    │
│  helpers, gaussian outliers, result consolidation       │
│  PyTorch 2.10.0 + CUDA 12.8                            │
├─────────────────────────────────────────────────────────┤
│                  vitpose2 (Python 3.8)                   │
│  ViTPose pose estimation                                │
│  mmcv-full 1.4.8, mmpose, PyTorch 2.4.1                │
├─────────────────────────────────────────────────────────┤
│                  parseq2 (Python 3.9)                    │
│  PARSeq scene text recognition                          │
│  pytorch-lightning 1.9.5, timm 1.0.25, PyTorch 2.8.0   │
├─────────────────────────────────────────────────────────┤
│                  centroids (Python 3.8)                  │
│  Centroid-ReID feature extraction                       │
│  pytorch-lightning 1.1.4, PyTorch 1.7.1                 │
└─────────────────────────────────────────────────────────┘
```

Communication between environments happens via **subprocess calls** and **file-based I/O** (JSON files, NumPy arrays, image files). The main `jersey` environment orchestrates everything through `subprocess.run()` with argument lists.

`configuration.py` manages building the correct Python executable path for each environment based on the OS.

---

## File Map

### Core Pipeline

| File | Purpose |
|------|---------|
| `main.py` | Entry point — orchestrates the full pipeline |
| `configuration.py` | All paths, conda env names, model URLs, OS/GPU detection |
| `helpers.py` | Utility functions — JSON generation, cropping, evaluation, Bayesian consolidation |

### ML Components

| File | Model | Task |
|------|-------|------|
| `legibility_classifier.py` | ResNet34 | Binary: is the jersey readable? |
| `centroid_reid.py` | ResNet50 (Centroid-ReID) | 128-dim identity feature extraction |
| `pose.py` | ViTPose-Huge | 17-keypoint body pose estimation |
| `str.py` | PARSeq | Scene text recognition (digit reading) |

### Model Definitions & Data

| File | Purpose |
|------|---------|
| `networks.py` | PyTorch model class definitions (ResNet, ViT, custom CNN variants) |
| `jersey_number_dataset.py` | Dataset classes for training and inference |
| `gaussian_outliers.py` | Statistical outlier filtering on ReID features |

### Setup

| File | Purpose |
|------|---------|
| `setup.py` | Creates conda envs, clones repos, downloads models |
| `jersey.yml` | Conda env spec for main pipeline |
| `vitpose.yml` | Conda env spec for pose estimation |
| `parseq2.yml` | Conda env spec for text recognition |
| `centroids.yml` | Conda env spec for ReID |

---

## Data Layout

```
project/
├── data/
│   ├── SoccerNet/jersey-2023/
│   │   ├── test/test/images/          # Test split tracklet images
│   │   ├── val/val/images/            # Validation split
│   │   ├── train/train/images/        # Training split
│   │   └── challenge/challenge/images/
│   └── Hockey/
│       ├── legibility_dataset/        # Legibility training data
│       └── jersey_number_dataset/     # LMDB jersey number data
├── models/                            # Pre-trained model weights
│   ├── legibility_resnet34_soccer_*.pth
│   ├── legibility_resnet34_hockey_*.pth
│   ├── parseq_epoch=24-*.ckpt        # SoccerNet STR
│   └── parseq_epoch=3-*.ckpt         # Hockey STR
├── out/SoccerNetResults/              # Pipeline outputs
│   ├── ball_tracks.json
│   ├── features/                      # ReID .npy files
│   ├── main_subject_gauss_*.json      # Filtered image lists
│   ├── legible.json / illegible.json
│   ├── pose_results.json
│   ├── crops/imgs/                    # Torso crop images
│   ├── jersey_id_results.json         # STR predictions
│   └── final_results.json             # Consolidated results
├── experiments/                       # Training checkpoints
├── pose/ViTPose/                      # Cloned ViTPose repo
├── str/parseq/                        # Cloned PARSeq repo
├── reid/centroids-reid/               # Cloned Centroid-ReID repo
└── sam2/                              # Cloned SAM optimizer repo
```

---

## How a Single Frame Becomes a Jersey Number

To make this concrete, here's what happens to one image of a player:

1. **Ball filter** checks image dimensions — it's 150×300px, so it passes (not a ball)
2. **Centroid-ReID** extracts a 128-dimensional feature vector from the image
3. **Gaussian filter** compares this vector to the tracklet's mean — distance is within 3.5σ, so it stays
4. **Legibility classifier** (ResNet34) outputs 0.82 after sigmoid — above 0.5 threshold, marked legible
5. **ViTPose** detects keypoints — left shoulder at (45, 60), right shoulder at (105, 58), left hip at (50, 140), right hip at (100, 138), all with confidence > 0.4
6. **Torso crop** computes bounding box from those 4 points → crops a 70×85px region from the original image
7. **PARSeq** reads the crop and predicts "23" with logits [0.1, 0.05, 0.8, 0.02, ...] for each position
8. **Consolidation** combines this prediction with 15 other frames from the same tracklet using log-likelihood — "23" wins with the highest aggregate score

Final answer for this tracklet: **Jersey #23**

---

## Training

### Legibility Classifier Training

```bash
# Train on Hockey dataset
python legibility_classifier.py --train --arch resnet34 --sam \
    --data <hockey-dataset> --trained_model_path ./experiments/hockey_legibility.pth

# Fine-tune on SoccerNet
python legibility_classifier.py --finetune --arch resnet34 --sam \
    --data <soccernet-dataset> --full_val_dir <val-dir> \
    --trained_model_path ./experiments/hockey_legibility.pth \
    --new_trained_model_path ./experiments/sn_legibility.pth
```

- Uses SAM optimizer (Sharpness-Aware Minimization) for better generalization
- Data augmentation: random grayscale, color jitter (brightness 0.5, hue 0.3)
- Can optionally balance classes during training

### PARSeq STR Fine-tuning

```bash
python main.py Hockey train --train_str
python main.py SoccerNet train --train_str
```

Fine-tunes the pre-trained PARSeq model on jersey number crops using LMDB datasets.

---

## Cross-Platform Support

| Feature | Windows | macOS | Linux |
|---------|---------|-------|-------|
| Python executable | `python.exe` | `python` | `python` |
| DataLoader workers | 0 | 4 | 4 |
| GPU | CUDA | MPS (Metal) | CUDA |
| Fallback | CPU | CPU | CPU |

Override defaults with environment variables:
- `JERSEY_PROJECT_ROOT` — project directory path
- `JERSEY_CONDA_BASE` — conda installation path
