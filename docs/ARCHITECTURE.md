# Jersey Number Recognition - Architecture

## Overview

The project implements an end-to-end pipeline for **jersey number recognition** in sports videos (SoccerNet and Hockey datasets). It combines multiple deep learning components: legibility classification, ReID feature extraction, pose estimation, torso cropping, and scene text recognition (STR).

## Pipeline Architecture

```mermaid
flowchart TB
    subgraph Input [Input Layer]
        Tracklets[Tracklet Directories]
        Images[Per-Tracklet Images]
    end

    subgraph Preprocessing [Preprocessing Stage]
        BallFilter[Soccer Ball Filter]
        ReID[Centroid-ReID Features]
        OutlierFilter[Gaussian Outlier Removal]
    end

    subgraph Legibility [Legibility Stage]
        LegClass[Legibility Classifier]
        Legible[Legible Images]
    end

    subgraph PoseCrop [Pose and Crop Stage]
        PoseJSON[Generate COCO JSON]
        ViTPose[ViTPose Keypoints]
        TorsoCrop[Torso Cropping]
    end

    subgraph STR [Scene Text Recognition]
        PARSeq[PARSeq Transformer]
        Predictions[Per-Image Predictions]
    end

    subgraph PostProcess [Post-Processing]
        Consolidate[Weighted Voting / Bayesian]
        Final[Final Jersey Number]
    end

    Tracklets --> BallFilter
    Tracklets --> ReID
    ReID --> OutlierFilter
    BallFilter --> LegClass
    OutlierFilter --> LegClass
    LegClass --> Legible
    Legible --> PoseJSON
    PoseJSON --> ViTPose
    ViTPose --> TorsoCrop
    TorsoCrop --> PARSeq
    PARSeq --> Predictions
    Predictions --> Consolidate
    Consolidate --> Final
```

## Component Dependency Graph

```mermaid
flowchart LR
    subgraph Core [Core Pipeline]
        main[main.py]
    end

    subgraph Config [Configuration]
        config[configuration.py]
    end

    subgraph Utils [Utilities]
        helpers[helpers.py]
    end

    subgraph Models [Model Components]
        lc[legibility_classifier.py]
        reid[centroid_reid.py]
        str_module[str.py]
        pose[pose.py]
        gauss[gaussian_outliers.py]
    end

    main --> config
    main --> helpers
    main --> lc
    main --> reid
    main --> str_module
    main --> pose
    main --> gauss
    lc --> networks
    str_module --> parseq
```

## File Inventory and Purposes

| File | Purpose |
|------|---------|
| `main.py` | Pipeline orchestrator; entry point `python main.py <dataset> <part>` |
| `configuration.py` | Paths, model URLs, conda env names (vitpose, parseq2, centroids) |
| `helpers.py` | Pose JSON generation, torso cropping, prediction consolidation, evaluation |
| `legibility_classifier.py` | Binary legibility classification (ResNet18/34 or ViT-B/16) |
| `centroid_reid.py` | ReID feature extraction via Centroid-ReID (ResNet50) |
| `str.py` | PARSeq STR wrapper; inference and temperature scaling |
| `pose.py` | ViTPose wrapper for keypoint detection |
| `gaussian_outliers.py` | Gaussian-based outlier removal for tracklet images |
| `networks.py` | Legibility and jersey number classifier architectures |
| `jersey_number_dataset.py` | PyTorch datasets for legibility and number classification |
| `number_classifier.py` | Alternative ResNet34-based classifier (non-STR) |
| `data.py` | SoccerNet dataset downloader |
| `setup.py` | Environment setup, model downloads, dependency cloning |

## External Dependencies (Cloned Repos)

- **str/parseq/** - PARSeq transformer for STR
- **pose/ViTPose/** - ViTPose pose estimation
- **reid/centroids-reid/** - Centroid-ReID
- **sam2/** - Sharpness-Aware Minimization optimizer

## Data Flow

### SoccerNet Pipeline

```
Tracklets → Ball Filter + ReID → Outliers → Legibility → Pose → Crops → PARSeq → Consolidate → Final
```

### Intermediate Data Formats

**Input:**
- Tracklet structure: `{tracklet_id}/image_*.jpg`
- Ground truth: JSON `{tracklet_id: jersey_number}`

**Intermediate:**
- Features: `{tracklet_id}_features.npy` (N×256)
- Legibility: `legible.json` `{tracklet_id: [image_paths]}`
- Pose: `pose_results.json` `{pose_results: [{img_name, keypoints}]}`
- Crops: `crops/imgs/{image_name}.jpg`

**Output:**
- Predictions: `jersey_id_results.json` `{image_name: {label, confidence, raw, logits}}`
- Final: `final_results.json` `{tracklet_id: jersey_number}`
