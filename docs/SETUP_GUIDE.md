# Complete Setup Guide - Jersey Number Recognition

This guide will walk you through setting up the Jersey Number Recognition project from scratch. Follow these steps carefully to ensure everything works correctly.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Automated Environment Setup](#automated-environment-setup)
4. [Manual Environment Setup (Alternative)](#manual-environment-setup-alternative)
5. [GPU Setup](#gpu-setup)
6. [Data Setup](#data-setup)
7. [Verify Installation](#verify-installation)
8. [Running the Pipeline](#running-the-pipeline)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
1. **Git** - For cloning the repository
   - Download: https://git-scm.com/downloads
   - Verify: `git --version`

2. **Conda (Miniconda or Anaconda)** - For managing Python environments
   - Download Miniconda: https://docs.conda.io/en/latest/miniconda.html
   - **IMPORTANT**: After installation, restart your terminal/command prompt
   - Verify: `conda --version`

3. **NVIDIA GPU with CUDA support** (recommended but optional)
   - Check your GPU: `nvidia-smi` (Windows/Linux)
   - For NVIDIA RTX 40xx/50xx series, you need CUDA 12.8+
   - For Apple Silicon Macs, MPS (Metal Performance Shaders) is automatically detected

### System Requirements
- **OS**: Windows 10/11, macOS (Intel/Apple Silicon), or Linux
- **RAM**: 16GB minimum, 32GB recommended
- **Disk Space**: ~20GB for code, models, and environments
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended (can run on CPU but slower)

---

## Initial Setup

### Step 1: Clone the Repository

```bash
git clone --recurse-submodules https://github.com/YOUR_USERNAME/Jersey-Number-Recognition-Project.git
cd Jersey-Number-Recognition-Project
```

**Note**: The `--recurse-submodules` flag is important as this project includes submodules.

### Step 2: Verify Directory Structure

After cloning, you should see:
```
Jersey-Number-Recognition-Project/
├── main.py
├── configuration.py
├── setup.py
├── pose/
├── reid/
├── str/
├── models/
├── data/
└── ...
```

---

## Automated Environment Setup

The easiest way to set up all dependencies is using the automated `setup.py` script.

### Option 1: Setup for SoccerNet (includes all components)

```bash
# This will setup everything: pose, reid, str, and download models
python setup.py SoccerNet
```


### What the Automated Setup Does:

1. **Creates 3 Conda Environments**:
   - `vitpose2` (Python 3.10) - For pose estimation
   - `parseq2` (Python 3.9) - For scene text recognition
   - `centroids` (Python 3.8) - For re-identification (SoccerNet only)
   - `jersey` (Python 3.10+) - Main environment (you need to create this manually, see below)

2. **Clones Required Repositories**:
   - ViTPose (pose estimation)
   - PARSeq (scene text recognition)
   - Centroid-ReID (re-identification)
   - SAM (Sharpness Aware Minimization optimizer)

3. **Downloads Pre-trained Models**:
   - ViTPose checkpoint
   - Fine-tuned STR models (SoccerNet/Hockey)
   - Legibility classifier weights
   - ReID model weights

**⏱️ Expected Time**: 15-30 minutes depending on your internet speed

---

## Manual Environment Setup (Alternative)

If the automated setup fails or you prefer manual control, follow these steps:

### 1. Create the Main `jersey` Environment

```bash
# For NVIDIA GPUs with CUDA 12.8+
conda create -n jersey python=3.10 -y
conda activate jersey
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python tqdm yacs pytorch-lightning==1.9.5 scikit-learn matplotlib seaborn pandas
```

```bash
# For Apple Silicon Macs (M1/M2/M3)
conda create -n jersey python=3.10 -y
conda activate jersey
pip install torch torchvision torchaudio
pip install opencv-python tqdm yacs pytorch-lightning==1.9.5 scikit-learn matplotlib seaborn pandas
```

```bash
# For CPU-only (no GPU)
conda create -n jersey python=3.10 -y
conda activate jersey
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python tqdm yacs pytorch-lightning==1.9.5 scikit-learn matplotlib seaborn pandas
```

### 2. Create `vitpose2` Environment (Pose Estimation)

```bash
conda create -n vitpose2 python=3.10 -y
conda activate vitpose2

# Install PyTorch with CUDA 12.8 (for NVIDIA GPUs)
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu128

# Install mmcv-full (required for ViTPose)
pip install openmim
mim install mmcv-full==1.7.2

# Install ViTPose
cd pose/ViTPose
pip install -v -e .
pip install timm==0.4.9 einops xtcocotools
cd ../..
```

**Important**: ViTPose requires older versions of mmcv and specific Python versions. If you get errors, use Python 3.10.

### 3. Create `parseq2` Environment (Scene Text Recognition)

```bash
conda create -n parseq2 python=3.9 -y
conda activate parseq2

# Navigate to parseq directory
cd str/parseq

# Install PyTorch
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# Install parseq requirements
pip install -r requirements/core.txt
pip install -e .[train,test]

# Install additional dependencies
pip install pytorch-lightning==1.9.5 lmdb timm nltk imgaug

cd ../..
```

### 4. Create `centroids` Environment (ReID - SoccerNet only)

```bash
conda create -n centroids python=3.8 -y
conda activate centroids

cd reid/centroids-reid
pip install -r requirements.txt
cd ../..
```



---

## GPU Setup

### For NVIDIA GPUs

1. **Check CUDA Version**:
   ```bash
   nvidia-smi
   ```
   Look for "CUDA Version" in the top-right corner.

2. **Install Matching PyTorch**:
   - CUDA 12.8+: Use the commands in the environment setup above
   - CUDA 11.8: Replace `cu128` with `cu118` in pip install commands

3. **Verify GPU Detection**:
   ```bash
   conda activate jersey
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```

   Expected output:
   ```
   CUDA available: True
   CUDA version: 12.8
   Device: NVIDIA GeForce RTX 5070 Ti
   ```

### For Apple Silicon (M1/M2/M3)

PyTorch automatically uses Metal Performance Shaders (MPS). Verify:
```bash
conda activate jersey
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### For CPU-Only

No additional setup needed. The code automatically falls back to CPU if no GPU is detected.

---

## Data Setup

### SoccerNet Dataset

1. **Download the Dataset**:
   - Visit: https://github.com/SoccerNet/sn-jersey
   - Follow instructions to download `jersey-2023` dataset
   
2. **Extract and Place**:
   ```
   Jersey-Number-Recognition-Project/
   └── data/
       └── SoccerNet/
           └── jersey-2023/
               ├── test/
               │   └── test/
               │       ├── images/
               │       └── test_gt.json
               ├── val/
               └── train/
   ```

3. **Verify Structure**:
   ```bash
   # Windows
   dir data\SoccerNet\jersey-2023\test\test\images
   
   # Mac/Linux
   ls data/SoccerNet/jersey-2023/test/test/images
   ```


---

## Verify Installation

### Step 1: Test Configuration

```bash
conda activate jersey
python -c "import configuration as config; print('Project root:', config._main_repo); print('Conda base:', config._conda_base)"
```

Expected output:
```
Project root: C:\Users\YourName\Jersey-Number-Recognition-Project
Conda base: C:\Users\YourName\miniconda3\envs
```

### Step 2: Test Each Environment

```bash
# Test jersey environment
conda activate jersey
python -c "import torch; import cv2; import pytorch_lightning; print('jersey: OK')"

# Test vitpose2 environment
conda activate vitpose2
python -c "import mmcv; import mmpose; print('vitpose2: OK')"

# Test parseq2 environment
conda activate parseq2
cd str/parseq
python -c "from strhub.models.parseq.system import PARSeq; print('parseq2: OK')"
cd ../..

# Test centroids environment (SoccerNet only)
conda activate centroids
python -c "import pytorch_lightning; print('centroids: OK')"
```

### Step 3: Verify Models Exist

```bash
# Check if all required models are downloaded
ls -lh models/
ls -lh pose/ViTPose/checkpoints/
ls -lh reid/centroids-reid/models/  # SoccerNet only
```

You should see:
- `models/sn_finetuned_jerseys.ckpt` (or `hockey_finetuned_jerseys.ckpt`)
- `models/sn_legibility.pth` (or `hockey_legibility.pth`)
- `pose/ViTPose/checkpoints/vitpose-h.pth`
- `reid/centroids-reid/models/market1501_resnet50_256_128_epoch_120.ckpt`

---

## Running the Pipeline

### Full Pipeline - SoccerNet Test Set

```bash
conda activate jersey
python main.py SoccerNet test
```

**Expected Output**:
```
Determine soccer ball: SKIPPED (output exists) OR Processing...
Generate features: using GPU (CUDA)
100%|██████████████| 1211/1211 [00:58<00:00, 20.67it/s]
Done generating features
...
Correct 1051 out of 1146. Accuracy 91.71%
```

**⏱️ Expected Time**:
- First run: 30-60 minutes (depending on GPU)
- Subsequent runs: 1-5 minutes (skips already processed steps)


### Selective Pipeline Execution

Edit `main.py` to run only specific steps. Find the `actions` dictionary:

```python
actions = {
    'ball': True,      # Detect soccer ball
    'reid': True,      # Generate ReID features
    'outliers': True,  # Remove outliers
    'legible': True,   # Classify legibility
    'pose': True,      # Detect pose
    'crops': True,     # Generate crops
    'str': True,       # Recognize jersey numbers
    'combine': True,   # Combine results
    'eval': True       # Evaluate accuracy
}
```

Set any step to `False` to skip it.

---
### Common errors

#### 1. "FileNotFoundError" or Path Issues

**Problem**: Files not found at expected locations

**Solution**:
Check `configuration.py` paths match your setup:

```bash
# Set environment variables if needed
export JERSEY_PROJECT_ROOT="/path/to/Jersey-Number-Recognition-Project"
export JERSEY_CONDA_BASE="/path/to/miniconda3/envs"

# Windows PowerShell
$env:JERSEY_PROJECT_ROOT="C:\path\to\Jersey-Number-Recognition-Project"
$env:JERSEY_CONDA_BASE="C:\path\to\miniconda3\envs"

# Windows CMD
set JERSEY_PROJECT_ROOT=C:\path\to\Jersey-Number-Recognition-Project
set JERSEY_CONDA_BASE=C:\path\to\miniconda3\envs
```

---

#### 2. "Process Stuck" or Hangs

**Problem**: Pipeline hangs at a step (e.g., "Generate features")

**Solutions**:
- On Windows, `Ctrl+C` may not work. Use `Ctrl+Break` or close the terminal.
- Check GPU usage: `nvidia-smi` (should show Python process using GPU)
- The pipeline has resume capability - restart and it will skip completed steps

---

## Quick Reference - Common Commands

```bash
# Activate main environment
conda activate jersey

# Run full pipeline (SoccerNet)
python main.py SoccerNet test

# Run full pipeline (Hockey)
python main.py Hockey test

# Check GPU status
nvidia-smi

# Verify PyTorch GPU
python -c "import torch; print(torch.cuda.is_available())"

# List conda environments
conda env list

# Remove an environment (if you need to start over)
conda env remove -n jersey

# Update a package
conda activate jersey
pip install --upgrade <package-name>
```

---

## Environment Summary

| Environment | Python | Purpose | Key Packages |
|-------------|--------|---------|--------------|
| `jersey` | 3.10+ | Main pipeline, legibility, ReID | torch, cv2, pytorch-lightning |
| `vitpose2` | 3.10 | Pose estimation | mmcv, mmpose, timm |
| `parseq2` | 3.9 | Scene text recognition | parseq, lmdb, nltk |
| `centroids` | 3.8 | Re-identification (SoccerNet) | pytorch-lightning, sklearn |

---

**Last Updated**: February 2026

