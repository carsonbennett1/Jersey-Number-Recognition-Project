## Environment & GPU Compatibility Fixes
**Modified files to fix this issue:**
- configuration.py
- pose.py
- centroid_reid.py
- str.py

**Reason:**  
Fix compatibility with RTX 5070 Ti / CUDA 12.8 and ensure correct environment + torch loading behavior.

---

## Dataset Path & Root Directory Errors
**Modified files to fix this issue:**
- configuration.py
- pose.py
- centroid_reid.py
- str.py
- main.py

**Reason:**  
Fix incorrect dataset paths, repo-root resolution, and broken relative paths causing file-not-found errors.

---

## Torch / Checkpoint Loading Errors
**Modified files to fix this issue:**
- legibility_classifier.py
- centroid_reid.py
- str.py

**Reason:**  
Fix PyTorch 2.x `torch.load` and Lightning checkpoint incompatibility (`weights_only=False`, Lightning patching).

---

## Windows / Performance & Inference Optimizations
**Modified files to fix this issue:**
- legibility_classifier.py
- centroid_reid.py
- pose.py

**Reason:**  
Improve GPU inference speed, remove unsafe AMP usage, optimize DataLoader and inference flow.

---

## Pipeline Execution & Resume Errors
**Modified files to fix this issue:**
- main.py
- helpers.py
- gaussian_outliers.py
- centroid_reid.py

**Reason:**  
Fix crashes from non-directory files, add skip/resume logic, ensure pipeline continues if outputs exist.

---

## Visualization / Optional Processing Fixes
**Modified files to fix this issue:**
- pose.py

**Reason:**  
Prevent crashes from visualization when output directory not provided and remove FP16 heatmap issues.

---

## Model Loading & Reuse Efficiency
**Modified files to fix this issue:**
- legibility_classifier.py
- main.py

**Reason:**  
Avoid repeated model loading per tracklet and improve runtime efficiency.