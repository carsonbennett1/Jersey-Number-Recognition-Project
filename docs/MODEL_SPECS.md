# Jersey Number Recognition - Model Specifications

## PARSeq Transformer (Scene Text Recognition)

The STR model is **PARSeq** (Permutation Autoregressive Sequence), configured for jersey numbers (0-99).

### Architecture Specs

Config source: `str/parseq/configs/model/parseq.yaml` and `str/parseq/configs/main.yaml`

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Input** | 32 x 128 (H x W) | Image size for jersey number crops |
| **Patch size** | 4 x 8 | Encoder patch dimensions |
| **Embed dim** | 384 | Hidden dimension |
| **Encoder** | | |
| - Depth | 12 layers | ViT encoder blocks |
| - Heads | 6 | Attention heads |
| - MLP ratio | 4 | FFN hidden = 4 * embed_dim |
| **Decoder** | | |
| - Depth | 1 layer | Transformer decoder |
| - Heads | 12 | Attention heads |
| - MLP ratio | 4 | FFN hidden = 4 * embed_dim |
| **Max label length** | 2 | Digits (0-99) |
| **Charset** | E0123456789 | E = end token, digits 0-9 |
| **Permutations** | 6 | Training permutation count |
| **Refine iters** | 1 | Iterative refinement at inference |
| **Decode mode** | AR (autoregressive) | Sequential decoding |

### Encoder (Vision Transformer)

- Extends `timm.models.vision_transformer.VisionTransformer`
- Patch embedding: 4x8 patches on 32x128 input → 4x16 = 64 tokens
- Output: 64 x 384 memory

### Decoder (Two-Stream Attention)

- Self-attention + cross-attention (XLNet-style)
- Pre-LayerNorm
- Learnable position queries: `(1, max_label_length+1, 384)` = (1, 3, 384)
- Output head: Linear(384 → 11) for charset size minus BOS/PAD

### Inference (str.py)

- Logits sliced to `[:, :3, :11]` (3 positions: BOS, digit1, digit2; 11 classes)
- Tokenizer decodes to string (e.g., "7", "23")
- Confidence = product of per-position softmax max values

---

## Legibility Classifier

### ResNet34-based (Default)

- Backbone: ResNet34 (ImageNet pretrained)
- Head: Linear(512 → 1)
- Activation: Sigmoid
- Loss: BCE
- Input: 224×224

### ResNet18-based

- Backbone: ResNet18 (ImageNet pretrained)
- Head: Linear(512 → 1)
- Activation: Sigmoid

### Vision Transformer (Optional)

- Backbone: ViT-B/16 (ImageNet pretrained)
- Head: Linear(768 → 1)
- Activation: Sigmoid
- Input: 224×224

---

## Centroid-ReID (Re-identification)

- Backbone: ResNet50
- Feature dimension: 256
- Trained on Market-1501 or DukeMTMC-ReID
- Output: 256-D feature vectors per image

---

## ViTPose (Pose Estimation)

- Architecture: ViT-Huge
- Input: 256×192
- Output: 17 COCO keypoints
- Used keypoints: shoulders (5, 6), hips (11, 12)

---

## Training Commands

### PARSeq STR (SoccerNet)

```bash
conda run -n parseq2 python train.py +experiment=parseq dataset=real \
  data.root_dir=<path_to_lmdb> trainer.max_epochs=25 trainer.devices=1 \
  trainer.val_check_interval=1 data.batch_size=128 data.max_label_length=2
```

### PARSeq STR (Hockey)

```bash
conda run -n parseq2 python train.py +experiment=parseq dataset=real \
  data.root_dir=<path_to_jersey_numbers_lmdb> trainer.max_epochs=25 \
  trainer.devices=1 trainer.val_check_interval=1 data.batch_size=128 \
  data.max_label_length=2
```

Or via main.py:

```bash
python main.py SoccerNet train --train_str
python main.py Hockey train --train_str
```

### Legibility Classifier

```bash
# Train (Hockey)
python legibility_classifier.py --train --arch resnet34 --sam --data <dataset_dir> \
  --trained_model_path ./experiments/hockey_legibility.pth

# Finetune (SoccerNet from Hockey)
python legibility_classifier.py --finetune --arch resnet34 --sam --data <dataset_dir> \
  --full_val_dir <dataset_dir>/val --trained_model_path ./experiments/hockey_legibility.pth \
  --new_trained_model_path ./experiments/sn_legibility.pth
```

---

## Model Checkpoints

| Model | SoccerNet | Hockey |
|-------|-----------|--------|
| Legibility | `models/legibility_resnet34_soccer_20240215.pth` | `models/legibility_resnet34_hockey_20240201.pth` |
| STR (PARSeq) | `models/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt` | `models/parseq_epoch=3-step=95-val_accuracy=98.7903-val_NED=99.3952.ckpt` |
| ViTPose | `pose/ViTPose/checkpoints/vitpose-h.pth` | same |
| ReID | `reid/centroids-reid/models/market1501_resnet50_256_128_epoch_120.ckpt` | N/A (Hockey skips ReID) |
