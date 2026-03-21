"""Download centroid-reid model weights (run: python download_reid_models.py)"""
import os
import gdown

models_dir = "./reid/centroids-reid/models"
os.makedirs(models_dir, exist_ok=True)

models = [
    ("1ZFywKEytpyNocUQd2APh2XqTe8X0HMom", "market1501_resnet50_256_128_epoch_120.ckpt"),
    ("1w9yzdP_5oJppGIM4gs3cETyLujanoHK8", "dukemtmcreid_resnet50_256_128_epoch_120.ckpt"),
]

for file_id, filename in models:
    path = os.path.join(models_dir, filename)
    if os.path.isfile(path) and os.path.getsize(path) > 10_000_000:
        print(f"{filename} already exists ({os.path.getsize(path) // 1_000_000}MB)")
        continue
    if os.path.isfile(path):
        os.remove(path)
    print(f"Downloading {filename}...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)
    print(f"Done: {os.path.getsize(path) // 1_000_000}MB")
