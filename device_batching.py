"""GPU/MPS-gated inference batch sizes for the SoccerNet pipeline.

Batched neural inference is only enabled when a CUDA GPU or Apple Silicon MPS
is available. On CPU, callers should use conservative batch sizes.
"""
import os

import torch


def accelerator_available() -> bool:
    """True if CUDA or Apple MPS can be used for accelerated inference."""
    if torch.cuda.is_available():
        return True
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return True
    return False


def inference_batch_size(*, default_gpu: int = 32, default_cpu: int = 1) -> int:
    """Batch size for inference: ``default_gpu`` on CUDA/MPS, else ``default_cpu``.

    Override on accelerated hardware with env ``JERSEY_INFERENCE_BATCH_GPU``
    (integer). CPU path ignores this env and always uses ``default_cpu``.
    """
    if accelerator_available():
        raw = os.environ.get("JERSEY_INFERENCE_BATCH_GPU")
        if raw is not None and raw.strip() != "":
            return max(1, int(raw))
        return max(1, int(default_gpu))
    return max(1, int(default_cpu))
