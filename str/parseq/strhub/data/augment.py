# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Jersey number recognition: imgaug removed (incompatible with NumPy 2.x).
# Custom RandAugment ops — GaussianBlur (PIL), motion blur (SciPy convolve1d),
# Gaussian/Poisson noise (NumPy). MotionBlur and GaussianNoise enabled;
# SharpnessIncreasing removed (it counteracts blur). Defaults match upstream
# parseq (magnitude=5, num_layers=3); pass larger values from the data module for a stronger schedule.

from functools import partial

import numpy as np
from PIL import ImageFilter, Image
from timm.data import auto_augment

from strhub.data import aa_overrides

aa_overrides.apply()

_OP_CACHE = {}


def _get_op(key, factory):
    try:
        op = _OP_CACHE[key]
    except KeyError:
        op = factory()
        _OP_CACHE[key] = op
    return op


def _get_param(level, img, max_dim_factor, min_level=1):
    max_level = max(min_level, max_dim_factor * max(img.size))
    return round(min(level, max_level))


def gaussian_blur(img, radius, **__):
    """PIL GaussianBlur; radius from RandAugment level via _get_param."""
    radius = _get_param(radius, img, 0.02)
    key = 'gaussian_blur_' + str(radius)
    op = _get_op(key, lambda: ImageFilter.GaussianBlur(radius))
    return img.filter(op)


def motion_blur(img, k, **__):
    """Horizontal motion blur: length-k box filter along width (HWC, axis=1)."""
    k = _get_param(k, img, 0.08, 3) | 1  # odd k for symmetric stencil
    k = max(3, k)
    kernel_1d = np.ones(k, dtype=np.float32) / k
    arr = np.asarray(img, dtype=np.float32)
    # Apply 1-D convolution along width axis for each channel
    from scipy.ndimage import convolve1d
    blurred = convolve1d(arr, kernel_1d, axis=1, mode='reflect')
    return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))


def gaussian_noise(img, scale, **_):
    """Additive N(0, scale) per pixel; scale clamped via _get_param."""
    scale = _get_param(scale, img, 0.25) | 1
    arr = np.asarray(img, dtype=np.float32)
    noise = np.random.normal(0, scale, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def poisson_noise(img, lam, **_):
    """Additive Poisson(lam) per pixel; lam clamped via _get_param."""
    lam = _get_param(lam, img, 0.2) | 1
    arr = np.asarray(img, dtype=np.float32)
    noise = np.random.poisson(lam, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _level_to_arg(level, _hparams, max):
    level = max * level / auto_augment._LEVEL_DENOM
    return level,


_RAND_TRANSFORMS = auto_augment._RAND_INCREASING_TRANSFORMS.copy()
_RAND_TRANSFORMS.remove('SharpnessIncreasing')  # remove, interferes with *blur ops
_RAND_TRANSFORMS.extend([
    'GaussianBlur',
    'MotionBlur',
    'GaussianNoise',
    'PoissonNoise'
])
auto_augment.LEVEL_TO_ARG.update({
    'GaussianBlur': partial(_level_to_arg, max=4),
    'MotionBlur': partial(_level_to_arg, max=20),
    'GaussianNoise': partial(_level_to_arg, max=0.1 * 255),
    'PoissonNoise': partial(_level_to_arg, max=40)
})
auto_augment.NAME_TO_OP.update({
    'GaussianBlur': gaussian_blur,
    'MotionBlur': motion_blur,
    'GaussianNoise': gaussian_noise,
    'PoissonNoise': poisson_noise
})


def rand_augment_transform(magnitude=5, num_layers=3):
    hparams = {
        'rotate_deg': 30,
        'shear_x_pct': 0.9,
        'shear_y_pct': 0.2,
        'translate_x_pct': 0.10,
        'translate_y_pct': 0.30
    }
    ra_ops = auto_augment.rand_augment_ops(magnitude, hparams=hparams, transforms=_RAND_TRANSFORMS)
    choice_weights = [1.0 / len(ra_ops) for _ in range(len(ra_ops))]
    return auto_augment.RandAugment(ra_ops, num_layers, choice_weights)
