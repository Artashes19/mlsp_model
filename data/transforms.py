from __future__ import annotations

"""Basic transforms: resize to 256x256 and normalization utilities."""

from typing import Tuple

import numpy as np
from PIL import Image
import torch


def resize_bilinear(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return img.resize(size[::-1], Image.BILINEAR)  # size=(H,W) -> PIL expects (W,H)


def resize_nearest(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return img.resize(size[::-1], Image.NEAREST)


def to_float32(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32)


def bit_depth_from_mode(mode: str) -> int:
    # Common modes: 'L' (8), 'RGB' (8), 'I;16' (16)
    if mode == "I;16":
        return 16
    return 8


def normalize_unit(arr: np.ndarray, bit_depth: int) -> np.ndarray:
    denom = float((1 << bit_depth) - 1)
    return (arr.astype(np.float32)) / denom


def nchw_tensor_from_hwc(arr_hwc: np.ndarray) -> torch.Tensor:
    # arr_hwc: [H,W,C]
    arr_chw = np.transpose(arr_hwc, (2, 0, 1)).astype(np.float32)
    return torch.from_numpy(arr_chw)



