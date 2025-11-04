from __future__ import annotations

"""
IndoorRadioMapDataset for Task_2 PNGs.
Pairs input RTD PNGs with output y PNGs (train) and yields X=[R,T,D], y, free_mask, meta.
"""

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import (
    resize_bilinear,
    resize_nearest,
    bit_depth_from_mode,
    normalize_unit,
    nchw_tensor_from_hwc,
)


@dataclass
class SamplePaths:
    input_path: Path
    output_path: Optional[Path]


class IndoorRadioMapDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        resize_hw: Tuple[int, int] = (256, 256),
        file_pairs: Optional[List[SamplePaths]] = None,
        y_db_max: float = 160.0,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.resize_hw = resize_hw
        self.y_db_max = float(y_db_max)
        self.samples: List[SamplePaths] = (
            file_pairs if file_pairs is not None else gather_task2_samples(self.root, split)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):  # type: ignore[override]
        sp = self.samples[idx]
        with Image.open(sp.input_path) as img_in:
            img_in = img_in.convert("RGB")
            # resize
            img_in = resize_bilinear(img_in, self.resize_hw)
            bit_in = bit_depth_from_mode(img_in.mode)
            x_arr = np.array(img_in)
            x_arr = normalize_unit(x_arr, bit_in)  # [H,W,3] in [0,1]

        if sp.output_path is not None and sp.output_path.exists():
            with Image.open(sp.output_path) as img_out:
                img_out = img_out.convert("L")
                img_out = resize_bilinear(img_out, self.resize_hw)
                y_arr = np.array(img_out).astype(np.float32)
                # Normalize outputs by dB max (default 160), NOT bit depth
                y_arr = y_arr / max(1.0, self.y_db_max)
        else:
            # test split: no outputs
            h, w = self.resize_hw
            y_arr = np.zeros((h, w), dtype=np.float32)

        # free mask: assume all free if not provided
        free_mask = np.ones_like(y_arr, dtype=np.float32)

        # to tensors (C,H,W)
        x = nchw_tensor_from_hwc(x_arr)
        y = torch.from_numpy(y_arr[None, ...].astype(np.float32))
        m = torch.from_numpy(free_mask[None, ...])

        meta = parse_meta(sp.input_path.name)
        return x, y, m, meta


def find_task2_paths(root: Path) -> Dict[str, Optional[Path]]:
    train_inputs = root / "train/Inputs/Task_2_ICASSP"
    train_outputs = root / "train/Outputs/Task_2_ICASSP"
    test_inputs = root / "test/Inputs/Task_2"
    return {
        "train_inputs": train_inputs if train_inputs.is_dir() else None,
        "train_outputs": train_outputs if train_outputs.is_dir() else None,
        "test_inputs": test_inputs if test_inputs.is_dir() else None,
    }


def list_pngs(folder: Optional[Path]) -> List[Path]:
    if folder is None:
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() == ".png" and p.is_file()])


def pair_train_files(inputs: List[Path], outputs: List[Path]) -> List[SamplePaths]:
    out_map: Dict[str, Path] = {p.stem: p for p in outputs}
    pairs: List[SamplePaths] = []
    for ip in inputs:
        op = out_map.get(ip.stem)
        if op is not None:
            pairs.append(SamplePaths(input_path=ip, output_path=op))
    return pairs


def gather_task2_samples(root: Path, split: str) -> List[SamplePaths]:
    paths = find_task2_paths(root)
    if split in ("train", "val"):
        # Trainer will perform actual split; here we just return all paired samples
        inputs = list_pngs(paths["train_inputs"])
        outputs = list_pngs(paths["train_outputs"])
        return pair_train_files(inputs, outputs)
    elif split == "test":
        inputs = list_pngs(paths["test_inputs"])
        return [SamplePaths(input_path=ip, output_path=None) for ip in inputs]
    else:
        raise ValueError(f"Unknown split: {split}")


_META_RE = re.compile(r"^B(?P<bld>\d+)_Ant(?P<ant>\d+)_f(?P<freq>\d+)_S(?P<idx>\d+)\.png$")


def parse_meta(filename: str) -> Dict[str, object]:
    m = _META_RE.match(filename)
    if not m:
        return {"scene_id": filename}
    d = m.groupdict()
    return {
        "building": int(d["bld"]),
        "antenna": int(d["ant"]),
        "freq": int(d["freq"]),
        "sample_idx": int(d["idx"]),
        "scene_id": filename,
    }


