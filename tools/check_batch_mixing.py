from __future__ import annotations

"""
Diagnostic: check whether training batches mix samples from different buildings
or appear ordered (unshuffled). Prints unique building IDs and counts per batch.
"""

import argparse
import ast
import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from data.dataset import IndoorRadioMapDataset, gather_task2_samples, parse_meta


def load_yaml(path: str | Path) -> Dict:
    try:
        import yaml  # type: ignore

        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        # Fallback: simple parser for key: value lines
        data: Dict[str, object] = {}
        with open(path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                if val == "":
                    continue
                low = val.lower()
                if low == "true":
                    data[key] = True
                    continue
                if low == "false":
                    data[key] = False
                    continue
                try:
                    data[key] = ast.literal_eval(val)
                except Exception:
                    data[key] = val
        return data


def split_pairs_by_hash(pairs: List, val_ratio: float, seed: int) -> Tuple[List, List]:
    def hkey(stem: str) -> float:
        h = hashlib.md5((stem + str(seed)).encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    train, val = [], []
    for sp in pairs:
        key = Path(sp.input_path).stem
        if hkey(key) < val_ratio:
            val.append(sp)
        else:
            train.append(sp)
    return train, val


def split_pairs_by_building(pairs: List, train_buildings: List[int], val_buildings: List[int]) -> Tuple[List, List]:
    train_set = set(int(b) for b in train_buildings)
    val_set = set(int(b) for b in val_buildings)
    if train_set & val_set:
        raise ValueError(f"train_buildings and val_buildings overlap: {sorted(train_set & val_set)}")

    train: List = []
    val: List = []
    for sp in pairs:
        meta = parse_meta(Path(sp.input_path).name)
        b = meta.get("building")
        if b in train_set:
            train.append(sp)
        elif b in val_set:
            val.append(sp)
    return train, val


def buildings_from_meta_batch(meta_batch) -> List[int]:
    # meta can be either a dict of lists (default collate) or a list of dicts
    if isinstance(meta_batch, dict):
        bvals = meta_batch.get("building", [])
        return [int(b) for b in bvals]
    if isinstance(meta_batch, list):
        return [int(m.get("building")) for m in meta_batch if isinstance(m, dict) and "building" in m]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Check batch mixing across buildings")
    parser.add_argument("--data", type=str, required=True, help="Path to data YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to training YAML config")
    parser.add_argument("--num-batches", type=int, default=8, help="Number of training batches to inspect")
    args = parser.parse_args()

    data_cfg = load_yaml(args.data)
    train_cfg = load_yaml(args.config)

    resize_hw = tuple(data_cfg.get("resize", [256, 256]))  # type: ignore[assignment]
    val_ratio = float(data_cfg.get("val_ratio", 0.1))
    seed = int(data_cfg.get("seed", 42))
    split_mode = str(data_cfg.get("split_mode", "hash")).lower()
    y_db_max = float(data_cfg.get("y_db_max", 160.0))

    all_pairs = gather_task2_samples(Path(data_cfg["data_root"]), split="train")
    if split_mode == "building":
        train_buildings = [int(b) for b in data_cfg.get("train_buildings", [])]
        val_buildings = [int(b) for b in data_cfg.get("val_buildings", [])]
        if not train_buildings or not val_buildings:
            raise ValueError("split_mode=building requires train_buildings and val_buildings in data config")
        train_pairs, _ = split_pairs_by_building(all_pairs, train_buildings, val_buildings)
        print(f"Split mode: building | train_buildings={sorted(set(train_buildings))}")
    else:
        train_pairs, _ = split_pairs_by_hash(all_pairs, val_ratio=val_ratio, seed=seed)
        print(f"Split mode: hash | val_ratio={val_ratio} | seed={seed}")

    ds_train = IndoorRadioMapDataset(
        root=data_cfg["data_root"],
        split="train",
        resize_hw=resize_hw,
        file_pairs=train_pairs,
        y_db_max=y_db_max,
    )

    batch_size = int(train_cfg.get("batch_size", 1))
    num_workers = int(train_cfg.get("num_workers", 4))
    pin_memory = bool(train_cfg.get("pin_memory", True))

    loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"Inspecting first {args.num_batches} batches | batch_size={batch_size} | shuffle=True")

    total_seen = 0
    cross_building_batches = 0
    single_building_batches = 0
    building_sequence: List[int] = []

    for bi, (x, y, m, meta) in enumerate(loader_train):
        if bi >= args.num_batches:
            break
        blds = buildings_from_meta_batch(meta)
        building_sequence.extend(blds)
        total_seen += len(blds)
        uniq = sorted(set(blds))
        counts = Counter(blds)
        if len(uniq) > 1:
            cross_building_batches += 1
        else:
            single_building_batches += 1
        print(f"Batch {bi:03d}: buildings={uniq} | counts={dict(counts)}")

    # Simple measure of ordering: fraction of adjacent pairs with the same building
    same_adjacent = 0
    for i in range(1, len(building_sequence)):
        if building_sequence[i] == building_sequence[i - 1]:
            same_adjacent += 1
    frac_same_adjacent = (same_adjacent / max(1, len(building_sequence) - 1)) if building_sequence else 0.0

    print("\nSummary:")
    print(f"  Batches inspected: {min(args.num_batches, len(loader_train))}")
    print(f"  Cross-building batches: {cross_building_batches}")
    print(f"  Single-building batches: {single_building_batches}")
    print(f"  Samples inspected: {total_seen}")
    print(f"  Adjacent same-building fraction (sequence seen): {frac_same_adjacent:.3f}")


if __name__ == "__main__":
    main()



