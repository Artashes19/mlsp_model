import json
import os
import random
import time
from collections import namedtuple
from typing import Optional

# Define the Split structure matching run.py usage
Split = namedtuple("Split", ["seed", "train_small", "train_full", "validation"])


def ensure_experiments_dir(root_name: str = "exps") -> str:
    """
    Ensures the root exps directory exists (e.g., /path/to/repo/exps).
    Returns absolute path.
    """
    # Assuming run.py is in root, we go relative to CWD or find git root
    root_abs = os.path.abspath(root_name)
    os.makedirs(root_abs, exist_ok=True)
    return root_abs


def ensure_exp_dir(exp_dir_name: Optional[str], root_dir: str) -> str:
    """
    Creates a specific experiment directory. 
    If exp_dir_name is None, creates a timestamped directory.
    """
    if exp_dir_name:
        path = os.path.join(root_dir, exp_dir_name)
    else:
        # Create timestamped dir: YYYY-MM-DD_HH-MM-SS
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(root_dir, timestamp)
    
    os.makedirs(path, exist_ok=True)
    return path


def generate_building_split(
    seed: int,
    val_buildings: list[int],
    n_buildings: int = 25,
    train_small_n: int = 7,
) -> Split:
    """
    Generates a consistent split of buildings [1..25].
    train_small is a subset of train_full.
    validation is the remaining buildings.
    """
    all_buildings = list(range(1, n_buildings + 1))
    
    rng = random.Random(seed)
    rng.shuffle(all_buildings)
    
    # First select the validation set (size = Total - Full Train)
    # Or better, select train_full first, then subset train_small from it?
    # run.py implies: train_small_n=7, train_full_n=20.
    # So we pick 20 for full training, and the remaining 5 are validation.
    # Then from the 20, we pick 7 for small training.
    
    validation = val_buildings
    train_full = sorted(set(all_buildings) - set(val_buildings))
    
    # Now select train_small as a subset of train_full
    # Use the same RNG state? Or re-seed? 
    # shuffle train_full to pick subset
    train_full_shuffled = list(train_full)
    rng.shuffle(train_full_shuffled)
    train_small = sorted(train_full_shuffled[:train_small_n])
    
    return Split(
        seed=seed,
        train_small=train_small,
        train_full=train_full,
        validation=validation
    )


def write_split_json(exp_dir: str, split: Split) -> None:
    """Saves the split to split.json in the experiment dir."""
    path = os.path.join(exp_dir, "split.json")
    data = {
        "seed": split.seed,
        "train_small": split.train_small,
        "train_full": split.train_full,
        "validation": split.validation
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_split_json(exp_dir: str) -> Optional[Split]:
    """Reads split.json if it exists."""
    path = os.path.join(exp_dir, "split.json")
    if not os.path.isfile(path):
        return None
    
    try:
        with open(path, "r") as f:
            data = json.load(f)
        
        return Split(
            seed=data.get("seed", 0),
            train_small=data.get("train_small", []),
            train_full=data.get("train_full", []),
            validation=data.get("validation", [])
        )
    except Exception:
        return None
