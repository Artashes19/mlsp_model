import random
from collections import namedtuple

# Define the Split structure matching run.py usage
Split = namedtuple("Split", ["seed", "train_small", "train_full", "validation"])


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
