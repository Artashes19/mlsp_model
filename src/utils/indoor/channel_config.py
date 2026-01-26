"""
Channel configuration for featurizer output tensors.

The CHANNEL_ORDER list defines which feature goes into which channel.
Index in the list = index in the output tensor.

To reorder channels: just reorder this list.
To add a channel: add it to the list.
To remove a channel: remove it from the list.

Available channel names:
    - "reflectance"
    - "transmittance"
    - "distance"
    - "antenna_gain"
    - "freq_sin_1", "freq_cos_1", "freq_sin_2", "freq_cos_2" (Fourier frequency encoding)
    - "mask"
    - "floor_plan"
    - "sparse"
"""

# ============================================================================
# Channel Order Configuration
# ============================================================================
# Position in this list = tensor channel index
CHANNEL_ORDER = [
    "reflectance",      # 0
    "transmittance",    # 1
    "distance",         # 2
    "antenna_gain",     # 3
    "freq_sin_1",       # 4
    "freq_cos_1",       # 5
    "freq_sin_2",       # 6
    "freq_cos_2",       # 7
    "mask",             # 8
    "floor_plan",       # 9
    "sparse",           # 10
]

# ============================================================================
# Derived Constants (computed from CHANNEL_ORDER)
# ============================================================================
NUM_CHANNELS = len(CHANNEL_ORDER)


def get_channel_index(name: str) -> int:
    """Get the tensor index for a channel by name. Raises ValueError if not found."""
    return CHANNEL_ORDER.index(name)


# Convenience constants for common lookups
REFLECTANCE_CHANNEL = get_channel_index("reflectance") if "reflectance" in CHANNEL_ORDER else None
TRANSMITTANCE_CHANNEL = get_channel_index("transmittance") if "transmittance" in CHANNEL_ORDER else None
DISTANCE_CHANNEL = get_channel_index("distance") if "distance" in CHANNEL_ORDER else None
ANTENNA_GAIN_CHANNEL = get_channel_index("antenna_gain") if "antenna_gain" in CHANNEL_ORDER else None
MASK_CHANNEL = get_channel_index("mask") if "mask" in CHANNEL_ORDER else None
FLOOR_PLAN_CHANNEL = get_channel_index("floor_plan") if "floor_plan" in CHANNEL_ORDER else None
SPARSE_CHANNEL = get_channel_index("sparse") if "sparse" in CHANNEL_ORDER else None

# Frequency channels
FREQ_SIN_1_CHANNEL = get_channel_index("freq_sin_1") if "freq_sin_1" in CHANNEL_ORDER else None
FREQ_COS_1_CHANNEL = get_channel_index("freq_cos_1") if "freq_cos_1" in CHANNEL_ORDER else None
FREQ_SIN_2_CHANNEL = get_channel_index("freq_sin_2") if "freq_sin_2" in CHANNEL_ORDER else None
FREQ_COS_2_CHANNEL = get_channel_index("freq_cos_2") if "freq_cos_2" in CHANNEL_ORDER else None
