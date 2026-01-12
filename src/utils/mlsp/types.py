import os
from dataclasses import asdict, dataclass
from typing import Optional, Union

import torch


@dataclass
class RadarSample:
    file_name: str
    task_idx: int
    pl_clip: Optional[torch.Tensor]
    use_approximator_feature: bool
    use_transmittance_loss: bool
    H: int
    W: int
    x_ant: float
    y_ant: float
    azimuth: float
    freq_MHz: float
    input_img: torch.Tensor  # In format (C, H, W)
    output_img: torch.Tensor  # In format (H, W) or (1, H, W)
    radiation_pattern: torch.Tensor
    pixel_size: float = 0.25
    mask: Union[torch.Tensor, None] = None
    floor_plan: Union[torch.Tensor, None] = None
    
    def asdict(self):
        return asdict(self)


@dataclass
class RadarSampleInputs:
    file_name: str
    freq_MHz: float
    input_file: str
    output_file: Union[str, None]
    position_file: str
    radiation_pattern_file: str
    sampling_position: int
    ids: Optional[tuple[int, int, int, int]] = None
    
    def asdict(self):
        return asdict(self)
    
    def __post_init__(self):
        if self.ids and not all(isinstance(i, int) for i in self.ids):
            raise ValueError("All IDs must be integers")
        
        if not isinstance(self.freq_MHz, (int, float)):
            raise ValueError("freq_MHz must be a number")
        
        for path_attr in ['input_file', 'position_file', 'radiation_pattern_file']:
            path = getattr(self, path_attr)
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
