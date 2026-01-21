import logging
import os
import random
import sys
import warnings

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from src.train import train_prep

log = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
project_root = os.path.dirname(os.path.abspath(__file__))

hydra.core.global_hydra.GlobalHydra.instance().clear()

# Ensure local src package is importable even if Hydra changes CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(config: DictConfig) -> None:
    from src import utils
    
    warnings.filterwarnings("ignore", ".*beta state*")
    
    terminal_col = config.get("terminal_col")
    if terminal_col:
        terminal_row = config.get("terminal_row", 24)
        utils.set_winsize(sys.stdin, terminal_col, terminal_row)
        utils.set_winsize(sys.stderr, terminal_col, terminal_row)
        utils.set_winsize(sys.stdout, terminal_col, terminal_row)
    
    if config.seed == -1:
        config.seed = random.randint(0, 10 ** 8)
    
    seed_everything(config.seed)
    log.info(f"Run dir: {os.path.realpath('./')}")
    
    # Performance: enable fast conv algo search and use fast matmul on Ampere+
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")
    
    if config["name"] == "train":
        return train_prep(config, project_root)
    elif config["name"] == "manifest":
        from src.manifest import manifest_prep
        
        if config.get("print_config"):
            utils.print_config(config, fields=tuple(config.keys()), resolve=True)
        return manifest_prep(config, project_root)
    elif config["name"] == "inference":
        from src.inference import inference_prep
        
        if config.get("print_config"):
            utils.print_config(config, fields=tuple(config.keys()), resolve=True)
        return inference_prep(config, project_root)


if __name__ == "__main__":
    main()
