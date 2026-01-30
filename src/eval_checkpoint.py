"""
Minimal checkpoint evaluation for Kaggle test tasks.
"""
import csv
import logging
import os
import re
import tempfile

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.algorithms.indoor import Indoor
from src.datamodules.datasets.indoor import PathlossDataset
from src.datamodules.indoor import IndoorDatamodule
from src.evaluate import submit_to_kaggle
from src.inference import generate_csv_rows
from src.utils.indoor.channel_config import NUM_CHANNELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


TASKS: dict[str, tuple[str, str]] = {
    "icassp_task_1": (
        "/nfs/dgx/raid/iot/data/icassp2025eval/manifests/icassp_test_Task_1.csv",
        "iprm-task-1",
    ),
    "icassp_task_2": (
        "/nfs/dgx/raid/iot/data/icassp2025eval/manifests/icassp_test_Task_2.csv",
        "indoor-pathloss-radio-map-prediction-task-2",
    ),
    "icassp_task_3": (
        "/nfs/dgx/raid/iot/data/icassp2025eval/manifests/icassp_test_Task_3.csv",
        "iprm-challenge",
    ),
    "mlsp_rate_0.02": (
        "/nfs/dgx/raid/iot/data/icassp2025eval/manifests/mlsp_test_rate0.02.csv",
        "the-sampling-assisted-pathloss-rm-prediction",
    ),
    "mlsp_rate_0.5": (
        "/nfs/dgx/raid/iot/data/icassp2025eval/manifests/mlsp_test_rate0.5.csv",
        "sampling-assisted-pathloss-rm-prediction-t-1-ii",
    ),
}


def load_manifest(manifest_path: str) -> list[dict]:
    return IndoorDatamodule.get_inputs_list(
        freqs_mhz=[868, 1800, 3500],
        freqs=[1, 2, 3],
        manifest_path=manifest_path,
    )


def load_model_from_checkpoint(ckpt_path: str, gpu: int) -> Indoor:
    log.info(f"[model] Loading checkpoint: {ckpt_path}")
    network_conf = OmegaConf.create({
        "_target_": "src.networks.TxUNetModel",
        "in_ch": NUM_CHANNELS,
        "out_ch": 1,
        "base_ch": 48,
        "depths": [4, 6, 6, 8],
        "heads": [4, 4, 8, 8],
        "expand": 2.66,
        "use_checkpoint": False,
        "ln_eps": 1e-5,
        "window0": None,
        "window0_stride": None,
        "sra0_enabled": True,
        "sra0_stride": 4,
    })
    compiled_conf = DictConfig({
        "_target_": "src.utils.CompileParams",
        "fullgraph": True,
        "dynamic": False,
        "backend": "inductor",
        "mode": "max-autotune",
        "options": None,
        "disable": True,
    })
    algorithm = Indoor(
        out_norm=160.0,
        use_sip2net=False,
        sip2net_params={},
        compiled=compiled_conf,
        optimizer_conf=None,
        scheduler_conf=None,
        network=None,
        network_conf=OmegaConf.to_yaml(network_conf),
        gpu=gpu,
    )
    ckpt = torch.load(ckpt_path, weights_only=False)
    algorithm.load_state_dict(ckpt["state_dict"])
    algorithm.eval()
    algorithm.cuda(gpu)
    return algorithm


def run_inference(
    algorithm: Indoor,
    dataset: PathlossDataset,
    output_dir: str,
) -> list[tuple[str, float]]:
    os.makedirs(output_dir, exist_ok=True)
    csv_rows = []
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Inference"):
            batch = dataset[idx]
            result = algorithm.pred(batch=batch)
            sample_meta = batch[3]
            file_name = sample_meta["file_name"]
            safe_file_name = re.sub(r"[^\w\-_.]", "_", str(file_name))
            output_path = os.path.join(output_dir, f"{safe_file_name}.npz")
            np.savez(
                output_path,
                pred=result["pred"],
                inputs=result["inputs"],
                targets=result["targets"],
                masks=result["masks"],
                file_name=result["sample"]["file_name"],
                pixel_size=result["sample"]["pixel_size"],
            )
            rows = generate_csv_rows(
                file_name=file_name,
                pred=result["pred"],
                mask=result["masks"],
            )
            csv_rows.extend(rows)
    return csv_rows


def write_csv(csv_rows: list[tuple[str, float]], output_path: str) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "PL"])
        for id_str, pl_value in csv_rows:
            writer.writerow([id_str, round(pl_value, 1)])


def evaluate_checkpoint(task_name: str, ckpt_path: str) -> float:
    manifest_path, competition_id = TASKS[task_name]
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.isfile(manifest_path):
        raise RuntimeError(f"Manifest not found: {manifest_path}")
    inputs_list = load_manifest(manifest_path=manifest_path)
    if len(inputs_list) == 0:
        raise RuntimeError(f"No valid inputs found in manifest: {manifest_path}")
    dataset = PathlossDataset(
        inputs_list=inputs_list,
        training=False,
        inference=True,
        augmentations=None,
        sparse_range=[0, 0],
        modality_dropout_prob=0,
        sparse_dropout_given_dropout=1.0,
        channels="rtdgps",
        force_drop_sparse=False,
        force_drop_trans_ref=False,
    )
    algorithm = load_model_from_checkpoint(ckpt_path=ckpt_path, gpu=0)
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_rows = run_inference(
            algorithm=algorithm,
            dataset=dataset,
            output_dir=tmpdir,
        )
        csv_path = os.path.join(tmpdir, "predictions.csv")
        write_csv(csv_rows=csv_rows, output_path=csv_path)
        mse = submit_to_kaggle(
            csv_path=csv_path,
            competition_id=competition_id,
        )
    if mse is None:
        raise RuntimeError("Kaggle submission failed or score unavailable")
    return float(np.sqrt(mse))
