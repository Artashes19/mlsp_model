import logging
import csv
import os
import pickle as pkl
import random
from typing import Optional, Union

import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

from src.datamodules.datasets import PathlossDataset
from src.datamodules.wair_d_base import WAIRDBaseDatamodule
from src.utils.mlsp.augmentations import AugmentationPipeline, GeometricAugmentation
from src.utils.mlsp.types import RadarSampleInputs

log = logging.getLogger(__name__)


class MLSPDatamodule(WAIRDBaseDatamodule):
    
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        drop_last: bool,
        data_dir: str,
        freqs_mhz: list[int],
        freqs: list[int],
        val_freq: list[int],
        val_buildings: list[int],
        kaggle_task1_path: Optional[str],
        kaggle_task2_path: Optional[str],
        kaggle_freqs_mhz: Optional[list[int]],
        icassp_train_path: Optional[str],
        aug_p: float,
        walls_aug_p: Optional[int],
        transmittance_range: Optional[tuple[int, int]],
        flip_vertical: bool,
        flip_horizontal: bool,
        angle_range: Optional[tuple[float, float]],
        cardinal_rotation: bool,
        scale_range: Optional[tuple[float, float]],
        inference: bool,
        multi_gpu: bool = False,
        validation_names: Optional[list[str]] = None,
        icassp_val_subsample_ratio: float = 0.25,
        *args, **kwargs
    ):
        self.freqs_mhz = freqs_mhz
        self.val_freq = val_freq
        self.val_buildings = val_buildings
        self.inference = inference
        self.kaggle: bool = bool(kaggle_task1_path) or bool(kaggle_task2_path)
        self.icassp_validation: bool = bool(icassp_train_path)
        self.icassp_val_subsample_ratio = icassp_val_subsample_ratio
        self.validation_names = validation_names or []

        self.aug_p = aug_p
        self.walls_aug_p = walls_aug_p
        self.transmittance_range = transmittance_range
        self.angle_range = angle_range
        self.scale_range = scale_range
        self.flip_vertical = flip_vertical
        self.flip_horizontal = flip_horizontal
        self.cardinal_rotation = cardinal_rotation
        
        # Optional experiment controls (not forwarded to dataset)
        self.split_save_path = kwargs.pop("split_save_path", "./train_val_split.pkl")
        self.train_subset_size = kwargs.pop("train_subset_size", None)
        self.train_subset_seed = kwargs.pop("train_subset_seed", 0)
        # New experiment controls
        self.train_buildings: Optional[list[int]] = kwargs.pop("train_buildings", None)
        self.val_buildings_override: Optional[list[int]] = kwargs.pop("val_buildings_override", None)
        self.use_synthetic_train: bool = bool(kwargs.pop("use_synthetic_train", False))
        self.synthetic_dir: Optional[str] = kwargs.pop("synthetic_dir", None)
        self.manifest_path: Optional[str] = kwargs.pop("manifest_path", None)
        
        # Always use dense ground truth outputs for ICASSP train-style data (Task_2_ICASSP layout)
        self.inputs_list = self.get_inputs_list(
            data_dir, freqs_mhz, freqs, task="Task_2_ICASSP", manifest_path=self.manifest_path
        )
        self.kaggle_task1_list = self.get_inputs_list(kaggle_task1_path, kaggle_freqs_mhz, [1], 0.5, "Task_1_ICASSP") if kaggle_task1_path else []
        self.kaggle_task2_list = self.get_inputs_list(kaggle_task2_path, kaggle_freqs_mhz, [1, 2], task="Task_2_ICASSP") if kaggle_task2_path else []
        self.kaggle_task1_set = None
        self.kaggle_task2_set = None
        self.icassp_train_list = self.get_inputs_list(icassp_train_path, freqs_mhz, freqs, "Task_2_ICASSP") if icassp_train_path else []
        self.icassp_val_set = None
        self.args = args
        self.kwargs = kwargs
        
        self.prepare_data()
        
        super().__init__(
            batch_size=batch_size, num_workers=num_workers, drop_last=drop_last, multi_gpu=multi_gpu,
            *args, **kwargs
        )
    
    @staticmethod
    def get_inputs_list(data_dir, freqs_mhz, freqs, task="Task_2_ICASSP", manifest_path: Optional[str] = None):
        inputs_list = []
        if not data_dir:
            return inputs_list

        # If manifest path is configured but missing, generate it via the helper script
        if manifest_path and not os.path.exists(manifest_path):
            try:
                from src.data_exploration.generate_manifest import generate_manifest
                parent_dir = os.path.dirname(manifest_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                _ = generate_manifest(data_dir, manifest_path, freqs_mhz)
            except Exception:
                pass

        # Fast path: load synthetic manifest if provided
        if manifest_path and os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", newline="") as fp:
                    reader = csv.DictReader(fp)
                    for row in reader:
                        npz_path = row.get("npz_file")
                        json_path = row.get("json_file")
                        sample_name = row.get("file_name") or (os.path.splitext(os.path.basename(npz_path))[0] if npz_path else None)
                        try:
                            b = int(row.get("building", 0))
                            ant = int(row.get("antenna", 0))
                        except Exception:
                            b, ant = 0, 0
                        # Resolve frequency index (1-based)
                        f_idx_internal = None
                        for key in ("freq_idx", "frequency_index"):
                            if row.get(key) not in (None, ""):
                                try:
                                    f_idx_internal = int(row.get(key))
                                except Exception:
                                    f_idx_internal = None
                                break
                        if f_idx_internal is None:
                            try:
                                freq_mhz_val = float(row.get("frequency_MHz"))
                                if freqs_mhz and len(freqs_mhz) > 0:
                                    diffs = [abs(freq_mhz_val - float(m)) for m in freqs_mhz]
                                    nearest = int(min(range(len(diffs)), key=lambda i: diffs[i]))
                                    f_idx_internal = 1 + nearest
                            except Exception:
                                f_idx_internal = 1
                        try:
                            sp = int(row.get("sample_index", 0))
                        except Exception:
                            sp = 0
                        # Filter by requested frequencies if provided
                        if freqs and f_idx_internal not in freqs:
                            continue
                        if npz_path and json_path and sample_name:
                            inputs_list.append({
                                "file_name": sample_name,
                                "npz_file": npz_path,
                                "json_file": json_path,
                                "ids": (b, ant, f_idx_internal, sp),
                            })
                return inputs_list
            except Exception:
                # fall back to filesystem scan
                inputs_list = []

        input_dir = os.path.join(data_dir, f"Inputs/{task}")
        output_dir = os.path.join(data_dir, f"Outputs/{task}")
        positions_dir = os.path.join(data_dir, "Positions/")
        radiation_patterns_dir = os.path.join(data_dir, "Radiation_Patterns/")
        # Expect these directories to exist for the ICASSP train-style dataset
        
        for b in range(1, 26):  # 25 buildings
            for ant in range(1, 3):  # 2 antenna types
                for f in freqs:
                    for sp in range(80):  # 80 sampling positions
                        input_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
                        output_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
                        radiation_file = f"Ant{ant}_Pattern.csv"
                        position_file = f"Positions_B{b}_Ant{ant}_f{f}.csv"
                        
                        if os.path.exists(os.path.join(input_dir, input_file)):
                            freq_mhz = freqs_mhz[f - 1]
                            input_img_path = os.path.join(input_dir, input_file)
                            output_img_path = os.path.join(output_dir, output_file)
                            positions_path = os.path.join(positions_dir, position_file)
                            radiation_pattern_file = os.path.join(radiation_patterns_dir, radiation_file)
                            # Always use dense Outputs
                            
                            radar_sample_inputs = RadarSampleInputs(
                                file_name=input_file,
                                freq_MHz=freq_mhz,
                                input_file=input_img_path,
                                output_file=output_img_path,
                                position_file=positions_path,
                                radiation_pattern_file=radiation_pattern_file,
                                sampling_position=sp,
                                ids=(b, ant, f, sp),
                            )
                            
                            inputs_list.append(radar_sample_inputs)
        
        # Synthetic dataset fallback: scan for per-sample NPZ+JSON pairs
        if not inputs_list and os.path.isdir(str(data_dir)):
            try:
                for root, dirs, files in os.walk(data_dir):
                    npz_files = [f for f in files if f.endswith('.npz')]
                    for npz in npz_files:
                        sample_name = os.path.splitext(npz)[0]
                        json_file = os.path.join(root, sample_name + '.json')
                        if not os.path.exists(json_file):
                            continue
                        npz_path = os.path.join(root, npz)
                        # Parse metadata for ids and optional filtering
                        try:
                            with open(json_file, 'r') as fp:
                                meta = __import__('json').load(fp)
                            ids = meta.get('ids', {})
                            b = int(ids.get('building', 0))
                            ant = int(ids.get('antenna', 0))
                            # Synthetic data may use 0-based frequency_index (0,1,2)
                            f_idx_raw = ids.get('frequency_index') if isinstance(ids, dict) and 'frequency_index' in ids else None
                            f_idx_internal = None
                            if f_idx_raw is not None:
                                try:
                                    f_idx_raw = int(f_idx_raw)
                                    if f_idx_raw in (0, 1, 2):
                                        # Map 0-based [0,1,2] -> 1-based [1,2,3] for internal filtering/ids
                                        f_idx_internal = f_idx_raw + 1
                                    else:
                                        f_idx_internal = f_idx_raw
                                except Exception:
                                    f_idx_internal = None
                            if f_idx_internal is None:
                                # Infer from frequency_MHz to nearest freqs_mhz index (1-based)
                                try:
                                    freq_mhz_val = float(meta.get('frequency_MHz'))
                                    if freqs_mhz and len(freqs_mhz) > 0:
                                        diffs = [abs(freq_mhz_val - float(m)) for m in freqs_mhz]
                                        nearest = int(min(range(len(diffs)), key=lambda i: diffs[i]))
                                        f_idx_internal = 1 + nearest
                                except Exception:
                                    # Fallback to 1 if all else fails
                                    f_idx_internal = 1
                            sp = int(ids.get('sample_index', 0))
                            # Filter by requested frequency indices if provided (expects 1-based indices)
                            if freqs and f_idx_internal not in freqs:
                                continue
                            ids_tuple = (b, ant, f_idx_internal, sp)
                        except Exception:
                            ids_tuple = (0, 0, 1, 0)
                        # Minimal dict for dataset reader
                        inputs_list.append({
                            "file_name": sample_name,
                            "npz_file": npz_path,
                            "json_file": json_file,
                            "ids": ids_tuple,
                        })
            except Exception:
                pass
        
        return inputs_list
    
    @staticmethod
    def split_data_task1(inputs_list, val_buildings: list[int], val_ratio=0.25, split_save_path=None, seed=None):
        def _get_ids(obj):
            return getattr(obj, 'ids', obj.get('ids') if isinstance(obj, dict) else None)
        building_ids = list(set([_get_ids(f)[0] for f in inputs_list if _get_ids(f) is not None]))
        np.random.seed(seed=seed)
        np.random.shuffle(building_ids)
        
        if val_buildings is None:
            n_buildings_total = len(building_ids)
            n_buildings_valid = int(n_buildings_total * val_ratio)
            
            if n_buildings_total == 0 or n_buildings_valid == 0:
                raise ValueError(
                    f"Invalid split, total number of buildings: {n_buildings_total}, ratio of validation set: {val_ratio}. Number of validation buildings {n_buildings_valid}"
                )
            
            val_buildings = building_ids[:n_buildings_valid]
        
        val_files, train_files = [], []
        for f in inputs_list:
            ids = _get_ids(f)
            if ids is None:
                continue
            if ids[0] in val_buildings:
                val_files.append(f)
            else:
                train_files.append(f)
        if split_save_path:
            parent_dir = os.path.dirname(split_save_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(split_save_path, "wb") as f:
                split_dict = {
                    "val_files": val_files,
                    "train_files": train_files,
                }
                pkl.dump(split_dict, f)
        return train_files, val_files
    
    @staticmethod
    def split_data_task2(inputs_list: list[RadarSampleInputs], val_freqs, val_buildings, split_save_path=None, seed=None):
        train_inputs, val_inputs = MLSPDatamodule.split_data_task1(inputs_list, val_buildings=val_buildings, seed=seed)
        def _get_f_idx(obj):
            ids = getattr(obj, 'ids', obj.get('ids') if isinstance(obj, dict) else None)
            return ids[2] if ids is not None else None
        val_inputs = [f for f in val_inputs if _get_f_idx(f) in val_freqs]
        # train_inputs = [f for f in train_inputs if f.ids[2] not in val_freqs]
        
        if split_save_path:
            parent_dir = os.path.dirname(split_save_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(split_save_path, "wb") as fp:
                pkl.dump(
                    {
                        "train_inputs": train_inputs,
                        "val_inputs": val_inputs,
                        "val_freqs": val_freqs
                    }, fp
                )
        return train_inputs, val_inputs
    
    def prepare_data(self) -> None:
        def _dataset_kwargs_filter(src_kwargs: dict) -> dict:
            allowed = {
                "mlsp_task1",
                "mlsp_task_idx",
                "pl_clip",
                "use_fspl",
                "use_transmittance_loss",
                "reps_per_epoch",
                "augment_val",
            }
            return {k: v for k, v in src_kwargs.items() if k in allowed}
        
        if self.inference:
            self._test_set = PathlossDataset(
                self.inputs_list,
                training=False,
                augmentations=None,
                inference=True,
                task_idx=-1,
                **_dataset_kwargs_filter(self.kwargs)
            )
            log.info(f"Prepared inference dataset: test={len(self._test_set)}")
        else:
            # Resolve validation buildings (override if provided)
            val_buildings_eff = self.val_buildings_override if self.val_buildings_override else self.val_buildings

            # Determine source for training inputs
            if self.use_synthetic_train and self.synthetic_dir:
                train_source_list = self.get_inputs_list(self.synthetic_dir, self.freqs_mhz, self.freqs, task="Task_2_ICASSP")
                # Validation is always from real data
                _, val_inputs = self.split_data_task2(
                    self.inputs_list,
                    val_freqs=self.val_freq,
                    val_buildings=val_buildings_eff,
                    split_save_path=None,
                    seed=self.train_subset_seed,
                )
                train_inputs = train_source_list
            else:
                # Real-only: use saved split if matches our needs, else recompute
                split_save_path = self.split_save_path
                if split_save_path and os.path.exists(split_save_path) and not self.val_buildings_override:
                    with open(split_save_path, "rb") as fp:
                        split_dict = pkl.load(fp)
                        train_inputs = split_dict["train_inputs"]
                        val_inputs = split_dict["val_inputs"]
                else:
                    train_inputs, val_inputs = self.split_data_task2(
                        self.inputs_list,
                        val_freqs=self.val_freq,
                        val_buildings=val_buildings_eff,
                        split_save_path=split_save_path if not self.val_buildings_override else None,
                        seed=self.train_subset_seed,
                    )

            # Optional training building whitelist (real training only)
            if not self.use_synthetic_train and self.train_buildings:
                tb = set(int(x) for x in self.train_buildings)
                def _get_b(obj):
                    ids = getattr(obj, 'ids', obj.get('ids') if isinstance(obj, dict) else None)
                    return ids[0] if ids is not None else None
                train_inputs = [f for f in train_inputs if _get_b(f) in tb]
            # Optional deterministic train subset
            if self.train_subset_size is not None and self.train_subset_size > 0:
                rng = random.Random(self.train_subset_seed)
                rng.shuffle(train_inputs)
                train_inputs = train_inputs[: self.train_subset_size]
            
            train_augmentations = AugmentationPipeline(
                [
                    GeometricAugmentation(
                        p=self.aug_p,
                        walls_p=self.walls_aug_p,
                        transmittance_range=self.transmittance_range,
                        angle_range=self.angle_range,
                        scale_range=self.scale_range,
                        flip_vertical=self.flip_vertical,
                        flip_horizontal=self.flip_horizontal,
                        cardinal_rotation=self.cardinal_rotation,
                    ),
                ]
            )
            self._train_set = PathlossDataset(
                train_inputs,
                training=True,
                augmentations=train_augmentations,
                inference=False,
                task_idx=-1,
                **_dataset_kwargs_filter(self.kwargs)
            )
            if val_inputs:
                val_augmentations = AugmentationPipeline(
                    [
                        GeometricAugmentation(
                            p=0,
                            walls_p=self.walls_aug_p,
                            transmittance_range=self.transmittance_range,
                            angle_range=(0, 0),
                            scale_range=(1, 1),
                            flip_vertical=self.flip_vertical,
                            flip_horizontal=self.flip_horizontal,
                            cardinal_rotation=self.cardinal_rotation,
                        ),
                    ]
                )
                self._val_set = PathlossDataset(
                    val_inputs,
                    training=False,
                    augmentations=val_augmentations,
                    inference=False,
                    task_idx=-1,
                    **_dataset_kwargs_filter(self.kwargs)
                )
                self._test_set = self._val_set
            if self.kaggle_task1_list:
                self.kaggle_task1_set = PathlossDataset(
                    self.kaggle_task1_list,
                    training=False,
                    augmentations=None,
                    inference=True,
                    task_idx=1,
                    **_dataset_kwargs_filter(self.kwargs)
                )
            if self.kaggle_task2_list:
                self.kaggle_task2_set = PathlossDataset(
                    self.kaggle_task2_list,
                    training=False,
                    augmentations=None,
                    inference=True,
                    task_idx=2,
                    **_dataset_kwargs_filter(self.kwargs)
                )
            if self.icassp_train_list:
                # Subsample ICASSP validation data for faster validation
                subsample_size = max(1, int(len(self.icassp_train_list) * self.icassp_val_subsample_ratio))
                subsampled_list = random.sample(self.icassp_train_list, subsample_size)

                self.icassp_val_set = PathlossDataset(
                    subsampled_list,
                    training=False,
                    augmentations=None,
                    inference=False,
                    task_idx=-2,  # Use -2 to distinguish from regular validation (-1) and kaggle tasks (1, 2)
                    **_dataset_kwargs_filter(self.kwargs)
                )
    
    @property
    def test_set(self):
        return self._test_set
    
    @property
    def val_set(self):
        if not self.kaggle:
            return self._val_set
        else:
            return self.kaggle_task1_set, self.kaggle_task2_set
    
    def val_dataloader(self) -> Union[DataLoader, list[DataLoader]]:
        dataloaders = []
        if self.kaggle_task1_set:
            dataloaders.append(
                DataLoader(
                    self.kaggle_task1_set,
                    batch_size=self._batch_size,
                    num_workers=0,
                    sampler=None,
                    collate_fn=self.collate_fn,
                    drop_last=False,
                    pin_memory=True,
                    shuffle=False
                )
            )
        if self.kaggle_task2_set:
            dataloaders.append(
                DataLoader(
                    self.kaggle_task2_set,
                    batch_size=self._batch_size,
                    num_workers=0,
                    sampler=None,
                    collate_fn=self.collate_fn,
                    drop_last=False,
                    pin_memory=True,
                    shuffle=False
                )
            )
        if self._val_set:
            sampler = DistributedSampler(self._val_set, shuffle=False) if self._multi_gpu else None
            dl_kwargs = dict(
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                sampler=sampler,
                collate_fn=self.collate_fn,
                drop_last=self._drop_last,
                pin_memory=True,
            )
            if self._num_workers and self._num_workers > 0:
                dl_kwargs.update(persistent_workers=True, prefetch_factor=4)
            dataloaders.append(DataLoader(self._val_set, **dl_kwargs))
        if self.icassp_val_set:
            sampler = DistributedSampler(self.icassp_val_set, shuffle=False) if self._multi_gpu else None
            dl_kwargs = dict(
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                sampler=sampler,
                collate_fn=self.collate_fn,
                drop_last=self._drop_last,
                pin_memory=True,
            )
            if self._num_workers and self._num_workers > 0:
                dl_kwargs.update(persistent_workers=True, prefetch_factor=4)
            dataloaders.append(DataLoader(self.icassp_val_set, **dl_kwargs))
        return dataloaders
