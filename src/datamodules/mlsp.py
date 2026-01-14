import csv
import logging
import os
import pickle as pkl
import random
import time
from typing import Optional, Union

import math
import numpy as np
import torch.distributed as dist
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
        use_synthetic_val: bool,
        synthetic_val_samples_per_epoch: Optional[int],
        inference: bool,
        multi_gpu: bool = False,
        validation_names: Optional[list[str]] = None,
        icassp_val_subsample_ratio: float = 0.25,
        *args, **kwargs
    ):
        self.freqs_mhz = freqs_mhz
        self.freqs = freqs
        self.val_freq = val_freq
        self.val_buildings = val_buildings
        self.data_dir = data_dir
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
        # Deprecated fallback; do not use implicit manifest_path
        self.manifest_path: Optional[str] = kwargs.pop("manifest_path", None)
        # Explicit manifest paths (set by orchestrator)
        self.real_manifest_path: Optional[str] = kwargs.pop("real_manifest_path", None)
        self.train_manifest_path: Optional[str] = kwargs.pop("train_manifest_path", None)
        self.val_manifest_path: Optional[str] = kwargs.pop("val_manifest_path", None)
        self.synthetic_manifest_path: Optional[str] = kwargs.pop("synthetic_manifest_path", None)
        # Debug/limit knobs (explicit only; default off)
        _lpb = kwargs.pop("icassp_limit_per_building", None)
        self.icassp_limit_per_building: Optional[int] = int(_lpb) if (
            _lpb is not None and str(_lpb).strip() != "") else None
        _slim = kwargs.pop("synthetic_limit", None)
        self.synthetic_limit: Optional[int] = int(_slim) if (_slim is not None and str(_slim).strip() != "") else None
        # Per-epoch sample budget (None => full dataset per epoch)
        _tspe = kwargs.pop("train_samples_per_epoch", None)
        self.train_samples_per_epoch: Optional[int] = int(_tspe) if (
            _tspe is not None and str(_tspe).strip() != "") else None
        
        self.use_synthetic_val = use_synthetic_val
        self.synthetic_val_samples_per_epoch = synthetic_val_samples_per_epoch
        
        # Sparse measurement controls - strictly from config
        self.sparse_range = kwargs.pop("sparse_range")
        
        # Modality dropout controls
        self.modality_dropout_prob = kwargs.pop("modality_dropout_prob")
        self.sparse_dropout_given_dropout = kwargs.pop("sparse_dropout_given_dropout")
        
        # Always use dense ground truth outputs for ICASSP train-style data (Task_2_ICASSP layout)
        # If explicit manifests are provided, skip enumerating the full ICASSP directory
        if not self.train_manifest_path and not self.val_manifest_path:
            self.inputs_list = self.get_inputs_list(
                data_dir, freqs_mhz, freqs, task="Task_2_ICASSP", manifest_path=self.real_manifest_path
            )
        else:
            self.inputs_list = []
        self.kaggle_task1_list = self.get_inputs_list(
            kaggle_task1_path, kaggle_freqs_mhz, [1], 0.5, "Task_1_ICASSP"
        ) if kaggle_task1_path else []
        self.kaggle_task2_list = self.get_inputs_list(
            kaggle_task2_path, kaggle_freqs_mhz, [1, 2], task="Task_2_ICASSP"
        ) if kaggle_task2_path else []
        self.kaggle_task1_set = None
        self.kaggle_task2_set = None
        self.icassp_train_list = self.get_inputs_list(
            icassp_train_path, freqs_mhz, freqs, "Task_2_ICASSP"
        ) if icassp_train_path else []
        self.icassp_val_set = None
        self.args = args
        self.kwargs = kwargs
        
        super().__init__(
            batch_size=batch_size, num_workers=num_workers, drop_last=drop_last, multi_gpu=multi_gpu,
            *args, **kwargs
        )
        
        self.prepare_data()
    
    @staticmethod
    def get_inputs_list(data_dir, freqs_mhz, freqs, task="Task_2_ICASSP", manifest_path: Optional[str] = None):
        t0 = time.perf_counter()
        log.info(
            f"[inputs] discover start: data_dir={data_dir}, task={task}, manifest_path={manifest_path or 'None'}, "
            f"freqs_mhz={list(freqs_mhz) if freqs_mhz else []}, freqs={list(freqs) if freqs else []}"
        )
        inputs_list = []
        if not data_dir:
            raise RuntimeError("data_dir is required for dataset discovery but was empty or None.")
        if not os.path.isdir(data_dir):
            raise RuntimeError(f"data_dir does not exist or is not a directory: {data_dir}")
        
        # If a manifest path is provided, do not attempt any fallback or regeneration here.
        # Reading the manifest is mandatory; failures must raise immediately.
        
        # Fast path: load synthetic manifest if present
        if manifest_path:
            n_rows = 0
            n_synth = 0
            n_real = 0
            with open(manifest_path, "r", newline="") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    n_rows += 1
                    # Common fields
                    b = int(row.get("building", 0))
                    ant = int(row.get("antenna", 0))
                    # Resolve frequency index (1-based)
                    f_idx_internal = None
                    for key in ("freq_idx", "frequency_index"):
                        if row.get(key) not in (None, ""):
                            f_idx_internal = int(row.get(key))
                            break
                    if f_idx_internal is None:
                        freq_mhz_val = float(row.get("freq_MHz", row.get("frequency_MHz")))
                        if freqs_mhz and len(freqs_mhz) > 0:
                            diffs = [abs(freq_mhz_val - float(m)) for m in freqs_mhz]
                            nearest = int(min(range(len(diffs)), key=lambda i: diffs[i]))
                            f_idx_internal = 1 + nearest
                        else:
                            f_idx_internal = 1
                    sp = int(row.get("sample_index", row.get("sampling_position", 0)))
                    # Filter by requested frequencies if provided
                    if freqs and f_idx_internal not in freqs:
                        continue
                    # Synthetic row
                    npz_path = row.get("npz_file")
                    json_path = row.get("json_file")
                    if npz_path and json_path:
                        sample_name = row.get("file_name") or (
                            os.path.splitext(os.path.basename(npz_path))[0] if npz_path else None)
                        inputs_list.append(
                            {
                                "file_name": sample_name,
                                "npz_file": npz_path,
                                "json_file": json_path,
                                "ids": (b, ant, f_idx_internal, sp),
                            }
                        )
                        n_synth += 1
                        continue
                    # ICASSP row
                    input_file = row.get("input_file")
                    output_file = row.get("output_file")
                    position_file = row.get("position_file")
                    radiation_pattern_file = row.get("radiation_pattern_file")
                    if input_file and position_file and radiation_pattern_file:
                        sample_name = row.get("file_name") or os.path.basename(input_file)
                        freq_mhz = float(
                            row.get("freq_MHz", freqs_mhz[f_idx_internal - 1] if f_idx_internal else freqs_mhz[0])
                        )
                        inputs_list.append(
                            {
                                "file_name": sample_name,
                                "freq_MHz": freq_mhz,
                                "input_file": input_file,
                                "output_file": output_file or "",
                                "position_file": position_file,
                                "radiation_pattern_file": radiation_pattern_file,
                                "sampling_position": sp,
                                "ids": (b, ant, f_idx_internal, sp),
                            }
                        )
                        n_real += 1
            dt = time.perf_counter() - t0
            log.info(
                f"[inputs] loaded from manifest={manifest_path} rows={n_rows}; parsed: synthetic={n_synth}, real={n_real} "
                f"(elapsed={dt:.2f}s)"
            )
            return inputs_list
        
        # ICASSP layout: Inputs/{task}/ and Outputs/{task}/
        input_dir = os.path.join(data_dir, f"Inputs/{task}")
        output_dir = os.path.join(data_dir, f"Outputs/{task}")
        positions_dir = os.path.join(data_dir, "Positions/")
        radiation_patterns_dir = os.path.join(data_dir, "Radiation_Patterns/")
        if not os.path.isdir(input_dir):
            log.warning(f"ICASSP input directory not found: {input_dir}")
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
        dt = time.perf_counter() - t0
        log.info(
            f"[inputs] discovered ICASSP samples from directory "
            f"(count={len(inputs_list)}, input_dir={input_dir}, output_dir={output_dir}, elapsed={dt:.2f}s)"
        )
        return inputs_list
    
    @staticmethod
    def split_data_task1(inputs_list, val_buildings: list[int], val_ratio=0.25, split_save_path=None, seed=None):
        t0 = time.perf_counter()
        log.info(
            f"[split_task1] start: total_inputs={len(inputs_list)}, "
            f"val_buildings={'auto' if val_buildings is None else len(val_buildings)}, "
            f"val_ratio={val_ratio}, split_save_path={split_save_path or 'None'}"
        )
        
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
        dt = time.perf_counter() - t0
        log.info(f"[split_task1] done: train={len(train_files)}, val={len(val_files)} (elapsed={dt:.2f}s)")
        return train_files, val_files
    
    @staticmethod
    def split_data_task2(
        inputs_list: list[RadarSampleInputs], val_freqs, val_buildings, split_save_path=None, seed=None):
        t0 = time.perf_counter()
        log.info(
            f"[split_task2] start: total_inputs={len(inputs_list)}, "
            f"val_freqs={list(val_freqs) if val_freqs is not None else []}, "
            f"val_buildings={'auto' if val_buildings is None else len(val_buildings)}, "
            f"split_save_path={split_save_path or 'None'}"
        )
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
        dt = time.perf_counter() - t0
        log.info(f"[split_task2] done: train={len(train_inputs)}, val={len(val_inputs)} (elapsed={dt:.2f}s)")
        return train_inputs, val_inputs
    
    def prepare_data(self) -> None:
        t0_prepare = time.perf_counter()
        
        def _summarize_counts(samples: list) -> tuple[int, dict]:
            total = len(samples)
            by_b: dict[int, int] = {}
            for it in samples:
                ids = _ids_of(it)
                b = ids[0] if ids is not None else None
                if b is None:
                    continue
                by_b[b] = by_b.get(b, 0) + 1
            # keep only top 5 for logging
            top5 = dict(sorted(by_b.items(), key=lambda kv: kv[1], reverse=True)[:5])
            return total, top5
        
        def _ids_of(obj):
            return getattr(obj, 'ids', obj.get('ids') if isinstance(obj, dict) else None)
        
        def _sort_key(obj):
            ids = _ids_of(obj)
            if ids is None:
                # fallback to file_name for deterministic order
                try:
                    return (str(getattr(obj, 'file_name', '')),
                            )
                except Exception:
                    return ('',)
            return tuple(ids)
        
        def _limit_per_building(inputs: list, limit: int) -> list:
            if not inputs or limit is None or limit <= 0:
                return inputs
            # Deterministic: sort by ids then keep first N per building
            sorted_inputs = sorted(inputs, key=_sort_key)
            seen: dict[int, int] = {}
            kept = []
            for it in sorted_inputs:
                ids = _ids_of(it)
                b = ids[0] if ids is not None else None
                if b is None:
                    continue
                c = seen.get(b, 0)
                if c < limit:
                    kept.append(it)
                    seen[b] = c + 1
            return kept
        
        def _dataset_kwargs_filter(src_kwargs: dict) -> dict:
            allowed = {
                "mlsp_task1",
                "mlsp_task_idx",
                "pl_clip",
                "use_approximator_feature",
                "use_transmittance_loss",
                "reps_per_epoch",
                "augment_val",
            }
            # Add sparse controls and modality dropout to dataset kwargs
            base = {k: v for k, v in src_kwargs.items() if k in allowed}
            base.update(
                {
                    "sparse_range": self.sparse_range,
                    "modality_dropout_prob": self.modality_dropout_prob,
                    "sparse_dropout_given_dropout": self.sparse_dropout_given_dropout
                }
            )
            return base
        
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
            # Resolve validation buildings (unused in manifest-driven flow; kept for fallback)
            val_buildings_eff = self.val_buildings_override if self.val_buildings_override else self.val_buildings
            
            # Determine source for training inputs
            if self.use_synthetic_train and self.synthetic_dir:
                # Synthetic train via manifest; deterministic cap if requested
                synth_manifest = self.synthetic_manifest_path or os.path.join(self.synthetic_dir, "samples.csv")
                train_source_list = self.get_inputs_list(
                    self.synthetic_dir, self.freqs_mhz, self.freqs, task="Task_2_ICASSP", manifest_path=synth_manifest
                )
                if self.synthetic_limit is not None and self.synthetic_limit > 0:
                    train_source_list = sorted(train_source_list, key=_sort_key)[: self.synthetic_limit]
                # Validation is always from real data via explicit manifest if provided
                if self.val_manifest_path:
                    val_inputs = self.get_inputs_list(
                        self.data_dir, self.freqs_mhz, self.freqs, task="Task_2_ICASSP",
                        manifest_path=self.val_manifest_path
                    )
                elif self.use_synthetic_val:
                    val_inputs = train_source_list[: self.synthetic_val_samples_per_epoch]
                else:
                    # Fallback: compute from enumeration (not preferred)
                    _, val_inputs = self.split_data_task2(
                        self.inputs_list,
                        val_freqs=self.val_freq,
                        val_buildings=val_buildings_eff,
                        split_save_path=None,
                        seed=self.train_subset_seed,
                    )
                train_inputs = train_source_list
            else:
                # Real-only: require explicit manifests; no runtime splitting
                if not self.train_manifest_path or not self.val_manifest_path:
                    raise RuntimeError(
                        "train_manifest_path and val_manifest_path must be provided for real-data training."
                    )
                train_inputs = self.get_inputs_list(
                    self.data_dir, self.freqs_mhz, self.freqs, task="Task_2_ICASSP",
                    manifest_path=self.train_manifest_path
                )
                val_inputs = self.get_inputs_list(
                    self.data_dir, self.freqs_mhz, self.freqs, task="Task_2_ICASSP",
                    manifest_path=self.val_manifest_path
                )
                # Optional deterministic per-building cap (manifests should already be capped, but keep deterministic guard)
                if self.icassp_limit_per_building is not None and self.icassp_limit_per_building > 0:
                    train_inputs = _limit_per_building(train_inputs, self.icassp_limit_per_building)
                    val_inputs = _limit_per_building(val_inputs, self.icassp_limit_per_building)
            
            # Optional training building whitelist (real training only)
            if not self.use_synthetic_train and self.train_buildings:
                tb = set(int(x) for x in self.train_buildings)
                
                def _get_b(obj):
                    ids = _ids_of(obj)
                    return ids[0] if ids is not None else None
                
                before_tb = len(train_inputs)
                train_inputs = [f for f in train_inputs if _get_b(f) in tb]
                after_tb = len(train_inputs)
                log.info(f"[ICASSPS] applied train_buildings whitelist={sorted(tb)}: {before_tb} -> {after_tb}")
            # Optional deterministic train subset
            if self.train_subset_size is not None and self.train_subset_size > 0:
                rng = random.Random(self.train_subset_seed)
                rng.shuffle(train_inputs)
                train_inputs = train_inputs[: self.train_subset_size]
                log.info(f"[ICASSPS] after train_subset_size={self.train_subset_size}: train={len(train_inputs)}")
            
            log.info(
                f"[datasets] final train={len(train_inputs)}, val={len(val_inputs) if 'val_inputs' in locals() else 0}, "
                f"use_synthetic_train={self.use_synthetic_train}"
            )
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
                log.info(f"[Kaggle] prepared task1 set: n={len(self.kaggle_task1_set)}")
            if self.kaggle_task2_list:
                self.kaggle_task2_set = PathlossDataset(
                    self.kaggle_task2_list,
                    training=False,
                    augmentations=None,
                    inference=True,
                    task_idx=2,
                    **_dataset_kwargs_filter(self.kwargs)
                )
                log.info(f"[Kaggle] prepared task2 set: n={len(self.kaggle_task2_set)}")
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
                log.info(
                    f"[ICASSPS] prepared ICASSP validation subsample: n={len(self.icassp_val_set)} "
                    f"(ratio={self.icassp_val_subsample_ratio})"
                )
        log.info(
            f"[prepare_data] done in {(time.perf_counter() - t0_prepare):.2f}s "
            f"(train_set={len(self._train_set) if self._train_set is not None else 0}, "
            f"val_set={len(self._val_set) if self._val_set is not None else 0}, "
            f"test_set={len(self._test_set) if self._test_set is not None else 0})"
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
                dl_kwargs.update(persistent_workers=True, prefetch_factor=2)
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
                dl_kwargs.update(persistent_workers=True, prefetch_factor=2)
            dataloaders.append(DataLoader(self.icassp_val_set, **dl_kwargs))
        return dataloaders
    
    def train_dataloader(self) -> DataLoader:
        # If not using per-epoch sample budget, fallback to base behavior
        if self.train_samples_per_epoch is None or int(self.train_samples_per_epoch) <= 0:
            return super().train_dataloader()
        
        # Determine rank/world size if DDP is active
        def _dist_info():
            if dist.is_available() and dist.is_initialized():
                try:
                    return dist.get_rank(), dist.get_world_size()
                except Exception:
                    return 0, 1
            return 0, 1
        
        rank, world_size = _dist_info()
        
        sampler = DistributedCyclicSampler(
            self._train_set,
            samples_per_epoch_total=int(self.train_samples_per_epoch),
            seed=int(self.train_subset_seed or 0),
            num_replicas=world_size,
            rank=rank,
        )
        dl_kwargs = dict(
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            sampler=sampler,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=self._drop_last,
            pin_memory=True,
        )
        if self._num_workers and self._num_workers > 0:
            dl_kwargs.update(persistent_workers=True, prefetch_factor=8)
        try:
            log.info(
                f"[dataloader/train] size={len(self._train_set) if self._train_set is not None else 0}, "
                f"batch_size={self._batch_size}, num_workers={self._num_workers}, "
                f"sampler={type(sampler).__name__}, multi_gpu={self._multi_gpu}, "
                f"samples_per_epoch_total={self.train_samples_per_epoch}"
            )
        except Exception:
            pass
        return DataLoader(self._train_set, **dl_kwargs)


class DistributedCyclicSampler(DistributedSampler):
    """
    DDP-aware cyclic sampler that:
      - Provides a continuous, non-resetting stream of indices across epochs
      - Uses a single fixed random permutation (seeded) for determinism
      - Partitions work equally across ranks each epoch
      - Advances a shared base pointer by world_size * per_rank each epoch
    """
    
    def __init__(
        self, dataset, samples_per_epoch_total: int, seed: int = 0, num_replicas: Optional[int] = None,
        rank: Optional[int] = None):
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        # Initialize as a DistributedSampler to avoid Lightning replacing our sampler
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False, drop_last=False)
        self._dataset_len = len(dataset)
        self._samples_per_epoch_total = max(0, int(samples_per_epoch_total))
        # Equalize steps across ranks for DDP; small overshoot vs total is acceptable
        self._per_rank = int(
            math.ceil(self._samples_per_epoch_total / float(self.num_replicas))
        ) if self.num_replicas > 0 else self._samples_per_epoch_total
        # Fixed shuffled order for cycling
        self._rng = np.random.RandomState(int(seed) if seed is not None else 0)
        self._order = np.arange(self._dataset_len, dtype=np.int64)
        if self._dataset_len > 0:
            self._rng.shuffle(self._order)
        # Global base pointer (for rank 0); ranks offset by +rank at iteration time
        self._base_pos = 0
    
    def __iter__(self):
        m = self._dataset_len
        if m == 0 or self._per_rank == 0:
            return iter(())
        start = self._base_pos
        # Yield this rank's strided slice
        for i in range(self._per_rank):
            gi = (start + i * self.num_replicas + self.rank) % m
            yield int(self._order[gi])
        # Advance global pointer by the total work done across all ranks
        self._base_pos = (self._base_pos + (self._per_rank * self.num_replicas) % m) % m
    
    def __len__(self):
        return self._per_rank
    
    def set_epoch(self, epoch: int):
        # No-op: keep continuous stream; do not reshuffle/reset
        return
    
    @property
    def position(self) -> int:
        return self._base_pos
    
    def set_position(self, pos: int):
        if self._dataset_len == 0:
            self._base_pos = 0
        else:
            self._base_pos = int(pos) % self._dataset_len
