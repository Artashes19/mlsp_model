import csv
import logging
import os
import time
from typing import Optional, Union

import pytorch_lightning as pl
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from src.datamodules.datasets import PathlossDataset
from src.samplers import DistributedCyclicSampler
from src.utils.mlsp.augmentations import AugmentationPipeline, GeometricAugmentation
from src.utils.mlsp.types import RadarSampleInputs

log = logging.getLogger(__name__)


class MLSPDatamodule(pl.LightningDataModule):
    
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        data_dir: str,
        freqs_mhz: list[int],
        freqs: list[int],
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
        train_subset_seed: int,
        train_buildings: Optional[list[int]],
        use_synthetic_train: bool,
        synthetic_dir: Optional[str],
        real_manifest_path: Optional[str],
        train_manifest_path: Optional[str],
        val_manifest_path: Optional[str],
        synthetic_manifest_path: Optional[str],
        synthetic_limit: Optional[int],
        train_samples_per_epoch: Optional[int],
        multi_gpu: bool,
        **kwargs
    ):
        self.freqs_mhz = freqs_mhz
        self.freqs = freqs
        self.data_dir = data_dir
        self.inference = inference
        
        self.aug_p = aug_p
        self.walls_aug_p = walls_aug_p
        self.transmittance_range = transmittance_range
        self.angle_range = angle_range
        self.scale_range = scale_range
        self.flip_vertical = flip_vertical
        self.flip_horizontal = flip_horizontal
        self.cardinal_rotation = cardinal_rotation
        
        # Experiment controls
        self.train_subset_seed = train_subset_seed
        self.train_buildings = train_buildings
        self.use_synthetic_train = use_synthetic_train
        self.synthetic_dir = synthetic_dir
        self.real_manifest_path = real_manifest_path
        self.train_manifest_path = train_manifest_path
        self.val_manifest_path = val_manifest_path
        self.synthetic_manifest_path = synthetic_manifest_path
        self.synthetic_limit = int(synthetic_limit) if synthetic_limit is not None else None
        self.train_samples_per_epoch = int(train_samples_per_epoch) if train_samples_per_epoch is not None else None
        
        self.use_synthetic_val = use_synthetic_val
        self.synthetic_val_samples_per_epoch = synthetic_val_samples_per_epoch
        
        # Store kwargs for dataset
        self.dataset_kwargs = kwargs
        
        # Always use dense ground truth outputs for ICASSP train-style data (Task_2_ICASSP layout)
        # If explicit manifests are provided, skip enumerating the full ICASSP directory
        if not self.train_manifest_path and not self.val_manifest_path:
            self.inputs_list = self.get_inputs_list(
                data_dir, freqs_mhz, freqs, task="Task_2_ICASSP", manifest_path=self.real_manifest_path
            )
        else:
            self.inputs_list = []
        self.icassp_train_list = self.get_inputs_list(
            icassp_train_path, freqs_mhz, freqs, "Task_2_ICASSP"
        ) if icassp_train_path else []
        self.icassp_val_set = None
        
        # Initialize base LightningDataModule
        super().__init__()
        
        # Initialize dataloader settings (previously from WAIRDBaseDatamodule)
        self._batch_size = batch_size
        # Coerce possibly-string env values (e.g., '8') to int
        self._num_workers = int(num_workers) if num_workers is not None else 0
        self._multi_gpu = multi_gpu
        
        self._train_set = None
        self._val_set = None
        self._test_set = None
        self._data_prepared = False
        
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
                    b = int(row.get("building", "0") or 0)
                    ant = int(row.get("antenna", "0") or 0)
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
    
    def prepare_data(self) -> None:
        if self._data_prepared:
            return
        
        t0_prepare = time.perf_counter()
        
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
        
        if self.inference:
            self._test_set = PathlossDataset(
                self.inputs_list,
                training=False,
                augmentations=None,
                inference=True,
                **self.dataset_kwargs
            )
            log.info(f"Prepared inference dataset: test={len(self._test_set)}")
        else:
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
                    raise RuntimeError(
                        "val_manifest_path must be provided or use_synthetic_val must be True for synthetic training."
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
                **self.dataset_kwargs
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
                    **self.dataset_kwargs
                )
                self._test_set = self._val_set
            if self.icassp_train_list:
                self.icassp_val_set = PathlossDataset(
                    self.icassp_train_list,
                    training=False,
                    augmentations=None,
                    inference=False,
                    **self.dataset_kwargs
                )
                log.info(f"[ICASSPS] prepared ICASSP validation set: n={len(self.icassp_val_set)}")
        log.info(
            f"[prepare_data] done in {(time.perf_counter() - t0_prepare):.2f}s "
            f"(train_set={len(self._train_set) if self._train_set is not None else 0}, "
            f"val_set={len(self._val_set) if self._val_set is not None else 0}, "
            f"test_set={len(self._test_set) if self._test_set is not None else 0})"
        )
        
        self._data_prepared = True
    
    @property
    def train_set(self):
        return self._train_set
    
    @property
    def test_set(self):
        return self._test_set
    
    @property
    def val_set(self):
        return self._val_set
    
    def val_dataloader(self) -> Union[DataLoader, list[DataLoader]]:
        dataloaders = []
        if self._val_set:
            sampler = DistributedSampler(self._val_set, shuffle=False) if self._multi_gpu else None
            dl_kwargs = dict(
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                sampler=sampler,
                drop_last=False,
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
                drop_last=False,
                pin_memory=True,
            )
            if self._num_workers and self._num_workers > 0:
                dl_kwargs.update(persistent_workers=True, prefetch_factor=2)
            dataloaders.append(DataLoader(self.icassp_val_set, **dl_kwargs))
        return dataloaders
    
    def train_dataloader(self) -> DataLoader:
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
            drop_last=True,
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
