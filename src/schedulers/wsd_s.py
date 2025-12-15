from collections.abc import Sequence
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

Phase = dict[str, Any]


class WSDScheduler(_LRScheduler):
    """
    Warmup-Stable-Decay (WSD-S) scheduler that follows a predefined step allocation.
    It ramps the learning rate from zero to the peak during warmup and then alternates
    between stable plateaus and decay ramps that connect the plateaus linearly.
    The final plateau is pinned at zero to ensure the schedule ends at the minimum LR.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        stable_steps: Sequence[int],
        decay_steps: Sequence[int],
        peak_lr: float,
        last_epoch: int,
        interval: str | None = None,
        frequency: int | None = None,
    ):
        self._last_lr = None
        self._validate_inputs(
            warmup_steps=warmup_steps,
            stable_steps=stable_steps,
            decay_steps=decay_steps,
            peak_lr=peak_lr,
        )
        self._warmup_steps = warmup_steps
        self._stable_steps = tuple(int(step) for step in stable_steps)
        self._decay_steps = tuple(int(step) for step in decay_steps)
        self._peak_lr = float(peak_lr)
        self._phases = self._build_phases()
        self._total_steps = self._phases[-1]["end"] if self._phases else 0
        self._first_step_call = True
        super().__init__(
            optimizer=optimizer,
            last_epoch=last_epoch,
        )
    
    def get_lr(
        self,
    ) -> list[float]:
        factor = self._factor_for_step(
            step=self.last_epoch,
        )
        peak_value: float = self._peak_lr * factor
        group_scales: list[float] = []
        for param_group in self.optimizer.param_groups:
            scale_value = param_group.get(
                "lr_scale",
                1.0,
            )
            group_scales.append(float(scale_value))
        return [
            peak_value * scale
            for scale in group_scales
        ]
    
    def step(
        self,
        epoch: int | None = None,
    ) -> None:
        if self._first_step_call:
            self._first_step_call = False
            lrs = [
                0.0
                for _ in self.optimizer.param_groups
            ]
        else:
            if epoch is None:
                self.last_epoch += 1
            else:
                self.last_epoch = epoch
            lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr
        self._last_lr = lrs
    
    @property
    def total_steps(
        self,
    ) -> int:
        return self._total_steps
    
    def _factor_for_step(
        self,
        step: int,
    ) -> float:
        if step < 0:
            return 0.0
        if not self._phases:
            return 0.0
        if step >= self._total_steps:
            return 0.0
        for phase in self._phases:
            if step < phase["end"]:
                return self._factor_for_phase(
                    phase=phase,
                    step=step,
                )
        return 0.0
    
    def _factor_for_phase(
        self,
        phase: Phase,
        step: int,
    ) -> float:
        phase_kind = phase["kind"]
        if phase_kind == "warmup":
            return self._interpolate(
                start_factor=0.0,
                end_factor=1.0,
                start_step=phase["start"],
                end_step=phase["end"],
                step=step,
            )
        if phase_kind == "stable":
            return 1.0
        if phase_kind == "decay":
            return self._interpolate(
                start_factor=1.0,
                end_factor=0.0,
                start_step=phase["start"],
                end_step=phase["end"],
                step=step,
            )
        raise RuntimeError(f"Unsupported phase kind {phase_kind}")
    
    def _interpolate(
        self,
        start_factor: float,
        end_factor: float,
        start_step: int,
        end_step: int,
        step: int,
    ) -> float:
        length = end_step - start_step
        if length <= 1:
            return end_factor
        offset = step - start_step
        ratio = offset / (length - 1)
        return start_factor + (end_factor - start_factor) * ratio
    
    def _build_phases(
        self,
    ) -> list[Phase]:
        phases: list[Phase] = []
        cursor = 0
        if self._warmup_steps > 0:
            phases.append(
                {
                    "kind": "warmup",
                    "start": cursor,
                    "end": cursor + self._warmup_steps,
                }
            )
            cursor += self._warmup_steps
        for index, stable_length in enumerate(self._stable_steps):
            phases.append(
                {
                    "kind": "stable",
                    "index": index,
                    "start": cursor,
                    "end": cursor + stable_length,
                }
            )
            cursor += stable_length
            if index < len(self._decay_steps):
                decay_length = self._decay_steps[index]
                phases.append(
                    {
                        "kind": "decay",
                        "index": index,
                        "start": cursor,
                        "end": cursor + decay_length,
                    }
                )
                cursor += decay_length
        return phases
    
    def _validate_inputs(
        self,
        warmup_steps: int,
        stable_steps: Sequence[int],
        decay_steps: Sequence[int],
        peak_lr: float,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if len(stable_steps) == 0:
            raise ValueError("stable_steps must not be empty")
        if any(step <= 0 for step in stable_steps):
            raise ValueError("stable_steps must contain only positive durations")
        if any(step <= 0 for step in decay_steps):
            raise ValueError("decay_steps must contain only positive durations")
        if peak_lr <= 0.0:
            raise ValueError("peak_lr must be positive")
