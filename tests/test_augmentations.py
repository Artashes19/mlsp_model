"""
Unit tests for augmentations and the augmentation pipeline.

Tests cover:
- CardinalRotationAugmentation: all 3 rotation angles, coordinate transforms,
  spatial consistency between input/output/mask/floor_plan, edge cases (None fields)
- AugmentationPipeline: sequencing, training guard, empty pipeline
- BaseAugmentation: interface contract

No real data is needed — all tests use synthetic RadarSample objects with
known pixel patterns so rotations can be verified deterministically.

Visual debugging:
  Before/after images are saved to tests/augmentation_visuals/ on every run.
"""
import copy
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Visual debug — always on, images saved to tests/augmentation_visuals/
# ---------------------------------------------------------------------------
VISUAL_DIR = Path(__file__).parent / "augmentation_visuals"
VISUAL_DIR.mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
from src.utils.indoor.types import RadarSample
from src.utils.indoor.augmentations import (
    AugmentationPipeline,
    BaseAugmentation,
    CardinalRotationAugmentation,
)


# ---------------------------------------------------------------------------
# Sample factories and helpers
# ---------------------------------------------------------------------------

def make_sample(
    H: int = 8,
    W: int = 8,
    x_ant: float = 3.0,
    y_ant: float = 2.0,
    azimuth: float = 45.0,
    with_output: bool = True,
    with_floor_plan: bool = True,
) -> RadarSample:
    """
    Build a RadarSample with deterministic, asymmetric content so that
    rotations can be verified by checking specific pixel values.

    The patterns used:
      - input_img channel 0: value = row index  (increases downward)
      - input_img channel 1: value = col index  (increases rightward)
      - input_img channel 2: value = row + col
      - output_img: value = row * 100 + col  (unique per pixel)
      - mask: 1 everywhere except a single zero at (0, 0)
      - floor_plan: 1 in the top-left quadrant, 0 elsewhere
    """
    rows = torch.arange(H, dtype=torch.float32).unsqueeze(1).expand(H, W)
    cols = torch.arange(W, dtype=torch.float32).unsqueeze(0).expand(H, W)

    input_img = torch.stack([rows, cols, rows + cols], dim=0)  # (3, H, W)

    output_img = None
    if with_output:
        output_img = rows * 100 + cols  # (H, W), unique per pixel

    mask = torch.ones(H, W, dtype=torch.float32)
    mask[0, 0] = 0.0  # asymmetric marker

    floor_plan = None
    if with_floor_plan:
        floor_plan = torch.zeros(H, W, dtype=torch.float32)
        floor_plan[: H // 2, : W // 2] = 1.0  # top-left quadrant

    return RadarSample(
        file_name="test_sample",
        H=H,
        W=W,
        x_ant=x_ant,
        y_ant=y_ant,
        azimuth=azimuth,
        freq_MHz=1800.0,
        input_img=input_img,
        output_img=output_img,
        radiation_pattern=torch.zeros(360, dtype=torch.float32),
        pixel_size=0.25,
        mask=mask,
        floor_plan=floor_plan,
    )


def clone_sample(sample: RadarSample) -> RadarSample:
    """Deep copy a RadarSample, cloning all tensor fields."""
    s = copy.copy(sample)
    s.input_img = sample.input_img.clone()
    if sample.output_img is not None and not isinstance(sample.output_img, str):
        s.output_img = sample.output_img.clone()
    if sample.mask is not None:
        s.mask = sample.mask.clone()
    if sample.floor_plan is not None:
        s.floor_plan = sample.floor_plan.clone()
    s.radiation_pattern = sample.radiation_pattern.clone()
    return s


# ---------------------------------------------------------------------------
# Visual debug helpers
# ---------------------------------------------------------------------------

def _plot_field(ax, data, title, x_ant=None, y_ant=None, cmap="viridis"):
    """Plot a 2D array on *ax* with colorbar, pixel values, and antenna marker."""
    if data is None:
        ax.text(
            0.5, 0.5, "N/A",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="gray",
        )
        ax.set_title(title, fontsize=9)
        ax.set_facecolor("#f0f0f0")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    im = ax.imshow(data, cmap=cmap, origin="upper", aspect="equal")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    H, W = data.shape

    # Annotate pixel values for small images
    if H <= 12 and W <= 12:
        vmin, vmax = float(data.min()), float(data.max())
        for r in range(H):
            for c in range(W):
                val = float(data[r, c])
                if vmax > vmin:
                    norm_val = (val - vmin) / (vmax - vmin)
                    color = "white" if norm_val < 0.5 else "black"
                else:
                    color = "black"
                fmt = f"{val:.0f}" if abs(val) < 1000 else f"{val:.0e}"
                ax.text(c, r, fmt, ha="center", va="center", fontsize=5, color=color)

    # Antenna marker (red X + circle)
    if x_ant is not None and y_ant is not None:
        ax.plot(
            x_ant, y_ant, marker="x", color="red",
            markersize=12, markeredgewidth=2.5, zorder=10,
        )
        ax.plot(
            x_ant, y_ant, marker="o", color="red",
            markersize=16, fillstyle="none", markeredgewidth=1.5, zorder=10,
        )

    # Pixel grid for small images
    if H <= 12 and W <= 12:
        ax.set_xticks(range(W))
        ax.set_yticks(range(H))
        ax.tick_params(labelsize=6)
        ax.grid(True, color="gray", linewidth=0.3, alpha=0.5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def _get_field_specs():
    """Return (name, extractor_fn, colormap) tuples for all spatial fields."""
    return [
        ("Reflectance (ch0)", lambda s: s.input_img[0].numpy(), "viridis"),
        ("Transmittance (ch1)", lambda s: s.input_img[1].numpy(), "viridis"),
        ("Distance (ch2)", lambda s: s.input_img[2].numpy(), "plasma"),
        (
            "Output (pathloss)",
            lambda s: s.output_img.numpy()
            if s.output_img is not None and not isinstance(s.output_img, str)
            else None,
            "hot",
        ),
        (
            "Mask",
            lambda s: s.mask.numpy() if s.mask is not None else None,
            "gray",
        ),
        (
            "Floor Plan",
            lambda s: s.floor_plan.numpy() if s.floor_plan is not None else None,
            "Greens",
        ),
    ]


def save_rotation_visual(before, after, k, filename):
    """Save a comprehensive before/after visual for a cardinal rotation."""

    deg = k * 90
    field_specs = _get_field_specs()
    n_cols = len(field_specs)

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 9))

    for i, (name, extractor, cmap) in enumerate(field_specs):
        _plot_field(
            axes[0, i], extractor(before),
            f"BEFORE\n{name}", before.x_ant, before.y_ant, cmap,
        )
        _plot_field(
            axes[1, i], extractor(after),
            f"AFTER\n{name}", after.x_ant, after.y_ant, cmap,
        )

    # Row labels
    axes[0, 0].set_ylabel("BEFORE", fontsize=12, fontweight="bold", labelpad=10)
    axes[1, 0].set_ylabel("AFTER", fontsize=12, fontweight="bold", labelpad=10)

    fig.suptitle(
        f"CardinalRotation  k={k}  ({deg}\u00b0 CCW)\n"
        f"Antenna: ({before.x_ant:.0f}, {before.y_ant:.0f}) \u2192 "
        f"({after.x_ant:.0f}, {after.y_ant:.0f})    "
        f"Azimuth: {before.azimuth:.0f}\u00b0 \u2192 {after.azimuth:.0f}\u00b0    "
        f"Shape: ({before.H}\u00d7{before.W}) \u2192 ({after.H}\u00d7{after.W})",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    out_path = VISUAL_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [visual] Saved: {out_path}")


def save_cycle_visual(states, k, filename):
    """Save a visual showing the 4-rotation cycle (original -> ... -> original)."""

    deg = k * 90
    n_states = len(states)  # 5: original + 4 rotations
    # Show 3 key fields per state
    field_specs = [
        ("Reflectance", lambda s: s.input_img[0].numpy(), "viridis"),
        (
            "Output",
            lambda s: s.output_img.numpy()
            if s.output_img is not None and not isinstance(s.output_img, str)
            else None,
            "hot",
        ),
        ("Mask", lambda s: s.mask.numpy() if s.mask is not None else None, "gray"),
    ]

    n_rows = len(field_specs)
    fig, axes = plt.subplots(n_rows, n_states, figsize=(4 * n_states, 4 * n_rows))

    col_labels = ["Original"] + [
        f"Rot {i+1}\n(+{deg}\u00b0)" for i in range(n_states - 1)
    ]

    for col, (state, label) in enumerate(zip(states, col_labels)):
        for row, (name, extractor, cmap) in enumerate(field_specs):
            title = f"{label}\n{name}\nant=({state.x_ant:.0f},{state.y_ant:.0f}) az={state.azimuth:.0f}\u00b0"
            _plot_field(
                axes[row, col], extractor(state),
                title, state.x_ant, state.y_ant, cmap,
            )

    fig.suptitle(
        f"Rotation Cycle:  k={k} ({deg}\u00b0 CCW) \u00d7 4 = identity\n"
        f"First and last columns should be identical",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = VISUAL_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [visual] Saved: {out_path}")


def save_pipeline_visual(before, after, pipeline_desc, filename):
    """Save a before/after visual for an augmentation pipeline."""

    field_specs = _get_field_specs()
    n_cols = len(field_specs)

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 9))

    for i, (name, extractor, cmap) in enumerate(field_specs):
        _plot_field(
            axes[0, i], extractor(before),
            f"BEFORE\n{name}", before.x_ant, before.y_ant, cmap,
        )
        _plot_field(
            axes[1, i], extractor(after),
            f"AFTER\n{name}", after.x_ant, after.y_ant, cmap,
        )

    axes[0, 0].set_ylabel("BEFORE", fontsize=12, fontweight="bold", labelpad=10)
    axes[1, 0].set_ylabel("AFTER", fontsize=12, fontweight="bold", labelpad=10)

    fig.suptitle(
        f"Pipeline: {pipeline_desc}\n"
        f"Antenna: ({before.x_ant:.0f}, {before.y_ant:.0f}) \u2192 "
        f"({after.x_ant:.0f}, {after.y_ant:.0f})    "
        f"Azimuth: {before.azimuth:.0f}\u00b0 \u2192 {after.azimuth:.0f}\u00b0    "
        f"Shape: ({before.H}\u00d7{before.W}) \u2192 ({after.H}\u00d7{after.W})",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    out_path = VISUAL_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [visual] Saved: {out_path}")


# ===========================================================================
# Tests
# ===========================================================================


class TestCardinalRotationAugmentation(unittest.TestCase):
    """Tests for CardinalRotationAugmentation (90/180/270 degree lossless rotations)."""

    # ------------------------------------------------------------------
    # Deterministic rotation tests (one per k value)
    # ------------------------------------------------------------------

    def _apply_rotation_k(self, k: int, sample: RadarSample) -> RadarSample:
        """Force a specific rotation by mocking random.choice."""
        aug = CardinalRotationAugmentation(p=1.0)
        with patch("src.utils.indoor.augmentations.random.choice", return_value=k):
            return aug(sample)

    # --- k = 1  (90 deg CCW) ---

    def test_rotation_90_input_img(self):
        s = make_sample(H=4, W=6)
        r = self._apply_rotation_k(1, s)

        # torch.rot90 k=1 dims=(1,2): new shape (C, W, H) = (3, 6, 4)
        self.assertEqual(r.input_img.shape, (3, 6, 4))
        self.assertEqual(r.H, 6)
        self.assertEqual(r.W, 4)

    def test_rotation_90_output_img(self):
        s = make_sample(H=4, W=6)
        orig_output = s.output_img.clone()
        r = self._apply_rotation_k(1, s)

        self.assertEqual(r.output_img.shape, (6, 4))
        expected = torch.rot90(orig_output, 1, (0, 1))
        self.assertTrue(torch.equal(r.output_img, expected))

    def test_rotation_90_mask(self):
        s = make_sample(H=4, W=6)
        orig_mask = s.mask.clone()
        r = self._apply_rotation_k(1, s)

        expected = torch.rot90(orig_mask, 1, (0, 1))
        self.assertTrue(torch.equal(r.mask, expected))

    def test_rotation_90_floor_plan(self):
        s = make_sample(H=4, W=6)
        orig_fp = s.floor_plan.clone()
        r = self._apply_rotation_k(1, s)

        expected = torch.rot90(orig_fp, 1, (0, 1))
        self.assertTrue(torch.equal(r.floor_plan, expected))

    def test_rotation_90_antenna_coords(self):
        s = make_sample(H=4, W=6, x_ant=2.0, y_ant=1.0)
        r = self._apply_rotation_k(1, s)

        # k=1: new_x = y_ant, new_y = old_W - x_ant - 1
        self.assertEqual(r.x_ant, 1.0)
        self.assertEqual(r.y_ant, 6 - 2.0 - 1)  # 3.0

    def test_rotation_90_azimuth(self):
        s = make_sample(azimuth=45.0)
        r = self._apply_rotation_k(1, s)
        self.assertAlmostEqual(r.azimuth, (45.0 + 90) % 360)

    # --- k = 2  (180 deg) ---

    def test_rotation_180_input_img(self):
        s = make_sample(H=4, W=6)
        r = self._apply_rotation_k(2, s)

        # 180 rotation preserves shape
        self.assertEqual(r.input_img.shape, (3, 4, 6))
        self.assertEqual(r.H, 4)
        self.assertEqual(r.W, 6)

    def test_rotation_180_output_img(self):
        s = make_sample(H=4, W=6)
        orig_output = s.output_img.clone()
        r = self._apply_rotation_k(2, s)

        expected = torch.rot90(orig_output, 2, (0, 1))
        self.assertTrue(torch.equal(r.output_img, expected))

    def test_rotation_180_mask(self):
        s = make_sample(H=4, W=6)
        orig_mask = s.mask.clone()
        r = self._apply_rotation_k(2, s)

        expected = torch.rot90(orig_mask, 2, (0, 1))
        self.assertTrue(torch.equal(r.mask, expected))

    def test_rotation_180_floor_plan(self):
        s = make_sample(H=4, W=6)
        orig_fp = s.floor_plan.clone()
        r = self._apply_rotation_k(2, s)

        expected = torch.rot90(orig_fp, 2, (0, 1))
        self.assertTrue(torch.equal(r.floor_plan, expected))

    def test_rotation_180_antenna_coords(self):
        s = make_sample(H=4, W=6, x_ant=2.0, y_ant=1.0)
        r = self._apply_rotation_k(2, s)

        # k=2: new_x = old_W - x_ant - 1, new_y = old_H - y_ant - 1
        self.assertEqual(r.x_ant, 6 - 2.0 - 1)  # 3.0
        self.assertEqual(r.y_ant, 4 - 1.0 - 1)  # 2.0

    def test_rotation_180_azimuth(self):
        s = make_sample(azimuth=200.0)
        r = self._apply_rotation_k(2, s)
        self.assertAlmostEqual(r.azimuth, (200.0 + 180) % 360)  # 20.0

    # --- k = 3  (270 deg CCW = 90 CW) ---

    def test_rotation_270_input_img(self):
        s = make_sample(H=4, W=6)
        r = self._apply_rotation_k(3, s)

        self.assertEqual(r.input_img.shape, (3, 6, 4))
        self.assertEqual(r.H, 6)
        self.assertEqual(r.W, 4)

    def test_rotation_270_output_img(self):
        s = make_sample(H=4, W=6)
        orig_output = s.output_img.clone()
        r = self._apply_rotation_k(3, s)

        expected = torch.rot90(orig_output, 3, (0, 1))
        self.assertTrue(torch.equal(r.output_img, expected))

    def test_rotation_270_mask(self):
        s = make_sample(H=4, W=6)
        orig_mask = s.mask.clone()
        r = self._apply_rotation_k(3, s)

        expected = torch.rot90(orig_mask, 3, (0, 1))
        self.assertTrue(torch.equal(r.mask, expected))

    def test_rotation_270_floor_plan(self):
        s = make_sample(H=4, W=6)
        orig_fp = s.floor_plan.clone()
        r = self._apply_rotation_k(3, s)

        expected = torch.rot90(orig_fp, 3, (0, 1))
        self.assertTrue(torch.equal(r.floor_plan, expected))

    def test_rotation_270_antenna_coords(self):
        s = make_sample(H=4, W=6, x_ant=2.0, y_ant=1.0)
        r = self._apply_rotation_k(3, s)

        # k=3: new_x = old_H - y_ant - 1, new_y = x_ant
        self.assertEqual(r.x_ant, 4 - 1.0 - 1)  # 2.0
        self.assertEqual(r.y_ant, 2.0)

    def test_rotation_270_azimuth(self):
        s = make_sample(azimuth=100.0)
        r = self._apply_rotation_k(3, s)
        self.assertAlmostEqual(r.azimuth, (100.0 + 270) % 360)  # 10.0

    # ------------------------------------------------------------------
    # Spatial consistency: the antenna pixel value in input_img and
    # output_img should follow the antenna after rotation.
    # ------------------------------------------------------------------

    def test_antenna_pixel_follows_rotation(self):
        """
        Place a unique marker at the antenna position in both input and
        output, rotate, and verify the marker moved to the new antenna
        position.
        """
        H, W = 8, 8
        x_ant, y_ant = 5, 3  # col=5, row=3
        MARKER = 999.0

        for k in [1, 2, 3]:
            with self.subTest(k=k):
                s = make_sample(H=H, W=W, x_ant=float(x_ant), y_ant=float(y_ant))

                # Plant marker at antenna location
                s.input_img[:, y_ant, x_ant] = MARKER
                s.output_img[y_ant, x_ant] = MARKER

                r = self._apply_rotation_k(k, s)

                new_x = int(r.x_ant)
                new_y = int(r.y_ant)

                # The marker should now live at the new antenna position
                for ch in range(3):
                    self.assertAlmostEqual(
                        r.input_img[ch, new_y, new_x].item(),
                        MARKER,
                        msg=f"k={k}: input_img channel {ch} marker not at new antenna pos",
                    )
                self.assertAlmostEqual(
                    r.output_img[new_y, new_x].item(),
                    MARKER,
                    msg=f"k={k}: output_img marker not at new antenna pos",
                )

    # ------------------------------------------------------------------
    # Full-cycle: rotating 4 times by the same k should restore original
    # ------------------------------------------------------------------

    def test_four_rotations_restore_original(self):
        """Rotating by the same k four times must return the original sample."""
        for k in [1, 2, 3]:
            with self.subTest(k=k):
                s = make_sample(H=6, W=10, x_ant=4.0, y_ant=2.0, azimuth=30.0)
                orig_input = s.input_img.clone()
                orig_output = s.output_img.clone()
                orig_mask = s.mask.clone()
                orig_fp = s.floor_plan.clone()
                orig_x, orig_y = s.x_ant, s.y_ant
                orig_az = s.azimuth

                for _ in range(4):
                    s = self._apply_rotation_k(k, s)

                self.assertTrue(torch.equal(s.input_img, orig_input))
                self.assertTrue(torch.equal(s.output_img, orig_output))
                self.assertTrue(torch.equal(s.mask, orig_mask))
                self.assertTrue(torch.equal(s.floor_plan, orig_fp))
                self.assertAlmostEqual(s.x_ant, orig_x)
                self.assertAlmostEqual(s.y_ant, orig_y)
                self.assertAlmostEqual(s.azimuth % 360, orig_az % 360)

    # ------------------------------------------------------------------
    # Probability gate
    # ------------------------------------------------------------------

    def test_p_zero_never_augments(self):
        aug = CardinalRotationAugmentation(p=0.0)
        s = make_sample()
        orig_input = s.input_img.clone()

        for _ in range(50):
            s_copy = make_sample()
            result = aug(s_copy)
            self.assertTrue(torch.equal(result.input_img, orig_input))

    def test_p_one_always_augments(self):
        """With p=1.0 and 3 possible rotations (90/180/270), every call must
        produce a rotated result (never identity)."""
        aug = CardinalRotationAugmentation(p=1.0)
        identical_count = 0
        N = 50

        for _ in range(N):
            s = make_sample(H=4, W=6)
            orig_input = s.input_img.clone()
            result = aug(s)
            if torch.equal(result.input_img, orig_input):
                identical_count += 1

        # With a non-square input, no rotation (90/180/270) can produce
        # the identical tensor, so identical_count should be 0.
        self.assertEqual(identical_count, 0)

    # ------------------------------------------------------------------
    # Edge cases: None / missing fields
    # ------------------------------------------------------------------

    def test_none_output_img(self):
        """Augmentation must not crash when output_img is None (kaggle test data)."""
        for k in [1, 2, 3]:
            with self.subTest(k=k):
                s = make_sample(with_output=False)
                r = self._apply_rotation_k(k, s)
                self.assertIsNone(r.output_img)

    def test_none_floor_plan(self):
        """Augmentation must not crash when floor_plan is None."""
        for k in [1, 2, 3]:
            with self.subTest(k=k):
                s = make_sample(with_floor_plan=False)
                r = self._apply_rotation_k(k, s)
                self.assertIsNone(r.floor_plan)

    # ------------------------------------------------------------------
    # Square image: verify dimensions don't change
    # ------------------------------------------------------------------

    def test_square_image_preserves_dimensions(self):
        for k in [1, 2, 3]:
            with self.subTest(k=k):
                s = make_sample(H=8, W=8)
                r = self._apply_rotation_k(k, s)
                self.assertEqual(r.H, 8)
                self.assertEqual(r.W, 8)
                self.assertEqual(r.input_img.shape, (3, 8, 8))

    # ------------------------------------------------------------------
    # Mask zero-pixel marker tracks correctly
    # ------------------------------------------------------------------

    def test_mask_zero_pixel_tracks_rotation(self):
        """The single zero in the mask at (0,0) should move to the expected
        position after each rotation."""
        H, W = 6, 10

        expected_zero_pos = {
            # (row, col) of the zero pixel after rotation
            1: (W - 1, 0),       # k=1: (0,0) -> row=W-0-1, col=0
            2: (H - 1, W - 1),   # k=2: (0,0) -> row=H-1, col=W-1
            3: (0, H - 1),       # k=3: (0,0) -> row=0, col=H-1
        }

        for k, (exp_row, exp_col) in expected_zero_pos.items():
            with self.subTest(k=k):
                s = make_sample(H=H, W=W)
                r = self._apply_rotation_k(k, s)

                self.assertEqual(
                    r.mask[exp_row, exp_col].item(),
                    0.0,
                    msg=f"k={k}: expected mask zero at ({exp_row},{exp_col})",
                )
                # And total number of zeros should still be 1
                self.assertEqual((r.mask == 0).sum().item(), 1)


class TestAugmentationPipeline(unittest.TestCase):
    """Tests for AugmentationPipeline sequencing and guards."""

    def test_training_false_skips_all(self):
        """Pipeline with training=False must return the sample unchanged."""
        aug = CardinalRotationAugmentation(p=1.0)
        pipeline = AugmentationPipeline(augmentations=[aug], training=False)

        s = make_sample(H=4, W=6)
        orig_input = s.input_img.clone()
        result = pipeline(s)

        self.assertTrue(torch.equal(result.input_img, orig_input))

    def test_empty_augmentation_list(self):
        """Pipeline with no augmentations returns sample unchanged."""
        pipeline = AugmentationPipeline(augmentations=[], training=True)

        s = make_sample()
        orig_input = s.input_img.clone()
        result = pipeline(s)

        self.assertTrue(torch.equal(result.input_img, orig_input))

    def test_augmentations_applied_in_order(self):
        """Verify that augmentations are called in the order they appear."""
        call_order = []

        class TrackerAug(BaseAugmentation):
            def __init__(self, name):
                self.name = name

            def __call__(self, sample):
                call_order.append(self.name)
                return sample

        pipeline = AugmentationPipeline(
            augmentations=[TrackerAug("first"), TrackerAug("second"), TrackerAug("third")],
            training=True,
        )
        pipeline(make_sample())
        self.assertEqual(call_order, ["first", "second", "third"])

    def test_pipeline_chains_mutations(self):
        """Each augmentation receives the output of the previous one."""

        class AddOneToOutput(BaseAugmentation):
            def __call__(self, sample):
                if sample.output_img is not None:
                    sample.output_img = sample.output_img + 1.0
                return sample

        pipeline = AugmentationPipeline(
            augmentations=[AddOneToOutput(), AddOneToOutput(), AddOneToOutput()],
            training=True,
        )
        s = make_sample()
        orig_output = s.output_img.clone()
        result = pipeline(s)

        # Three increments of 1.0
        self.assertTrue(torch.allclose(result.output_img, orig_output + 3.0))

    def test_pipeline_with_real_rotation(self):
        """Smoke test: pipeline containing a CardinalRotationAugmentation produces
        a valid RadarSample (no crashes, correct types)."""
        pipeline = AugmentationPipeline(
            augmentations=[CardinalRotationAugmentation(p=1.0)],
            training=True,
        )
        s = make_sample()
        result = pipeline(s)

        self.assertIsInstance(result, RadarSample)
        self.assertEqual(result.input_img.ndim, 3)
        self.assertEqual(result.output_img.ndim, 2)
        self.assertEqual(result.mask.ndim, 2)
        self.assertEqual(result.input_img.shape[1], result.H)
        self.assertEqual(result.input_img.shape[2], result.W)


class TestBaseAugmentation(unittest.TestCase):
    """Test the BaseAugmentation interface contract."""

    def test_not_implemented(self):
        base = BaseAugmentation()
        with self.assertRaises(NotImplementedError):
            base(make_sample())


class TestCardinalRotationStatistical(unittest.TestCase):
    """Statistical tests to verify uniform distribution of rotation angles."""

    def test_uniform_distribution_of_k(self):
        """Over many calls, each k (1, 2, 3) should be chosen roughly equally."""
        aug = CardinalRotationAugmentation(p=1.0)
        H, W = 4, 6  # non-square so we can distinguish k by output shape

        shape_counts = {}
        N = 600

        for _ in range(N):
            s = make_sample(H=H, W=W)
            r = aug(s)
            key = (r.H, r.W)
            shape_counts[key] = shape_counts.get(key, 0) + 1

        # k=1 and k=3 produce (W, H) = (6, 4), k=2 produces (H, W) = (4, 6)
        # So we expect ~2/3 of samples to have shape (6, 4) and ~1/3 to have (4, 6)
        count_swapped = shape_counts.get((W, H), 0)  # k=1 or k=3
        count_same = shape_counts.get((H, W), 0)  # k=2

        # With N=600, expected: swapped=400, same=200. Allow generous margin.
        self.assertGreater(count_swapped, 250, "k=1/k=3 combined should appear often")
        self.assertGreater(count_same, 100, "k=2 should appear reasonably often")
        self.assertEqual(count_swapped + count_same, N)


# ===========================================================================
# Visual debug tests
# ===========================================================================


class TestVisualDebug(unittest.TestCase):
    """
    Visual debugging tests that save before/after images for each augmentation.
    Images are saved to tests/augmentation_visuals/.
    """

    def _apply_rotation_k(self, k: int, sample: RadarSample) -> RadarSample:
        aug = CardinalRotationAugmentation(p=1.0)
        with patch("src.utils.indoor.augmentations.random.choice", return_value=k):
            return aug(sample)

    # ------------------------------------------------------------------
    # One comprehensive image per rotation angle
    # ------------------------------------------------------------------

    def test_visual_rotation_90(self):
        """Save before/after visual for 90-degree CCW rotation."""
        s = make_sample(H=6, W=10, x_ant=7.0, y_ant=2.0, azimuth=45.0)
        before = clone_sample(s)
        after = self._apply_rotation_k(1, s)
        save_rotation_visual(before, after, k=1, filename="rotation_k1_90ccw.png")

    def test_visual_rotation_180(self):
        """Save before/after visual for 180-degree rotation."""
        s = make_sample(H=6, W=10, x_ant=7.0, y_ant=2.0, azimuth=45.0)
        before = clone_sample(s)
        after = self._apply_rotation_k(2, s)
        save_rotation_visual(before, after, k=2, filename="rotation_k2_180.png")

    def test_visual_rotation_270(self):
        """Save before/after visual for 270-degree CCW (= 90 CW) rotation."""
        s = make_sample(H=6, W=10, x_ant=7.0, y_ant=2.0, azimuth=45.0)
        before = clone_sample(s)
        after = self._apply_rotation_k(3, s)
        save_rotation_visual(before, after, k=3, filename="rotation_k3_270ccw.png")

    # ------------------------------------------------------------------
    # Antenna tracking — marker planted at antenna position
    # ------------------------------------------------------------------

    def test_visual_antenna_tracking(self):
        """
        Plant a bright 999-marker at the antenna pixel, rotate, and
        visually verify the marker moved to the new antenna position.
        """
        H, W = 8, 12
        x_ant, y_ant = 9, 3
        MARKER = 999.0

        for k in [1, 2, 3]:
            with self.subTest(k=k):
                s = make_sample(H=H, W=W, x_ant=float(x_ant), y_ant=float(y_ant))
                s.input_img[:, y_ant, x_ant] = MARKER
                s.output_img[y_ant, x_ant] = MARKER
                before = clone_sample(s)
                after = self._apply_rotation_k(k, s)
                save_rotation_visual(
                    before, after, k=k,
                    filename=f"antenna_tracking_k{k}.png",
                )

    # ------------------------------------------------------------------
    # Rotation cycle: 4 x same rotation = identity
    # ------------------------------------------------------------------

    def test_visual_rotation_cycle(self):
        """
        Show the full 4-rotation cycle for each k, verifying visually
        that the last state matches the original.
        """
        for k in [1, 2, 3]:
            with self.subTest(k=k):
                s = make_sample(H=6, W=10, x_ant=4.0, y_ant=2.0, azimuth=30.0)
                states = [clone_sample(s)]
                for _ in range(4):
                    s = self._apply_rotation_k(k, s)
                    states.append(clone_sample(s))
                save_cycle_visual(states, k=k, filename=f"cycle_k{k}.png")

    # ------------------------------------------------------------------
    # Pipeline visual — rotation through the pipeline
    # ------------------------------------------------------------------

    def test_visual_pipeline(self):
        """
        Show before/after of a sample going through AugmentationPipeline
        with a CardinalRotationAugmentation (random k).
        """
        pipeline = AugmentationPipeline(
            augmentations=[CardinalRotationAugmentation(p=1.0)],
            training=True,
        )
        s = make_sample(H=6, W=10, x_ant=7.0, y_ant=2.0, azimuth=45.0)
        before = clone_sample(s)
        after = pipeline(s)
        save_pipeline_visual(
            before, after,
            pipeline_desc="CardinalRotation(p=1.0)",
            filename="pipeline_cardinal_rotation.png",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
