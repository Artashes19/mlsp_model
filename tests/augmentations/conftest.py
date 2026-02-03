"""
Shared fixtures and helpers for augmentation tests.

Provides:
- make_sample / clone_sample: deterministic RadarSample factories
- Visual debug helpers: _plot_field, save_rotation_visual, save_cycle_visual, save_pipeline_visual
- VISUAL_DIR constant pointing to tests/augmentation_visuals/
"""
import copy
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.indoor.types import RadarSample

# ---------------------------------------------------------------------------
# Visual output directory
# ---------------------------------------------------------------------------
VISUAL_DIR = Path(__file__).parent.parent / "augmentation_visuals"
VISUAL_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Sample factories
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
