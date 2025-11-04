
# Radio Map Modeling — Comprehensive Build Specification

> **Scope:** This document specifies how to **build, train, and evaluate** two interchangeable models for **indoor path‑loss radio‑map prediction** using the same data, sampler, and trainer.  
> **Models:**
> - **Model A — TxUNet (radio‑paper style):** U‑Net encoder–decoder with efficient Transformer blocks (1×1 + depthwise projections in attention and FFN), tailored to the radio task.
> - **Model B — Restormer‑Port:** CVPR’22 Restormer (MDTA + GDFN + pixel‑(un)shuffle + optional refinement) adapted to the same 4‑channel input and 1‑channel output.

---

## 0. Problem Contract

### 0.1 Task
Predict a dense **path‑loss map** \( y \in \mathbb{R}^{1	imes H	imes W} \) from a 4‑channel feature stack
\[
X = [R, T, D, P] \in \mathbb{R}^{4	imes H	imes W},
\]
where:
- **R** — reflection coefficient per pixel,
- **T** — transmission coefficient per pixel,
- **D** — distance (e.g., Tx→pixel path length) per pixel,
- **P** — sparse ground‑truth path‑loss samples located at selected pixels; **0 elsewhere**.

**Resizing:** Default to **256×256** inputs for parity and throughput.  
**Frequencies:** Optionally train across multiple frequency bands; evaluation may be band‑specific.

### 0.2 Sampling budgets
Let \(N = H	imes W\). With a sampling rate \(
ho\), select \(k=\mathrm{round}(
ho N)\) pixels to populate `P`. Standard budgets:
- **0.00% (k=0)** — no samples (P=0),
- **0.02%** — tiny budget,
- **0.5%** — larger budget.

### 0.3 Metrics & losses
- **Optimization:** Mean Absolute Error (L1),
  \[
  \mathcal{L}_{\mathrm{L1}} = rac{1}{|\Omega|}\sum_{(i,j)\in\Omega} \left| \hat y_{ij} - y_{ij} 
ight|
  \]
  where \(\Omega\) is the set of valid pixels.
- **Reporting:** **RMSE** (and optional weighted RMSE),
  \[
  \mathrm{RMSE}=\sqrt{rac{1}{|\Omega|}\sum_{(i,j)\in\Omega}(\hat y_{ij}-y_{ij})^2 }.
  \]

---

## 1. Repository Layout

```
radio-task/
  cfgs/
    data.yaml
    train.yaml
    model_txunet.yaml
    model_restormer.yaml
  data/
    dataset.py            # builds X=[R,T,D,P], y, free_mask, meta
    sampling.py           # stratified grid sampler + P injection
    transforms.py         # resize/normalize
  models/
    radio_unet_tx/        # Model A
      blocks.py           # EfficientAttention + GatedDepthwiseFFN
      unet.py
    restormer_port/       # Model B
      restormer_arch.py   # MDTA + GDFN + pixel-(un)shuffle
      wrapper.py          # in_ch=4, out_ch=1
  losses/l1_rmse.py
  train.py
  eval.py
  tests/
    test_sampler.py
    test_shapes.py
    test_io.py
  README.md (project overview)
  docs/
    MODEL_BUILD.md  (this file)
```

---

## 2. Data Layer

### 2.1 Dataset API
**`IndoorRadioMapDataset`** returns:
```python
(X, y, free_mask, meta)
# X:  float32 [4,H,W], channel order [R,T,D,P]
# y:  float32 [1,H,W]
# free_mask: uint8/bool [1,H,W] (1=free space, 0=obstacle)
# meta: dict, e.g. {"freq": 868, "tx": (x,y), "scene_id": "..."}
```

**Normalization (recommended):**
- Scale `D` into \([0,1]\) by dividing by the map diagonal \(\sqrt{H^2+W^2}\) or a physically meaningful max distance.
- Optionally z‑score normalize `R` and `T` if their range is wide.
- Leave `P` as raw path‑loss values at sampled locations; zeros elsewhere.

### 2.2 Transforms
- Resize to **256×256** (bilinear for continuous maps; nearest for masks).
- Optional random flips/rotations if consistent with the physics (disable at first for strict parity).

---

## 3. Sampling Module

### 3.1 Stratified grid sampler
For a target budget \(k\), define \( s=\lceil\sqrt{k}
ceil \). Partition the free‑space domain into an \(s 	imes s\) grid in image coordinates and select **one valid pixel** per cell if available. If a cell has no free pixels, reassign that quota to a global fallback (e.g., uniform over all free pixels excluding already selected ones).

```python
def stratified_k_points(free_mask: np.ndarray, k: int) -> np.ndarray:
    """
    Returns K unique (row, col) indices within free space.
    s = ceil(sqrt(k)); 1 pick per cell; fallback if a cell is empty.
    """
```

### 3.2 P‑channel injection
```python
def inject_samples(P: np.ndarray, y: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    """
    Zero P and write y at sampled indices.
    P.shape == y.shape == (H, W)
    idxs: array of shape [k, 2] with (row, col) positions.
    """
```

---

## 4. Model A — TxUNet (Radio‑Paper Style)

### 4.1 High‑level design
- **Stem:** \(3	imes3\) conv → base width **C** (default C=48).
- **Encoder:** 4 levels; each level has \(N_i\) **efficient Transformer blocks**; downsample by stride‑2.
- **Skip connections:** apply **1×1 conv on skip BEFORE concat** for channel alignment.
- **Decoder:** upsample (nearest + 1×1 or pixelshuffle), concat aligned skip, apply \(N_i\) blocks.
- **Head:** \(3	imes3 
ightarrow 1	imes1 
ightarrow\) output \(1\) channel.
- **Channel schedule:** C, 2C, 4C, 8C (bottleneck), then mirror; penultimate decoder width ≈ 2C.

### 4.2 Stage‑wise shapes (input 256×256, C=48)
```
Level 0 (stem):           [B, C,   256, 256]
Down1  -> Level 1:        [B, 2C,  128, 128]
Down2  -> Level 2:        [B, 4C,   64,  64]
Down3  -> Level 3:        [B, 8C,   32,  32]   # bottleneck
Up3 + skip2 -> Level 2d:  [B, 4C,   64,  64]
Up2 + skip1 -> Level 1d:  [B, 2C,  128, 128]
Up1 + skip0 -> Level 0d:  [B, 2C,  256, 256]   # penultimate 2C
Head -> Output:           [B, 1,   256, 256]
```

### 4.3 Efficient Transformer block

**Block structure (per block):**
```
x → LN → EfficientAttention(dim, heads)
  → (+) residual
  → LN → GatedDepthwiseFFN(dim, expand)
  → (+) residual
```

#### 4.3.1 EfficientAttention
- **Projections:** Q, K, V via **1×1 conv** to `dim`, then **depthwise 3×3** on Q and K to inject locality.
- **Affinity:** standard scaled dot‑product attention across spatial tokens (reshape to [B, HW, Hn, Ch] or [B, Hn, HW, Ch]).
- **Complexity reduction:** pointwise + depthwise keep FLOPs/memory manageable at high resolution.

_Pseudocode (shape‑aware):_
```python
def efficient_attention(x, heads, dim):
    B, C, H, W = x.shape
    Q = dw3x3(conv1x1_q(x))  # [B, Cq, H, W]
    K = dw3x3(conv1x1_k(x))  # [B, Ck, H, W]
    V = conv1x1_v(x)         # [B, Cv, H, W]
    # reshape: [B, heads, HW, C_head], scale, attn, and fold back to [B, dim, H, W]
    ...
```

#### 4.3.2 GatedDepthwiseFFN (GDFN‑like)
- **Expand:** 1×1 conv to `γ·dim` (γ≈2–4),
- **Depthwise:** 3×3 conv,
- **Gate:** split into two halves `(u, v)`, output `u ⊙ σ(v)` (σ: GELU or SiLU),
- **Project:** 1×1 conv back to `dim`.

_Pseudocode:_
```python
def gdfn(x, expand=2.66):
    z = conv1x1_expand(x)    # [B, γC, H, W]
    z = dw3x3(z)             # depthwise conv
    u, v = z.chunk(2, dim=1)
    z = u * activation(v)
    out = conv1x1_project(z) # [B, C, H, W]
    return out
```

### 4.4 Training defaults
- **Loss:** L1
- **Optimizer:** Adam, **lr=3e‑4**
- **Batch size:** 1
- **Epochs:** ~100
- **Input size:** 256×256
- **Multi‑band:** optional, but recommended for generalization.

---

## 5. Model B — Restormer‑Port (for Radio)

### 5.1 High‑level design
- **Backbone:** Restormer encoder–decoder with **pixel‑unshuffle (down)** and **pixel‑shuffle (up)** for efficient scaling.
- **Blocks:** **MDTA** (multi‑DConv‑head **transposed** attention across channels) and **GDFN**.
- **Refinement stage (optional):** several blocks at full resolution after the decoder to polish details.

### 5.2 Radio adaptation
- Change input conv to **`in_ch=4`** and final head to **`out_ch=1`**.
- Keep MDTA/GDFN and pixel‑(un)shuffle as in the original design.

### 5.3 MDTA (conceptual)
- Attention is computed **across channels** (C×C) rather than HW×HW tokens (“transposed” attention).
- Q/K/V produced by 1×1 + depthwise conv; a learnable scaling factor stabilizes training.
- Works well at high spatial resolutions.

### 5.4 Training recipe
- **Optimizer:** AdamW, cosine decay (e.g., 3e‑4 → 1e‑6),
- **Patch‑based training** if memory constrained; evaluation at 256×256 for comparability.

---

## 6. Unified Training & Evaluation

### 6.1 Config (Hydra‑friendly)
```yaml
# cfgs/data.yaml
resize: [256, 256]
normalize: true

# cfgs/train.yaml
optimizer: Adam        # or AdamW
lr: 3e-4
epochs: 100
batch_size: 1
scheduler: none        # or cosine
budget: 0.0002         # {0.0000, 0.0002, 0.0050}
sampling: stratified   # {none, stratified, random}

# cfgs/model_txunet.yaml
name: radio_unet_tx
in_ch: 4
out_ch: 1
base_ch: 48
depths: [4,6,6,8]
heads: [4,4,8,8]
expand: 2.66

# cfgs/model_restormer.yaml
name: restormer
in_ch: 4
out_ch: 1
base_ch: 48            # starting width at level-0
heads: [1,2,4,8]
refinement_blocks: 4
expand: 2.66
```

### 6.2 Trainer flow
1. Load dataset & transforms.
2. Draw sample indices with the chosen sampler and budget.
3. Inject `P` and build `X=[R,T,D,P]`.
4. Forward through the selected model (`txunet` or `restormer`).
5. Compute **L1**; step optimizer/scheduler.
6. Validate: compute **RMSE**; save best checkpoint.
7. Export qualitative grids: structure, GT, pred @0.02%, @0.5%.

### 6.3 CLI examples
```bash
# Model A (TxUNet)
python train.py --model cfgs/model_txunet.yaml --data cfgs/data.yaml     --config cfgs/train.yaml --sampling stratified --budget 0.0002

# Model B (Restormer-Port)
python train.py --model cfgs/model_restormer.yaml --data cfgs/data.yaml     --config cfgs/train.yaml --optimizer AdamW --scheduler cosine     --sampling stratified --budget 0.0050

# Eval with visualization
python eval.py --ckpt runs/txunet_best.ckpt --data cfgs/data.yaml     --sampling stratified --budget 0.0002 --save_viz
```

---

## 7. Sanity Checks (must pass before long runs)

- **Overfit a single example**: training RMSE → ~0 for both models.
- **Monotonicity vs budget**: RMSE(P=0.5%) < RMSE(P=0.02%) < RMSE(P=0.0%).
- **Sampler correctness**: exactly `k` unique points; all in `free_mask==1`.

---

## 8. Ablations & Improvements

1. **Position encodings**  
   Add 2D sine/cosine channels (or CoordConv) at the stem to mitigate resizing artifacts.

2. **P‑channel encoding**  
   Diffuse point samples into a smooth “potential/loss field” (Gaussian or Laplacian filtering) before feeding to the network; stabilizes tiny‑budget training.

3. **Sampler variants**  
   Compare stratified vs random; optionally simple uncertainty‑guided selection (e.g., pick next k points at maxima of residual magnitude from a warm‑start model).

4. **Restormer toggles**  
   - With/without **refinement stage**,  
   - Pixel‑(un)shuffle vs stride‑2 convs,  
   - Heads schedule and GDFN expansion (γ≈2–3).

---

## 9. Unit Tests

- `test_sampler.py`  
  - returns **exactly k** indices, all unique;
  - all indices lie within `free_mask==1`;
  - robust for edge cases (very small k, thin corridors).
- `test_shapes.py`  
  - forward on dummy batch for both models; verify output shape `[B,1,H,W]`;
  - parameterized for different H,W, base_ch, heads.
- `test_io.py`  
  - asserts channel order `[R,T,D,P]`;
  - checks normalization ranges and absence of NaNs/Infs.

---

## 10. Reproducibility

- Fix global seeds (Python, NumPy, PyTorch).
- Turn off non‑deterministic CuDNN ops (or log their state).
- Save:
  - full Hydra/YAML config,
  - git commit hash,
  - best checkpoints,
  - qualitative PNGs,
  - a CSV of metrics per epoch and per budget.

**Run directory layout (example):**
```
runs/2025-11-02_18-05-44_txunet_rho0p0002/
  config.yaml
  model_best.ckpt
  train_log.csv
  val_metrics.csv
  viz/
    scene12_budget0p02_pred.png
    scene12_gt.png
```

---

## 11. Troubleshooting

- **NaNs during attention:** lower learning rate; add eps in LayerNorm; clip gradients (e.g., 1.0).  
- **Exploding loss at tiny budgets:** smooth the P‑channel (Gaussian σ≈1–2 px) or warm‑start from a P=0.5% model.  
- **Shape mismatches on skips:** ensure **1×1 conv** on skip **before concat** in TxUNet; in Restormer, keep the official concat→1×1 policy if you port it that way.  
- **CUDA OOM:** reduce base_ch; use gradient checkpointing; switch to patch training for Restormer.

---

## 12. API Reference (minimal)

```python
# models/radio_unet_tx/unet.py
class TxUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, base_ch=48,
                 depths=(4,6,6,8), heads=(4,4,8,8), expand=2.66):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,4,H,W] → y_hat: [B,1,H,W]
        ...

# models/restormer_port/wrapper.py
class RestormerRadio(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, base_ch=48,
                 heads=(1,2,4,8), refinement_blocks=4, expand=2.66):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

---

## 13. Glossary

- **MDTA:** Multi‑DConv‑Head Transposed Attention — attention across channels with depthwise‑conv‑enhanced projections.  
- **GDFN:** Gated Depthwise Feed‑Forward Network — depthwise conv inside a gated MLP.  
- **Pixel‑(un)shuffle:** invertible reshape to trade spatial resolution for channels (down/up sampling).  
- **TxUNet:** U‑Net with Transformer‑style blocks designed for the radio task.

---

## 14. License & Attribution

- Code you write here: your chosen license.
- If you copy portions of Restormer, retain their MIT headers in copied files.

---

### End of Specification
