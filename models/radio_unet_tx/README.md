# Radio U-Net with Transformer Blocks (TxUNet)

A Transformer-based U-Net architecture for radio map prediction, implementing efficient global spatial attention with gated feed-forward networks.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Input/Output Specification](#inputoutput-specification)
3. [Component Details](#component-details)
4. [Dimension Reference](#dimension-reference)
5. [Usage](#usage)

---

## Architecture Overview

```
Input [B, 4, H, W]
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEM: 3×3 Conv (4 → C)                                          │
│ Output: F₀ [B, C, H, W]  ─────────────────────────────────────┐ │
└─────────────────────────────────────────────────────────────┐ │ │
                                                              │ │ │
┌─────────────────────────────────────────────────────────────┼─┼─┤
│ ENCODER                                                     │ │ │
│                                                             │ │ │
│  Level 0: Transformer × N₁  [B, C, H, W]      ──────────────┼─┼─┤──► skip to Dec0
│      │                                                      │ │ │
│      ▼ Downsample (C → 2C)                                  │ │ │
│                                                             │ │ │
│  Level 1: Transformer × N₂  [B, 2C, H/2, W/2] ──► 1×1 ──────┼─┼─┤──► skip to Dec1
│      │                                                      │ │ │
│      ▼ Downsample (2C → 4C)                                 │ │ │
│                                                             │ │ │
│  Level 2: Transformer × N₃  [B, 4C, H/4, W/4] ──► 1×1 ──────┼─┼─┤──► skip to Dec2
│      │                                                      │ │ │
│      ▼ Downsample (4C → 8C)                                 │ │ │
│                                                             │ │ │
│  Bottleneck: Transformer × N₄  [B, 8C, H/8, W/8]            │ │ │
└─────────────────────────────────────────────────────────────┼─┼─┤
                                                              │ │ │
┌─────────────────────────────────────────────────────────────┼─┼─┤
│ DECODER                                                     │ │ │
│                                                             │ │ │
│  Level 2: Upsample (8C → 4C)                                │ │ │
│           Concat with skip2 → [B, 8C, H/4, W/4]        ◄────┘ │ │
│           1×1 Conv (8C → 4C)                                  │ │
│           Transformer × N₃                                    │ │
│      │                                                        │ │
│      ▼                                                        │ │
│  Level 1: Upsample (4C → 2C)                                  │ │
│           Concat with skip1 → [B, 4C, H/2, W/2]          ◄────┘ │
│           1×1 Conv (4C → 2C)                                    │
│           Transformer × N₂                                      │
│      │                                                          │
│      ▼                                                          │
│  Level 0: Upsample (2C → C)                                     │
│           Concat with E₀ (NO 1×1) → [B, 2C, H, W]          ◄────┘
│           Transformer × N₁
│           Transformer × 1  (extra single block)
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ HEAD                                                            │
│   3×3 Conv (2C → C)                                             │
│   Add F₀ residual  ◄──────────────────────────────────────────┘ │
│   3×3 Conv (C → out_ch)                                         │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
Output [B, 1, H, W]
```

---

## Input/Output Specification

### Input: 4 Channels
| Channel | Name | Description |
|---------|------|-------------|
| 0 | R | Reflection coefficient at each grid point |
| 1 | T | Transmission coefficient at each grid point |
| 2 | D | Distance from transmitter to each point |
| 3 | P | Ground-truth pathloss at sampled points (masked elsewhere) |

### Output: 1 Channel
- Predicted pathloss map at full resolution

---

## Component Details

### 1. DConvBlock

Basic building block: 1×1 pointwise convolution followed by 3×3 depthwise convolution.

```
Input [B, in_ch, H, W]
    │
    ▼
1×1 Conv (in_ch → out_ch)
    │
    ▼
3×3 Depthwise Conv (out_ch → out_ch, groups=out_ch)
    │
    ▼
Output [B, out_ch, H, W]
```

### 2. LayerNorm2d

Channel-wise layer normalization for 2D feature maps with float32 upcasting for numerical stability.

```
Input [B, C, H, W]
    │
    ▼
Permute to [B, H, W, C]
    │
    ▼
Upcast to float32
    │
    ▼
LayerNorm over C dimension
    │
    ▼
Cast back + Permute to [B, C, H, W]
```

### 3. EfficientGlobalAttention

Global spatial self-attention with depthwise locality on Q, K, V.

```
Input X [B, C, H, W]
    │
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
DConvBlock     DConvBlock     DConvBlock
    │              │              │
    ▼              ▼              ▼
    Q              K              V
 [B,C,H,W]     [B,C,H,W]     [B,C,H,W]
    │              │              │
    └──────────────┴──────────────┘
                   │
                   ▼
    Scaled Dot-Product Attention
    (Flash Attention on CUDA, streaming on CPU)
                   │
                   ▼
            1×1 Output Proj
                   │
                   ▼
            Output [B, C, H, W]
```

### 4. GatedDepthwiseFFN

Gated feed-forward network with two separate DConvBlock paths and internal residual.

```
Input Y [B, C, H, W]
    │
    ├────────────────────┐
    │                    │
    ▼                    ▼
DConvBlock (C→Hid)   DConvBlock (C→Hid)
    │                    │
    │                    ▼
    │                  GELU
    │                    │
    ▼                    ▼
    u                    v
 [B,Hid,H,W]         [B,Hid,H,W]
    │                    │
    └────────⊙───────────┘
             │
             ▼
      g = u × v (gate)
         [B,Hid,H,W]
             │
             ▼
      1×1 Conv (Hid → C)
             │
             ▼
         Add Y (residual)
             │
             ▼
      Output [B, C, H, W]
```

Where `Hid = round(C × expand)` with default expand = 2.66

### 5. TransformerBlock

Complete transformer block combining attention and FFN.

```
Input X [B, C, H, W]
    │
    ├─────────────────────────────┐
    │                             │
    ▼                             │
LayerNorm2d                       │
    │                             │
    ▼                             │
EfficientGlobalAttention          │
    │                             │
    ▼                             │
    ◄─────────────────────────────┘ (Add residual)
    │
    │ (This becomes input to FFN)
    │
    ▼
GatedDepthwiseFFN (has internal residual)
    │
    ▼
Output [B, C, H, W]
```

### 6. Downsample

Spatial 2× downsampling with channel expansion.

```
Input [B, in_ch, H, W]
    │
    ▼
3×3 Conv, stride=2, padding=1 (in_ch → out_ch)
    │
    ▼
Output [B, out_ch, H/2, W/2]
```

### 7. Upsample

Spatial 2× upsampling with channel reduction.

```
Input [B, in_ch, H, W]
    │
    ▼
Nearest Neighbor Upsample (×2)
    │
    ▼
1×1 Conv (in_ch → out_ch)
    │
    ▼
Output [B, out_ch, 2H, 2W]
```

---

## Dimension Reference

Default configuration: `base_ch=48`, `depths=(4,6,6,8)`, `heads=(4,4,8,8)`, `expand=2.66`

### Encoder Dimensions

| Stage | Output Shape | Channels |
|-------|--------------|----------|
| Input | [B, 4, H, W] | 4 |
| Stem (F₀) | [B, 48, H, W] | C=48 |
| Enc0 (E₀) | [B, 48, H, W] | C |
| Down1 | [B, 96, H/2, W/2] | 2C |
| Enc1 (E₁) | [B, 96, H/2, W/2] | 2C |
| Down2 | [B, 192, H/4, W/4] | 4C |
| Enc2 (E₂) | [B, 192, H/4, W/4] | 4C |
| Down3 | [B, 384, H/8, W/8] | 8C |
| Bottleneck | [B, 384, H/8, W/8] | 8C |

### Decoder Dimensions

| Stage | Output Shape | Channels |
|-------|--------------|----------|
| Up3 | [B, 192, H/4, W/4] | 4C |
| Concat2 | [B, 384, H/4, W/4] | 8C |
| Fuse2 | [B, 192, H/4, W/4] | 4C |
| Dec2 | [B, 192, H/4, W/4] | 4C |
| Up2 | [B, 96, H/2, W/2] | 2C |
| Concat1 | [B, 192, H/2, W/2] | 4C |
| Fuse1 | [B, 96, H/2, W/2] | 2C |
| Dec1 | [B, 96, H/2, W/2] | 2C |
| Up1 | [B, 48, H, W] | C |
| Concat0 | [B, 96, H, W] | 2C |
| Dec0 | [B, 96, H, W] | 2C |
| Dec0_extra | [B, 96, H, W] | 2C |

### Head Dimensions

| Stage | Output Shape | Channels |
|-------|--------------|----------|
| Head Conv1 | [B, 48, H, W] | C |
| + F₀ | [B, 48, H, W] | C |
| Head Conv2 | [B, 1, H, W] | out_ch |

---

## Usage

```python
from models.radio_unet_tx import TxUNet

# Create model with default parameters
model = TxUNet(
    in_ch=4,              # Input channels: R, T, D, P
    out_ch=1,             # Output channels: predicted pathloss
    base_ch=48,           # Base channel count (C)
    depths=(4, 6, 6, 8),  # Transformer blocks per level: N₁, N₂, N₃, N₄
    heads=(4, 4, 8, 8),   # Attention heads per level
    expand=2.66,          # FFN expansion ratio
    use_checkpoint=True,  # Gradient checkpointing for memory efficiency
    ln_eps=1e-5,          # LayerNorm epsilon
)

# Input: [B, 4, H, W]
x = torch.randn(2, 4, 256, 256)

# Output: [B, 1, H, W]
y = model(x)
print(y.shape)  # torch.Size([2, 1, 256, 256])
```

### Memory Efficiency Features

1. **Gradient Checkpointing**: Enabled by default during training to reduce memory at the cost of compute.

2. **Flash Attention**: Automatically uses PyTorch's `scaled_dot_product_attention` with Flash/Memory-Efficient kernels on CUDA.

3. **Streaming Softmax**: Falls back to a memory-efficient streaming attention implementation on CPU.

---

## File Structure

```
models/radio_unet_tx/
├── __init__.py      # Module exports
├── blocks.py        # DConvBlock, LayerNorm2d, Attention, FFN, TransformerBlock
├── unet.py          # Downsample, Upsample, TxUNet
└── README.md        # This file
```

