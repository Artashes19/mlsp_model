"""Comprehensive benchmarking for TxUNet training."""

import torch
import triton
from pathlib import Path
import yaml
from torch.utils.data import DataLoader
import time

from data.dataset import IndoorRadioMapDataset, gather_task2_samples, parse_meta
from models.radio_unet_tx import TxUNet


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def split_by_building(pairs, train_buildings, val_buildings):
    train_set, val_set = set(train_buildings), set(val_buildings)
    train, val = [], []
    for sp in pairs:
        meta = parse_meta(Path(sp.input_path).name)
        b = meta.get("building")
        if b in train_set:
            train.append(sp)
        elif b in val_set:
            val.append(sp)
    return train, val


def bench(fn, warmup=5, rep=20):
    """Benchmark using triton.testing.do_bench for accurate GPU timing."""
    times = triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="all")
    return sum(times) / len(times)


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)
    
    # ========== 1. DATA LOADING BENCHMARK ==========
    print("\n[1] DATA LOADING BENCHMARK")
    print("-" * 50)
    
    data_cfg = load_yaml("cfgs/data.yaml")
    all_pairs = gather_task2_samples(Path(data_cfg["data_root"]), split="train")
    train_buildings = [int(b) for b in data_cfg["train_buildings"]]
    val_buildings = [int(b) for b in data_cfg["val_buildings"]]
    train_pairs, _ = split_by_building(all_pairs, train_buildings, val_buildings)
    
    ds = IndoorRadioMapDataset(data_cfg["data_root"], "train", (256, 256), train_pairs[:100])
    
    # Test different num_workers
    for num_workers in [0, 2, 4, 8]:
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=num_workers, pin_memory=True)
        
        # Warmup
        for i, batch in enumerate(loader):
            if i >= 2:
                break
        
        # Time
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i, (x, y, m, _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            torch.cuda.synchronize()
            if i >= 20:
                break
        elapsed = (time.perf_counter() - t0) / 20
        print(f"  num_workers={num_workers}: {elapsed*1000:.1f} ms/batch")
    
    # ========== 2. MODEL SETUP ==========
    print("\n[2] MODEL SETUP")
    print("-" * 50)
    
    model_cfg = load_yaml("cfgs/model_txunet.yaml")
    model = TxUNet(
        in_ch=model_cfg.get("in_ch", 4),
        out_ch=1,
        base_ch=48,
        depths=(4, 6, 6, 8),
        heads=(4, 4, 8, 8),
        use_checkpoint=False,  # Disable for accurate timing
    ).to(device)
    model.train()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    x = torch.randn(1, 4, 256, 256, device=device)  # batch_size=1 due to memory
    
    # ========== 3. FORWARD PASS BENCHMARK ==========
    print("\n[3] FORWARD PASS (TOTAL)")
    print("-" * 50)
    
    fwd_time = bench(lambda: model(x), warmup=5, rep=20)
    print(f"  Forward: {fwd_time:.2f} ms")
    
    # ========== 4. BACKWARD PASS BENCHMARK ==========
    print("\n[4] BACKWARD PASS")
    print("-" * 50)
    
    def fwd_bwd():
        y = model(x)
        y.sum().backward()
    
    total_time = bench(fwd_bwd, warmup=5, rep=20)
    bwd_time = total_time - fwd_time
    print(f"  Backward: {bwd_time:.2f} ms")
    print(f"  Total (fwd+bwd): {total_time:.2f} ms")
    
    # ========== 5. PER-LAYER FORWARD TIMING ==========
    print("\n[5] FORWARD PASS PER LAYER")
    print("-" * 50)
    
    model.eval()
    with torch.no_grad():
        # Stem
        stem_time = bench(lambda: model.stem(x), warmup=5, rep=20)
        f0 = model.stem(x)
        print(f"  Stem:      {stem_time:.2f} ms  | shape: {list(f0.shape)}")
        
        # Encoder Level 0
        enc0_time = bench(lambda: model.enc0(f0), warmup=5, rep=20)
        x0 = model.enc0(f0)
        print(f"  Enc0 (4 blocks @ 256x256): {enc0_time:.2f} ms  | shape: {list(x0.shape)}")
        
        # Down1 + Enc1
        x1_pre = model.down1(x0)
        enc1_time = bench(lambda: model.enc1(x1_pre), warmup=5, rep=20)
        x1 = model.enc1(x1_pre)
        print(f"  Enc1 (6 blocks @ 128x128): {enc1_time:.2f} ms  | shape: {list(x1.shape)}")
        
        # Down2 + Enc2
        x2_pre = model.down2(x1)
        enc2_time = bench(lambda: model.enc2(x2_pre), warmup=5, rep=20)
        x2 = model.enc2(x2_pre)
        print(f"  Enc2 (6 blocks @ 64x64):   {enc2_time:.2f} ms  | shape: {list(x2.shape)}")
        
        # Down3 + Bottleneck
        x3_pre = model.down3(x2)
        enc3_time = bench(lambda: model.enc3(x3_pre), warmup=5, rep=20)
        x3 = model.enc3(x3_pre)
        print(f"  Enc3 (8 blocks @ 32x32):   {enc3_time:.2f} ms  | shape: {list(x3.shape)}")
        
        # Decoder
        s2 = model.skip2(x2)
        y2 = model.fuse2(torch.cat([model.up3(x3), s2], dim=1))
        dec2_time = bench(lambda: model.dec2(y2), warmup=5, rep=20)
        print(f"  Dec2 (6 blocks @ 64x64):   {dec2_time:.2f} ms")
        
        y2_out = model.dec2(y2)
        s1 = model.skip1(x1)
        y1 = model.fuse1(torch.cat([model.up2(y2_out), s1], dim=1))
        dec1_time = bench(lambda: model.dec1(y1), warmup=5, rep=20)
        print(f"  Dec1 (6 blocks @ 128x128): {dec1_time:.2f} ms")
        
        y1_out = model.dec1(y1)
        y0 = torch.cat([model.up1(y1_out), x0], dim=1)
        dec0_time = bench(lambda: model.dec0(y0), warmup=5, rep=20)
        dec0_extra_time = bench(lambda: model.dec0_extra(model.dec0(y0)), warmup=5, rep=20) - dec0_time
        print(f"  Dec0 (4 blocks @ 256x256): {dec0_time:.2f} ms")
        print(f"  Dec0_extra (1 block):      {dec0_extra_time:.2f} ms")
    
    # ========== 6. SINGLE TRANSFORMER BLOCK BREAKDOWN ==========
    print("\n[6] SINGLE TRANSFORMER BLOCK @ 256x256")
    print("-" * 50)
    
    block = model.enc0[0]  # First block of enc0
    test_input = torch.randn(1, 48, 256, 256, device=device)
    
    # Norm
    norm_time = bench(lambda: block.norm(test_input), warmup=5, rep=20)
    normed = block.norm(test_input)
    print(f"  LayerNorm:   {norm_time:.2f} ms")
    
    # Attention
    attn_time = bench(lambda: block.attn(normed), warmup=5, rep=20)
    print(f"  Attention:   {attn_time:.2f} ms  <-- MAIN BOTTLENECK")
    
    # FFN
    attn_out = normed + block.attn(normed)
    ffn_time = bench(lambda: block.ffn(attn_out), warmup=5, rep=20)
    print(f"  FFN:         {ffn_time:.2f} ms")
    
    # ========== 7. ATTENTION BREAKDOWN ==========
    print("\n[7] ATTENTION MODULE BREAKDOWN @ 256x256")
    print("-" * 50)
    
    attn_module = block.attn
    attn_input = normed
    
    # Q, K, V projection (DConvBlocks)
    q_time = bench(lambda: attn_module.q_block(attn_input), warmup=5, rep=20)
    k_time = bench(lambda: attn_module.k_block(attn_input), warmup=5, rep=20)
    v_time = bench(lambda: attn_module.v_block(attn_input), warmup=5, rep=20)
    print(f"  Q DConvBlock: {q_time:.2f} ms")
    print(f"  K DConvBlock: {k_time:.2f} ms")
    print(f"  V DConvBlock: {v_time:.2f} ms")
    print(f"  Total QKV:    {q_time + k_time + v_time:.2f} ms")
    
    # Full attention (includes SDPA or streaming)
    q = attn_module.q_block(attn_input)
    k = attn_module.k_block(attn_input)
    v = attn_module.v_block(attn_input)
    
    # Measure the actual attention computation part
    attn_only = attn_time - q_time - k_time - v_time
    print(f"  Attention compute (QK^T, softmax, @V): {attn_only:.2f} ms  <-- THE BOTTLENECK")
    
    # ========== 8. THEORETICAL FLOPS ==========
    print("\n[8] THEORETICAL FLOPs ANALYSIS")
    print("-" * 50)
    
    B = 1  # batch size (reduced due to memory)
    
    def calc_attention_flops(h, w, heads, dim):
        """Calculate attention FLOPs for one transformer block."""
        n = h * w  # sequence length
        d = dim // heads  # dim per head
        # QK^T: 2 * B * heads * n * d * n (matmul)
        # Softmax: ~5 * B * heads * n * n (approx)
        # Attn @ V: 2 * B * heads * n * n * d (matmul)
        qkt = 2 * B * heads * n * d * n
        softmax = 5 * B * heads * n * n
        attn_v = 2 * B * heads * n * n * d
        return qkt + softmax + attn_v
    
    def calc_dconv_flops(in_ch, out_ch, h, w):
        """Calculate DConvBlock FLOPs (1x1 conv + 3x3 depthwise)."""
        conv1x1 = 2 * in_ch * out_ch * h * w
        dw3x3 = 2 * 9 * out_ch * h * w
        return conv1x1 + dw3x3
    
    def calc_ffn_flops(dim, hidden, h, w):
        """Calculate FFN FLOPs (2 DConvBlocks + gating + proj)."""
        branch1 = calc_dconv_flops(dim, hidden, h, w)
        branch2 = calc_dconv_flops(dim, hidden, h, w)
        proj = 2 * hidden * dim * h * w
        return branch1 + branch2 + proj
    
    # Calculate per level
    levels = [
        ("Level 0 (256x256)", 256, 256, 48, 4, 4+4+1),  # enc + dec + extra
        ("Level 1 (128x128)", 128, 128, 96, 4, 6+6),
        ("Level 2 (64x64)", 64, 64, 192, 8, 6+6),
        ("Level 3 (32x32)", 32, 32, 384, 8, 8),
    ]
    
    total_attn_flops = 0
    total_ffn_flops = 0
    
    print(f"  {'Level':<22} {'Attn GFLOPs':>12} {'FFN GFLOPs':>12} {'Total GFLOPs':>12}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12}")
    
    for name, h, w, dim, heads, num_blocks in levels:
        attn_flops = calc_attention_flops(h, w, heads, dim) * num_blocks
        hidden = int(dim * 2.66)
        ffn_flops = calc_ffn_flops(dim, hidden, h, w) * num_blocks
        total_attn_flops += attn_flops
        total_ffn_flops += ffn_flops
        print(f"  {name:<22} {attn_flops/1e9:>12.1f} {ffn_flops/1e9:>12.1f} {(attn_flops+ffn_flops)/1e9:>12.1f}")
    
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'TOTAL':<22} {total_attn_flops/1e9:>12.1f} {total_ffn_flops/1e9:>12.1f} {(total_attn_flops+total_ffn_flops)/1e9:>12.1f}")
    
    # ========== 9. SUMMARY ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    batches_per_epoch = 750
    epoch_time_min = batches_per_epoch * total_time / 60000
    
    print(f"  Forward:         {fwd_time:.2f} ms")
    print(f"  Backward:        {bwd_time:.2f} ms")
    print(f"  Total/batch:     {total_time:.2f} ms")
    print(f"  Batches/sec:     {1000/total_time:.2f}")
    print(f"  Est. epoch time: {epoch_time_min:.1f} min ({epoch_time_min/60:.2f} hours)")
    print()
    
    level0_time = enc0_time + dec0_time + dec0_extra_time
    print(f"  Level 0 time (enc0+dec0+extra): {level0_time:.0f} ms")
    print(f"  Level 0 is {100*level0_time/fwd_time:.0f}% of forward time")
    print()
    print(f"  Total theoretical FLOPs: {(total_attn_flops+total_ffn_flops)/1e12:.2f} TFLOPs")
    print(f"  Attention FLOPs: {total_attn_flops/1e12:.2f} TFLOPs ({100*total_attn_flops/(total_attn_flops+total_ffn_flops):.0f}%)")


if __name__ == "__main__":
    main()

