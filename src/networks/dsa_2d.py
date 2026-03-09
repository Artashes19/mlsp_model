from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.ops.dsa_indexer import act_quant_reference_safe, stable_topk, weighted_relu_index_score
from src.ops.dsa_rope import apply_partial_rope_2d_interleaved
from src.ops.dsa_sparse_mla import gather_sparse_mla_tokens


@dataclass(frozen=True)
class DSA2DMLAConfig:
    dim: int
    n_heads: int
    n_kv_heads: int
    q_lora_rank: int = 512
    kv_lora_rank: int = 256
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    index_n_heads: int = 4
    index_head_dim: int = 128
    index_topk: int = 64

    def __post_init__(self) -> None:
        positive_fields = {
            "dim": self.dim,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "index_n_heads": self.index_n_heads,
            "index_head_dim": self.index_head_dim,
            "index_topk": self.index_topk,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0")

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA")

        if self.qk_rope_head_dim % 4 != 0:
            raise ValueError("qk_rope_head_dim must be divisible by 4 for 2D partial RoPE")

        if self.index_head_dim % 2 != 0:
            raise ValueError("index_head_dim must be even for a balanced rope/nope split")

        index_rope_head_dim = self.index_head_dim // 2
        if index_rope_head_dim % 4 != 0:
            raise ValueError("index rope-active dim must be divisible by 4 for 2D partial RoPE")


class DSA2DIndexer(nn.Module):
    def __init__(self, cfg: DSA2DMLAConfig) -> None:
        super().__init__()
        self.index_n_heads = cfg.index_n_heads
        self.index_head_dim = cfg.index_head_dim
        self.index_topk = cfg.index_topk

    def forward(self, q: torch.Tensor, k: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_q, q_scale = act_quant_reference_safe(q)
        k_q, k_scale = act_quant_reference_safe(k)
        logits = weighted_relu_index_score(
            q_q.to(dtype=torch.float32) * q_scale,
            k_q.to(dtype=torch.float32) * k_scale,
            w,
        )
        topk = min(self.index_topk, logits.shape[-1])
        idx = stable_topk(logits, k=topk)
        return logits, idx


class DSA2DMLAAttention(nn.Module):
    def __init__(self, cfg: DSA2DMLAConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.dim = cfg.dim
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.gqa_group_size = cfg.n_heads // cfg.n_kv_heads

        self.q_lora_rank = cfg.q_lora_rank
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim

        self.index_n_heads = cfg.index_n_heads
        self.index_head_dim = cfg.index_head_dim
        self.index_rope_head_dim = self.index_head_dim // 2
        self.index_nope_head_dim = self.index_head_dim - self.index_rope_head_dim
        self.index_topk = cfg.index_topk
        self.indexer = DSA2DIndexer(cfg)
        self.index_q_proj = nn.Linear(self.dim, self.index_n_heads * self.index_head_dim, bias=False)
        self.index_k_proj = nn.Linear(self.dim, self.index_n_heads * self.index_head_dim, bias=False)
        self.index_w_proj = nn.Linear(self.dim, self.index_n_heads, bias=False)

        self.q_proj_out_dim = self.n_heads * self.qk_head_dim
        self.kv_a_out_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.kv_proj_out_dim = self.n_kv_heads * (self.qk_nope_head_dim + self.v_head_dim)
        self.attn_out_dim = self.n_heads * self.v_head_dim

        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = nn.LayerNorm(self.q_lora_rank)
        self.wq_b = nn.Linear(self.q_lora_rank, self.q_proj_out_dim, bias=False)

        self.wkv_a = nn.Linear(self.dim, self.kv_a_out_dim, bias=False)
        self.kv_norm = nn.LayerNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.kv_proj_out_dim, bias=False)

        self.proj = nn.Linear(self.attn_out_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_head_dim ** -0.5

    def _flatten_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W] input, got shape={tuple(x.shape)}")
        batch, channels, height, width = x.shape
        if channels != self.dim:
            raise ValueError(f"Expected channel dim {self.dim}, got C={channels}")
        tokens = x.flatten(start_dim=2).transpose(1, 2)
        return tokens, height, width

    def _dense_mla_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        tokens, height, width = self._flatten_tokens(x)
        batch, seq_len, _ = tokens.shape

        q_latent = self.q_norm(self.wq_a(tokens))
        q = self.wq_b(q_latent).view(batch, seq_len, self.n_heads, self.qk_head_dim).permute(0, 2, 1, 3)
        q_nope = q[..., :self.qk_nope_head_dim]
        q_pe = apply_partial_rope_2d_interleaved(
            q[..., self.qk_nope_head_dim:],
            H=height,
            W=width,
            rope_dim=self.qk_rope_head_dim,
        )
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv_latent_and_pe = self.wkv_a(tokens)
        kv_latent = self.kv_norm(kv_latent_and_pe[..., :self.kv_lora_rank])
        k_pe_shared = kv_latent_and_pe[..., self.kv_lora_rank:].view(batch, 1, seq_len, self.qk_rope_head_dim)
        k_pe_shared = apply_partial_rope_2d_interleaved(
            k_pe_shared,
            H=height,
            W=width,
            rope_dim=self.qk_rope_head_dim,
        )

        kv = self.wkv_b(kv_latent).view(
            batch,
            seq_len,
            self.n_kv_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ).permute(0, 2, 1, 3)
        k_nope = kv[..., :self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim:]
        k = torch.cat([k_nope, k_pe_shared.expand(-1, self.n_kv_heads, -1, -1)], dim=-1)
        return q, k, v, height, width

    def forward_dense_reference(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v, height, width = self._dense_mla_qkv(x)
        k = k.repeat_interleave(self.gqa_group_size, dim=1)
        v = v.repeat_interleave(self.gqa_group_size, dim=1)

        attn_scores = torch.matmul(q * self.softmax_scale, k.transpose(-1, -2))
        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], height * width, self.attn_out_dim)
        out = self.proj(out)
        return out.transpose(1, 2).reshape(x.shape[0], self.dim, height, width)

    def forward_sparse_from_indices(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        q, k, v, height, width = self._dense_mla_qkv(x)
        batch = x.shape[0]
        k = k.repeat_interleave(self.gqa_group_size, dim=1)
        v = v.repeat_interleave(self.gqa_group_size, dim=1)

        k_selected = gather_sparse_mla_tokens(k, idx)
        v_selected = gather_sparse_mla_tokens(v, idx)

        attn_scores = torch.einsum("bhtd,bhtkd->bhtk", q * self.softmax_scale, k_selected)
        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.einsum("bhtk,bhtkd->bhtd", attn, v_selected)
        out = out.permute(0, 2, 1, 3).reshape(batch, height * width, self.attn_out_dim)
        out = self.proj(out)
        return out.transpose(1, 2).reshape(batch, self.dim, height, width)

    def forward_sparse_with_forced_topk(self, x: torch.Tensor, *, topk_equals_t: bool = False) -> torch.Tensor:
        if not topk_equals_t:
            raise NotImplementedError("Only the topk=T sparse-equivalence path is implemented at this stage.")

        batch = x.shape[0]
        seq_len = x.shape[-2] * x.shape[-1]
        full_idx = torch.arange(seq_len, device=x.device, dtype=torch.int64).view(1, 1, seq_len)
        full_idx = full_idx.expand(batch, seq_len, seq_len)
        return self.forward_sparse_from_indices(x, full_idx)

    def build_dense_teacher_distribution(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q, k, _, _, _ = self._dense_mla_qkv(x)
            k = k.repeat_interleave(self.gqa_group_size, dim=1)
            attn_scores = torch.matmul(q * self.softmax_scale, k.transpose(-1, -2))
            attn = torch.softmax(attn_scores, dim=-1)
            teacher = attn.sum(dim=1)
            return teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def build_indexer_logits(self, x: torch.Tensor, *, detach_inputs: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, _, _ = self._flatten_tokens(x)
        if detach_inputs:
            tokens = tokens.detach()
        batch, seq_len, _ = tokens.shape
        q = self.index_q_proj(tokens).view(batch, seq_len, self.index_n_heads, self.index_head_dim).permute(0, 2, 1, 3)
        k = self.index_k_proj(tokens).view(batch, seq_len, self.index_n_heads, self.index_head_dim).permute(0, 2, 1, 3)
        w = self.index_w_proj(tokens)
        return self.indexer(q, k, w)

    def prepare_warmup_indexer_inputs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, _, _ = self._flatten_tokens(x)
        return {"tokens": tokens.detach()}

    def assert_warmup_detach_contract(self) -> None:
        self.zero_grad(set_to_none=True)
        x = torch.randn(
            1,
            self.dim,
            2,
            2,
            dtype=self.proj.weight.dtype,
            device=self.proj.weight.device,
            requires_grad=True,
        )
        teacher = self.build_dense_teacher_distribution(x)
        logits, _ = self.build_indexer_logits(x, detach_inputs=True)
        loss = self.indexer_alignment_kl_loss(logits, teacher)
        loss.backward()

        if x.grad is not None:
            raise AssertionError("Warm-up indexer inputs must be detached from the main model graph")
        if not any(param.grad is not None for name, param in self.named_parameters() if name.startswith("index_")):
            raise AssertionError("Warm-up step should still produce gradients for indexer parameters")

    def indexer_alignment_kl_loss(self, logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
        return F.kl_div(
            F.log_softmax(logits, dim=-1),
            teacher_probs,
            reduction="batchmean",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, idx = self.build_indexer_logits(x, detach_inputs=True)
        return self.forward_sparse_from_indices(x, idx)
