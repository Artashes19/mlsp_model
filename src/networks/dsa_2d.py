from __future__ import annotations

from dataclasses import dataclass

from torch import nn


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

        if self.index_head_dim % 2 != 0:
            raise ValueError("index_head_dim must be even for a 64/64 rope split")


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

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Dense MLA math is intentionally deferred to later tasks.")
