from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.ops.dsa_indexer import (
    act_quant_reference_safe,
    fwht_last_dim,
    stable_topk,
    streaming_weighted_relu_topk,
    weighted_relu_index_score,
)
from src.ops.dsa_deepgemm import deepgemm_is_supported, deepgemm_weighted_relu_logits
from src.ops.dsa_rope import apply_partial_rope_2d_interleaved, apply_partial_rope_2d_non_interleaved
from src.ops.dsa_flashmla import flashmla_is_supported, flashmla_sparse_mla_forward
import src.ops.dsa_sparse_mla_autograd as dsa_sparse_mla_autograd
from src.ops.dsa_sparse_mla import (
    streaming_sparse_mla_reference,
    streaming_sparse_mla_reference_from_runtime,
)


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
    indexer_mode: str = "dense"
    indexer_backend: str = "auto"
    sparse_backend: str = "auto"

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
        if self.index_head_dim & (self.index_head_dim - 1):
            raise ValueError("index_head_dim must be a power of two for FWHT")

        if self.indexer_mode not in {"dense", "streaming"}:
            raise ValueError(f"Unsupported indexer_mode={self.indexer_mode!r}")
        if self.indexer_backend not in {"auto", "reference", "deepgemm"}:
            raise ValueError(f"Unsupported indexer_backend={self.indexer_backend!r}")
        if self.sparse_backend not in {"auto", "reference", "flashmla"}:
            raise ValueError(f"Unsupported sparse_backend={self.sparse_backend!r}")


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
        self.runtime_qk_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.runtime_v_dim = self.kv_lora_rank

        self.index_n_heads = cfg.index_n_heads
        self.index_head_dim = cfg.index_head_dim
        self.index_rope_head_dim = self.index_head_dim // 2
        self.index_nope_head_dim = self.index_head_dim - self.index_rope_head_dim
        self.index_topk = cfg.index_topk
        self.indexer_mode = cfg.indexer_mode
        self.indexer_backend = cfg.indexer_backend
        self.sparse_backend = cfg.sparse_backend
        self.index_softmax_scale = self.index_head_dim ** -0.5
        self.indexer = DSA2DIndexer(cfg)
        self.index_wq_b = nn.Linear(self.q_lora_rank, self.index_n_heads * self.index_head_dim, bias=False)
        self.index_wk = nn.Linear(self.dim, self.index_head_dim, bias=False)
        self.index_k_norm = nn.RMSNorm(self.index_head_dim, eps=1e-5)
        self.index_weights_proj = nn.Linear(self.dim, self.index_n_heads, bias=False)
        self._selector_frozen = False

        self.q_proj_out_dim = self.n_heads * self.qk_head_dim
        self.kv_a_out_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.kv_proj_out_dim = self.n_kv_heads * (self.qk_nope_head_dim + self.v_head_dim)
        self.attn_out_dim = self.n_heads * self.v_head_dim

        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=1e-5)
        self.wq_b = nn.Linear(self.q_lora_rank, self.q_proj_out_dim, bias=False)

        self.wkv_a = nn.Linear(self.dim, self.kv_a_out_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=1e-5)
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

    def _kv_projection_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.wkv_b.weight.view(
            self.n_kv_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        return (
            weights[:, :self.qk_nope_head_dim, :],
            weights[:, self.qk_nope_head_dim:, :],
        )

    def _mla_components(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
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
        return q_nope, q_pe, kv_latent, k_pe_shared, kv, height, width

    def _dense_mla_runtime(self, x: torch.Tensor) -> dict[str, torch.Tensor | int | str]:
        q_nope, q_pe, kv_latent, k_pe_shared, _, height, width = self._mla_components(x)
        batch, _, seq_len, _ = q_nope.shape
        w_uk, _ = self._kv_projection_weights()

        q_nope_grouped = q_nope.view(
            batch,
            self.n_kv_heads,
            self.gqa_group_size,
            seq_len,
            self.qk_nope_head_dim,
        )
        q_absorbed = torch.einsum("bngtd,ndr->bngtr", q_nope_grouped, w_uk)
        q_absorbed = q_absorbed.reshape(batch, self.n_heads, seq_len, self.kv_lora_rank)
        q_runtime = torch.cat([q_absorbed, q_pe], dim=-1)

        kv_latent_runtime = kv_latent[:, None, :, :].expand(-1, self.n_kv_heads, -1, -1)
        k_pe_runtime = k_pe_shared.expand(-1, self.n_kv_heads, -1, -1)
        kv_runtime = torch.cat([kv_latent_runtime, k_pe_runtime], dim=-1)

        return {
            "q": q_runtime,
            "kv": kv_runtime,
            "height": height,
            "width": width,
            "d_qk": self.runtime_qk_dim,
            "d_v": self.runtime_v_dim,
            "kv_layout": "latent_then_rope",
        }

    def selector_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.indexer,
            self.index_wq_b,
            self.index_wk,
            self.index_k_norm,
            self.index_weights_proj,
        )

    def selector_parameters(self):
        for module in self.selector_modules():
            yield from module.parameters()

    def _apply_uv_output_projection(self, latent_out: torch.Tensor) -> torch.Tensor:
        if latent_out.ndim != 4:
            raise ValueError(f"Expected latent_out as [B, h_q, T, r], got shape={tuple(latent_out.shape)}")
        if latent_out.shape[1] != self.n_heads:
            raise ValueError(f"Expected {self.n_heads} query heads, got {latent_out.shape[1]}")
        if latent_out.shape[-1] != self.kv_lora_rank:
            raise ValueError(f"Expected latent rank {self.kv_lora_rank}, got {latent_out.shape[-1]}")

        _, w_uv = self._kv_projection_weights()
        batch, _, seq_len, _ = latent_out.shape
        latent_grouped = latent_out.view(
            batch,
            self.n_kv_heads,
            self.gqa_group_size,
            seq_len,
            self.kv_lora_rank,
        )
        projected = torch.einsum("bngtr,nvr->bngtv", latent_grouped, w_uv)
        return projected.reshape(batch, self.n_heads, seq_len, self.v_head_dim)

    def _dense_mla_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        q_nope, q_pe, _, k_pe_shared, kv_proj, height, width = self._mla_components(x)
        q = torch.cat([q_nope, q_pe], dim=-1)
        k_nope = kv_proj[..., :self.qk_nope_head_dim]
        v = kv_proj[..., self.qk_nope_head_dim:]
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
        runtime = self._dense_mla_runtime(x)
        height = int(runtime["height"])
        width = int(runtime["width"])
        batch = x.shape[0]
        should_try_flashmla = self.sparse_backend in {"auto", "flashmla"} and not torch.is_grad_enabled()
        if should_try_flashmla and flashmla_is_supported(
            device=x.device,
            n_kv_heads=self.n_kv_heads,
        ):
            out = flashmla_sparse_mla_forward(
                runtime,
                idx,
                gqa_group_size=self.gqa_group_size,
                softmax_scale=self.softmax_scale,
            )
        else:
            out = streaming_sparse_mla_reference_from_runtime(
                runtime,
                idx,
                gqa_group_size=self.gqa_group_size,
                softmax_scale=self.softmax_scale,
            )
        out = self._apply_uv_output_projection(out)
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

    def freeze_selector_parameters(self) -> None:
        self._selector_frozen = True
        for param in self.selector_parameters():
            param.requires_grad_(False)

    def unfreeze_selector_parameters(self) -> None:
        self._selector_frozen = False
        for param in self.selector_parameters():
            param.requires_grad_(True)

    def forward_sparse_with_frozen_selector(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, idx = self.build_indexer_selection(x, detach_inputs=True)
        if self.sparse_backend == "reference":
            return self.forward_sparse_from_indices(x, idx)
        runtime = self._dense_mla_runtime(x)
        out = dsa_sparse_mla_autograd.packed_sparse_mla_autograd_forward(
            runtime,
            idx,
            gqa_group_size=self.gqa_group_size,
            softmax_scale=self.softmax_scale,
        )
        out = self._apply_uv_output_projection(out)
        batch = x.shape[0]
        height = int(runtime["height"])
        width = int(runtime["width"])
        out = out.permute(0, 2, 1, 3).reshape(batch, height * width, self.attn_out_dim)
        out = self.proj(out)
        return out.transpose(1, 2).reshape(batch, self.dim, height, width)

    def build_dense_teacher_distribution(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q, k, _, _, _ = self._dense_mla_qkv(x)
            k = k.repeat_interleave(self.gqa_group_size, dim=1)
            attn_scores = torch.matmul(q * self.softmax_scale, k.transpose(-1, -2))
            attn = torch.softmax(attn_scores, dim=-1)
            teacher = attn.sum(dim=1)
            return teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _prepare_indexer_qkw(
        self,
        x: torch.Tensor,
        *,
        detach_inputs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, height, width = self._flatten_tokens(x)
        if detach_inputs:
            tokens = tokens.detach()
        batch, seq_len, _ = tokens.shape
        q_latent = self.q_norm(self.wq_a(tokens))
        if detach_inputs:
            q_latent = q_latent.detach()
        q = self.index_wq_b(q_latent).view(batch, seq_len, self.index_n_heads, self.index_head_dim).permute(0, 2, 1, 3)
        k = self.index_wk(tokens)
        k = self.index_k_norm(k).view(batch, 1, seq_len, self.index_head_dim)
        q = apply_partial_rope_2d_non_interleaved(
            q,
            H=height,
            W=width,
            rope_dim=self.index_rope_head_dim,
        )
        k = apply_partial_rope_2d_non_interleaved(
            k,
            H=height,
            W=width,
            rope_dim=self.index_rope_head_dim,
        )
        q = fwht_last_dim(q)
        k = fwht_last_dim(k).expand(-1, self.index_n_heads, -1, -1)
        w = self.index_weights_proj(tokens).to(dtype=torch.float32)
        w = w * (self.index_n_heads ** -0.5) * self.index_softmax_scale
        return q, k, w

    def build_indexer_logits(self, x: torch.Tensor, *, detach_inputs: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, w = self._prepare_indexer_qkw(x, detach_inputs=detach_inputs)
        return self.indexer(q, k, w)

    def build_indexer_selection(self, x: torch.Tensor, *, detach_inputs: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        should_try_deepgemm = self.indexer_backend in {"auto", "deepgemm"} and not torch.is_grad_enabled()
        if should_try_deepgemm and deepgemm_is_supported(
            device=x.device,
            n_kv_heads=self.n_kv_heads,
            index_n_heads=self.index_n_heads,
            index_head_dim=self.index_head_dim,
        ):
            q, k, w = self._prepare_indexer_qkw(x, detach_inputs=detach_inputs)
            logits = deepgemm_weighted_relu_logits(q, k, w)
            idx = stable_topk(logits, k=min(self.index_topk, logits.shape[-1]))
            scores = torch.gather(logits, dim=-1, index=idx)
            return scores, idx

        if self.indexer_mode == "dense":
            logits, idx = self.build_indexer_logits(x, detach_inputs=detach_inputs)
            scores = torch.gather(logits, dim=-1, index=idx)
            return scores, idx

        q, k, w = self._prepare_indexer_qkw(x, detach_inputs=detach_inputs)
        q_q, q_scale = act_quant_reference_safe(q)
        k_q, k_scale = act_quant_reference_safe(k)
        return streaming_weighted_relu_topk(
            q_q.to(dtype=torch.float32) * q_scale,
            k_q.to(dtype=torch.float32) * k_scale,
            w,
            topk=self.index_topk,
            block_s=self.index_topk,
        )

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
        if self.wq_a.weight.grad is not None or self.q_norm.weight.grad is not None:
            raise AssertionError("Warm-up should not backpropagate through the shared MLA query path")
        if not any(param.grad is not None for param in self.selector_parameters()):
            raise AssertionError("Warm-up step should still produce gradients for indexer parameters")

    def indexer_alignment_kl_loss(self, logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
        per_token_kl = F.kl_div(
            F.log_softmax(logits, dim=-1),
            teacher_probs,
            reduction="none",
        )
        return per_token_kl.sum(dim=-1).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._selector_frozen:
            return self.forward_sparse_with_frozen_selector(x)
        _, idx = self.build_indexer_selection(x, detach_inputs=True)
        return self.forward_sparse_from_indices(x, idx)
