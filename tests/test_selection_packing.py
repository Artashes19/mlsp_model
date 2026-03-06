import torch

from src.ops.selection_attention_2d_per_query import _build_packed_patch_metadata


def _unpack_per_head(unique_patch_ids: torch.Tensor, cu_unique_counts: torch.Tensor, packed_idx: torch.Tensor) -> torch.Tensor:
    B, h_kv, T, top_n = packed_idx.shape
    out = torch.empty_like(packed_idx)
    for bh in range(B * h_kv):
        start = int(cu_unique_counts[bh].item())
        end = int(cu_unique_counts[bh + 1].item())
        table = unique_patch_ids[start:end]
        b = bh // h_kv
        h = bh % h_kv
        out[b, h] = table[packed_idx[b, h].to(torch.long)]
    return out


def test_build_packed_patch_metadata_round_trips_single_head():
    block_idx = torch.tensor(
        [[[[5, 2, 5], [1, 2, 3]]]],
        dtype=torch.int32,
    )

    unique_patch_ids, cu_unique_counts, packed_idx = _build_packed_patch_metadata(block_idx)

    assert torch.equal(unique_patch_ids, torch.tensor([1, 2, 3, 5], dtype=torch.int32))
    assert torch.equal(cu_unique_counts, torch.tensor([0, 4], dtype=torch.int32))
    assert torch.equal(
        packed_idx,
        torch.tensor([[[[3, 1, 3], [0, 1, 2]]]], dtype=torch.int32),
    )
    assert torch.equal(_unpack_per_head(unique_patch_ids, cu_unique_counts, packed_idx), block_idx)


def test_build_packed_patch_metadata_is_per_kv_head():
    block_idx = torch.tensor(
        [
            [
                [[7, 3], [7, 1]],
                [[2, 4], [2, 4]],
            ]
        ],
        dtype=torch.int32,
    )

    unique_patch_ids, cu_unique_counts, packed_idx = _build_packed_patch_metadata(block_idx)

    assert torch.equal(unique_patch_ids, torch.tensor([1, 3, 7, 2, 4], dtype=torch.int32))
    assert torch.equal(cu_unique_counts, torch.tensor([0, 3, 5], dtype=torch.int32))
    assert torch.equal(
        packed_idx,
        torch.tensor(
            [
                [
                    [[2, 1], [2, 0]],
                    [[0, 1], [0, 1]],
                ]
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(_unpack_per_head(unique_patch_ids, cu_unique_counts, packed_idx), block_idx)
