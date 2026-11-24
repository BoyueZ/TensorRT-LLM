# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse-attention key selection: each query row's `topk` best-scoring causal keys,
as the K and V pool rows the token-sparse attention kernel reads."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .index_pool_rows import index_pool_rows
from .index_scores import index_scores


class IndexTopkRows(OpWrapper):
    """For every row `r`: score request `row_req[r]`'s cached index keys at positions
    `[0, row_len[r])` against `q[r]` (``index_scores``), keep the `topk` highest
    (TensorRT-LLM's indexer top-k, ``torch.ops.trtllm.indexer_topk_decode``; a row with at
    most `topk` keys keeps all of them, padded with -1), and write their K and V pool rows
    (``index_pool_rows``) into `out[0, r]`. Returns `out` `[1, N, 2, topk]`.

    `offsets` is the attention pool's block offsets `[B, 2, max_blocks]`, `slot_scale` its
    page-index scale (an index-key page's slot is a K offset divided by it) and
    `layer_in_pool` the layer's position in it. `logits` (fp32, at least `[N, max row_len]`)
    and `indices` (int32 `[>= N, topk]`, contiguous) are caller-owned so the step can be
    captured in a CUDA graph. With the split-work scratch `aux_indices` / `aux_logits` (int32 /
    fp32, `N * 10 * topk` elements) the selection is the decode top-k, which splits a long row
    over several CTAs; without it (None), the prefill top-k, one CTA per row. The row tiles are
    ``index_scores``'s.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode rows, short and long histories",
            spec=dict(tpb=32, rows=(1, 1, 1), past=(100, 700, 5000)),
        ),
        Cell(
            why="short chunks from two requests", spec=dict(tpb=64, rows=(3, 3), past=(509, 3000))
        ),
        Cell(why="a prefill chunk crossing the top-k", spec=dict(tpb=64, rows=(200,), past=(400,))),
        Cell(
            why="prefill rows on the one-CTA top-k",
            spec=dict(tpb=32, rows=(130, 2), past=(450, 3000), prefill=True),
        ),
    )

    note = """
    The top-k returns positions in no particular order and breaks exact ties
    by its radix passes; the reference compares the selected sets, and a test
    that pits it against torch.topk has to allow ties at the cut. Both
    selections are over the same fp32 scores.
    """

    def __call__(
        self,
        q: torch.Tensor,
        cache: torch.Tensor,
        offsets: torch.Tensor,
        slot_scale: int,
        layer_in_pool: int,
        row_req: torch.Tensor,
        row_len: torch.Tensor,
        tile_row0: torch.Tensor,
        tile_rows: torch.Tensor,
        logits: torch.Tensor,
        indices: torch.Tensor,
        aux_indices: torch.Tensor | None,
        aux_logits: torch.Tensor | None,
        out: torch.Tensor,
        splits: int = 1,
        block_m: int = 16,
    ) -> torch.Tensor:
        n = q.shape[0]
        topk = indices.shape[1]
        index_scores(
            q,
            cache,
            offsets[:, 0],
            slot_scale,
            row_req,
            row_len,
            tile_row0,
            tile_rows,
            logits[:n],
            splits,
            block_m,
        )
        if aux_indices is None:
            starts = torch.zeros_like(row_len)
            torch.ops.trtllm.indexer_topk_prefill(logits[:n], starts, row_len, indices[:n], topk)
        else:
            torch.ops.trtllm.indexer_topk_decode(
                logits[:n], row_len, indices[:n], 1, topk, 1, aux_indices, aux_logits, None
            )
        return index_pool_rows(indices[:n], row_req, offsets, layer_in_pool, cache.shape[1], out)

    def reference(
        self,
        q,
        cache,
        offsets,
        slot_scale,
        layer_in_pool,
        row_req,
        row_len,
        tile_row0,
        tile_rows,
        logits,
        indices,
        aux_indices,
        aux_logits,
        out,
        splits=1,
        block_m=16,
    ) -> torch.Tensor:
        """Same contract with torch scores and torch.topk; the selected rows come out sorted."""
        n, topk = q.shape[0], indices.shape[1]
        tpb = cache.shape[1]
        res = torch.full((1, n, 2, topk), -1, dtype=torch.int32, device=q.device)
        for r, (b, length) in enumerate(zip(row_req.tolist(), row_len.tolist())):
            pos = torch.arange(length, device=q.device)
            slots = offsets[b, 0, pos // tpb].long() // slot_scale
            scores = cache[slots, pos % tpb, 0].float() @ q[r].float()
            keep = pos if length <= topk else scores.topk(topk).indices
            keep = keep.sort().values
            page, within = keep // tpb, keep % tpb
            for kv in range(2):
                rows = (offsets[b, kv, page].long() + 2 * layer_in_pool) * tpb + within
                res[0, r, kv, : keep.numel()] = rows.to(torch.int32)
        return res

    def is_valid(
        self,
        q,
        cache,
        offsets,
        slot_scale,
        layer_in_pool,
        row_req,
        row_len,
        tile_row0,
        tile_rows,
        logits,
        indices,
        aux_indices,
        aux_logits,
        out,
        splits=1,
        block_m=16,
    ) -> None:
        n, topk = q.shape[0], indices.shape[1]
        assert (
            indices.dtype == torch.int32
            and indices.is_contiguous()
            and indices.shape[0] >= n
            and topk % 4 == 0
            and row_len.dtype == torch.int32
            and row_len.numel() == n
            and tuple(out.shape) == (1, n, 2, topk)
            and offsets.dim() == 3
            and slot_scale > 0
            and logits.shape[0] >= n
            and logits.stride(1) == 1
            and (aux_indices is None) == (aux_logits is None)
            and (aux_indices is None or aux_indices.numel() >= n * 10 * topk)
            and (aux_logits is None or aux_logits.numel() >= n * 10 * topk)
        ), "IndexTopkRows received unsupported inputs"


index_topk_rows = IndexTopkRows()
