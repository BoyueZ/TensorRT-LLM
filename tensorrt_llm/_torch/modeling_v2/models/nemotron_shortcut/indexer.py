# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan causal dot-product selection over the one-head, paged index-key cache."""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import torch
import triton

from tensorrt_llm._torch.modeling_v2.catalog.attention.thop_attention import thop_attention
from tensorrt_llm._utils import maybe_pin_memory

from .ops.attention.index_key_store import index_key_store
from .ops.attention.index_topk_rows import index_topk_rows

_PREFILL_ROWS = 64
_DECODE_ROWS = 16
_SCORE_COLS = 64
_PREFILL_SCORE_BYTES = 256 << 20
_DECODE_SCORE_BYTES = 1 << 30
_SPLIT_TOPK_MIN_WIDTH = 200_000
_TOPK_SPLITS = 10


class _Layout:
    """The physical index-key views and K/V page addressing for one cache manager."""

    def __init__(self, manager, md, layers: list[int], index_dim: int):
        self.manager = weakref.ref(manager)
        self.views = {}
        self.layer_in_pool = {}
        pools = set()
        for layer in layers:
            if layer not in manager.layer_offsets:
                continue
            view = manager.get_index_k_buffer(layer, head_dim=index_dim)
            if view is None:
                raise ValueError(f"Attention layer {layer} has no registered index-key buffer")
            self.views[layer] = view
            row = manager.layer_offsets[layer]
            pool, offset = md.host_kv_cache_pool_mapping[row].tolist()
            pools.add(pool)
            self.layer_in_pool[layer] = offset
        if len(pools) != 1:
            raise ValueError(f"Expected one attention K/V pool, found {sorted(pools)}")
        self.pool = pools.pop()
        self.slot_scale = int(manager.index_scales[self.pool])


@dataclass
class _Pass:
    start: int
    end: int
    tile_starts: torch.Tensor
    tile_sizes: torch.Tensor
    splits: int
    width: int
    tile_rows: int
    split_topk: bool = False


@dataclass
class _DecodeBuffers:
    indices: torch.Tensor
    pool_rows: torch.Tensor
    logits: torch.Tensor
    aux_indices: torch.Tensor | None
    aux_logits: torch.Tensor | None


class IndexPlan:
    """One step shared by the attention layers. Each row names a request and an absolute position."""

    def __init__(
        self,
        layout: _Layout,
        offsets: torch.Tensor,
        requests: torch.Tensor,
        positions: torch.Tensor,
        num_ctx_tokens: int,
        topk: int,
        passes: list[_Pass],
        buffers: _DecodeBuffers | None = None,
    ):
        self.layout = layout
        self.offsets = offsets
        self.requests = requests
        self.positions = positions
        self.lengths = positions + 1
        self.num_ctx_tokens = num_ctx_tokens
        self.topk = topk
        self.passes = passes
        self.sparse = bool(passes)
        self.buffers = buffers

    def store(self, layer: int, keys: torch.Tensor) -> None:
        """Store this layer's [tokens, index_dim] keys, including those from dense steps."""
        index_key_store(
            keys,
            self.layout.views[layer],
            self.offsets[:, 0],
            self.layout.slot_scale,
            self.requests,
            self.positions,
        )

    def select(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        """Return the selected K and V pool rows shaped [1, tokens, 2, topk]."""
        n = q.shape[0]
        if self.buffers is None:
            out = torch.empty(1, n, 2, self.topk, dtype=torch.int32, device=q.device)
            indices = torch.empty(n, self.topk, dtype=torch.int32, device=q.device)
        else:
            out = self.buffers.pool_rows[:, :n]
            indices = self.buffers.indices[:n]
        for work in self.passes:
            rows = slice(work.start, work.end)
            if self.buffers is None:
                logits = torch.empty(
                    work.end - work.start, work.width, dtype=torch.float32, device=q.device
                )
            else:
                logits = self.buffers.logits
            index_topk_rows(
                q[rows],
                self.layout.views[layer],
                self.offsets,
                self.layout.slot_scale,
                self.layout.layer_in_pool[layer],
                self.requests[rows],
                self.lengths[rows],
                work.tile_starts,
                work.tile_sizes,
                logits,
                indices[rows],
                self.buffers.aux_indices if self.buffers is not None and work.split_topk else None,
                self.buffers.aux_logits if self.buffers is not None and work.split_topk else None,
                out[:, rows],
                work.splits,
                work.tile_rows,
            )
        return out

    def attend(self, qkv: torch.Tensor, out: torch.Tensor, rows: torch.Tensor, args: dict) -> None:
        """Run the context and generation portions using their own token-sparse dispatcher."""
        n_ctx = self.num_ctx_tokens
        qkv = qkv.contiguous()
        sparse = dict(
            args,
            use_paged_context_fmha=True,
            sparse_attn_indices_block_size=1,
            num_sparse_topk=self.topk,
        )
        if n_ctx:
            thop_attention(
                q=qkv[:n_ctx],
                output=out[:n_ctx],
                **dict(sparse, attention_input_type=1, sparse_attn_indices=rows[:, :n_ctx]),
            )
        if n_ctx < qkv.shape[0]:
            thop_attention(
                q=qkv[n_ctx:],
                output=out[n_ctx:],
                **dict(sparse, attention_input_type=2, sparse_attn_indices=rows[:, n_ctx:]),
            )


class IndexState:
    """Index-key cache binding and persistent selection buffers for single-token generation."""

    def __init__(self, topk: int, layers: list[int], num_sms: int, *, index_dim: int):
        if topk <= 0 or topk % 4:
            raise ValueError("Token-sparse attention requires a positive top-k divisible by four")
        self.topk = topk
        self.layers = layers
        self.index_dim = index_dim
        self.num_sms = num_sms
        self._layout: _Layout | None = None
        self._rows: torch.Tensor | None = None
        self._buffers: _DecodeBuffers | None = None

    def bind(self, md, device: torch.device) -> None:
        """Allocate decode buffers during profiling so KV-cache sizing accounts for the memory."""
        rows, width = md.max_num_sequences, md.max_seq_len
        self._rows = torch.arange(rows, dtype=torch.int32, device=device)
        if width <= self.topk:
            self._buffers = None
            return
        score_rows = max(1, min(rows, _DECODE_SCORE_BYTES // (4 * width)))
        aux_indices = aux_logits = None
        if width >= _SPLIT_TOPK_MIN_WIDTH:
            elements = min(score_rows, self.num_sms) * _TOPK_SPLITS * self.topk
            aux_indices = torch.empty(elements, dtype=torch.int32, device=device)
            aux_logits = torch.empty(elements, dtype=torch.float32, device=device)
        self._buffers = _DecodeBuffers(
            torch.empty(rows, self.topk, dtype=torch.int32, device=device),
            torch.empty(1, rows, 2, self.topk, dtype=torch.int32, device=device),
            torch.empty(score_rows, width, dtype=torch.float32, device=device),
            aux_indices,
            aux_logits,
        )

    def _cache_layout(self, md) -> _Layout:
        manager = md.kv_cache_manager
        if self._layout is None or self._layout.manager() is not manager:
            self._layout = _Layout(manager, md, self.layers, self.index_dim)
        return self._layout

    def _prefill_passes(self, seq_lens: list[int], width: int, device: torch.device) -> list[_Pass]:
        """Partition contiguous rows of each request without exceeding the score-buffer budget."""
        tiles = []
        offset = 0
        for length in seq_lens:
            for row in range(0, length, _PREFILL_ROWS):
                tiles.append((offset + row, min(_PREFILL_ROWS, length - row)))
            offset += length
        budget = max(_PREFILL_ROWS, _PREFILL_SCORE_BYTES // (4 * width))
        passes = []
        first = 0
        while first < len(tiles):
            start = tiles[first][0]
            end = first + 1
            while end < len(tiles) and tiles[end][0] + tiles[end][1] - start <= budget:
                end += 1
            group = tiles[first:end]
            grid = maybe_pin_memory(
                torch.tensor(
                    [[row - start for row, _ in group], [size for _, size in group]],
                    dtype=torch.int32,
                )
            ).to(device, non_blocking=True)
            splits = min(triton.cdiv(2 * self.num_sms, len(group)), triton.cdiv(width, _SCORE_COLS))
            last_start, last_size = group[-1]
            passes.append(
                _Pass(start, last_start + last_size, grid[0], grid[1], splits, width, _PREFILL_ROWS)
            )
            first = end
        return passes

    def prefill_plan(self, md, n_rows: int) -> IndexPlan:
        """Build an eager plan including cached prefixes, chunks and mixed generation rows."""
        layout = self._cache_layout(md)
        seq = md.seq_lens_cuda[: md.num_seqs]
        kv = md.kv_lens_cuda_runtime[: md.num_seqs]
        requests = torch.repeat_interleave(
            torch.arange(md.num_seqs, dtype=torch.int32, device=seq.device), seq, output_size=n_rows
        )
        starts = torch.cumsum(seq, 0, dtype=torch.int32) - seq
        rows = torch.arange(n_rows, dtype=torch.int32, device=seq.device)
        positions = (kv - seq)[requests] + rows - starts[requests]
        width = int(md.kv_lens_runtime[: md.num_seqs].max()) if n_rows else 0
        passes = []
        if width > self.topk:
            passes = self._prefill_passes(md.seq_lens[: md.num_seqs].tolist(), width, seq.device)
        return IndexPlan(
            layout,
            md.kv_cache_block_offsets[layout.pool],
            requests,
            positions.to(torch.int32),
            md.num_ctx_tokens,
            self.topk,
            passes,
        )

    def decode_plan(self, md, n_rows: int) -> IndexPlan:
        """Build a graph-capturable generation plan using device-resident runtime lengths."""
        if n_rows != md.num_seqs:
            raise ValueError("Single-token decoding requires one input row per sequence")
        if self._rows is None:
            raise RuntimeError("IndexState.bind must run before decoding")
        layout = self._cache_layout(md)
        requests = self._rows[:n_rows]
        positions = (md.kv_lens_cuda_runtime[:n_rows] - 1).to(torch.int32)
        buffers = self._buffers
        passes = []
        if buffers is not None:
            group_size, width = buffers.logits.shape
            sizes = torch.ones(min(n_rows, group_size), dtype=torch.int32, device=requests.device)
            for start in range(0, n_rows, group_size):
                end = min(n_rows, start + group_size)
                size = end - start
                splits = min(triton.cdiv(32 * self.num_sms, size), triton.cdiv(width, _SCORE_COLS))
                passes.append(
                    _Pass(
                        start,
                        end,
                        requests[:size],
                        sizes[:size],
                        splits,
                        width,
                        _DECODE_ROWS,
                        width >= _SPLIT_TOPK_MIN_WIDTH and size < self.num_sms,
                    )
                )
        return IndexPlan(
            layout,
            md.kv_cache_block_offsets[layout.pool],
            requests,
            positions,
            0,
            self.topk,
            passes,
            buffers,
        )
