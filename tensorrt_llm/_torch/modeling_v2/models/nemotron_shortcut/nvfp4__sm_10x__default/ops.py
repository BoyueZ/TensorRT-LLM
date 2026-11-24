# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native FP8 projections and NVFP4 projections with per-expert calibration."""

from dataclasses import dataclass

import torch

from tensorrt_llm._torch.modeling_v2.catalog.gemm.nvfp4_gemm import nvfp4_gemm
from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import fp4_quantize

from ..moe import clamped_relu2


@dataclass
class Fp8Projection:
    weight: torch.Tensor
    input_scale: torch.Tensor
    weight_scale: torch.Tensor

    def apply(self, x):
        if x.shape[0] == 0:
            return x.new_empty((0, self.weight.shape[0]))
        packed, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            x.contiguous(), self.input_scale
        )
        return torch.ops.trtllm.cublas_scaled_mm(
            packed,
            self.weight.t(),
            scale_a=self.input_scale,
            scale_b=self.weight_scale,
            bias=None,
            out_dtype=x.dtype,
            output_buffer_kind=0,
            group=None,
        )


@dataclass
class Nvfp4Projection:
    weight: torch.Tensor
    block_scales: torch.Tensor
    inverse_input_scale: torch.Tensor
    alpha: torch.Tensor
    out_features: int

    def apply(self, x):
        if x.shape[0] == 0:
            return x.new_empty((0, self.out_features))
        packed, scales = fp4_quantize(x.contiguous(), self.inverse_input_scale, 16)
        out = nvfp4_gemm(packed, self.weight, scales, self.block_scales, self.alpha, x.dtype)
        return out[:, : self.out_features]

    def apply_experts(self, x, counts, start, end):
        from flashinfer.gemm import grouped_gemm_nt_masked
        from flashinfer.quantization.fp4_quantization import scaled_fp4_grouped_quantize

        lhs = scaled_fp4_grouped_quantize(
            x, counts, self.inverse_input_scale[start:end], nvfp4_4over6=None
        )
        w = self.weight[start:end]
        experts, padded_n, half_k = w.shape
        # The physical layout of these scales is already the 128x4 GPU swizzle.
        scales = (
            self.block_scales[start:end]
            .view(torch.float8_e4m3fn)
            .view(experts, padded_n // 128, half_k // 32, 32, 4, 4)
        )
        rhs = w.permute(1, 2, 0), scales.permute(3, 4, 1, 5, 2, 0)
        out = x.new_empty((experts, x.shape[1], padded_n))
        grouped_gemm_nt_masked(
            lhs,
            rhs,
            out.permute(1, 2, 0),
            counts,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
            alpha=self.alpha[start:end],
            alpha_dtype="float32",
        )
        return out[:, :, : self.out_features]


def grouped_experts(x, expert_ids, scores, up, down, ep_rank, clamp):
    """Bound padded scratch while retaining the calibrated FC1/FC2 stages for each slot."""
    ntokens, topk = expert_ids.shape
    if ntokens == 0:
        return x.float()
    num_local = up.weight.shape[0]
    local = expert_ids.flatten() - ep_rank * num_local
    local = torch.where((local >= 0) & (local < num_local), local, num_local)
    ordered, perm = local.sort(stable=True)
    boundaries = torch.searchsorted(
        ordered, torch.arange(num_local + 1, device=x.device, dtype=ordered.dtype)
    )
    counts = boundaries.diff().to(torch.int32)
    # Every token chooses distinct experts. No expert can have more than ntokens
    # assignments. Up to 1024 padded rows fit in the per-pair activation workspace.
    rows = min(ntokens, 128)
    # FlashInfer requires a 16-byte aligned FP32 alpha pointer at the start
    # of each consecutive expert batch. Slice at multiples of four experts.
    per_batch = min(num_local, (1024 // rows // 4) * 4)
    # For large eager prefill a single count read gives the number of tiles in
    # each group. Graph replay must permit routes to change after capture.
    eager_counts = None
    if ntokens > rows and not torch.cuda.is_current_stream_capturing():
        eager_counts = counts.tolist()

    sink = ntokens * topk
    slots = x.new_zeros((sink + 1, x.shape[-1]))
    row = torch.arange(rows, device=x.device)
    for first in range(0, num_local, per_batch):
        last = min(first + per_batch, num_local)
        max_count = ntokens if eager_counts is None else max(eager_counts[first:last])
        for base in range(0, max_count, rows):
            mask = (counts[first:last] - base).clamp(0, rows)
            indices = (boundaries[first:last, None] + base + row).clamp_max(sink - 1)
            routed_slots = perm[indices]
            inputs = x[routed_slots // topk]
            hidden = up.apply_experts(inputs, mask, first, last)
            weighted = (
                clamped_relu2(hidden, clamp).float() * scores.flatten()[routed_slots, None]
            ).to(x.dtype)
            result = down.apply_experts(weighted.contiguous(), mask, first, last)
            # Each active slot has exactly one writer. All padding writes land on
            # the discarded final row. Non-local slots retain zero for the EP sum.
            dest = torch.where(row < mask[:, None], routed_slots, sink)
            slots.index_copy_(0, dest.flatten(), result.reshape(-1, x.shape[-1]))
    return slots[:sink].view(ntokens, topk, -1).float().sum(1)
