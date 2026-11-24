# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expert selection and ungated BF16 routed experts."""

import torch
from torch.nn import functional as F


def routing_weights(logits, bias, topk, scale):
    """Select on corrected logits and weight the selected experts by their unbiased scores."""
    ids = torch.topk(logits + bias, topk, dim=-1, sorted=False).indices
    weights = torch.sigmoid(logits).gather(1, ids)
    if topk > 1:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return weights * scale, ids.int()


def clamped_relu2(x, clamp):
    limited = (clamp * torch.tanh(x.float() / clamp)).to(x.dtype)
    return limited.relu().square()


def grouped_experts(x, expert_ids, scores, up, down, ep_rank, clamp):
    """Apply the routing weight before the down projection and sum slots in FP32."""
    num_local = up.shape[0]
    local = expert_ids.flatten() - ep_rank * num_local
    # Remote experts sort after the last local group. Keep a fixed number of rows
    # so graph replay can change every token's routing decision.
    local = torch.where((local >= 0) & (local < num_local), local, num_local)
    ordered, perm = local.sort(stable=True)
    groups = torch.arange(num_local, device=x.device, dtype=ordered.dtype)
    ends = torch.searchsorted(ordered, groups, right=True, out_int32=True)
    tokens = perm // expert_ids.shape[1]
    # grouped_mm permits an unused suffix. Retain one spare row even on EP1.
    rows = F.pad(x[tokens], (0, 0, 0, 1))
    hidden = F.grouped_mm(rows, up.transpose(1, 2), offs=ends)
    scale = F.pad(scores.flatten()[perm], (0, 1))
    hidden = (clamped_relu2(hidden, clamp).float() * scale[:, None]).to(x.dtype)
    result = F.grouped_mm(hidden, down.transpose(1, 2), offs=ends)[:-1]
    result = torch.where((ordered < num_local)[:, None], result, 0)
    slots = torch.empty_like(result).index_copy_(0, perm, result)
    return slots.view(x.shape[0], expert_ids.shape[1], x.shape[1]).float().sum(1)
