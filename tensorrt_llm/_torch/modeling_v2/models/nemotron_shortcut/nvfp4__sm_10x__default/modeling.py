# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixed-precision text-generation target for Blackwell."""

import torch

from tensorrt_llm._torch.modeling_v2 import disable_v2_by_default

from ..decoder import ShortcutDecoder, ShortcutForCausalLM
from ..manifest import _manifest, excluded_prefixes
from .ops import grouped_experts
from .weights import load, projection_parts


class MixedDecoder(ShortcutDecoder):
    def __init__(self, model_config):
        self.recipe = model_config.quant_config_dict or {}
        self.projections = {}
        self.weight_manifest = None
        super().__init__(model_config)
        head = self.recipe.get("lm_head")
        if head is None or head.quant_algo != "FP8":
            raise ValueError("The mixed-precision decoder requires a calibrated FP8 output head")
        used = {"lm_head"}
        for parts in self.projections.values():
            used.update(part.source.rsplit(".", 1)[0] for part in parts)
        ignored = excluded_prefixes(self.mtp_enabled)
        extra = {key for key in self.recipe if not key.startswith(ignored)} - used
        if extra:
            raise ValueError(f"Unexpected ModelOpt text projection rules: {sorted(extra)[:8]}")

    def _declare_matrix(self, weights, name, shape, P):
        if self.weight_manifest is None:
            self.weight_manifest = _manifest(self)
        parts = projection_parts(name, shape, self.weight_manifest[name], self.recipe)
        if not parts:
            super()._declare_matrix(weights, name, shape, P)
            return
        self.projections[name] = parts
        for part in parts:
            part.declare(weights, P)

    def _matrix_view(self, name):
        if name not in self.projections:
            return super()._matrix_view(name)
        parts = [part.view(self.w) for part in self.projections[name]]
        return parts[0] if len(parts) == 1 else parts

    def _mm(self, x, weight, **kwargs):
        if isinstance(weight, torch.Tensor):
            return super()._mm(x, weight, **kwargs)
        if isinstance(weight, list):
            return torch.cat([part.apply(x) for part in weight], dim=-1)
        return weight.apply(x)

    _grouped_experts = staticmethod(grouped_experts)


@disable_v2_by_default
class ShortcutMixedForCausalLM(ShortcutForCausalLM):
    decoder_cls = MixedDecoder
    weight_loader = staticmethod(load)
