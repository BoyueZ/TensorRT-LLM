# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nemotron Shortcut GDP state and sparse-attention index pages for the hybrid executor."""

import torch

from tensorrt_llm._torch.pyexecutor.config_utils import (
    build_mamba_kv_cache_params,
    get_nemotron_shortcut_layer_masks,
    mamba_effective_tp_size,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm.bindings import DataType
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig


def get_nemotron_shortcut_cache_params(config, *, spec_config=None, quant_config=None):
    """Express the packed Householder histories and one [head, V, K] state per GDP block."""
    config = getattr(config, "text_config", None) or config
    attention, recurrent = get_nemotron_shortcut_layer_masks(config)
    value_width = config.gdp_num_heads * config.gdp_head_dim
    # The hybrid manager sizes its conv plane as value_width + 2 * n_groups * key_head_dim.
    groups, remainder = divmod(config.gdp_conv_dim - value_width, 2 * config.gdp_key_head_dim)
    if groups < 0 or remainder:
        raise ValueError("GDP convolution width cannot be represented by the hybrid cache manager")
    return build_mamba_kv_cache_params(
        config,
        state_size=config.gdp_key_head_dim,
        conv_kernel=config.gdp_conv_kernel,
        num_heads=config.gdp_num_heads,
        n_groups=groups,
        head_dim=config.gdp_head_dim,
        mamba_mask=recurrent,
        target_full_attn_mask=attention,
        spec_config=spec_config,
        quant_config=quant_config,
    )


class NemotronShortcutCacheManager(MambaHybridCacheManagerV2):
    """A BF16 K/V and index-K page per attention block and an independent slot per GDP block."""

    _supports_mixed_precision_kv = False

    def __init__(self, *args, pretrained_config, dtype, **kwargs):
        config = getattr(pretrained_config, "text_config", None) or pretrained_config
        if dtype != DataType.BF16:
            raise ValueError("Nemotron Shortcut requires BF16 attention KV cache")
        self.index_key_dim = config.index_head_dim if config.index_topk is not None else 0
        super().__init__(
            *args,
            pretrained_config=config,
            dtype=dtype,
            conv_section_widths=config.gdp_conv_section_widths,
            **kwargs,
        )

    @classmethod
    def _estimate_attention_cache_cost(cls, model_config, mapping, **kwargs):
        config = model_config.pretrained_config
        config = getattr(config, "text_config", None) or config
        attention, _ = get_nemotron_shortcut_layer_masks(config)
        layer_count = sum(attention[layer] for layer in mapping.pp_layers(len(attention)))
        # One complete KV head is replicated when TP exceeds the number of KV heads.
        tp = mamba_effective_tp_size(mapping)
        kv_heads = max(1, config.num_key_value_heads // tp)
        index_dim = config.index_head_dim if config.index_topk is not None else 0
        elements_per_token = 2 * kv_heads * config.head_dim + index_dim
        return layer_count * elements_per_token * torch.bfloat16.itemsize, 0, 0

    def _extra_buffers_per_layer(self, *, tokens_per_block):
        if not self.index_key_dim:
            return {}
        size = self.index_key_dim * torch.bfloat16.itemsize * tokens_per_block
        return {
            local: [BufferConfig(role=Role.INDEX_KEY, size=size)]
            for local in range(self.num_local_layers)
            if not self._is_local_mamba_layer(local)
        }

    def get_layer_bytes_per_token(self, local_layer_idx, data_role):
        index_bytes = (
            0
            if self._is_local_mamba_layer(local_layer_idx)
            else self.index_key_dim * torch.bfloat16.itemsize
        )
        if data_role == Role.INDEX_KEY:
            return index_bytes
        base = super().get_layer_bytes_per_token(local_layer_idx, data_role)
        return base + index_bytes if data_role == Role.ALL else base

    def _attention_cache_bytes_per_token(self):
        return sum(
            self.get_layer_bytes_per_token(i, Role.ALL) for i in range(self.num_local_layers)
        )
