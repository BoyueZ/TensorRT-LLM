# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.configs.nemotron_shortcut import NemotronShortcutTextConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.nemotron_shortcut.cache_manager import NemotronShortcutCacheManager
from tensorrt_llm._torch.pyexecutor._util import _create_kv_cache_manager, get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.config_utils import extract_mamba_kv_cache_params
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping


def _config():
    return NemotronShortcutTextConfig(
        layer_types=["linear_attention", "mlp", "shortcut_moe_attn", "shortcut_moe_gdp"],
        architectures=[NemotronShortcutTextConfig.causal_lm_architecture],
        dtype=torch.bfloat16,
        head_dim=64,
        index_topk=32,
        index_head_dim=32,
        gdp_num_heads=8,
        gdp_num_groups=8,
        gdp_num_householder=5,
        gdp_head_dim=16,
        gdp_key_head_dim=32,
        gdp_conv_kernel=3,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("tp, attention_dp, local_heads", [(4, False, 2), (2, True, 8)])
def test_sparse_kv_replication_and_gdp_sharding(tp, attention_dp, local_heads):
    mapping = Mapping(world_size=tp, tp_size=tp, enable_attention_dp=attention_dp)
    model = ModelConfig(pretrained_config=_config(), mapping=mapping)
    params = extract_mamba_kv_cache_params(model.pretrained_config)
    assert model.get_num_attention_layers() == 1
    assert model.get_num_mamba_layers() == 2
    # Five (V, K) updates and a final query keep two BF16 convolution taps.
    conv_bytes = local_heads * (5 * (16 + 32) + 32) * 2 * torch.bfloat16.itemsize
    temporal_bytes = local_heads * 16 * 32 * torch.float32.itemsize
    assert params.get_states_bytes_per_layer(mapping) == conv_bytes + temporal_bytes
    assert NemotronShortcutCacheManager._estimate_attention_cache_cost(model, mapping) == (
        (2 * 64 + 32) * torch.bfloat16.itemsize,
        0,
        0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA-backed cache pages")
def test_gdp_and_index_page_views():
    config = _config()
    mapping = Mapping()
    model = ModelConfig(pretrained_config=config, mapping=mapping)
    cache = KvCacheConfig(max_tokens=128, enable_block_reuse=False, use_kv_cache_manager_v2=True)
    manager_cls = get_kv_cache_manager_cls(model, cache)
    assert manager_cls is NemotronShortcutCacheManager
    manager = _create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=manager_cls,
        mapping=mapping,
        kv_cache_config=cache,
        tokens_per_block=32,
        max_seq_len=64,
        max_batch_size=1,
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=64,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=model,
        dtype=torch.bfloat16,
    )
    try:
        assert manager.pp_layers == [0, 2, 3]
        assert manager.conv_section_dims == [128] * 5 + [256] * 6
        state = manager.mamba_layer_cache(3)
        assert state.conv.shape[1:] == (2176, 2)
        assert state.conv.dtype == torch.bfloat16
        assert state.temporal.shape[1:] == (8, 16, 32)
        assert state.temporal.dtype == torch.float32
        index = manager.get_index_k_buffer(2, num_heads=1, head_dim=config.index_head_dim)
        assert index.shape[1:] == (32, 1, 32)
        assert index.dtype == torch.bfloat16
        assert manager.get_index_k_buffer(3, head_dim=config.index_head_dim) is None
        assert manager.get_cache_bytes_per_token() == (2 * 64 + 32) * torch.bfloat16.itemsize
    finally:
        manager.shutdown()
