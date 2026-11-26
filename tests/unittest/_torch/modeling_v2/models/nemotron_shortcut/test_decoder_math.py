# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.decoder import _rms_norm
from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.moe import routing_weights


@pytest.mark.cpu_only
def test_indexer_adapter_cannot_alias_attention_query():
    from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.decoder import ShortcutForCausalLM
    from tensorrt_llm._torch.peft.lora.loaders import get_all_hf_lora_weights

    owner = ShortcutForCausalLM.lora_owner
    targets = {hf for _, hf in owner.hf_targets}
    key = "base_model.model.model.language_model.layers.5.self_attn.indexer.q_proj.lora_A.weight"
    with pytest.raises(ValueError, match="Unsupported decoder LoRA projection"):
        get_all_hf_lora_weights({key: torch.empty(8, 16)}, targets, owner=owner)


@pytest.mark.cpu_only
def test_fused_nemo_adapter_is_rejected_before_target_inference(tmp_path):
    from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.decoder import ShortcutForCausalLM
    from tensorrt_llm._torch.peft.lora.config import LoraConfig

    config = LoraConfig(lora_ckpt_source="nemo", lora_dir=[str(tmp_path)])
    with pytest.raises(ValueError, match="requires HF LoRA checkpoints with separate projections"):
        ShortcutForCausalLM.lora_owner.configure_hf_targets(config)


@pytest.mark.cpu_only
def test_logit_correction_only_affects_expert_selection():
    # Correcting sigmoid scores would select expert 1 instead of expert 0.
    logits = torch.tensor([[7.0, 2.0, 3.0, -2.0]])
    bias = torch.tensor([-0.5, 0.6, 0.2, 0.0])
    weights, ids = routing_weights(logits, bias, topk=2, scale=3.0)
    order = ids.argsort(1)
    assert ids.gather(1, order).tolist() == [[0, 2]]
    scores = [1 / (1 + math.exp(-v)) for v in (7.0, 3.0)]
    expected = torch.tensor([[3.0 * score / sum(scores) for score in scores]])
    torch.testing.assert_close(weights.gather(1, order), expected)


@pytest.mark.cpu_only
def test_branch_norm_rounds_before_applying_weight():
    x = torch.tensor([[1, 3, 5, 7]], dtype=torch.bfloat16)
    weight = torch.tensor([1.25, 1.5, 1.75, 2.25], dtype=x.dtype)
    expected = torch.tensor([[0.271484375, 0.984375, 1.9140625, 3.4375]], dtype=x.dtype)
    torch.testing.assert_close(_rms_norm(x, weight, 1e-6), expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA-backed GDP state")
def test_prefill_retains_the_clamped_state():
    from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.gdp import _prefill

    # With no keys or values, the only state change is the decay once per token.
    # A large resumed state makes an epsilon added to the linear gate observable.
    with torch.device("cuda"):
        q = torch.zeros(2, 10, 128, dtype=torch.bfloat16)
        k = torch.zeros(2, 3, 10, 128, dtype=q.dtype)
        v = torch.zeros(2, 3, 10, 64, dtype=q.dtype)
        log_decay = torch.full((2, 10), -20.0)
        beta = torch.ones(2, 3, 10)
        state = torch.full((2, 10, 64, 128), 1e9)
        _prefill(
            q,
            k,
            v,
            log_decay,
            beta,
            state,
            torch.tensor([0, 1]),
            torch.tensor([True, False]),
            torch.tensor([0, 1, 2]),
        )
        expected = torch.full_like(state[0], 1e9 * math.exp(-20.0))
        torch.testing.assert_close(state[0], expected, rtol=1e-4, atol=1e-5)
        assert torch.count_nonzero(state[1]) == 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="requires the Blackwell sparse indexer",
)
def test_sparse_selection_preserves_logical_order_at_cutoff():
    from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.ops.attention.index_topk_rows import (
        index_topk_rows,
    )

    width, topk, page_size = 2813, 2048, 64
    pages = math.ceil(width / page_size)
    with torch.device("cuda"):
        # The last key wins and the other keys tie. The second row sees only five keys.
        keys = torch.zeros((pages, page_size, 1, 128), dtype=torch.bfloat16)
        keys.flatten(0, 1)[width - 1].fill_(1)
        query = torch.ones((2, 128), dtype=keys.dtype)
        page = torch.arange(pages, dtype=torch.int32) * 2
        offsets = torch.stack((page, page + 1)).unsqueeze(0)
        lengths = torch.tensor([width, 5], dtype=torch.int32)
        scores = torch.empty((2, width + 3), dtype=torch.float32)[:, :width]
        indices = torch.empty((2, topk), dtype=torch.int32)
        out = torch.empty((1, 2, 2, topk), dtype=torch.int32)
        row = torch.arange(2, dtype=torch.int32)
        arguments = (
            query,
            keys,
            offsets,
            2,
            0,
            torch.zeros_like(row),
            lengths,
            row,
            torch.ones_like(row),
            scores,
            indices,
            out,
        )
        index_topk_rows(*arguments)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            index_topk_rows(*arguments)
        indices.fill_(-2)
        graph.replay()
        expected = torch.full_like(indices, -1)
        expected[0, :-1] = torch.arange(topk - 1, dtype=torch.int32)
        expected[0, -1] = width - 1
        expected[1, :5] = torch.arange(5, dtype=torch.int32)
        torch.testing.assert_close(indices, expected, atol=0, rtol=0)
