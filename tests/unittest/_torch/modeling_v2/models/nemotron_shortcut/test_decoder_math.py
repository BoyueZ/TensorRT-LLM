# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.decoder import rms_norm
from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.lora import (
    HF_PROJECTIONS,
    ShortcutLoraOwner,
)
from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.moe import routing_weights


@pytest.mark.cpu_only
def test_indexer_adapter_cannot_alias_attention_query():
    from tensorrt_llm._torch.peft.lora.loaders import get_all_hf_lora_weights

    owner = ShortcutLoraOwner(module_path="model", head_size=8, attention_tp=4)
    key = "base_model.model.model.language_model.layers.5.self_attn.indexer.q_proj.lora_A.weight"
    with pytest.raises(ValueError, match="Unsupported decoder LoRA projection"):
        get_all_hf_lora_weights(
            {key: torch.empty(8, 16)}, set(HF_PROJECTIONS.values()), owner=owner
        )


@pytest.mark.cpu_only
def test_fused_nemo_adapter_is_rejected_before_target_inference(tmp_path):
    from tensorrt_llm._torch.peft.lora.config import LoraConfig

    config = LoraConfig(lora_ckpt_source="nemo", lora_dir=[str(tmp_path)])
    owner = ShortcutLoraOwner(module_path="model", head_size=8, attention_tp=1)
    with pytest.raises(ValueError, match="requires HF LoRA checkpoints with separate projections"):
        owner.configure_hf_targets(config)


@pytest.mark.cpu_only
@pytest.mark.parametrize("tp", [1, 4])
def test_kv_adapter_rows_follow_attention_tp(tp):
    from tensorrt_llm._torch.peft.lora.loaders import get_all_hf_lora_weights

    dim, rank = 8, 2
    owner = ShortcutLoraOwner(module_path="model", head_size=dim, attention_tp=tp)
    proj = "self_attn.k_proj"
    prefix = f"base_model.model.model.layers.0.{proj}"
    a = torch.arange(rank * 16).view(rank, 16)
    b = torch.arange(dim * rank).view(dim, rank)
    rows = get_all_hf_lora_weights(
        {f"{prefix}.lora_A.weight": a, f"{prefix}.lora_B.weight": b}, {proj}, owner=owner
    )[0][proj]
    assert rows["in"] is a
    for shard in rows["out"].chunk(tp):
        torch.testing.assert_close(shard, b)
    assert owner.prepare_hf_weight(proj, "out", rows["out"]) is rows["out"]


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
    torch.testing.assert_close(rms_norm(x, weight, 1e-6), expected, atol=0, rtol=0)


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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="requires SM103 NVFP4 paged attention",
)
def test_native_nvfp4_kv_continuation_across_pages():
    from tensorrt_llm._torch.modeling_v2.catalog.attention.qsa_paged_kv_store_nvfp4 import (
        qsa_paged_kv_store_nvfp4,
    )
    from tensorrt_llm._torch.modeling_v2.catalog.attention.qsa_paged_sparse_gqa_nvfp4 import (
        qsa_paged_sparse_gqa_nvfp4,
    )
    from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.indexer import IndexPlan

    page_size, head_dim, query_heads = 32, 256, 8
    topk, batch = 2048, 16
    with torch.device("cuda"):
        table = torch.tensor([[1, 0]], dtype=torch.int32)
        slot_scale = 24
        offsets = torch.stack([table * slot_scale, table * slot_scale + 1], dim=1)
        generator = torch.Generator(device="cuda").manual_seed(5)
        slots = topk // page_size + batch - 1
        data = torch.randint(
            256, (slots, 2, 1, page_size, head_dim // 2), dtype=torch.uint8, generator=generator
        )
        scales = torch.randint(
            0x28,
            0x3B,
            (slots, 2, 1, page_size, head_dim // 16),
            dtype=torch.uint8,
            generator=generator,
        )
        planes = (data[:, 0], data[:, 1], scales[:, 0], scales[:, 1])
        layout = SimpleNamespace(nvfp4_planes={5: planes}, slot_scale=slot_scale)
        calibration = torch.tensor([1.0, 0.25, 0.5])
        for start, count in ((0, page_size), (page_size, 2)):
            positions = torch.arange(start, start + count, dtype=torch.int32)
            req = torch.zeros_like(positions)
            plan = IndexPlan(layout, offsets, req, positions, count if not start else 0, 64, [])
            qkv = torch.randn(
                count, query_heads + 2, head_dim, dtype=torch.bfloat16, generator=generator
            )
            q, k, v = qkv.split((query_heads, 1, 1), dim=1)
            packed = qsa_paged_kv_store_nvfp4.reference(
                k,
                v,
                *planes,
                req,
                positions,
                table,
                page_size,
                calibration.reciprocal(),
            )
            keys = torch.arange(64, dtype=torch.int32)
            selected = torch.where(keys <= positions[:, None], keys, -1)
            expected = qsa_paged_sparse_gqa_nvfp4.reference(
                q,
                *packed,
                calibration,
                table,
                selected,
                req,
                page_size,
                head_dim**-0.5,
            )
            actual = plan.attend_nvfp4(
                5, qkv.flatten(1), None, calibration, calibration.reciprocal()
            )
            for cached, reference in zip(planes, packed):
                torch.testing.assert_close(cached, reference, rtol=0, atol=0)
            # The sparse kernel rounds probabilities before multiplying V.
            torch.testing.assert_close(actual, expected.flatten(1), rtol=0.015625, atol=0.015625)

        # The requests share a full prefix and own distinct tail pages. A query
        # must keep the same reduction order when other generation rows join it.
        pages = torch.arange(topk // page_size, dtype=torch.int32).repeat(batch, 1)
        pages[:, -1] += torch.arange(batch, dtype=torch.int32)
        positions = topk - batch + torch.arange(batch, dtype=torch.int32)
        qkv = torch.randn(
            batch, (query_heads + 2) * head_dim, dtype=torch.bfloat16, generator=generator
        )
        results = []
        for request_slice in (slice(7, 8), slice(None)):
            table = pages[request_slice]
            offsets = torch.stack([table * slot_scale, table * slot_scale + 1], dim=1)
            requests = torch.arange(len(table), dtype=torch.int32)
            plan = IndexPlan(
                layout,
                offsets,
                requests,
                positions[request_slice],
                0,
                topk,
                [],
                batch_invariant=True,
            )
            results.append(
                plan.attend_nvfp4(
                    5, qkv[request_slice], None, calibration, calibration.reciprocal()
                )
            )
        torch.testing.assert_close(results[0][0], results[1][7], rtol=0, atol=0)
