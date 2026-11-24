# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

from ....conftest import llm_models_root
from ...accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from .._env import scoped_env_overrides


class _ShortcutTextAccuracy(LlmapiAccuracyTestHarness):
    @pytest.mark.skip_less_device(4)
    def test_gsm8k(self):
        if get_sm_version() != 103:
            pytest.skip("Four-rank text generation is qualified on GB300")
        checkpoint = Path(llm_models_root()) / self.MODEL_NAME
        if not checkpoint.is_dir():
            pytest.skip(f"Checkpoint unavailable: {checkpoint}")
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(
                str(checkpoint),
                modeling_v2="require",
                env_overrides=rank_env,
                trust_remote_code=True,
                disable_mm_encoder=True,
                tensor_parallel_size=4,
                moe_tensor_parallel_size=2,
                moe_expert_parallel_size=2,
                enable_attention_dp=False,
                enable_chunked_prefill=False,
                max_seq_len=4352,
                max_num_tokens=8192,
                max_batch_size=64,
                cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=64),
                kv_cache_config=KvCacheConfig(
                    dtype="bfloat16", use_kv_cache_manager_v2=True, enable_block_reuse=False
                ),
            ) as llm,
        ):
            GSM8K(self.MODEL_NAME).evaluate(
                llm,
                extra_evaluator_kwargs={"num_fewshot": 5, "apply_chat_template": False},
                sampling_params=SamplingParams(
                    temperature=0, max_tokens=256, truncate_prompt_tokens=4096
                ),
            )
