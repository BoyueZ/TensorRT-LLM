# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm._torch.modeling_v2 import disable_v2_by_default
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin

from ..multimodal import ShortcutMedia
from .modeling import ShortcutMxfp8ForCausalLM


@disable_v2_by_default
class ShortcutMxfp8ForConditionalGeneration(
    ShortcutMedia, ShortcutMxfp8ForCausalLM, MultimodalModelMixin
):
    """Generate text conditioned on visual inputs."""
