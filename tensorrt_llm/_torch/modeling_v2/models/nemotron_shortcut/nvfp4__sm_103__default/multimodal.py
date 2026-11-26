# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin

from ..multimodal import ShortcutMedia
from .modeling import ShortcutMixedForCausalLM


class ShortcutMixedForConditionalGeneration(
    ShortcutMedia, ShortcutMixedForCausalLM, MultimodalModelMixin
):
    """Generate text conditioned on visual inputs."""
