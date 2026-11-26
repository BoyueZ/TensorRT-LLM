# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 text-generation entry point."""

from tensorrt_llm._torch.modeling_v2 import disable_v2_by_default

from ..decoder import ShortcutDecoder, ShortcutForCausalLM
from .weights import load


@disable_v2_by_default
class ShortcutBf16ForCausalLM(ShortcutForCausalLM):
    decoder_cls = ShortcutDecoder
    weight_loader = staticmethod(load)
