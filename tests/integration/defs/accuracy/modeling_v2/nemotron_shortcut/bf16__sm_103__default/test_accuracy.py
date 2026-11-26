# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from .._common import _ShortcutTextAccuracy


class TestShortcutBf16(_ShortcutTextAccuracy):
    MODEL_NAME = "nemotron_shortcut/bf16"

    def test_gsm8k(self):
        self.evaluate_gsm8k()
