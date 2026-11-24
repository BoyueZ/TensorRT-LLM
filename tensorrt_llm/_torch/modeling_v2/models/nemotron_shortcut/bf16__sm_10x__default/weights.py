# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank-local BF16 weight loader for the text decoder."""

import torch

from ..manifest import _SKIPPED_PREFIXES, _manifest, _rows


@torch.no_grad()
def load(model, weights):
    """Load only the local shards and check both checkpoint and destination coverage."""
    core = model.model
    manifest = _manifest(core)
    parameters = dict(core.w.items())
    if set(manifest) != set(parameters):
        raise ValueError(f"Parameter manifest mismatch: {sorted(set(manifest) ^ set(parameters))}")

    head = model.lm_head.weight
    local_vocab = head.shape[0]
    if local_vocab * core.tp_size != parameters["embed"].shape[0]:
        raise ValueError("The vocabulary must divide evenly across the output projection ranks.")
    head_window = _rows(core.slices["tp_rank"] * local_vocab, local_vocab)
    parameters["lm_head"] = head
    manifest["lm_head"] = [("lm_head.weight", head_window, None, None)]

    needed = {key for entries in manifest.values() for key, _, _, _ in entries}
    provided = {key for key in weights if not key.startswith(_SKIPPED_PREFIXES)}
    if needed != provided:
        missing = sorted(needed - provided)
        extra = sorted(provided - needed)
        raise ValueError(f"Checkpoint manifest mismatch: missing {missing[:8]}, extra {extra[:8]}")

    for name, entries in manifest.items():
        for key, window, rows, transform in entries:
            source = weights[key][:] if window is None else weights[key][window]
            if transform is not None:
                source = transform(source)
            dest = parameters[name] if rows is None else parameters[name][rows]
            if source.shape != dest.shape or source.dtype != dest.dtype:
                raise ValueError(
                    f"{key} -> {name}: expected {dest.shape} {dest.dtype}, "
                    f"got {source.shape} {source.dtype}"
                )
            dest.copy_(source, non_blocking=True)
    if head.is_cuda:
        torch.cuda.current_stream(head.device).synchronize()
