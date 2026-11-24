# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load a mixed ModelOpt recipe using logical, rank-local decoder windows."""

from dataclasses import dataclass

import torch

from ..manifest import _SKIPPED_PREFIXES
from .ops import Fp8Projection, Nvfp4Projection


@dataclass
class Projection:
    name: str
    source: str
    algorithm: str
    shape: tuple
    # Source and destination windows in uncompressed matrix coordinates.
    windows: list

    @property
    def is_bank(self):
        return len(self.shape) == 3

    @property
    def suffixes(self):
        if self.algorithm == "NVFP4":
            return ("weight", "weight_scale", "input_scale", "weight_scale_2")
        return ("weight", "weight_scale", "input_scale")

    def declare(self, weights, P):
        *bank, rows, cols = self.shape
        scalars = bank or [1]
        weights[self.name + "_input_scale"] = P(*scalars, dtype=torch.float32)
        if self.algorithm == "FP8":
            weights[self.name] = P(rows, cols, dtype=torch.float8_e4m3fn)
            weights[self.name + "_weight_scale"] = P(1, dtype=torch.float32)
            return
        # The grouped kernel addresses 128x4-swizzled 16-value scale blocks.
        if cols % 64:
            raise ValueError(f"{self.source}: local NVFP4 K={cols} must divide into 64-value tiles")
        padded = (rows + 127) // 128 * 128
        weights[self.name] = P(*bank, padded, cols // 2, dtype=torch.uint8)
        weights[self.name + "_weight_scale"] = P(*bank, padded * (cols // 16), dtype=torch.uint8)
        weights[self.name + "_weight_scale_2"] = P(*scalars, dtype=torch.float32)

    def view(self, w):
        scale = w[self.name + "_input_scale"]
        block = w[self.name + "_weight_scale"]
        if self.algorithm == "FP8":
            return Fp8Projection(w[self.name], scale, block)
        return Nvfp4Projection(
            w[self.name],
            block,
            scale.reciprocal(),
            scale * w[self.name + "_weight_scale_2"],
            self.shape[-2],
        )

    def module(self, expert=None):
        if not self.is_bank:
            return self.source.removesuffix(".weight")
        group, proj = self.source.rsplit(".", 1)
        return f"{group}.{expert}.{proj}"


def projection_parts(name, shape, entries, recipe):
    """Separate Q/K/V and index Q/K because their calibrated scales can differ."""
    sources = {}
    for key, window, dest, _ in entries:
        sources.setdefault(key, []).append((window, dest))
    parts = []
    bank = len(shape) == 3
    for i, (key, windows) in enumerate(sources.items()):
        module = key.rsplit(".", 1)[0]
        config = recipe.get(module)
        if name.endswith("_router"):
            if config is not None:
                raise ValueError(f"{module}: routing requires a BF16 weight and FP32 logits")
            return []
        if config is None:
            raise ValueError(f"Missing ModelOpt projection recipe for {module}")
        algo = config.quant_algo
        if algo not in ("FP8", "NVFP4") or (bank and algo != "NVFP4"):
            raise ValueError(f"{module}: unsupported projection algorithm {algo}")
        if algo == "NVFP4" and config.group_size != 16:
            raise ValueError(f"{module}: NVFP4 requires group_size=16")
        part_name = name
        part_shape = shape
        if len(sources) > 1:
            ((window, dest),) = windows
            rows = dest[0].stop - dest[0].start
            part_name = f"{name}_{i}"
            part_shape = (rows, shape[-1])
            windows = [(window, None)]
        parts.append(Projection(part_name, key, algo, part_shape, windows))
    return parts


def _k_window(window, elements_per_value):
    if not window or len(window) == 1:
        return window
    rows, cols = window
    if cols.start % elements_per_value or cols.stop % elements_per_value:
        raise ValueError(f"Cannot shard {cols} on {elements_per_value}-element boundaries")
    return rows, slice(cols.start // elements_per_value, cols.stop // elements_per_value)


def _copy(target, source, key):
    if source.shape != target.shape or source.dtype != target.dtype:
        raise ValueError(
            f"{key}: expected {tuple(target.shape)} {target.dtype}, "
            f"got {tuple(source.shape)} {source.dtype}"
        )
    target.copy_(source, non_blocking=True)


def _copy_scalar(weights, key, target):
    # The actual export stores zero-dimensional scales. A lazy safetensors
    # slice accepts an empty index tuple for a scalar, whereas [:] is invalid.
    _copy(target.reshape(()), weights[key][()], key)


def _load_projection(part, w, weights):
    if part.is_bank:
        ((window, _),) = part.windows
        local = range(window[0].start, window[0].stop)
        windows = [(window[1:], None)]
    else:
        local = (None,)
        windows = part.windows
    for i, expert in enumerate(local):
        module = part.module(expert)
        name = part.name
        weight = w[name][i] if part.is_bank else w[name]
        scale = w[name + "_weight_scale"]
        inp_scale = w[name + "_input_scale"][i]
        _copy_scalar(weights, module + ".input_scale", inp_scale)
        nvfp4 = part.algorithm == "NVFP4"
        rows, cols = part.shape[-2:]
        if nvfp4:
            _copy_scalar(weights, module + ".weight_scale_2", w[name + "_weight_scale_2"][i])
            if part.is_bank:
                scale = scale[i]
            # Assemble GDP's discontiguous head windows before swizzling.
            linear = torch.empty(
                (weight.shape[0], cols // 16), device=weight.device, dtype=torch.uint8
            )
            linear[rows:].zero_()
            weight[rows:].zero_()
        else:
            _copy_scalar(weights, module + ".weight_scale", scale)
        for win, dst in windows:
            win = win or (slice(None),)
            dest = dst[0] if dst else slice(0, rows)
            key = module + ".weight"
            _copy(weight[dest], weights[key][_k_window(win, 2 if nvfp4 else 1)], key)
            if nvfp4:
                key = module + ".weight_scale"
                source = weights[key][_k_window(win, 16)]
                _copy(linear[dest].view(torch.float8_e4m3fn), source, key)
        if nvfp4:
            swizzled = torch.ops.trtllm.block_scale_interleave(linear)
            scale.copy_(swizzled.reshape(-1).view(torch.uint8))


def _check_keys(core, weights):
    # The checkpoint dictionary describes every rank. Validate the full set of
    # expert keys without ever opening an off-rank expert.
    keys = {"lm_head." + k for k in ("weight", "weight_scale", "input_scale")}
    for name, entries in core.weight_manifest.items():
        if name not in core.projections:
            keys.update(key for key, *_ in entries)
            continue
        for part in core.projections[name]:
            experts = range(core.num_experts) if part.is_bank else (None,)
            for expert in experts:
                module = part.module(expert)
                keys.update(module + "." + suffix for suffix in part.suffixes)
            # BF16 KV bypasses the published cache-only scales.
            if part.source.endswith(".self_attn.k_proj.weight"):
                keys.add(part.module() + ".k_scale")
            if part.source.endswith(".self_attn.v_proj.weight"):
                keys.add(part.module() + ".v_scale")
    actual = {key for key in weights if not key.startswith(_SKIPPED_PREFIXES)}
    if keys != actual:
        raise ValueError(
            f"Checkpoint manifest mismatch: missing {sorted(keys - actual)[:8]}, "
            f"extra {sorted(actual - keys)[:8]}"
        )


@torch.no_grad()
def load(model, weights):
    core = model.model
    _check_keys(core, weights)
    for name, entries in core.weight_manifest.items():
        if name in core.projections:
            for part in core.projections[name]:
                _load_projection(part, core.w, weights)
            continue
        for key, window, rows, transform in entries:
            source = weights[key][window or (slice(None),)]
            if transform is not None:
                source = transform(source)
            dest = core.w[name] if rows is None else core.w[name][rows]
            _copy(dest, source, key)

    head = model.lm_head
    if head.weight.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "The mixed-precision decoder requires the native calibrated FP8 output head"
        )
    width = head.weight.shape[0]
    if width * core.tp_size != core.w["embed"].shape[0]:
        raise ValueError("The vocabulary must divide evenly across output-head ranks")
    start = core.slices["tp_rank"] * width
    _copy(head.weight, weights["lm_head.weight"][start : start + width], "lm_head.weight")
    _copy_scalar(weights, "lm_head.input_scale", head.input_scale)
    _copy_scalar(weights, "lm_head.weight_scale", head.weight_scale)
    head.inv_input_scale.copy_(head.input_scale.reciprocal())
    head.kv_scales.fill_(1)
    head.inv_kv_scales.fill_(1)
    if head.weight.is_cuda:
        torch.cuda.current_stream(head.weight.device).synchronize()
