# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind Shortcut's expert banks to the executor's standard expert load balancer."""

from tensorrt_llm._torch.moe.fused_moe.impl_blocks import MoEEplbWeightLayoutMixin
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import get_moe_load_balancer


class ExpertPlacement(MoEEplbWeightLayoutMixin):
    def __init__(self, config, manager, layer_idx, num_experts, topk):
        self.num_experts = num_experts
        placement = config.get_layer_initial_global_assignments(layer_idx)
        if placement is None:
            resident = num_experts // config.ep_size
            placement = [
                (rank * resident + slot) % num_experts
                for rank in range(config.ep_size)
                for slot in range(config.num_local_slots)
            ]
        if set(placement) != set(range(num_experts)):
            raise ValueError(
                f"Expert placement for layer {layer_idx} must cover every logical expert"
            )
        self.initial_local_expert_ids = placement[config.slot_start : config.slot_end]
        self.layer_load_balancer = manager.add_layer(num_experts, topk, config.num_local_slots)
        self.layer_load_balancer.set_initial_weight_assignments(placement)

    def bind_weights(self, parameters, prefix, weights_for_host):
        # Resolve the current parameters when the loader registers their CUDA slots.
        self._parameters = parameters
        self._prefix = prefix
        self.register_all_parameter_slot_and_to_fix_weight_fns(
            weights_for_host, checkpoint_banks=True
        )

    def _slot_parameter(self, name):
        return self._parameters[self._prefix + "experts_" + name]

    def route(self, ids):
        layer = self.layer_load_balancer
        if layer.is_dynamic_routing():
            layer.start_wait_gpu_stage()
            layer.done_wait_gpu_stage()
            # The decoder already gathered the logical IDs from all attention-DP ranks.
            layer.update_statistic_with_global_ids(ids, is_first_stage=True, is_last_stage=True)
        return layer.route(ids, offset_by_ep_rank=False)

    def finish(self):
        layer = self.layer_load_balancer
        if layer.is_dynamic_routing():
            layer.start_set_cpu_stage()
            layer.done_set_cpu_stage()


def create_placements(model_config, kinds, moe_kinds, num_experts, topk):
    config = model_config.moe_load_balancer
    manager = get_moe_load_balancer()
    if manager is None:
        raise ValueError(
            "Shortcut expert placement requires an active expert-parallel load balancer"
        )
    if config.mode != "standard" or config.warmup_placement is not None:
        raise ValueError("Shortcut supports standard static or online expert placement")
    if config.num_slots < num_experts:
        raise ValueError("The number of physical expert slots must cover every logical expert")
    return {
        f"l{i}_": ExpertPlacement(config, manager, i, num_experts, topk)
        for i, kind in enumerate(kinds)
        if kind in moe_kinds
    }
