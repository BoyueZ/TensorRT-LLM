# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint configuration for the Nemotron Shortcut decoder."""

from transformers.configuration_utils import PretrainedConfig
from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig

from tensorrt_llm.checkpoint_format import NEMOTRON_SHORTCUT_MODEL_TYPE

_MIXER_TYPES = {
    "linear_attention": "linear_attention",
    "full_attention": "full_attention",
    "mlp": "dense",
    "shortcut_moe_attn": "full_attention",
    "shortcut_moe_gdp": "linear_attention",
}
_SHORTCUT_TYPES = {
    "linear_attention": "shortcut_moe_gdp",
    "full_attention": "shortcut_moe_attn",
}


def _decoder_blocks(layer_types: list[str]) -> list[str]:
    """Fold a standalone MoE into its preceding mixer to get decoder block indices."""
    blocks = []
    for kind in layer_types:
        if kind == "moe":
            if not blocks or blocks[-1] not in _SHORTCUT_TYPES:
                raise ValueError("A MoE must immediately follow an attention or GDP mixer.")
            blocks[-1] = _SHORTCUT_TYPES[blocks[-1]]
        elif kind in _MIXER_TYPES:
            blocks.append(kind)
        else:
            raise ValueError(f"Unsupported decoder block type: {kind!r}")
    if not blocks:
        raise ValueError("The decoder must contain at least one block.")
    return blocks


class NemotronShortcutTextConfig(PretrainedConfig):
    """Decoder block order, GDP state geometry and expert routing configuration."""

    model_type = "nemotron4_text"
    base_config_key = "text_config"
    causal_lm_architecture = "NemotronShortcutForCausalLM"

    layer_types: list[str] | None = None
    block_types: list[str] | None = None
    vocab_size: int = 262272
    hidden_size: int = 512
    max_position_embeddings: int = 8192
    tie_word_embeddings: bool = False
    use_cache: bool = False
    rms_norm_eps: float = 1e-5
    residual_in_fp32: bool = False
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    head_dim: int = 128
    index_topk: int | None = None
    index_head_dim: int = 128
    intermediate_size: int = 2048
    mlp_hidden_act: str = "relu2"
    activation_clamp_value: float | None = 16.0
    gdp_num_heads: int = 8
    gdp_num_groups: int = 8
    gdp_num_householder: int = 3
    gdp_head_dim: int = 64
    gdp_key_head_dim: int = 128
    gdp_conv_kernel: int = 4
    gdp_norm_eps: float = 1e-5
    gdp_g_clamp_value: float | None = None
    n_routed_experts: int = 16
    num_experts_per_tok: int = 4
    moe_intermediate_size: int = 512
    moe_shared_expert_intermediate_size: int | None = 512
    moe_latent_size: int = 128
    moe_use_norm_before_up_proj: bool = True
    routed_scaling_factor: float = 3.16
    moe_shortcut_post_norm: bool = False
    moe_router_bias_space: str = "score"
    wide_residual_num_streams: int = 3
    wide_residual_learned_retention: bool = True
    wide_residual_retention_max_forget: float = 0.1

    def __post_init__(self, **kwargs):
        # The serialized in-tree config carries both block order and the mixer
        # vocabulary Transformers uses for layer validation and cache dispatch.
        raw = self.block_types if self.block_types is not None else self.layer_types
        if raw is None:
            raw = [
                "linear_attention",
                "mlp",
                "shortcut_moe_attn",
                "shortcut_moe_gdp",
                "linear_attention",
                "mlp",
            ]
        self.block_types = _decoder_blocks(raw)
        self.layer_types = [_MIXER_TYPES[kind] for kind in self.block_types]
        kwargs.pop("num_hidden_layers", None)
        self.num_hidden_layers = len(self.block_types)
        if self.num_key_value_heads <= 0 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_key_value_heads must be positive and divide num_attention_heads.")
        if self.gdp_num_groups <= 0 or self.gdp_num_heads % self.gdp_num_groups:
            raise ValueError("gdp_num_groups must be positive and divide gdp_num_heads.")
        if not 0 < self.num_experts_per_tok <= self.n_routed_experts:
            raise ValueError("num_experts_per_tok must be between 1 and n_routed_experts.")
        if self.index_topk is not None and self.index_topk <= 0:
            raise ValueError("index_topk must be positive when set.")
        if self.gdp_g_clamp_value is not None and not self.gdp_g_clamp_value > 0:
            raise ValueError("gdp_g_clamp_value must be positive when set.")
        if self.moe_router_bias_space not in ("score", "logit"):
            raise ValueError("moe_router_bias_space must be 'score' or 'logit'.")
        super().__post_init__(**kwargs)

    @property
    def gdp_conv_section_widths(self) -> list[int]:
        """Global conv sections, each sharded independently over attention TP."""
        value_width = self.gdp_num_heads * self.gdp_head_dim
        key_width = self.gdp_num_groups * self.gdp_key_head_dim
        return [value_width] * self.gdp_num_householder + [key_width] * (
            self.gdp_num_householder + 1
        )

    @property
    def gdp_conv_dim(self) -> int:
        return sum(self.gdp_conv_section_widths)

    def get_input_processor_cls(self, *, disable_mm_encoder: bool):
        if disable_mm_encoder:
            from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.input_processor import (
                ShortcutTextInputProcessor,
            )

            return ShortcutTextInputProcessor
        return None


class NemotronShortcutConfig(PretrainedConfig):
    """Composite checkpoint wrapper exposing the decoder via get_text_config()."""

    model_type = NEMOTRON_SHORTCUT_MODEL_TYPE
    sub_configs = {"text_config": NemotronShortcutTextConfig, "vision_config": PixtralVisionConfig}
    text_config: dict | NemotronShortcutTextConfig | None = None
    vision_config: dict | PixtralVisionConfig | None = None
    tie_word_embeddings: bool = False

    @classmethod
    def from_checkpoint_dict(cls, config: dict, **kwargs):
        if config["model_type"] == cls.model_type:
            return cls.from_dict(config, **kwargs)

        # Older flat checkpoints store the vision fields beside the decoder
        # fields or under media_config when saved from a loaded configuration.
        flat = config.copy()
        media = dict(flat.pop("media_config", None) or {})
        if flat.get("vision_config"):
            for key in (
                "vision_config",
                "vision_feature_layer",
                "image_token_id",
                "multimodal_projector_bias",
                "projector_hidden_act",
                "projector_norm_eps",
                "spatial_merge_size",
                "video_temporal_patch_size",
                "separate_video_embedder",
            ):
                if key in flat:
                    media[key] = flat.pop(key)
        text = NemotronShortcutTextConfig.from_dict(flat, **kwargs)
        if media.get("vision_config") is not None:
            return cls(text_config=text, dtype=text.dtype, **media)
        return text

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = NemotronShortcutTextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = NemotronShortcutTextConfig(**self.text_config)
        if isinstance(self.vision_config, dict):
            self.vision_config = PixtralVisionConfig.from_dict(self.vision_config)
        super().__post_init__(**kwargs)
        if self.text_config.dtype is None:
            self.text_config.dtype = self.dtype

    def get_input_processor_cls(self, *, disable_mm_encoder: bool):
        if disable_mm_encoder or self.vision_config is None:
            return self.text_config.get_input_processor_cls(disable_mm_encoder=disable_mm_encoder)
        # The registered model module also supplies the chat placeholder metadata.
        from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.multimodal import (
            ShortcutInputProcessor,
        )

        return ShortcutInputProcessor

    def normalize_model_kwargs(self, overrides: dict) -> dict:
        """Apply legacy flat decoder overrides and keep the executor's mirrored fields in sync."""
        nested = overrides.get("text_config", {})
        if not isinstance(nested, dict):
            return overrides
        decoder = {
            key: value
            for key, value in (overrides | nested).items()
            if hasattr(self.text_config, key) or key == "quantization_config"
        }
        mirrored = {
            key: value
            for key, value in decoder.items()
            if key in nested and key not in ("model_type", "architectures")
        }
        return {**overrides, **mirrored, "text_config": decoder | nested}
