# Copyright 2024 The AI Edge Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example of building a Gemma3n model."""

from typing import Callable, Dict, List, Optional, Tuple, Union
import math

from ai_edge_torch.generative.layers import attention
from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.generative.layers import normalization
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.layers.rotary_position_embedding as rotary_pos_emb
from ai_edge_torch.generative.utilities import export_config as export_cfg
from ai_edge_torch.generative.utilities import model_builder
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn

TENSOR_NAMES_SEP_QKV = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="model.layers.{}.mlp.up_proj",
    ff_down_proj="model.layers.{}.mlp.down_proj",
    ff_gate_proj="model.layers.{}.mlp.gate_proj",
    attn_query_proj="model.layers.{}.self_attn.q_proj",
    attn_key_proj="model.layers.{}.self_attn.k_proj",
    attn_value_proj="model.layers.{}.self_attn.v_proj",
    attn_output_proj="model.layers.{}.self_attn.o_proj",
    pre_attn_norm="model.layers.{}.input_layernorm",
    post_attn_norm="model.layers.{}.post_attention_layernorm",
    pre_ff_norm="model.layers.{}.pre_feedforward_layernorm",
    post_ff_norm="model.layers.{}.post_feedforward_layernorm",
    embedding="model.embed_tokens",
    final_norm="model.norm",
)

# Gemma3n has specific tensors for AltUp and Laurel
TENSOR_NAMES_GEMMA3N = TENSOR_NAMES_SEP_QKV
# Note: Additional tensors like altup, laurel, per_layer_inputs are handled by custom loader logic or added here if possible.
# But ModelLoader.TensorNames is a fixed namedtuple. We might need to handle specific tensors manually in the model.

class Laurel(nn.Module):
    """Learned Augmented Residual Layer (Laurel)"""
    def __init__(self, embedding_dim: int, config: cfg.LaurelConfig, norm_config: cfg.NormalizationConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        self.linear_left = nn.Linear(embedding_dim, config.rank, bias=False)
        self.linear_right = nn.Linear(config.rank, embedding_dim, bias=False)
        self.post_laurel_norm = builder.build_norm(embedding_dim, norm_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        laurel_hidden_states = self.linear_left(hidden_states)
        laurel_hidden_states = self.linear_right(laurel_hidden_states)
        normed_laurel_hidden_states = self.post_laurel_norm(laurel_hidden_states)
        # Note: Residual addition happens in the block
        return normed_laurel_hidden_states


class AltUp(nn.Module):
    """Alternating Updates (AltUp)"""
    def __init__(self, embedding_dim: int, config: cfg.AltUpConfig, norm_config: cfg.NormalizationConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim

        self.correct_output_scale = nn.Parameter(torch.zeros(embedding_dim))
        self.correction_coefs = nn.Linear(config.num_inputs, config.num_inputs, bias=False)
        self.prediction_coefs = nn.Linear(config.num_inputs, config.num_inputs**2, bias=False)
        self.modality_router = nn.Linear(embedding_dim, config.num_inputs, bias=False)
        self.router_norm = builder.build_norm(embedding_dim, norm_config)
        self.register_buffer("router_input_scale", torch.tensor(embedding_dim**-1.0), persistent=False)

    def compute_router_modalities(self, x: torch.Tensor) -> torch.Tensor:
        router_inputs = self.router_norm(x) * self.router_input_scale
        routed = self.modality_router(router_inputs)
        return torch.tanh(routed.float()).type_as(x)

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [num_altup_inputs, batch_size, num_tokens, hidden_size]
        Returns:
            predictions: [num_altup_inputs, batch_size, num_tokens, hidden_size]
        """
        # hidden_states[self.config.active_idx] is [batch, seq, dim]
        modalities = self.compute_router_modalities(hidden_states[self.config.active_idx])

        if self.training and self.config.coef_clip is not None:
             self.prediction_coefs.weight.data.clamp_(-self.config.coef_clip, self.config.coef_clip)

        # Reshape and permute prediction coefs
        # prediction_coefs: [batch, seq, num_inputs^2] -> [batch, seq, num_inputs, num_inputs]
        all_coefs = (
            self.prediction_coefs(modalities)
            .reshape(*modalities.shape[:-1], self.config.num_inputs, self.config.num_inputs)
            .permute(0, 1, 3, 2)
        )

        # hidden_states: [num_inputs, batch, seq, dim] -> permute to [batch, seq, dim, num_inputs]
        # all_coefs: [batch, seq, num_inputs, num_inputs]
        # matmul: [batch, seq, dim, num_inputs] @ [batch, seq, num_inputs, num_inputs] -> [batch, seq, dim, num_inputs]
        predictions = torch.matmul(hidden_states.permute(1, 2, 3, 0), all_coefs)
        predictions = predictions.permute(3, 0, 1, 2) # [num_inputs, batch, seq, dim]
        predictions = predictions + hidden_states
        return predictions.contiguous().type_as(hidden_states)

    def correct(self, predictions: torch.Tensor, activated: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [num_inputs, batch, seq, dim]
            activated: [batch, seq, dim]
        """
        modalities = self.compute_router_modalities(activated)
        innovation = activated - predictions[self.config.active_idx]
        innovation = innovation.expand(self.config.num_inputs, -1, -1, -1) # Expand on dim 0

        if self.config.coef_clip is not None:
            self.correction_coefs.weight.data.clamp_(-self.config.coef_clip, self.config.coef_clip)

        # correction_coefs: [batch, seq, num_inputs]
        all_coefs = self.correction_coefs(modalities) + 1.0
        # Permute to [num_inputs, batch, seq, 1] for broadcasting
        all_coefs = all_coefs.permute(2, 0, 1).unsqueeze(-1)

        corrected = torch.mul(innovation, all_coefs)
        corrected = corrected + predictions
        return corrected.contiguous().type_as(activated)

    def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor:
        return (corrected.type_as(self.correct_output_scale) * self.correct_output_scale).type_as(corrected)


class Gemma3nBlock(nn.Module):
    """
    Gemma3n Decoder Layer.
    Distinct from TransformerBlock because the forward pass logic is Matformer specific (AltUp/Laurel).
    """
    def __init__(self, config: cfg.TransformerBlockConfig, model_config: cfg.ModelConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = model_config.embedding_dim

        # Standard components
        self.atten_func = attention.CausalSelfAttention(
            self.embedding_dim,
            config.attn_config,
            model_config.enable_hlfb,
        )
        self.ff = builder.build_ff(self.embedding_dim, config.ff_config)

        # Norms
        self.input_layernorm = builder.build_norm(self.embedding_dim, config.pre_attention_norm_config)
        self.post_attention_layernorm = builder.build_norm(self.embedding_dim, config.post_attention_norm_config)
        # Note: pre/post feedforward norms are handled inside the FF block builder in ai_edge_torch usually,
        # but Gemma3n logic is specific.
        # In `modeling_gemma3n.py`:
        # attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        # attn_ffw = self.mlp(attn_norm)
        # attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        #
        # `builder.build_ff` attaches pre_ff_norm and post_ff_norm to the FF module.
        # So `self.ff` already includes them if configured in `ff_config`.
        # We need to ensure `ff_config` has these norms set.

        # Matformer components
        self.altup = AltUp(self.embedding_dim, config.altup_config, config.pre_attention_norm_config) # Reuse norm config for router_norm
        self.laurel = Laurel(self.embedding_dim, config.laurel_config, config.pre_attention_norm_config) # Reuse norm config for laurel norm

        # Per-layer inputs components
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.per_layer_input_gate = nn.Linear(self.embedding_dim, self.hidden_size_per_layer_input, bias=False)
        self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.embedding_dim, bias=False)
        self.post_per_layer_input_norm = builder.build_norm(self.embedding_dim, config.pre_attention_norm_config) # Reuse norm config

        self.act_fn = builder.get_activation(config.ff_config.activation) # Reuse activation from FF config?
        # modeling_gemma3n uses config.hidden_activation. Let's assume FF config matches.

    def forward(
        self,
        hidden_states: torch.Tensor, # This is [num_altup_inputs, batch, seq, dim]
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        per_layer_input: torch.Tensor = None,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: kv_utils.KVCacheEntry = None,
    ) -> Tuple[torch.Tensor, kv_utils.KVCacheEntry]:

        # 1. AltUp Predict
        predictions = self.altup.predict(hidden_states)
        active_prediction = predictions[self.config.altup_config.active_idx]

        # 2. Laurel
        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        # 3. Attention
        # atten_func expects [batch, seq, dim]
        attn_out, kv = self.atten_func(active_prediction_normed, rope, mask, input_pos, kv_cache)
        attn_out = self.post_attention_layernorm(attn_out)

        # 4. Gating and Mixing
        attn_gated = active_prediction + attn_out
        attn_laurel = (attn_gated + laurel_output) / math.sqrt(2)

        # 5. FeedForward
        # self.ff includes pre_ff_norm and post_ff_norm if configured
        attn_ffw_norm = self.ff(attn_laurel)

        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm

        # 6. AltUp Correct
        corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)

        # 7. Per-layer inputs
        first_prediction = corrected_predictions[self.config.altup_config.active_idx].clone()
        if self.config.altup_config.correct_scale:
            first_prediction = self.altup.scale_corrected_output(first_prediction)

        # gate
        first_prediction = self.per_layer_input_gate(first_prediction)
        first_prediction = self.act_fn(first_prediction)
        if per_layer_input is not None:
             first_prediction = torch.mul(first_prediction, per_layer_input)

        # project back
        first_prediction = self.per_layer_projection(first_prediction)
        first_prediction = self.post_per_layer_input_norm(first_prediction)

        # Add to all other streams (except 0?)
        # modeling_gemma3n.py: corrected_predictions[1:] += first_prediction
        # This implies it modifies in-place.
        # To avoid in-place issues in torch export sometimes, we might need to be careful,
        # but let's follow the reference logic.

        # We need to construct a new tensor or modify.
        # corrected_predictions is [num_inputs, B, T, D]
        # first_prediction is [B, T, D]

        # split
        preds_list = list(corrected_predictions.unbind(0))
        for i in range(1, len(preds_list)):
            preds_list[i] = preds_list[i] + first_prediction

        corrected_predictions = torch.stack(preds_list, dim=0)

        return corrected_predictions, kv


class Gemma3n(nn.Module):
    """Gemma3n Model."""

    def __init__(self, config: cfg.ModelConfig, mask_cache_size: int = 0):
        super().__init__()
        self.config = config

        # Embeddings
        self.tok_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        # Gemma3n has a separate embedding for per-layer inputs
        self.embed_tokens_per_layer = nn.Embedding(
            config.vocab_size_per_layer_input,
            config.num_layers * config.block_config(0).hidden_size_per_layer_input,
            padding_idx=0
        )

        # Per layer projection (global)
        self.per_layer_model_projection = nn.Linear(
             config.embedding_dim,
             config.num_layers * config.block_config(0).hidden_size_per_layer_input,
             bias=False
        )
        self.per_layer_projection_norm = builder.build_norm(
            config.block_config(0).hidden_size_per_layer_input,
            config.block_config(0).pre_attention_norm_config
        )

        # AltUp Projections
        num_altup_inputs = config.block_config(0).altup_config.num_inputs
        self.altup_projections = nn.ModuleList(
            [nn.Linear(config.embedding_dim, config.embedding_dim, bias=False) for _ in range(1, num_altup_inputs)]
        )
        self.altup_unembed_projections = nn.ModuleList(
            [nn.Linear(config.embedding_dim, config.embedding_dim, bias=False) for _ in range(1, num_altup_inputs)]
        )

        # Layers
        self.transformer_blocks = nn.ModuleList(
            Gemma3nBlock(config.block_config(idx), config)
            for idx in range(config.num_layers)
        )

        self.final_norm = builder.build_norm(config.embedding_dim, config.final_norm_config)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=config.lm_head_use_bias)
        if config.lm_head_share_weight_with_embedding:
             self.lm_head.weight = self.tok_embedding.weight

        # Scales
        self.register_buffer("per_layer_projection_scale", torch.tensor(config.embedding_dim**-0.5), persistent=False)
        self.register_buffer("per_layer_input_scale", torch.rsqrt(torch.tensor(2.0)), persistent=False)

        self.build_mask_cache(mask_cache_size)

    def build_mask_cache(self, mask_cache_size: int):
        if mask_cache_size <= 0:
            self.mask_cache = None
            self.sliding_window_mask_cache = None
            return
        self.mask_cache = attn_utils.build_causal_mask_cache(mask_cache_size)
        self.sliding_window_mask_cache = attn_utils.build_sliding_window_mask_cache(
            size=mask_cache_size,
            window_size=self.config.block_config(0).attn_config.sliding_window_size,
        )

    def get_attention_mask(self, attn_type: cfg.AttentionType, input_pos: torch.Tensor) -> torch.Tensor:
        if attn_type == cfg.AttentionType.LOCAL_SLIDING:
            return self.sliding_window_mask_cache.index_select(2, input_pos)
        return self.mask_cache.index_select(2, input_pos)

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Input ids for per layer inputs need to be derived or passed.
        # Logic from modeling_gemma3n.py:
        # per_layer_inputs_mask = torch.logical_and(input_ids >= 0, input_ids < self.vocab_size_per_layer_input)
        # per_layer_inputs_tokens = torch.where(per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids))
        vocab_size_per_layer = self.config.vocab_size_per_layer_input
        mask = (input_ids >= 0) & (input_ids < vocab_size_per_layer)
        tokens = torch.where(mask, input_ids, torch.zeros_like(input_ids))

        embeds = self.embed_tokens_per_layer(tokens)
        # Reshape: [B, T, num_layers * hidden_per_layer] -> [B, T, num_layers, hidden_per_layer]
        return embeds.reshape(
            *tokens.shape,
            self.config.num_layers,
            self.config.block_config(0).hidden_size_per_layer_input
        )

    def project_per_layer_inputs(self, inputs_embeds: torch.Tensor, per_layer_inputs: torch.Tensor) -> torch.Tensor:
        proj = self.per_layer_model_projection(inputs_embeds)
        proj = proj * self.per_layer_projection_scale

        hidden_per_layer = self.config.block_config(0).hidden_size_per_layer_input
        proj = proj.reshape(*inputs_embeds.shape[:-1], self.config.num_layers, hidden_per_layer)
        proj = self.per_layer_projection_norm(proj)

        if per_layer_inputs is None:
             return proj

        # Scale
        return (proj + per_layer_inputs) * self.per_layer_input_scale

    @torch.inference_mode
    def forward(
        self,
        tokens: torch.Tensor,
        input_pos: torch.Tensor,
        kv_cache: kv_utils.KVCache,
        mask: Optional[torch.Tensor] = None,
        export_config: Optional[export_cfg.ExportConfig] = None,
    ) -> dict[torch.Tensor, kv_utils.KVCache]:

        input_embeds = self.tok_embedding(tokens)
        if self.config.embedding_scale is not None:
             input_embeds = input_embeds * self.config.embedding_scale

        # Per layer inputs
        per_layer_inputs = self.get_per_layer_inputs(tokens)
        per_layer_inputs = self.project_per_layer_inputs(input_embeds, per_layer_inputs)

        # AltUp Input Expansion
        target_magnitude = torch.mean(input_embeds**2, dim=-1, keepdim=True) ** 0.5
        epsilon = 1e-5

        temp_hidden_states = [input_embeds]
        num_altup_inputs = self.config.block_config(0).altup_config.num_inputs

        for i in range(1, num_altup_inputs):
            altup_proj = self.altup_projections[i-1](input_embeds)

            new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, torch.tensor(epsilon).to(target_magnitude)))
            current_hidden_state = altup_proj * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)

        hidden_states = torch.stack(temp_hidden_states, dim=0) # [num_inputs, B, T, D]

        # RoPE
        attn_config = self.config.block_config(0).attn_config
        n_elem = int(attn_config.rotary_percentage * attn_config.head_dim)
        rope = rotary_pos_emb.build_rope(input_pos, n_elem, attn_config.rotary_base)

        if mask is None:
             mask = [
                 self.get_attention_mask(self.config.block_config(i).attn_config.attn_type, input_pos)
                 for i in range(self.config.num_layers)
             ]

        updated_kv_entries = []
        for i, block in enumerate(self.transformer_blocks):
             mask_entry = mask[i] if isinstance(mask, list) else mask
             kv_entry = kv_cache.caches[i] if kv_cache else None

             # Extract per_layer_input for this layer: [B, T, D_per_layer]
             layer_input = per_layer_inputs[:, :, i, :]

             hidden_states, kv_entry = block(hidden_states, rope, layer_input, mask_entry, input_pos, kv_entry)
             if kv_entry:
                  updated_kv_entries.append(kv_entry)

        updated_kv_cache = kv_utils.KVCache(tuple(updated_kv_entries))

        # Output collapse
        target_magnitude = torch.mean(hidden_states[0] ** 2, dim=-1, keepdim=True) ** 0.5
        temp_hidden_states = [hidden_states[0]]

        for i in range(1, num_altup_inputs):
             altup_unemb_proj = self.altup_unembed_projections[i-1](hidden_states[i])

             new_magnitude = torch.mean(altup_unemb_proj**2, dim=-1, keepdim=True)
             new_magnitude = torch.sqrt(torch.maximum(new_magnitude, torch.tensor(epsilon).to(target_magnitude)))
             current_hidden_state = altup_unemb_proj * target_magnitude / new_magnitude
             temp_hidden_states.append(current_hidden_state)

        hidden_states = torch.stack(temp_hidden_states)
        hidden_states = torch.mean(hidden_states, dim=0)
        hidden_states = self.final_norm(hidden_states)

        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcap is not None:
             logits = logits / self.config.final_logit_softcap
             logits = torch.tanh(logits)
             logits = logits * self.config.final_logit_softcap

        if export_config is not None:
             if torch.numel(input_pos) > 1 and not export_config.output_logits_on_prefill:
                  return {"kv_cache": updated_kv_cache}

        return {"logits": logits, "kv_cache": updated_kv_cache}


def get_model_config(
    num_layers=26,
    embedding_dim=2304,
    hidden_dim=9216,
    num_heads=8,
    head_dim=256,
    vocab_size=256000,
    vocab_size_per_layer_input=32768, # Default from HF config usually
    altup_num_inputs=2, # Default
) -> cfg.ModelConfig:
  """Returns a model config for Gemma3n. Defaults need to be tuned for specific model sizes."""

  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM, epsilon=1e-6, zero_centered=False, with_scale=True
  )

  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH), # Gemma3n uses GeLU
      intermediate_size=hidden_dim,
      pre_ff_norm_config=norm_config,
      post_ff_norm_config=norm_config,
  )

  altup_config = cfg.AltUpConfig(
      num_inputs=altup_num_inputs,
      active_idx=0,
      correct_scale=True, # Often true
  )

  laurel_config = cfg.LaurelConfig(rank=128) # Default rank?

  def get_block_config(idx: int) -> cfg.TransformerBlockConfig:
    attn_config = cfg.AttentionConfig(
        num_heads=num_heads,
        head_dim=head_dim,
        num_query_groups=num_heads // 2, # Just an example, check actual config
        rotary_base=10000,
        rotary_percentage=1.0,
        qkv_transpose_before_split=True,
        qkv_fused_interleaved=False,
        logit_softcap=50.0,
        sliding_window_size=4096,
        attn_type=(
            cfg.AttentionType.GLOBAL
            if (idx + 1) % 2 == 0
            else cfg.AttentionType.LOCAL_SLIDING
        ),
        # Norms for QKV
        query_norm_config=norm_config,
        key_norm_config=norm_config,
        value_norm_config=cfg.NormalizationConfig(
            type=cfg.NormalizationType.RMS_NORM, epsilon=1e-6, zero_centered=False, with_scale=False
        ),
    )
    return cfg.TransformerBlockConfig(
        attn_config=attn_config,
        ff_config=ff_config,
        pre_attention_norm_config=norm_config,
        post_attention_norm_config=norm_config,
        altup_config=altup_config,
        laurel_config=laurel_config,
        hidden_size_per_layer_input=1024, # Need actual default
    )

  config = cfg.ModelConfig(
      vocab_size=vocab_size,
      num_layers=num_layers,
      max_seq_len=8192,
      embedding_dim=embedding_dim,
      embedding_scale=embedding_dim**0.5,
      block_configs=[get_block_config(i) for i in range(num_layers)],
      final_norm_config=norm_config,
      lm_head_use_bias=False,
      final_logit_softcap=30.0,
      vocab_size_per_layer_input=vocab_size_per_layer_input,
  )
  return config

def build_model_e2b(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:
    # This function needs to load the actual config from the checkpoint or a config file
    # For now, we use a default config for demonstration
    config = get_model_config()

    # Use model_builder if possible, but we need to map tensors carefully.
    # Since Gemma3n structure is complex, we might rely on the standard loader
    # but we need to ensure TENSOR_NAMES_GEMMA3N is exhaustive.
    # It likely isn't because of AltUp/Laurel parameters.
    #
    # If standard model_builder cannot be used due to missing tensor mappings in TensorNames tuple,
    # we might need to manually load state dict.

    model = Gemma3n(config, mask_cache_size)

    # Logic to load weights would go here.
    # Since we don't have the checkpoint, we return the initialized model.
    # In a real scenario, we would use loading_utils.load_model(...)

    return model
