# Copyright 2025 The AI Edge Torch Authors.
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

"""Gemma3n model implementation.

Gemma3n is a MatFormer (Matryoshka Transformer) architecture with several
unique components:
- Per-Layer Embeddings (PLE): Token-specific embeddings injected at each layer
- LAuReL (Learned Augmented Residual Layer): Low-rank residual connections
- AltUp (Alternating Updates): Sparse update mechanism for efficiency
- Hybrid attention: Mix of sliding window (local) and full (global) attention
- KV cache sharing: Later layers share KV cache with earlier layers

References:
- MatFormer: https://arxiv.org/pdf/2310.07707
- LAuReL: https://arxiv.org/pdf/2411.07501
- AltUp: https://arxiv.org/pdf/2301.13310
"""

import math
from typing import Callable, Dict, List, Optional, Tuple, Union

from ai_edge_torch.generative.layers import attention
from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.layers.rotary_position_embedding as rotary_pos_emb
from ai_edge_torch.generative.utilities import export_config as export_cfg
from ai_edge_torch.generative.utilities import model_builder
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn
import torch.nn.functional as F


# Tensor name mappings for loading checkpoints
TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="model.layers.{}.mlp.up_proj",
    ff_down_proj="model.layers.{}.mlp.down_proj",
    ff_gate_proj="model.layers.{}.mlp.gate_proj",
    attn_query_proj="model.layers.{}.self_attn.q_proj",
    attn_key_proj="model.layers.{}.self_attn.k_proj",
    attn_value_proj="model.layers.{}.self_attn.v_proj",
    attn_output_proj="model.layers.{}.self_attn.o_proj",
    attn_query_norm="model.layers.{}.self_attn.q_norm",
    attn_key_norm="model.layers.{}.self_attn.k_norm",
    pre_attn_norm="model.layers.{}.input_layernorm",
    post_attn_norm="model.layers.{}.post_attention_layernorm",
    pre_ff_norm="model.layers.{}.pre_feedforward_layernorm",
    post_ff_norm="model.layers.{}.post_feedforward_layernorm",
    embedding="model.embed_tokens",
    final_norm="model.norm",
    lm_head=None,
)


def load_gemma3n_weights(model: nn.Module, checkpoint_path: str):
  """Loads Gemma3n weights from checkpoint.

  This function handles the complete weight loading for Gemma3n, including
  standard transformer weights and custom Gemma3n-specific weights (AltUp,
  Laurel, PerLayer embeddings).
  """
  # Load raw state dict
  state_dict = loading_utils.load_safetensors(checkpoint_path)
  if "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]

  converted_state = {}

  # === Global embeddings ===
  converted_state["tok_embedding.weight"] = state_dict["model.embed_tokens.weight"]
  converted_state["final_norm.weight"] = state_dict["model.norm.weight"]

  # Per-layer embeddings
  converted_state["per_layer_embedding.embedding.weight"] = state_dict["model.embed_tokens_per_layer.weight"]
  converted_state["per_layer_model_projection.weight"] = state_dict["model.per_layer_model_projection.weight"]
  converted_state["per_layer_projection_norm.weight"] = state_dict["model.per_layer_projection_norm.weight"]

  # Global AltUp projections
  for i in range(len(model.altup_projections)):
    converted_state[f"altup_projections.{i}.weight"] = state_dict[f"model.altup_projections.{i}.weight"]
  for i in range(len(model.altup_unembed_projections)):
    converted_state[f"altup_unembed_projections.{i}.weight"] = state_dict[f"model.altup_unembed_projections.{i}.weight"]

  # === Per-layer weights ===
  for i in range(model.config.num_layers):
    src = f"model.layers.{i}"
    dst = f"transformer_blocks.{i}"

    # Norms (sibling to atten_func and ff)
    converted_state[f"{dst}.pre_atten_norm.weight"] = state_dict[f"{src}.input_layernorm.weight"]
    converted_state[f"{dst}.post_atten_norm.weight"] = state_dict[f"{src}.post_attention_layernorm.weight"]
    converted_state[f"{dst}.pre_ff_norm.weight"] = state_dict[f"{src}.pre_feedforward_layernorm.weight"]
    converted_state[f"{dst}.post_ff_norm.weight"] = state_dict[f"{src}.post_feedforward_layernorm.weight"]

    # Attention
    attn_config = model.config.block_config(i).attn_config
    q = state_dict[f"{src}.self_attn.q_proj.weight"]
    k = state_dict[f"{src}.self_attn.k_proj.weight"]
    v = state_dict[f"{src}.self_attn.v_proj.weight"]
    # Fuse QKV (non-interleaved)
    converted_state[f"{dst}.atten_func.qkv_projection.weight"] = torch.cat([q, k, v], dim=0)
    converted_state[f"{dst}.atten_func.output_projection.weight"] = state_dict[f"{src}.self_attn.o_proj.weight"]

    # Attention norms (Q/K norms)
    if f"{src}.self_attn.q_norm.weight" in state_dict:
      converted_state[f"{dst}.atten_func.query_norm.weight"] = state_dict[f"{src}.self_attn.q_norm.weight"]
    if f"{src}.self_attn.k_norm.weight" in state_dict:
      converted_state[f"{dst}.atten_func.key_norm.weight"] = state_dict[f"{src}.self_attn.k_norm.weight"]

    # Feedforward (gated: w1=gate, w2=down, w3=up)
    converted_state[f"{dst}.ff.w1.weight"] = state_dict[f"{src}.mlp.gate_proj.weight"]
    converted_state[f"{dst}.ff.w2.weight"] = state_dict[f"{src}.mlp.down_proj.weight"]
    converted_state[f"{dst}.ff.w3.weight"] = state_dict[f"{src}.mlp.up_proj.weight"]

    # Laurel
    converted_state[f"{dst}.laurel.linear_left.weight"] = state_dict[f"{src}.laurel.linear_left.weight"]
    converted_state[f"{dst}.laurel.linear_right.weight"] = state_dict[f"{src}.laurel.linear_right.weight"]
    converted_state[f"{dst}.laurel.post_laurel_norm.weight"] = state_dict[f"{src}.laurel.post_laurel_norm.weight"]

    # AltUp (per-layer)
    converted_state[f"{dst}.altup.correct_output_scale"] = state_dict[f"{src}.altup.correct_output_scale"]
    converted_state[f"{dst}.altup.correction_coefs.weight"] = state_dict[f"{src}.altup.correction_coefs.weight"]
    converted_state[f"{dst}.altup.prediction_coefs.weight"] = state_dict[f"{src}.altup.prediction_coefs.weight"]
    converted_state[f"{dst}.altup.modality_router.weight"] = state_dict[f"{src}.altup.modality_router.weight"]
    converted_state[f"{dst}.altup.router_norm.weight"] = state_dict[f"{src}.altup.router_norm.weight"]

    # Per-layer input integration
    converted_state[f"{dst}.per_layer_input_gate.weight"] = state_dict[f"{src}.per_layer_input_gate.weight"]
    converted_state[f"{dst}.per_layer_projection.weight"] = state_dict[f"{src}.per_layer_projection.weight"]
    converted_state[f"{dst}.post_per_layer_norm.weight"] = state_dict[f"{src}.post_per_layer_input_norm.weight"]

  model.load_state_dict(converted_state, strict=False)



class RMSNorm(nn.Module):
  """RMS Normalization layer."""

  def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
    super().__init__()
    self.eps = eps
    self.with_scale = with_scale
    if self.with_scale:
      self.weight = nn.Parameter(torch.ones(dim))
    else:
      self.register_buffer("weight", torch.tensor(1.0), persistent=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    output = x.float() / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
    return (output * self.weight.float()).type_as(x)


class LaurelBlock(nn.Module):
  """Learned Augmented Residual Layer (LAuReL).

  A low-rank residual connection that provides more expressivity than
  simple additive residuals while remaining computationally efficient.

  Reference: https://arxiv.org/pdf/2411.07501
  """

  def __init__(self, hidden_size: int, laurel_rank: int, eps: float = 1e-6):
    super().__init__()
    self.linear_left = nn.Linear(hidden_size, laurel_rank, bias=False)
    self.linear_right = nn.Linear(laurel_rank, hidden_size, bias=False)
    self.post_laurel_norm = RMSNorm(hidden_size, eps=eps)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    laurel_out = self.linear_left(hidden_states)
    laurel_out = self.linear_right(laurel_out)
    laurel_out = self.post_laurel_norm(laurel_out)
    return hidden_states + laurel_out


class AltUp(nn.Module):
  """Alternating Updates (AltUp) module.

  AltUp wraps transformer layers to enable sparse updates across multiple
  prediction streams. The 'predict' step modifies the input, and the 'correct'
  step propagates the output to sparsely updated dimensions.

  Reference: https://arxiv.org/pdf/2301.13310
  """

  def __init__(
      self,
      hidden_size: int,
      num_inputs: int = 4,
      active_idx: int = 0,
      coef_clip: float = 120.0,
      eps: float = 1e-6,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_inputs = num_inputs
    self.active_idx = active_idx
    self.coef_clip = coef_clip

    self.correct_output_scale = nn.Parameter(torch.zeros(hidden_size))
    self.correction_coefs = nn.Linear(num_inputs, num_inputs, bias=False)
    self.prediction_coefs = nn.Linear(num_inputs, num_inputs**2, bias=False)
    self.modality_router = nn.Linear(hidden_size, num_inputs, bias=False)
    self.router_norm = RMSNorm(hidden_size, eps=eps)
    self.register_buffer(
        "router_input_scale", torch.tensor(hidden_size**-1.0), persistent=False
    )

  def compute_router_modalities(self, x: torch.Tensor) -> torch.Tensor:
    router_inputs = self.router_norm(x) * self.router_input_scale
    routed = self.modality_router(router_inputs)
    return torch.tanh(routed.float()).type_as(x)

  def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Predicts outputs using a trainable map.

    Args:
      hidden_states: [num_inputs, batch, seq_len, hidden_size]

    Returns:
      Predictions of same shape.
    """
    modalities = self.compute_router_modalities(hidden_states[self.active_idx])

    all_coefs = (
        self.prediction_coefs(modalities)
        .reshape(*modalities.shape[:-1], self.num_inputs, self.num_inputs)
        .permute(0, 1, 3, 2)
    )

    predictions = torch.matmul(hidden_states.permute(1, 2, 3, 0), all_coefs)
    predictions = predictions.permute(3, 0, 1, 2)
    predictions = predictions + hidden_states
    return predictions.contiguous().type_as(hidden_states)

  def correct(
      self, predictions: torch.Tensor, activated: torch.Tensor
  ) -> torch.Tensor:
    """Corrects predictions relative to the activated output.

    Args:
      predictions: [num_inputs, batch, seq_len, hidden_size]
      activated: [batch, seq_len, hidden_size]

    Returns:
      Corrected predictions of same shape as predictions.
    """
    modalities = self.compute_router_modalities(activated)
    innovation = activated - predictions[self.active_idx]
    innovation = innovation.unsqueeze(0).expand(self.num_inputs, -1, -1, -1)

    all_coefs = self.correction_coefs(modalities) + 1.0
    all_coefs = all_coefs.permute(2, 0, 1).unsqueeze(-1)

    corrected = torch.mul(innovation, all_coefs)
    corrected = corrected + predictions
    return corrected.contiguous().type_as(activated)

  def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor:
    """Scales the corrected output."""
    return (
        corrected.type_as(self.correct_output_scale) * self.correct_output_scale
    ).type_as(corrected)


class PerLayerEmbedding(nn.Module):
  """Per-Layer Embedding module.

  Provides token-specific embeddings that are injected at each transformer
  layer. This allows the model to retrieve layer-specific information on-demand
  rather than encoding everything in the initial embedding.
  """

  def __init__(
      self,
      vocab_size: int,
      num_layers: int,
      per_layer_dim: int,
      padding_idx: int = 0,
  ):
    super().__init__()
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.per_layer_dim = per_layer_dim

    # Embedding table: vocab_size -> num_layers * per_layer_dim
    self.embedding = nn.Embedding(
        vocab_size, num_layers * per_layer_dim, padding_idx=padding_idx
    )
    self.embed_scale = per_layer_dim**0.5

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    """Get per-layer embeddings for input tokens.

    Args:
      input_ids: [batch, seq_len]

    Returns:
      Per-layer embeddings: [batch, seq_len, num_layers, per_layer_dim]
    """
    # Clamp to valid vocab range
    clamped_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
    embeddings = self.embedding(clamped_ids) * self.embed_scale
    return embeddings.reshape(
        *input_ids.shape, self.num_layers, self.per_layer_dim
    )


class Gemma3nDecoderBlock(nn.Module):
  """Gemma3n decoder block with LAuReL, AltUp, and per-layer embeddings."""

  def __init__(
      self,
      config: cfg.TransformerBlockConfig,
      model_config: cfg.ModelConfig,
      layer_idx: int,
      hidden_size: int,
      per_layer_dim: int,
      laurel_rank: int,
      altup_num_inputs: int,
      eps: float = 1e-6,
  ):
    super().__init__()
    self.layer_idx = layer_idx
    self.hidden_size = hidden_size
    self.per_layer_dim = per_layer_dim
    self.altup_active_idx = 0
    self.altup_num_inputs = altup_num_inputs

    # Standard transformer components
    self.pre_atten_norm = builder.build_norm(
        model_config.embedding_dim, config.pre_attention_norm_config
    )
    self.atten_func = attention.CausalSelfAttention(
        model_config.embedding_dim,
        config.attn_config,
        model_config.enable_hlfb,
    )
    self.post_atten_norm = builder.build_norm(
        model_config.embedding_dim, config.post_attention_norm_config
    )

    # Pre/post feedforward norms (applied separately from ff layer)
    self.pre_ff_norm = builder.build_norm(
        model_config.embedding_dim, config.ff_config.pre_ff_norm_config
    )
    self.post_ff_norm = builder.build_norm(
        model_config.embedding_dim, config.ff_config.post_ff_norm_config
    )

    # Build feedforward WITHOUT internal norms (we apply them separately)
    ff_config_no_norms = cfg.FeedForwardConfig(
        type=config.ff_config.type,
        activation=config.ff_config.activation,
        intermediate_size=config.ff_config.intermediate_size,
        use_bias=config.ff_config.use_bias,
        use_separate_gating=True,
        pre_ff_norm_config=None,  # No internal norms
        post_ff_norm_config=None,
    )
    self.ff = builder.build_ff(model_config.embedding_dim, ff_config_no_norms)

    # Gemma3n-specific components
    self.laurel = LaurelBlock(hidden_size, laurel_rank, eps=eps)
    self.altup = AltUp(
        hidden_size, num_inputs=altup_num_inputs, active_idx=0, eps=eps
    )

    # Per-layer embedding integration
    self.per_layer_input_gate = nn.Linear(hidden_size, per_layer_dim, bias=False)
    self.per_layer_projection = nn.Linear(per_layer_dim, hidden_size, bias=False)
    self.post_per_layer_norm = RMSNorm(hidden_size, eps=eps)

    self.config = config
    self.act_fn = lambda x: F.gelu(x, approximate="tanh")

  def forward(
      self,
      hidden_states: torch.Tensor,
      rope: Tuple[torch.Tensor, torch.Tensor],
      mask: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCacheEntry,
      per_layer_input: torch.Tensor,
  ) -> Tuple[torch.Tensor, kv_utils.KVCacheEntry]:
    """Forward pass of Gemma3n decoder block.

    Args:
      hidden_states: [num_altup_inputs, batch, seq_len, hidden_size]
      rope: Rotary position embeddings (cos, sin)
      mask: Attention mask
      input_pos: Input positions for KV cache
      kv_cache: KV cache entry
      per_layer_input: Per-layer embeddings [batch, seq_len, per_layer_dim]

    Returns:
      Updated hidden states and KV cache entry.
    """
    # AltUp predict step
    predictions = self.altup.predict(hidden_states)
    active_prediction = predictions[self.altup_active_idx]

    # Pre-attention normalization and LAuReL
    active_normed = self.pre_atten_norm(active_prediction)
    laurel_output = self.laurel(active_normed)

    # Self-attention
    attn_out, kv = self.atten_func(
        active_normed, rope, mask, input_pos, kv_cache
    )
    attn_out = self.post_atten_norm(attn_out)

    # Combine attention with LAuReL
    attn_gated = active_prediction + attn_out
    attn_laurel = (attn_gated + laurel_output) / math.sqrt(2)

    # Feed-forward with explicit pre/post norms
    ff_input = self.pre_ff_norm(attn_laurel)
    ff_out = self.ff(ff_input)
    ff_out = self.post_ff_norm(ff_out)
    attn_ff = attn_laurel + ff_out

    # AltUp correct step
    corrected = self.altup.correct(predictions, attn_ff)

    # Apply per-layer embeddings to non-active predictions
    first_prediction = corrected[self.altup_active_idx]
    scaled_prediction = self.altup.scale_corrected_output(first_prediction)

    gate = self.act_fn(self.per_layer_input_gate(scaled_prediction))
    per_layer_out = gate * per_layer_input
    per_layer_out = self.per_layer_projection(per_layer_out)
    per_layer_out = self.post_per_layer_norm(per_layer_out)

    # Create new tensor with updated values (avoid in-place modification)
    # corrected[1:] += per_layer_out becomes:
    result_parts = [corrected[0:1]]  # Keep first unchanged
    for i in range(1, self.altup_num_inputs):
      result_parts.append(corrected[i:i+1] + per_layer_out.unsqueeze(0))
    result = torch.cat(result_parts, dim=0)

    return result, kv


class Decoder(nn.Module):
  """Gemma3n decoder model."""

  def __init__(self, config: cfg.ModelConfig, gemma3n_config: dict):
    super().__init__()
    self.config = config
    self.gemma3n_config = gemma3n_config

    hidden_size = config.embedding_dim
    num_layers = config.num_layers
    per_layer_dim = gemma3n_config.get("hidden_size_per_layer_input", 256)
    laurel_rank = gemma3n_config.get("laurel_rank", 64)
    altup_num_inputs = gemma3n_config.get("altup_num_inputs", 4)
    vocab_size_per_layer = gemma3n_config.get("vocab_size_per_layer_input", 262144)
    eps = config.block_config(0).attn_config.query_norm_config.epsilon

    # Token embedding
    self.tok_embedding = nn.Embedding(
        config.vocab_size, config.embedding_dim, padding_idx=0
    )

    # Per-layer embeddings
    self.per_layer_embedding = PerLayerEmbedding(
        vocab_size=vocab_size_per_layer,
        num_layers=num_layers,
        per_layer_dim=per_layer_dim,
        padding_idx=0,
    )

    # Per-layer projection from main hidden state
    self.per_layer_model_projection = nn.Linear(
        hidden_size, num_layers * per_layer_dim, bias=False
    )
    self.per_layer_projection_norm = RMSNorm(per_layer_dim, eps=eps)
    self.register_buffer(
        "per_layer_projection_scale",
        torch.tensor(hidden_size**-0.5),
        persistent=False,
    )
    self.register_buffer(
        "per_layer_input_scale",
        torch.rsqrt(torch.tensor(2.0)),
        persistent=False,
    )

    # AltUp projections for expanding/contracting hidden states
    self.altup_projections = nn.ModuleList([
        nn.Linear(hidden_size, hidden_size, bias=False)
        for _ in range(1, altup_num_inputs)
    ])
    self.altup_unembed_projections = nn.ModuleList([
        nn.Linear(hidden_size, hidden_size, bias=False)
        for _ in range(1, altup_num_inputs)
    ])

    # Transformer blocks
    self.transformer_blocks = nn.ModuleList([
        Gemma3nDecoderBlock(
            config.block_config(idx),
            config,
            layer_idx=idx,
            hidden_size=hidden_size,
            per_layer_dim=per_layer_dim,
            laurel_rank=laurel_rank,
            altup_num_inputs=altup_num_inputs,
            eps=eps,
        )
        for idx in range(num_layers)
    ])

    # Final layers
    self.final_norm = builder.build_norm(
        config.embedding_dim, config.final_norm_config
    )
    self.lm_head = nn.Linear(
        config.embedding_dim, config.vocab_size, bias=config.lm_head_use_bias
    )
    # Tied weights
    self.lm_head.weight.data = self.tok_embedding.weight.data

    self.altup_num_inputs = altup_num_inputs
    self.altup_active_idx = 0
    self.mask_cache = None

  def get_per_layer_inputs(
      self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor
  ) -> torch.Tensor:
    """Compute per-layer inputs from token IDs and embeddings."""
    # Get per-layer embeddings from vocabulary
    per_layer_embeds = self.per_layer_embedding(input_ids)

    # Project main embeddings to per-layer dimension
    per_layer_proj = self.per_layer_model_projection(inputs_embeds)
    per_layer_proj = per_layer_proj * self.per_layer_projection_scale
    per_layer_proj = per_layer_proj.reshape(
        *inputs_embeds.shape[:-1],
        self.config.num_layers,
        self.gemma3n_config.get("hidden_size_per_layer_input", 256),
    )
    per_layer_proj = self.per_layer_projection_norm(per_layer_proj)

    # Combine embeddings and projections
    combined = (per_layer_proj + per_layer_embeds) * self.per_layer_input_scale
    return combined

  def expand_hidden_states(
      self, hidden_states: torch.Tensor
  ) -> torch.Tensor:
    """Expand hidden states for AltUp processing.

    Args:
      hidden_states: [batch, seq_len, hidden_size]

    Returns:
      Expanded states: [num_altup_inputs, batch, seq_len, hidden_size]
    """
    target_magnitude = torch.mean(hidden_states**2, dim=-1, keepdim=True) ** 0.5
    epsilon = torch.tensor(1e-5, device=hidden_states.device, dtype=hidden_states.dtype)

    expanded = [hidden_states]
    for proj in self.altup_projections:
      projected = proj(hidden_states)
      new_magnitude = torch.mean(projected**2, dim=-1, keepdim=True)
      new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon))
      normalized = projected * target_magnitude / new_magnitude
      expanded.append(normalized)

    return torch.stack(expanded, dim=0)

  def collapse_hidden_states(
      self, hidden_states: torch.Tensor
  ) -> torch.Tensor:
    """Collapse AltUp hidden states back to single stream.

    Args:
      hidden_states: [num_altup_inputs, batch, seq_len, hidden_size]

    Returns:
      Collapsed states: [batch, seq_len, hidden_size]
    """
    target_magnitude = (
        torch.mean(hidden_states[0] ** 2, dim=-1, keepdim=True) ** 0.5
    )
    epsilon = torch.tensor(1e-5, device=hidden_states.device, dtype=hidden_states.dtype)

    collapsed = [hidden_states[0]]
    for i, proj in enumerate(self.altup_unembed_projections):
      projected = proj(hidden_states[i + 1])
      new_magnitude = torch.mean(projected**2, dim=-1, keepdim=True)
      new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon))
      normalized = projected * target_magnitude / new_magnitude
      collapsed.append(normalized)

    stacked = torch.stack(collapsed, dim=0)
    return torch.mean(stacked, dim=0)

  def get_attention_mask(
      self,
      mask: torch.Tensor,
      attn_config: cfg.AttentionConfig,
      input_pos: torch.Tensor,
      kv_cache_max_len: int,
  ) -> torch.Tensor:
    """Get attention mask with optional sliding window."""
    if attn_config.attn_type == cfg.AttentionType.LOCAL_SLIDING:
      sliding_window = attn_config.sliding_window_size
      if sliding_window is not None:
        sliding_mask = self._create_sliding_mask(
            input_pos, kv_cache_max_len, sliding_window
        )
        # Combine masks using minimum (preserves -inf)
        return torch.minimum(mask, sliding_mask)
    return mask

  def _create_sliding_mask(
      self,
      input_pos: torch.Tensor,
      cache_len: int,
      sliding_window_size: int,
  ) -> torch.Tensor:
    """Creates mask for sliding window attention."""
    cache_positions = torch.arange(cache_len, dtype=torch.int32, device=input_pos.device)
    cache_positions = cache_positions.view(1, 1, 1, -1)
    segment_pos = input_pos.view(1, 1, -1, 1)

    left_boundary = cache_positions > segment_pos - sliding_window_size
    right_boundary = cache_positions < segment_pos + sliding_window_size

    sliding_mask_bool = left_boundary & right_boundary

    sliding_mask = torch.where(
        sliding_mask_bool,
        torch.zeros_like(sliding_mask_bool, dtype=torch.float32),
        torch.full_like(sliding_mask_bool, float("-inf"), dtype=torch.float32),
    )
    return sliding_mask

  @torch.inference_mode
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      mask: Optional[torch.Tensor] = None,
      export_config: Optional[export_cfg.ExportConfig] = None,
  ) -> Dict[str, Union[torch.Tensor, kv_utils.KVCache]]:
    """Forward pass of the decoder.

    Args:
      tokens: Input token IDs [batch, seq_len]
      input_pos: Position indices for the input tokens
      kv_cache: KV cache for attention
      mask: Optional attention mask
      export_config: Export configuration

    Returns:
      Dictionary with 'logits' and 'kv_cache'.
    """
    _, seq_len = tokens.size()
    assert self.config.max_seq_len >= seq_len

    # Token embeddings with scaling
    input_embeds = self.tok_embedding(tokens)
    if self.config.embedding_scale is not None:
      input_embeds = input_embeds * self.config.embedding_scale


    # Get per-layer inputs
    per_layer_inputs = self.get_per_layer_inputs(tokens, input_embeds)

    # Expand for AltUp
    hidden_states = self.expand_hidden_states(input_embeds)

    # Build RoPE for each layer (different rotary base for local vs global)
    attn_config = self.config.block_config(0).attn_config
    rope_list = [
        rotary_pos_emb.build_rope(
            input_pos,
            attn_config.head_dim,
            self.config.block_config(i).attn_config.rotary_base,
        )
        for i in range(self.config.num_layers)
    ]

    # Handle mask
    if mask is None:
      assert self.mask_cache is not None, "Mask cache must be built."
      kv_cache_max_len = kv_cache.get_max_seq_len()
      mask = self.mask_cache.index_select(2, input_pos)
      mask = mask[:, :, :, :kv_cache_max_len]
    else:
      kv_cache_max_len = mask.size(-1)

    # Build per-layer masks with sliding window handling
    mask_list = [
        self.get_attention_mask(
            mask,
            self.config.block_config(i).attn_config,
            input_pos,
            kv_cache_max_len,
        )
        for i in range(self.config.num_layers)
    ]

    # Process through transformer blocks
    updated_kv_entries = []
    for i, block in enumerate(self.transformer_blocks):
      kv_entry = kv_cache.caches[i] if kv_cache else None
      per_layer_input = per_layer_inputs[:, :, i, :]

      hidden_states, kv_entry = block(
          hidden_states,
          rope_list[i],
          mask_list[i],
          input_pos,
          kv_entry,
          per_layer_input,
      )

      if kv_entry:
        updated_kv_entries.append(kv_entry)

    updated_kv_cache = kv_utils.KVCache(tuple(updated_kv_entries))

    # Collapse AltUp hidden states
    hidden_states = self.collapse_hidden_states(hidden_states)

    # Skip logits computation during prefill if configured
    if export_config is not None:
      if (
          torch.numel(input_pos) > 1
          and not export_config.output_logits_on_prefill
      ):
        return {"kv_cache": updated_kv_cache}

    # Final norm and logit computation
    hidden_states = self.final_norm(hidden_states)
    logits = self.lm_head(hidden_states)

    # Apply final logit softcapping
    final_softcap = self.config.final_logit_softcap
    if final_softcap is not None:
      logits = logits / final_softcap
      logits = torch.tanh(logits)
      logits = logits * final_softcap

    return {"logits": logits, "kv_cache": updated_kv_cache}


def get_model_config_e2b() -> Tuple[cfg.ModelConfig, dict]:
  """Returns the model config for Gemma3n E2B model."""
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM,
      epsilon=1e-6,
      zero_centered=False,
  )

  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
      intermediate_size=8192,
      use_separate_gating=True,
      pre_ff_norm_config=norm_config,
      post_ff_norm_config=norm_config,
  )

  # Layer types: every 5th layer is full attention, rest are sliding window
  num_layers = 30

  def get_block_config(idx: int) -> cfg.TransformerBlockConfig:
    is_global = (idx + 1) % 5 == 0
    attn_config = cfg.AttentionConfig(
        num_heads=8,
        head_dim=256,
        num_query_groups=2,
        rotary_base=1_000_000 if is_global else 10_000,
        rotary_percentage=1.0,
        qkv_transpose_before_split=True,
        query_norm_config=norm_config,
        key_norm_config=norm_config,
        logit_softcap=None,
        sliding_window_size=512,
        attn_type=(
            cfg.AttentionType.GLOBAL
            if is_global
            else cfg.AttentionType.LOCAL_SLIDING
        ),
    )
    return cfg.TransformerBlockConfig(
        attn_config=attn_config,
        ff_config=ff_config,
        pre_attention_norm_config=norm_config,
        post_attention_norm_config=norm_config,
    )

  embedding_dim = 2048
  model_config = cfg.ModelConfig(
      vocab_size=262400,
      num_layers=num_layers,
      max_seq_len=32768,
      embedding_dim=embedding_dim,
      embedding_scale=embedding_dim**0.5,
      block_configs=[get_block_config(i) for i in range(num_layers)],
      final_norm_config=norm_config,
      lm_head_use_bias=False,
      final_logit_softcap=30.0,
  )

  # Gemma3n-specific config
  gemma3n_config = {
      "hidden_size_per_layer_input": 256,
      "vocab_size_per_layer_input": 262144,
      "laurel_rank": 64,
      "altup_num_inputs": 4,
      "altup_active_idx": 0,
      "altup_coef_clip": 120.0,
      "altup_correct_scale": True,
      "num_kv_shared_layers": 10,
      "activation_sparsity_pattern": [0.95] * 10 + [0.0] * 20,
  }

  return model_config, gemma3n_config


def get_model_config_e4b() -> Tuple[cfg.ModelConfig, dict]:
  """Returns the model config for Gemma3n E4B model."""
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM,
      epsilon=1e-6,
      zero_centered=False,
  )

  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
      intermediate_size=16384,
      use_separate_gating=True,
      pre_ff_norm_config=norm_config,
      post_ff_norm_config=norm_config,
  )

  num_layers = 35

  def get_block_config(idx: int) -> cfg.TransformerBlockConfig:
    is_global = (idx + 1) % 5 == 0
    attn_config = cfg.AttentionConfig(
        num_heads=8,
        head_dim=256,
        num_query_groups=2,
        rotary_base=1_000_000 if is_global else 10_000,
        rotary_percentage=1.0,
        qkv_transpose_before_split=True,
        query_norm_config=norm_config,
        key_norm_config=norm_config,
        logit_softcap=None,
        sliding_window_size=512,
        attn_type=(
            cfg.AttentionType.GLOBAL
            if is_global
            else cfg.AttentionType.LOCAL_SLIDING
        ),
    )
    return cfg.TransformerBlockConfig(
        attn_config=attn_config,
        ff_config=ff_config,
        pre_attention_norm_config=norm_config,
        post_attention_norm_config=norm_config,
    )

  embedding_dim = 2048
  model_config = cfg.ModelConfig(
      vocab_size=262400,
      num_layers=num_layers,
      max_seq_len=32768,
      embedding_dim=embedding_dim,
      embedding_scale=embedding_dim**0.5,
      block_configs=[get_block_config(i) for i in range(num_layers)],
      final_norm_config=norm_config,
      lm_head_use_bias=False,
      final_logit_softcap=30.0,
  )

  gemma3n_config = {
      "hidden_size_per_layer_input": 256,
      "vocab_size_per_layer_input": 262144,
      "laurel_rank": 64,
      "altup_num_inputs": 4,
      "altup_active_idx": 0,
      "altup_coef_clip": 120.0,
      "altup_correct_scale": True,
      "num_kv_shared_layers": 15,
      "activation_sparsity_pattern": [0.95] * 10 + [0.0] * 25,
  }

  return model_config, gemma3n_config


def _build_mask_cache(max_seq_len: int) -> torch.Tensor:
  """Build causal mask cache."""
  return attn_utils.build_causal_mask_cache(max_seq_len)


def build_model_e2b(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> Decoder:
  """Builds a Gemma3n E2B model.

  Args:
    checkpoint_path: Path to the model checkpoint.
    custom_loader: Optional custom checkpoint loader function.
    mask_cache_size: Size of the attention mask cache.

  Returns:
    The constructed Decoder model.
  """
  model_config, gemma3n_config = get_model_config_e2b()
  model = Decoder(model_config, gemma3n_config)

  if mask_cache_size > 0:
    model.mask_cache = _build_mask_cache(mask_cache_size)

  load_gemma3n_weights(model, checkpoint_path)

  model.eval()
  return model


def build_model_e4b(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> Decoder:
  """Builds a Gemma3n E4B model.

  Args:
    checkpoint_path: Path to the model checkpoint.
    custom_loader: Optional custom checkpoint loader function.
    mask_cache_size: Size of the attention mask cache.

  Returns:
    The constructed Decoder model.
  """
  model_config, gemma3n_config = get_model_config_e4b()
  model = Decoder(model_config, gemma3n_config)

  if mask_cache_size > 0:
    model.mask_cache = _build_mask_cache(mask_cache_size)

  load_gemma3n_weights(model, checkpoint_path)

  model.eval()
  return model
