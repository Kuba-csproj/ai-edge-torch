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
from typing import Callable, Dict, List, Optional, Tuple

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
    output = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    return output * self.weight


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
    innovation = innovation.repeat(self.num_inputs, 1, 1, 1)

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
    self.ff = builder.build_ff(model_config.embedding_dim, config.ff_config)

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
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      mask: Optional[torch.Tensor] = None,
      input_pos: Optional[torch.Tensor] = None,
      kv_cache: kv_utils.KVCacheEntry = None,
      per_layer_input: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, Optional[kv_utils.KVCacheEntry]]:
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

    # Feed-forward
    ff_out = self.ff(attn_laurel)
    attn_ff = attn_laurel + ff_out

    # AltUp correct step
    corrected = self.altup.correct(predictions, attn_ff)

    # Apply per-layer embeddings
    first_prediction = corrected[self.altup_active_idx].clone()
    first_prediction = self.altup.scale_corrected_output(first_prediction)

    if per_layer_input is not None:
      gate = self.act_fn(self.per_layer_input_gate(first_prediction))
      per_layer_out = gate * per_layer_input
      per_layer_out = self.per_layer_projection(per_layer_out)
      per_layer_out = self.post_per_layer_norm(per_layer_out)
      corrected[1:] = corrected[1:] + per_layer_out

    return corrected, kv


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
    self.build_mask_cache(0)

  def build_mask_cache(self, mask_cache_size: int):
    if mask_cache_size <= 0:
      self.mask_cache = None
    else:
      self.mask_cache = attn_utils.build_causal_mask_cache(mask_cache_size)

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
    epsilon = torch.tensor(1e-5, device=hidden_states.device)

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
    epsilon = torch.tensor(1e-5, device=hidden_states.device)

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
      mask: Optional[torch.Tensor],
      input_pos: torch.Tensor,
      kv_cache: Optional[kv_utils.KVCache],
      attn_config: cfg.AttentionConfig,
  ) -> torch.Tensor:
    """Get attention mask with optional sliding window."""
    if mask is not None:
      return mask

    assert self.mask_cache is not None, "Mask cache must be built."
    assert kv_cache is not None, "KV cache must be provided."

    base_mask = self.mask_cache.index_select(2, input_pos)
    kv_cache_max_len = kv_cache.get_max_seq_len()
    base_mask = base_mask[:, :, :, :kv_cache_max_len]

    # Apply sliding window if configured
    if attn_config.attn_type == cfg.AttentionType.LOCAL_SLIDING:
      sliding_window = attn_config.sliding_window_size
      if sliding_window is not None:
        seq_len = input_pos.shape[0]
        cache_positions = torch.arange(kv_cache_max_len, device=input_pos.device)
        cache_positions = cache_positions.view(1, 1, 1, -1)
        input_pos_expanded = input_pos.view(1, 1, -1, 1)

        left_bound = cache_positions > input_pos_expanded - sliding_window
        right_bound = cache_positions <= input_pos_expanded

        sliding_mask = torch.where(
            left_bound & right_bound,
            torch.zeros_like(base_mask),
            torch.full_like(base_mask, float("-inf")),
        )
        base_mask = torch.minimum(base_mask, sliding_mask)

    return base_mask

  @torch.inference_mode
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      mask: Optional[torch.Tensor] = None,
      export_config: Optional[export_cfg.ExportConfig] = None,
  ) -> dict:
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

    # Build RoPE embeddings (use first layer's config for parameters)
    attn_config = self.config.block_config(0).attn_config
    n_elem = int(attn_config.rotary_percentage * attn_config.head_dim)

    # Process through transformer blocks
    updated_kv_entries = []
    for i, block in enumerate(self.transformer_blocks):
      block_attn_config = self.config.block_config(i).attn_config

      # Get appropriate RoPE for this layer's attention type
      rope = self.config.build_rope(
          input_pos, n_elem, block_attn_config.rotary_base
      )

      # Get mask for this layer
      layer_mask = self.get_attention_mask(
          mask, input_pos, kv_cache, block_attn_config
      )

      kv_entry = kv_cache.caches[i] if kv_cache else None
      per_layer_input = per_layer_inputs[:, :, i, :]

      hidden_states, kv_entry = block(
          hidden_states,
          rope,
          layer_mask,
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


def build_model_e2b(
    checkpoint_path: str,
    mask_cache_size: int = 0,
) -> Decoder:
  """Builds a Gemma3n E2B model.

  Args:
    checkpoint_path: Path to the model checkpoint.
    mask_cache_size: Size of the attention mask cache.

  Returns:
    The constructed Decoder model.
  """
  model_config, gemma3n_config = get_model_config_e2b()
  model = Decoder(model_config, gemma3n_config)

  if checkpoint_path:
    model.build_mask_cache(mask_cache_size)
    # Note: Custom loader would be needed for Gemma3n's unique weight structure
    # For now, this creates an uninitialized model that can be loaded separately

  model.eval()
  return model


def build_model_e4b(
    checkpoint_path: str,
    mask_cache_size: int = 0,
) -> Decoder:
  """Builds a Gemma3n E4B model.

  Args:
    checkpoint_path: Path to the model checkpoint.
    mask_cache_size: Size of the attention mask cache.

  Returns:
    The constructed Decoder model.
  """
  model_config, gemma3n_config = get_model_config_e4b()
  model = Decoder(model_config, gemma3n_config)

  if checkpoint_path:
    model.build_mask_cache(mask_cache_size)

  model.eval()
  return model
