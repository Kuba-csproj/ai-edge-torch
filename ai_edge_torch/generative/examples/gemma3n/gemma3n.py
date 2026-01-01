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
# Gemma3n model implementation.

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Callable

from ai_edge_torch.generative.layers import attention
from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.generative.layers import normalization
from ai_edge_torch.generative.layers import model_config as cfg
from ai_edge_torch.generative.layers import rotary_position_embedding as rotary_pos_emb
from ai_edge_torch.generative.layers import attention_utils as attn_utils
from ai_edge_torch.generative.utilities import export_config as export_cfg
from ai_edge_torch.generative.utilities import model_builder
from ai_edge_torch.generative.utilities import loader as loading_utils

from ai_edge_torch.generative.examples.gemma3n.blocks import LaurelBlock, AltupRouter

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

class Gemma3nBlock(attention.TransformerBlock):
    """Gemma3n transformer block with AltUp and Laurel."""

    def __init__(
        self,
        config: cfg.TransformerBlockConfig,
        model_config: cfg.ModelConfig,
        layer_idx: int,
    ) -> None:
        super().__init__(config, model_config)
        self.layer_idx = layer_idx
        self.dim = model_config.embedding_dim

        # Laurel block
        if config.laurel_rank:
            self.laurel = LaurelBlock(self.dim, config.laurel_rank)
        else:
            self.laurel = None

        self.residual_scale = 0.57735  # sqrt(1/3)

        # AltUp setup
        self.altup_config = config.altup_config
        if self.altup_config:
            self.altup_router_predict = AltupRouter(self.dim, self.altup_config.num_inputs)
            self.altup_router_correct = AltupRouter(self.dim, self.altup_config.num_inputs)

            # These are specific to AltUp. Since they are simple linears, we define them here.
            # Assuming AltUp uses 3 experts as per reference implementation loop range(3)
            self.altup_proj = nn.ModuleList([
                nn.Linear(self.dim, self.dim, bias=False) for _ in range(3)
            ])
            self.altup_unproj = nn.ModuleList([
                nn.Linear(self.dim, self.dim, bias=False) for _ in range(3)
            ])

        # Per-layer embeddings
        if model_config.hidden_size_per_layer_input:
            self.per_layer_gate = nn.Linear(self.dim, model_config.hidden_size_per_layer_input, bias=False)
            self.per_layer_proj = nn.Linear(model_config.hidden_size_per_layer_input, self.dim, bias=False)
            self.per_layer_norm = normalization.RMSNorm(self.dim, eps=1e-6)

    def altup_predict(self, x):
        """ALTUP prediction step"""
        # Get routing weights
        router_weights = self.altup_router_predict(x)
        router_probs = F.softmax(router_weights, dim=-1)

        # Apply ALTUP projections
        experts = []
        for i in range(self.altup_config.num_inputs - 1):
            expert = self.altup_proj[i](x if i == 0 else experts[i-1])
            expert = torch.clamp(expert, -self.altup_config.coef_clip, self.altup_config.coef_clip)
            experts.append(expert)

        # Stack and mix experts. x is the first expert input but not transformed?
        # Re-reading reference: experts_stack = torch.stack([x] + experts, dim=2)
        # So it mixes x and the 3 transformed experts.
        experts_stack = torch.stack([x] + experts, dim=2)  # [B, T, 4, D]
        mixed = torch.matmul(router_probs.unsqueeze(2), experts_stack).squeeze(2)

        return mixed

    def altup_correct(self, predicted, actual):
        """ALTUP correction step"""
        # Compute error
        error = actual - predicted

        # Get routing weights for correction
        router_weights = self.altup_router_correct(actual)
        router_probs = F.softmax(router_weights + 0.5, dim=-1)  # Bias seen in model

        # Apply correction
        correction = router_probs.unsqueeze(-1) * error.unsqueeze(2)
        corrected = predicted.unsqueeze(2) + correction

        # Unproject through ALTUP
        outputs = []
        for i in range(self.altup_config.num_inputs - 1):
            output = self.altup_unproj[i](corrected[:, :, i+1]) # corrected idx 1, 2, 3 correspond to experts 0, 1, 2
            output = torch.clamp(output, -self.altup_config.coef_clip, self.altup_config.coef_clip)
            outputs.append(output)

        # Mix outputs. Reference: final = torch.stack([corrected[:, :, 0]] + outputs, dim=0).mean(dim=0)
        # corrected[:, :, 0] corresponds to x (residual)
        final = torch.stack([corrected[:, :, 0]] + outputs, dim=0).mean(dim=0)

        return final

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: kv_utils.KVCacheEntry = None,
        per_layer_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[kv_utils.KVCacheEntry]]:

        # ALTUP predict
        if self.altup_config:
            predicted = self.altup_predict(x)
        else:
            predicted = x

        # Attention block
        # Gemma3n uses pre-norm for attention input relative to the predicted state
        # In the reference:
        # h = self.pre_attn_norm(predicted)
        # h = self.attention(...)
        # h = self.post_attn_norm(h)
        # h = predicted + h

        x_norm = self.pre_atten_norm(predicted)
        # Standard attention from base class
        # Note: atten_func (CausalSelfAttention) returns (output, kv_cache) if kv_cache is not None,
        # or just output if kv_cache is None in some implementations?
        # Checked attention.py: CausalSelfAttention.forward returns:
        # return y if kv_cache is None else (y, kv_cache)
        atten_func_out = self.atten_func(x_norm, rope, mask, input_pos, kv_cache)

        if kv_cache is None:
            attn_out = atten_func_out
            kv = None
        else:
            attn_out, kv = atten_func_out

        attn_out_norm = self.post_atten_norm(attn_out)
        h = predicted + attn_out_norm

        # Laurel residual
        if self.laurel:
            h = self.laurel(h)

        h = h * self.residual_scale

        # MLP block
        # Reference:
        # h_norm = self.pre_mlp_norm(h)
        # mlp_out = self.mlp(h_norm)
        # mlp_out = self.post_mlp_norm(mlp_out)
        # h = h + mlp_out

        # self.ff wraps GatedFeedForward with pre/post norms configured in build_model_e2b.
        mlp_out = self.ff(h)
        h = h + mlp_out

        # ALTUP correct
        if self.altup_config:
            corrected = self.altup_correct(predicted, h)
        else:
            corrected = h

        # Add per-layer embeddings
        if per_layer_emb is not None and hasattr(self, 'per_layer_gate'):
            gate = F.gelu(self.per_layer_gate(corrected))
            # per_layer_emb: [B, T, NumLayers, D] -> slice for this layer
            # We assume per_layer_emb passed here is for all layers or just this one?
            # If it's [B, T, D] specific to this layer, that's easier.
            # Reference: per_layer_out = self.per_layer_proj(gate * per_layer_emb[:, :, self.layer_idx])
            per_layer_out = self.per_layer_proj(gate * per_layer_emb)
            per_layer_out = self.per_layer_norm(per_layer_out)
            return corrected + per_layer_out, kv

        return corrected, kv

class Gemma3n(nn.Module):
    """Gemma3n model."""

    def __init__(self, config: cfg.ModelConfig, mask_cache_size: int = 0):
        super().__init__()
        self.config = config

        self.tok_embedding = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=0
        )
        self.lm_head = nn.Linear(
            config.embedding_dim, config.vocab_size, bias=config.lm_head_use_bias
        )
        if config.lm_head_share_weight_with_embedding:
            self.lm_head.weight = self.tok_embedding.weight

        # Per-layer embedding components
        if config.hidden_size_per_layer_input:
             # This corresponds to `PerLayerEmbedder` in reference, but integrated.
             # Actually reference has a separate `PerLayerEmbedder` and projection.
             # Reference:
             # self.per_layer_proj = nn.Linear(30 * 256, 30 * 256, bias=False)
             # self.per_layer_norm = LayerNorm(256)

             # Also `PerLayerEmbedder` class produces [B, T, NumLayers, 256]

             self.per_layer_embeddings = nn.ModuleList([
                 nn.Embedding(
                     config.per_layer_input_vocab_size if config.per_layer_input_vocab_size else config.vocab_size,
                     config.hidden_size_per_layer_input
                 )
                 for _ in range(config.num_layers)
             ])

             # The projection in `GeminiModel` (Gemma3n)
             # self.per_layer_proj = nn.Linear(30 * 256, 30 * 256, bias=False)
             self.per_layer_proj_global = nn.Linear(
                 config.num_layers * config.hidden_size_per_layer_input,
                 config.num_layers * config.hidden_size_per_layer_input,
                 bias=False
             )
             self.per_layer_norm_global = normalization.RMSNorm(
                 config.hidden_size_per_layer_input, eps=1e-6
             )
             self.per_layer_scaling = 0.167


        self.transformer_blocks = nn.ModuleList(
            Gemma3nBlock(config.block_config(idx), config, idx)
            for idx in range(config.num_layers)
        )

        self.final_norm = builder.build_norm(
            config.embedding_dim, config.final_norm_config
        )

        self.output_scale = 30.0 # From reference

        self.build_mask_cache(mask_cache_size)

    def build_mask_cache(self, mask_cache_size: int):
        if mask_cache_size <= 0:
            self.mask_cache = None
        else:
            self.mask_cache = attn_utils.build_causal_mask_cache(mask_cache_size)

    @torch.inference_mode
    def forward(
        self,
        tokens: torch.Tensor,
        input_pos: torch.Tensor,
        kv_cache: kv_utils.KVCache,
        export_config: Optional[export_cfg.ExportConfig] = None,
    ) -> Dict[str, torch.Tensor]:

        # Embeddings
        x = self.tok_embedding(tokens)
        if self.config.embedding_scale:
            x = x * self.config.embedding_scale

        # Per-layer embeddings processing
        per_layer_emb = None
        if hasattr(self, 'per_layer_embeddings'):
            # Gather embeddings
            # tokens: [B, T]
            # output: [B, T, NumLayers, D_per_layer]

            # Clamp tokens if using a smaller vocab for per-layer embeddings
            vocab_limit = self.per_layer_embeddings[0].num_embeddings
            tokens_clamped = torch.clamp(tokens, 0, vocab_limit - 1)

            # Using loop over module list
            emb_list = [emb(tokens_clamped) for emb in self.per_layer_embeddings]
            per_layer_raw = torch.stack(emb_list, dim=2) # [B, T, L, D]

            B, T, L, D = per_layer_raw.shape

            # Global projection
            flat = per_layer_raw.view(B, T, L * D)
            projected = self.per_layer_proj_global(flat)
            projected = projected.view(B, T, L, D)

            # Norm
            normed = self.per_layer_norm_global(projected)

            # Residual + Scaling
            per_layer_emb = (normed + per_layer_raw) * self.per_layer_scaling

        # RoPE
        attn_config = self.config.block_config(0).attn_config
        rope = [
            rotary_pos_emb.build_rope(
                input_pos,
                attn_config.head_dim,
                self.config.block_config(i).attn_config.rotary_base,
            )
            for i in range(self.config.num_layers)
        ]

        # Mask
        if self.mask_cache is not None and kv_cache is not None:
             kv_cache_max_len = kv_cache.get_max_seq_len()
             mask = self.mask_cache.index_select(2, input_pos)
             mask = mask[:, :, :, :kv_cache_max_len]
        else:
             # Basic causal mask if cache not used (for simple testing)
             S = x.size(1)
             mask = torch.triu(torch.full((S, S), float('-inf'), device=x.device), diagonal=1)
             mask = mask.unsqueeze(0).unsqueeze(0)

        # Layers
        updated_kv_entries = []
        for i, block in enumerate(self.transformer_blocks):
            # Pass correct per-layer embedding slice
            ple = per_layer_emb[:, :, i, :] if per_layer_emb is not None else None

            kv_entry = kv_cache.caches[i] if kv_cache else None

            # Mask handling for sliding window?
            # Reuse logic from Gemma3 Decoder if needed.
            # For now assuming global mask or handled by attention.

            x, kv_entry = block(x, rope[i], mask, input_pos, kv_entry, per_layer_emb=ple)
            if kv_entry:
                updated_kv_entries.append(kv_entry)

        updated_kv_cache = kv_utils.KVCache(tuple(updated_kv_entries))

        x = self.final_norm(x)
        logits = self.lm_head(x)

        # Output scaling from reference
        logits = logits / self.output_scale
        logits = torch.tanh(logits) * self.output_scale

        return {"logits": logits, "kv_cache": updated_kv_cache}


def build_model_e2b(checkpoint_path: str = None) -> Gemma3n:
    """Builds the Gemma3n E2B model."""

    norm_config = cfg.NormalizationConfig(
        type=cfg.NormalizationType.RMS_NORM, epsilon=1e-6, zero_centered=False,
    )

    # 2048 hidden size, 8 heads -> 256 head dim
    # 8192 intermediate size
    ff_config = cfg.FeedForwardConfig(
        type=cfg.FeedForwardType.GATED,
        activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
        intermediate_size=8192,
        pre_ff_norm_config=norm_config,
        post_ff_norm_config=norm_config,
    )

    altup_config = cfg.AltUpConfig(
        num_inputs=4,
        active_idx=0,
        coef_clip=120.0,
        correct_scale=True
    )

    def get_block_config(idx: int) -> cfg.TransformerBlockConfig:
        # layer_types: ["sliding_attention", ..., "full_attention"]
        # pattern repeats every 5 layers: 4 sliding, 1 full.
        is_global = (idx + 1) % 5 == 0

        attn_config = cfg.AttentionConfig(
            num_heads=8,
            head_dim=256,
            num_query_groups=2, # GQA
            rotary_base=1_000_000 if is_global else 10_000, # rope_theta for global, rope_local_base_freq for local
            qkv_transpose_before_split=True,
            query_norm_config=norm_config,
            key_norm_config=norm_config,
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
            laurel_rank=64,
            altup_config=altup_config,
        )

    num_layers = 30
    embedding_dim = 2048

    config = cfg.ModelConfig(
        vocab_size=262400,
        num_layers=num_layers,
        max_seq_len=32768,
        embedding_dim=embedding_dim,
        embedding_scale=None, # Not in config json
        block_configs=[get_block_config(i) for i in range(num_layers)],
        final_norm_config=norm_config,
        lm_head_use_bias=False,
        hidden_size_per_layer_input=256,
        per_layer_input_vocab_size=262144,
    )

    if checkpoint_path:
        loader = loading_utils.ModelLoader(checkpoint_path, TENSOR_NAMES)
        model = loader.load(Gemma3n, config, strict=False)
    else:
        model = Gemma3n(config, mask_cache_size=0)

    model.eval()

    return model
