# Gemma 3n Implementation Walkthrough

This document provides a comprehensive guide to the Gemma 3n (Matformer) implementation in AI Edge Torch, including the architecture, conversion process, and implementation details.

## Table of Contents

1. [Overview](#overview)
2. [Architecture Background](#architecture-background)
3. [Implementation Structure](#implementation-structure)
4. [Conversion Pipeline](#conversion-pipeline)
5. [Component Details](#component-details)
6. [Usage Guide](#usage-guide)
7. [Known Limitations](#known-limitations)
8. [References](#references)

---

## Overview

Gemma 3n is Google's **Matformer (Matryoshka Transformer)** architecture designed for efficient on-device inference. This implementation enables conversion of Gemma 3n models to TFLite format for deployment on edge devices via MediaPipe's LLM Inference API.

### Supported Model Variants

| Variant | Layers | Embedding Dim | Intermediate Size | Description |
|---------|--------|---------------|-------------------|-------------|
| **E2B** | 30 | 2048 | 8192 | ~2 billion parameters |
| **E4B** | 35 | 2048 | 16384 | ~4 billion parameters |

---

## Architecture Background

Gemma 3n introduces several novel mechanisms that differentiate it from standard transformers:

### 1. AltUp (Alternating Updates)

**Reference**: [arxiv.org/pdf/2301.13310](https://arxiv.org/pdf/2301.13310)

AltUp enables sparse updates across multiple prediction streams, reducing computational cost while maintaining model expressivity. Instead of processing a single hidden state, AltUp maintains multiple parallel "streams" and only fully processes one at a time.

```
Input Hidden States → [num_altup_inputs, batch, seq_len, hidden_size]
                     ↓
              Predict Step (expand predictions using learned coefficients)
                     ↓
              Process Active Stream (attention + FFN)
                     ↓
              Correct Step (propagate updates to other streams)
                     ↓
Output Hidden States → [num_altup_inputs, batch, seq_len, hidden_size]
```

### 2. LAuReL (Learned Augmented Residual Layer)

**Reference**: [arxiv.org/pdf/2411.07501](https://arxiv.org/pdf/2411.07501)

LAuReL provides more expressivity than simple additive residuals through a low-rank projection:

```
input → Linear(hidden_size → laurel_rank) → Linear(laurel_rank → hidden_size) → RMSNorm → + input
```

### 3. Per-Layer Embeddings (PLE)

Each transformer layer receives additional token-specific embeddings that provide layer-specific information. This allows the model to retrieve context on-demand rather than encoding everything in the initial embedding.

```
Token IDs → Embedding(vocab_size → num_layers * per_layer_dim)
         → reshape to [batch, seq_len, num_layers, per_layer_dim]
```

### 4. Hybrid Attention

Gemma 3n uses a mix of:
- **Local Sliding Window Attention** (window_size=512) for most layers
- **Global Full Attention** for every 5th layer

This pattern balances computational efficiency with long-range context modeling.

---

## Implementation Structure

### File Organization

```
ai_edge_torch/generative/examples/gemma3n/
├── __init__.py                    # Module exports
├── gemma3n.py                     # Core model implementation (973 lines)
└── convert_gemma3n_to_tflite.py   # Conversion script
```

### Key Classes and Functions

| Component | Location | Purpose |
|-----------|----------|---------|
| `RMSNorm` | gemma3n.py:151 | RMS normalization layer |
| `LaurelBlock` | gemma3n.py:168 | LAuReL residual connection |
| `AltUp` | gemma3n.py:190 | Alternating updates module |
| `PerLayerEmbedding` | gemma3n.py:280 | Per-layer embedding lookup |
| `SparseGatedMLP` | gemma3n.py:323 | MLP with Gaussian top-k activation sparsity |
| `Gemma3nDecoderBlock` | gemma3n.py:376 | Single decoder layer with KV sharing support |
| `Decoder` | gemma3n.py:520 | Full decoder model with KV sharing map |
| `build_model_e2b` | gemma3n.py:980 | E2B model builder |
| `build_model_e4b` | gemma3n.py:1010 | E4B model builder |
| `load_gemma3n_weights` | gemma3n.py:70 | Custom weight loader |

---

## Conversion Pipeline

### Step-by-Step Conversion Flow

```mermaid
graph TD
    A[HuggingFace Checkpoint] --> B[build_model_e2b/e4b]
    B --> C[load_gemma3n_weights]
    C --> D[Decoder Model]
    D --> E[converter.build_and_convert_to_tflite_from_flags]
    E --> F[ExportableModule wrapper]
    F --> G[Add prefill signatures]
    F --> H[Add decode signature]
    G --> I[ai_edge_torch.convert]
    H --> I
    I --> J[Quantization]
    J --> K[.tflite output]
```

### 1. Model Building

The conversion starts by calling the model builder function:

```python
from ai_edge_torch.generative.examples.gemma3n import gemma3n

# Build the model from a checkpoint
pytorch_model = gemma3n.build_model_e2b(checkpoint_path)
```

Internally, `build_model_e2b` performs:

1. **Get configuration**: Calls `get_model_config_e2b()` to create `ModelConfig` and `gemma3n_config` dict
2. **Instantiate model**: Creates `Decoder(model_config, gemma3n_config)`
3. **Build mask cache**: If `mask_cache_size > 0`, pre-builds causal attention mask
4. **Load weights**: Calls `load_gemma3n_weights(model, checkpoint_path)`
5. **Set eval mode**: `model.eval()`

### 2. Weight Loading

The `load_gemma3n_weights` function handles the complete mapping from HuggingFace checkpoint format to our internal structure. It supports both direct text model and multimodal checkpoint formats:

**Supported checkpoint formats:**
- **Direct text model**: `model.embed_tokens`, `model.layers.{i}`
- **Multimodal (HuggingFace)**: `model.language_model.embed_tokens`, `model.language_model.layers.{i}`

The function auto-detects the format by checking for the presence of either prefix in the state dict.

```python
# HuggingFace format → Our format
"model.layers.{i}.self_attn.q_proj.weight" → "transformer_blocks.{i}.atten_func.qkv_projection.weight" (fused with K, V)
"model.layers.{i}.mlp.gate_proj.weight" → "transformer_blocks.{i}.ff.gate_proj.weight"
"model.layers.{i}.laurel.linear_left.weight" → "transformer_blocks.{i}.laurel.linear_left.weight"
"model.layers.{i}.altup.correction_coefs.weight" → "transformer_blocks.{i}.altup.correction_coefs.weight"
```

Key transformations:
- **QKV fusion**: Q, K, V projections are concatenated into a single `qkv_projection`
- **Norm naming**: `input_layernorm` → `pre_atten_norm`
- **Laurel/AltUp**: Direct mapping with prefix change
- **Format detection**: Automatic prefix detection (`model` vs `model.language_model`)

### 3. Conversion to TFLite

The conversion uses the standard AI Edge Torch converter:

```python
from ai_edge_torch.generative.utilities import converter

converter.build_and_convert_to_tflite_from_flags(builder)
```

This function:

1. **Builds the model** with the provided builder function
2. **Creates sample inputs** for each signature:
   - Prefill signatures: `(tokens, input_pos, kv_cache, [mask])`
   - Decode signature: `(tokens, input_pos, kv_cache, [mask])`
3. **Wraps the model** in `ExportableModule` to capture extra kwargs
4. **Adds signatures** to the converter for each prefill length and decode
5. **Applies quantization** based on the `--quantize` flag
6. **Exports** to `.tflite` file

### 4. Signature Structure

The output TFLite model contains multiple signatures:

| Signature Name | Input Shape | Purpose |
|----------------|-------------|---------|
| `prefill_8` | tokens: [1, 8], input_pos: [8] | Short prefill |
| `prefill_64` | tokens: [1, 64], input_pos: [64] | Medium prefill |
| `prefill_128` | tokens: [1, 128], input_pos: [128] | Default prefill |
| ... | ... | Additional prefill lengths |
| `decode` | tokens: [1, 1], input_pos: [1] | Single token decode |

---

## Component Details

### Decoder Forward Pass

The `Decoder.forward` method orchestrates the full inference:

```python
def forward(self, tokens, input_pos, kv_cache, mask=None, export_config=None):
    # 1. Token embedding with scaling
    input_embeds = self.tok_embedding(tokens)
    input_embeds = input_embeds * self.config.embedding_scale  # sqrt(embedding_dim)
    
    # 2. Compute per-layer inputs (PLE)
    per_layer_inputs = self.get_per_layer_inputs(tokens, input_embeds)
    
    # 3. Expand hidden states for AltUp
    hidden_states = self.expand_hidden_states(input_embeds)
    # Shape: [num_altup_inputs, batch, seq_len, hidden_size]
    
    # 4. Build RoPE embeddings for each layer
    rope_list = [build_rope(input_pos, head_dim, rotary_base) for each layer]

    # 5. Process through transformer blocks with KV cache sharing
    shared_kv_storage = {}  # Store KV cache for shared layers
    for i, block in enumerate(self.transformer_blocks):
        # Get shared KV if this is a KV-shared layer
        shared_kv = None
        if i in self.kv_sharing_map:
            source_layer = self.kv_sharing_map[i]
            shared_kv = shared_kv_storage.get(source_layer)

        hidden_states, kv = block(
            hidden_states, rope_list[i], mask_list[i],
            input_pos, kv_cache[i], per_layer_inputs[:,:,i,:],
            shared_kv=shared_kv
        )

        # Store KV for layers that are sources for sharing
        if i in self.kv_source_layers:
            shared_kv_storage[i] = kv
    
    # 6. Collapse AltUp hidden states
    hidden_states = self.collapse_hidden_states(hidden_states)
    
    # 7. Final norm and logit computation
    hidden_states = self.final_norm(hidden_states)
    logits = self.lm_head(hidden_states)
    
    # 8. Apply logit softcapping
    logits = tanh(logits / 30.0) * 30.0
    
    return {"logits": logits, "kv_cache": updated_kv_cache}
```

### Decoder Block Forward Pass

Each `Gemma3nDecoderBlock` performs:

```python
def forward(self, hidden_states, rope, mask, input_pos, kv_cache, per_layer_input, shared_kv=None):
    # 1. AltUp predict step
    predictions = self.altup.predict(hidden_states)
    active_prediction = predictions[0]  # active_idx = 0

    # 2. Pre-attention norm + LAuReL
    active_normed = self.pre_atten_norm(active_prediction)
    laurel_output = self.laurel(active_normed)

    # 3. Self-attention with optional KV cache sharing
    # If this is a KV-shared layer, use shared KV from an earlier layer
    effective_kv = shared_kv if (self.is_kv_shared_layer and shared_kv is not None) else kv_cache
    attn_out, kv = self.atten_func(active_normed, rope, mask, input_pos, effective_kv)
    # For shared layers, don't update KV cache - return original unchanged
    if self.is_kv_shared_layer and shared_kv is not None:
        kv = kv_cache
    attn_out = self.post_atten_norm(attn_out)
    
    # 4. Combine attention with LAuReL
    attn_gated = active_prediction + attn_out
    attn_laurel = (attn_gated + laurel_output) / sqrt(2)

    # 5. Feed-forward with pre/post norms and activation sparsity
    ff_input = self.pre_ff_norm(attn_laurel)
    ff_out = self.ff(ff_input)  # SparseGatedMLP applies Gaussian top-k if sparsity > 0
    ff_out = self.post_ff_norm(ff_out)
    attn_ff = attn_laurel + ff_out
    
    # 6. AltUp correct step
    corrected = self.altup.correct(predictions, attn_ff)
    
    # 7. Apply per-layer embeddings to non-active predictions
    gate = gelu(self.per_layer_input_gate(corrected[0] * scale))
    per_layer_out = self.per_layer_projection(gate * per_layer_input)
    per_layer_out = self.post_per_layer_norm(per_layer_out)
    corrected[1:] += per_layer_out  # Add to non-active streams
    
    return corrected, kv
```

### SparseGatedMLP with Precomputed Sparsity

The `SparseGatedMLP` class implements gated MLP with optional Gaussian top-k activation sparsity. A key optimization for TFLite compatibility is the **precomputation of `std_multiplier`** during initialization:

```python
class SparseGatedMLP:
    def __init__(self, hidden_size, intermediate_size, activation_sparsity=0.0):
        # ...
        # Precompute std_multiplier to avoid erfinv at runtime (not supported by TFLite)
        if self.activation_sparsity > 0.0:
            normal_dist = torch.distributions.normal.Normal(0, 1)
            std_mult_value = float(normal_dist.icdf(torch.tensor(self.activation_sparsity)))
            self.register_buffer("std_multiplier", torch.tensor(std_mult_value), persistent=False)

    def _gaussian_topk(self, inputs):
        # Threshold: mean + std * icdf(sparsity) - where icdf is precomputed
        inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
        cutoff = inputs_mean + inputs_std * self.std_multiplier.to(inputs.dtype)
        return F.relu(inputs - cutoff)
```

**Why precompute?**
- The inverse CDF computation (`icdf`/`erfinv`) is not supported by TFLite's operation set
- By computing this value once during model initialization and storing it as a buffer, we avoid runtime errors during conversion
- For 95% sparsity (used in first 10 layers), `std_multiplier ≈ 1.6449`

**Sparsity pattern in Gemma3n:**
- First 10 layers: 95% activation sparsity (only top 5% of activations pass through)
- Remaining layers: Dense (0% sparsity)

### KV Cache Sharing

The decoder implements KV cache sharing between layers to reduce memory footprint:

**Layer types:**
- **Global layers**: Every 5th layer (layers 4, 9, 14, etc.) - uses full attention
- **Sliding window layers**: All other layers - uses local attention with window_size=512

**Sharing mechanism:**
- Later layers (controlled by `num_kv_shared_layers`) reuse KV cache from earlier layers of the same type
- A `kv_sharing_map` dict maps shared layer indices to their source layer: `{28: 8, 29: 9}` means layer 28 shares KV from layer 8
- Source layers store their KV cache in `shared_kv_storage` for reuse
- Shared layers receive `shared_kv` parameter and don't update the KV cache

```python
# In Decoder.forward():
for i, block in enumerate(self.transformer_blocks):
    # Get shared KV if this is a shared layer
    shared_kv = None
    if i in self.kv_sharing_map:
        source_layer = self.kv_sharing_map[i]
        shared_kv = shared_kv_storage.get(source_layer)

    # Process block with optional shared KV
    hidden_states, kv_entry = block(..., shared_kv=shared_kv)

    # Store KV for layers that are sources for sharing
    if i in self.kv_source_layers:
        shared_kv_storage[i] = kv_entry
```

---

## Usage Guide

### Basic Conversion

```bash
python -m ai_edge_torch.generative.examples.gemma3n.convert_gemma3n_to_tflite \
    --checkpoint_path=/path/to/gemma3n-e2b \
    --output_path=/output/directory \
    --model_variant=e2b \
    --quantize=dynamic_int8 \
    --kv_cache_max_len=1280 \
    --prefill_seq_lens=8,64,128,256
```

### Programmatic Usage

```python
from ai_edge_torch.generative.examples.gemma3n import gemma3n
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities.export_config import ExportConfig
from ai_edge_torch.generative.layers import kv_cache

# Build the model
pytorch_model = gemma3n.build_model_e2b("/path/to/checkpoint")

# Configure export
export_config = ExportConfig()
export_config.kvcache_layout = kv_cache.KV_LAYOUT_TRANSPOSED
export_config.mask_as_input = True

# Convert
converter.convert_to_tflite(
    pytorch_model,
    output_path="/output/directory",
    output_name_prefix="gemma3n-e2b",
    prefill_seq_len=[8, 64, 128],
    kv_cache_max_len=1280,
    quantize="dynamic_int8",
    export_config=export_config,
)
```

### Conversion Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint_path` | ~/Downloads/llm_data/gemma3n-e2b | Path to model checkpoint |
| `--output_path` | /tmp/ | Output directory |
| `--model_variant` | e2b | Model variant: 'e2b' or 'e4b' |
| `--quantize` | dynamic_int8 | Quantization type |
| `--kv_cache_max_len` | 1280 | Maximum KV cache length |
| `--prefill_seq_lens` | 8,64,128,256,512,1024 | Prefill sequence lengths |
| `--mask_as_input` | True | Pass mask as input |
| `--transpose_kv_cache` | True | Use transposed KV cache layout |

---

## Known Limitations

> [!WARNING]
> This implementation is experimental and has the following limitations:

1. **Text-only support**: This implementation only supports the text decoder. Vision and audio encoders from the full Gemma 3n model are not included.

2. **No verification against reference**: The implementation has not been verified against the original HuggingFace implementation for numerical accuracy.

3. **Weight loading assumes specific format**: The weight loader expects HuggingFace SafeTensors format. Other checkpoint formats may require modifications.

### Implemented Features ✓

- **KV Cache Sharing**: Later layers reuse KV cache from earlier layers of the same attention type (sliding/global). Configured via `num_kv_shared_layers` in gemma3n_config.

- **Activation Sparsity**: Gaussian top-k sparsity applied to MLP gate projections. First 10 layers use 0.95 sparsity, rest are dense. Configured via `activation_sparsity_pattern`.

---

## References

### Architecture Papers

- **MatFormer**: [https://arxiv.org/pdf/2310.07707](https://arxiv.org/pdf/2310.07707)
- **LAuReL**: [https://arxiv.org/pdf/2411.07501](https://arxiv.org/pdf/2411.07501)
- **AltUp**: [https://arxiv.org/pdf/2301.13310](https://arxiv.org/pdf/2301.13310)

### Source Files

- [gemma3n.py](ai_edge_torch/generative/examples/gemma3n/gemma3n.py) - Main model implementation
- [convert_gemma3n_to_tflite.py](ai_edge_torch/generative/examples/gemma3n/convert_gemma3n_to_tflite.py) - Conversion script
- [converter.py](ai_edge_torch/generative/utilities/converter.py) - Conversion utilities
- [references/gemma3n-hf/](references/gemma3n-hf/) - HuggingFace reference implementation

### Related Examples

- [Gemma 3 implementation](ai_edge_torch/generative/examples/gemma3/) - Similar architecture without Matformer components
- [Examples README](ai_edge_torch/generative/examples/README.md) - Overview of all model examples
