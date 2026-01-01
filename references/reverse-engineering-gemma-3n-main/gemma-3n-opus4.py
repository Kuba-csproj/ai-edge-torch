# MatFormer: https://arxiv.org/pdf/2310.07707	
# AltUp Layers: https://arxiv.org/pdf/2301.13310	
# Laurel Blocks: https://arxiv.org/pdf/2411.07501	

import torch	
import torch.nn as nn	
import torch.nn.functional as F	
import math	
from typing import Optional, Tuple	

class LayerNorm(nn.Module):	
    """Layer normalization matching the STABLEHLO_COMPOSITE operations"""	
    def __init__(self, dim, eps=1e-6):	
        super().__init__()	
        self.weight = nn.Parameter(torch.ones(dim))	
        self.eps = eps	

    def forward(self, x):	
        # Compute x^2, mean, and normalize	
        x_squared = x * x	
        mean_squared = x_squared.mean(dim=-1, keepdim=True)	
        rms = torch.pow(mean_squared, 0.5)	
        rms = torch.maximum(rms, torch.tensor(self.eps))	
        return (x / rms) * self.weight	

class RotaryPositionEmbedding(nn.Module):	
    """RoPE implementation based on the sin/cos operations in the model"""	
    def __init__(self, dim, max_seq_len=4096):	
        super().__init__()	
        self.dim = dim	

        # Precompute sin/cos embeddings	
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))	
        position = torch.arange(max_seq_len).float()	
        sincos = torch.einsum('i,j->ij', position, inv_freq)	
        self.register_buffer('sin', sincos.sin())	
        self.register_buffer('cos', sincos.cos())	

    def forward(self, x, position_ids):	
        # x shape: [batch, seq_len, num_heads, head_dim]	
        batch, seq_len, num_heads, head_dim = x.shape	

        # Split into two halves for rotation	
        x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]	

        # Get sin/cos for the positions	
        cos = self.cos[position_ids].unsqueeze(1).unsqueeze(2)	
        sin = self.sin[position_ids].unsqueeze(1).unsqueeze(2)	

        # Apply rotation	
        rx1 = x1 * cos - x2 * sin	
        rx2 = x2 * cos + x1 * sin	

        return torch.cat([rx1, rx2], dim=-1)	

class Attention(nn.Module):	
    """Multi-head attention with RoPE"""	
    def __init__(self, dim, num_heads=8, head_dim=256):	
        super().__init__()	
        self.num_heads = num_heads	
        self.head_dim = head_dim	
        self.scale = 0.0625  # 1/16, matching the model's scaling	

        # Q, K, V projections (quantized in the model)	
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)	
        self.k_proj = nn.Linear(dim, num_heads * head_dim, bias=False) 	
        self.v_proj = nn.Linear(dim, num_heads * head_dim, bias=False)	
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)	

        # Norms	
        self.q_norm = LayerNorm(head_dim)	
        self.k_norm = LayerNorm(head_dim)	

        # RoPE	
        self.rope = RotaryPositionEmbedding(head_dim)	

    def forward(self, x, position_ids, kv_cache=None):	
        batch, seq_len, _ = x.shape	

        # Project to Q, K, V	
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)	
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)	
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)	

        # Apply norms	
        q = self.q_norm(q)	
        k = self.k_norm(k)	

        # Apply RoPE	
        q = self.rope(q, position_ids)	
        k = self.rope(k, position_ids)	

        # Scale query	
        q = q * self.scale * 0.88388  # Additional scaling seen in model	

        # Reshape for attention	
        q = q.reshape(batch, 2, seq_len//2, self.head_dim)	
        k = k.reshape(batch, 2, seq_len//2, self.head_dim)	
        v = v.reshape(batch, 2, seq_len//2, self.head_dim).transpose(-2, -1)	

        # Compute attention	
        scores = torch.matmul(q, k.transpose(-2, -1))	

        # Add causal mask if needed	
        if kv_cache is None:  # Only for prefill	
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)	
            scores = scores + mask.to(scores.device)	

        # Softmax and apply to values	
        attn_weights = F.softmax(scores, dim=-1)	
        attn_output = torch.matmul(attn_weights, v.transpose(-2, -1))	

        # Reshape and project	
        attn_output = attn_output.reshape(batch, seq_len, self.num_heads * self.head_dim)	
        return self.o_proj(attn_output)	

class MLP(nn.Module):	
    """Feed-forward network with GELU activation"""	
    def __init__(self, dim, hidden_dim=8192):	
        super().__init__()	
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)	
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)	
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)	

    def forward(self, x):	
        gate = F.gelu(self.gate_proj(x))	
        up = self.up_proj(x)	
        return self.down_proj(gate * up)	

class LaurelBlock(nn.Module):	
    """Laurel residual connection block"""	
    def __init__(self, dim, bottleneck_dim=64):	
        super().__init__()	
        self.down_proj = nn.Linear(dim, bottleneck_dim, bias=False)	
        self.up_proj = nn.Linear(bottleneck_dim, dim, bias=False)	
        self.norm = LayerNorm(dim)	

    def forward(self, x):	
        residual = x	
        x = self.down_proj(x)	
        x = self.up_proj(x)	
        x = self.norm(x)	
        return residual + x	

class AltupRouter(nn.Module):	
    """ALTUP routing mechanism"""	
    def __init__(self, dim, num_experts=4):	
        super().__init__()	
        self.norm = LayerNorm(dim)	
        self.router = nn.Linear(dim, num_experts, bias=False)	
        self.scale = 3.0  # Scaling factor seen in model	

    def forward(self, x):	
        x = self.norm(x)	
        logits = self.router(x)	
        logits = logits * self.scale	
        return torch.tanh(logits)
    
class TransformerLayer(nn.Module):
    """Single transformer layer with ALTUP"""
    def __init__(self, dim, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Pre-attention
        self.pre_attn_norm = LayerNorm(dim)
        self.attention = Attention(dim)
        self.post_attn_norm = LayerNorm(dim)
        
        # Laurel block
        self.laurel = LaurelBlock(dim)
        
        # MLP
        self.pre_mlp_norm = LayerNorm(dim)
        self.mlp = MLP(dim)
        self.post_mlp_norm = LayerNorm(dim)
        
        # ALTUP routers
        self.altup_router_predict = AltupRouter(dim)
        self.altup_router_correct = AltupRouter(dim)
        
        # ALTUP projections
        self.altup_proj = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(3)
        ])
        self.altup_unproj = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(3)
        ])
        
        # Per-layer embeddings
        self.per_layer_gate = nn.Linear(dim, 256, bias=False)
        self.per_layer_proj = nn.Linear(256, dim, bias=False)
        self.per_layer_norm = LayerNorm(dim)
        
        # Scaling factors
        self.residual_scale = 0.57735  # sqrt(1/3)
        
    def altup_predict(self, x, per_layer_emb):
        """ALTUP prediction step"""
        # Get routing weights
        router_weights = self.altup_router_predict(x)
        router_probs = F.softmax(router_weights, dim=-1)
        
        # Apply ALTUP projections
        experts = []
        for i in range(3):
            expert = self.altup_proj[i](x if i == 0 else experts[i-1])
            expert = torch.clamp(expert, -10, 10)  # Clipping seen in model
            experts.append(expert)
            
        # Stack and mix experts
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
        for i in range(3):
            output = self.altup_unproj[i](corrected[:, :, i+1])
            output = torch.clamp(output, -10, 10)
            outputs.append(output)
            
        # Mix outputs
        final = torch.stack([corrected[:, :, 0]] + outputs, dim=0).mean(dim=0)
        
        return final
    
    def forward(self, x, position_ids, per_layer_emb):
        # ALTUP predict
        predicted = self.altup_predict(x, per_layer_emb)
        
        # Attention block
        h = self.pre_attn_norm(predicted)
        h = self.attention(h, position_ids)
        h = self.post_attn_norm(h)
        h = predicted + h
        
        # Laurel residual
        h = self.laurel(h)
        h = h * self.residual_scale
        
        # MLP block  
        h_norm = self.pre_mlp_norm(h)
        mlp_out = self.mlp(h_norm)
        mlp_out = self.post_mlp_norm(mlp_out)
        h = h + mlp_out
        
        # ALTUP correct
        corrected = self.altup_correct(predicted, h)
        
        
        # Add per-layer embeddings
        gate = F.gelu(self.per_layer_gate(corrected))
        per_layer_out = self.per_layer_proj(gate * per_layer_emb[:, :, self.layer_idx])
        per_layer_out = self.per_layer_norm(per_layer_out)
        
        return corrected + per_layer_out
    

class GeminiModel(nn.Module):	
    """Complete Gemini model"""	
    def __init__(self, vocab_size=262144, dim=2048, num_layers=30):	
        super().__init__()	
        self.vocab_size = vocab_size	
        self.dim = dim	
        self.num_layers = num_layers	

        # Per-layer embedding projection	
        self.per_layer_proj = nn.Linear(30 * 256, 30 * 256, bias=False)	
        self.per_layer_norm = LayerNorm(256)	

        # Transformer layers	
        self.layers = nn.ModuleList([	
            TransformerLayer(dim, i) for i in range(num_layers)	
        ])	

        # Final norm and output	
        self.final_norm = LayerNorm(dim)	
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)	

        # Output scaling	
        self.output_scale = 30.0	


    def forward(self, embeddings, per_layer_embeddings, position_ids):	
        # Process per-layer embeddings	
        B, T, num_layers, emb_dim = per_layer_embeddings.shape	
        per_layer_emb = per_layer_embeddings.view(B, T, -1)	
        per_layer_emb = self.per_layer_proj(per_layer_emb)	
        per_layer_emb = per_layer_emb.view(B, T, num_layers, emb_dim)	
        per_layer_emb = self.per_layer_norm(per_layer_emb)	
        per_layer_emb = per_layer_emb + per_layer_embeddings	
        per_layer_emb = per_layer_emb * 0.167  # Scaling factor	


        # Add per-layer embeddings
        h = embeddings	        
        
        for layer in self.layers:	        
            h = layer(h, position_ids, per_layer_emb)

        # Final output	
        h = self.final_norm(h)	
        logits = self.lm_head(h)	
        logits = logits / self.output_scale	
        logits = torch.tanh(logits) * self.output_scale	

        return logits	

class PerLayerEmbedder(nn.Module):	
    """Per-layer embedding lookup matching TF_LITE_PER_LAYER_EMBEDDER"""	
    def __init__(self, vocab_size=262144, embedding_dim=256, num_layers=30):	
        super().__init__()	
        self.num_layers = num_layers	
        self.embeddings = nn.ModuleList([	
            nn.Embedding(vocab_size, embedding_dim) for _ in range(num_layers)	
        ])	

    def forward(self, token_ids):	
        # Clamp token ids to valid range	
        token_ids = torch.clamp(token_ids, 0, 262143)	


        # Get embeddings for each layer	        return corrected + per_layer_out
        layer_embeddings = []	
        for emb in self.embeddings:	
            layer_embeddings.append(emb(token_ids))	

        # Stack to [batch, seq_len, num_layers, embedding_dim]	
        return torch.stack(layer_embeddings, dim=2)	

# Example usage	
if __name__ == "__main__":	
    batch_size = 1	
    seq_len = 1	

    # Initialize models	
    embedder = PerLayerEmbedder()	
    model = GeminiModel()	

    # Example inputs	
    token_ids = torch.randint(0, 262144, (batch_size, seq_len))	
    embeddings = torch.randn(batch_size, seq_len, 2048)	
    position_ids = torch.arange(seq_len)	

    # Get per-layer embeddings	
    per_layer_emb = embedder(token_ids)	

    # Run model	
    logits = model(embeddings, per_layer_emb, position_ids)	
    print(f"Output shape: {logits.shape}")	
