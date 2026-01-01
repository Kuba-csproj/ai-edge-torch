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
# Connection blocks for Gemma3n.

import torch
from torch import nn
import torch.nn.functional as F
from ai_edge_torch.generative.layers import normalization

class LaurelBlock(nn.Module):
    """Laurel residual connection block.

    References:
        https://arxiv.org/pdf/2411.07501
    """
    def __init__(self, dim: int, bottleneck_dim: int):
        super().__init__()
        self.down_proj = nn.Linear(dim, bottleneck_dim, bias=False)
        self.up_proj = nn.Linear(bottleneck_dim, dim, bias=False)
        # Using RMSNorm as per reference implementation structure and existing layers
        self.norm = normalization.RMSNorm(dim, eps=1e-6)

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.up_proj(x)
        x = self.norm(x)
        return residual + x

class AltupRouter(nn.Module):
    """ALTUP routing mechanism.

    References:
        https://arxiv.org/pdf/2301.13310
    """
    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        # Using RMSNorm as per reference implementation structure and existing layers
        self.norm = normalization.RMSNorm(dim, eps=1e-6)
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.scale = 3.0  # Scaling factor seen in model

    def forward(self, x):
        x = self.norm(x)
        logits = self.router(x)
        logits = logits * self.scale
        return torch.tanh(logits)
