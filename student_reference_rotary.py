"""Rotary decoder with causal cache inference.

RMSNorm/RoPE/SwiGLU, bias-free projections, tied embeddings and residual
dropout are used. Gate/up are fused and native FP32 RMSNorm is used.
Inference cache uses a strictly causal probability mixture.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RotaryBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, hidden = config['width'], config['ffn_hidden']
        self.heads = config['heads']
        self.norm1 = nn.RMSNorm(width, eps=1e-5)
        self.norm2 = nn.RMSNorm(width, eps=1e-5)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(config.get('residual_dropout', .2))

    @staticmethod
    def rotate(tensor, cosine, sine):
        even, odd = tensor[..., 0::2], tensor[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2)

    def forward(self, x, cosine, sine):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(
            batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        q, k = self.rotate(q, cosine, sine), self.rotate(k, cosine, sine)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        gate, up = self.gate_up(self.norm2(x)).chunk(2, -1)
        return x + self.dropout(self.down(F.silu(gate) * up))


class ReferenceRotary(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width, heads = config['width'], config['heads']
        if width % heads or (width // heads) % 2:
            raise ValueError('Attention head width must be an even integer')
        dropout = config.get('residual_dropout', .2)
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('Residual dropout must be in [0, 1)')
        self.cache_weight = config.get('cache_weight', 0.)
        self.cache_theta = config.get('cache_theta', 13.)
        self.neural_temperature = config.get('neural_temperature', 1.)
        if not 0 <= self.cache_weight < 1 or not math.isfinite(self.cache_weight):
            raise ValueError('Cache weight must be in [0, 1)')
        if any(not math.isfinite(v) or v <= 0 for v in (self.cache_theta, self.neural_temperature)):
            raise ValueError('Temperature and cache theta must be positive')
        self.token = nn.Embedding(config['vocab'], width)
        self.blocks = nn.ModuleList([RotaryBlock(config) for _ in range(config['depth'])])
        self.norm = nn.RMSNorm(width, eps=1e-5)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight
        head_width = width // heads
        angles = torch.arange(self.context).float()[:, None] * (
            10000. ** (-torch.arange(0, head_width, 2).float() / head_width))[None, :]
        self.register_buffer('cosine', angles.cos(), persistent=False)
        self.register_buffer('sine', angles.sin(), persistent=False)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)

    def features(self, ids):
        length = ids.shape[1]
        if not 0 < length <= self.context:
            raise ValueError('Input length exceeds context')
        x = self.token(ids)
        for block in self.blocks:
            x = block(x, self.cosine[:length], self.sine[:length])
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        features = self.features(ids)
        logits = self.head(features).float() / self.neural_temperature
        if self.cache_weight == 0:
            return F.log_softmax(logits, -1)
        probabilities = F.softmax(logits, -1)
        first = probabilities[:, 0].clone()
        probabilities.mul_(1. - self.cache_weight)
        length = ids.shape[1]
        if length > 1:
            unit = F.normalize(features.float(), dim=-1)
            similarity = self.cache_theta * (unit @ unit.transpose(-1, -2))
            query = torch.arange(1, length, device=ids.device)[:, None]
            key = torch.arange(length - 1, device=ids.device)[None, :]
            weights = similarity[:, 1:, :-1].masked_fill(key >= query, float('-inf')).softmax(-1)
            observed = ids[:, None, 1:].expand(ids.shape[0], length - 1, length - 1)
            probabilities[:, 1:].scatter_add_(-1, observed, weights * self.cache_weight)
        probabilities[:, 0] = first
        return probabilities.clamp_min_(torch.finfo(torch.float32).tiny).log_()


def build_model(config):
    return ReferenceRotary(config)
