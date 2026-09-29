"""Parameter-matched feed-forward experiments on the classroom GPT.

Default: SwiGLU with learned absolute positions. Set ffn_type='silu' for the
ungated activation control, ffn_type='gelu' for the original FFN, or
position_encoding='rope' for rotary Q/K positions. Set tie_embeddings=False
to train independent input and output token weights. Explicit configs live in
configs/. Normalization and tied embeddings remain the backward-compatible
defaults.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from model import GPT


class SwiGLU(nn.Module):
    """Fuse the two input projections into one linear operation on CPU."""

    def __init__(self, width, hidden):
        super().__init__()
        self.gate_up = nn.Linear(width, 2 * hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class RMSNorm(nn.Module):
    """FP32 RMS normalization with a learned scale and no mean subtraction."""

    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def causal_cache_probs(features, ids, theta):
    """For position t, retrieve only (h_j, ids[j+1]) pairs with j < t.

    Row zero has no history and is returned as all zeros; the caller must
    use the neural distribution there. Everything is local to this call.
    """
    batch, length, _ = features.shape
    probabilities = features.new_zeros(batch, length, 2048, dtype=torch.float32)
    if length < 2:
        return probabilities
    unit = F.normalize(features.float(), dim=-1)
    similarity = theta * (unit @ unit.transpose(-1, -2))
    query = torch.arange(1, length, device=ids.device)[:, None]
    key = torch.arange(length - 1, device=ids.device)[None, :]
    weights = similarity[:, 1:, :-1].masked_fill(key >= query, float('-inf')).softmax(-1)
    observed = ids[:, None, 1:].expand(batch, length - 1, length - 1)
    probabilities[:, 1:].scatter_add_(-1, observed, weights)
    return probabilities


class StudentGPT(GPT):
    def __init__(self, config):
        ffn_type = config.get('ffn_type', 'swiglu')
        if ffn_type not in ('swiglu', 'silu', 'gelu'):
            raise ValueError(f'Unknown ffn_type: {ffn_type!r}')
        width = config['width']
        # 3*d*h ~= 2*d*(4*d); round h upwards to a multiple of eight.
        default_hidden = 8 * ((width + 2) // 3) if ffn_type == 'swiglu' else 4 * width
        hidden = config.get('ffn_hidden', default_hidden)
        if type(hidden) is not int or hidden <= 0:
            raise ValueError('ffn_hidden must be a positive integer.')
        if ffn_type != 'swiglu' and hidden != 4 * width:
            raise ValueError('Ungated controls keep ffn_hidden=4*width.')
        position_encoding = config.get('position_encoding', 'learned')
        if position_encoding not in ('learned', 'rope'):
            raise ValueError(f'Unknown position_encoding: {position_encoding!r}')
        tie_embeddings = config.get('tie_embeddings', True)
        if type(tie_embeddings) is not bool:
            raise ValueError('tie_embeddings must be a boolean.')
        norm_type = config.get('norm_type', 'layernorm')
        if norm_type not in ('layernorm', 'rmsnorm'):
            raise ValueError('norm_type must be layernorm or rmsnorm.')
        residual_init_scale = config.get('residual_init_scale', 1.)
        if (isinstance(residual_init_scale, bool) or not isinstance(residual_init_scale, (int, float))
                or not math.isfinite(residual_init_scale) or residual_init_scale <= 0):
            raise ValueError('residual_init_scale must be finite and positive.')
        cache_weight = config.get('cache_weight', 0.)
        cache_theta = config.get('cache_theta', 5.)
        if not isinstance(cache_weight, (int, float)) or not 0 <= cache_weight < 1:
            raise ValueError('cache_weight must be in [0, 1).')
        if not isinstance(cache_theta, (int, float)) or not math.isfinite(cache_theta) or cache_theta <= 0:
            raise ValueError('cache_theta must be finite and positive.')
        rope_theta = config.get('rope_theta', 10000.)
        if (isinstance(rope_theta, bool) or not isinstance(rope_theta, (int, float))
                or not math.isfinite(rope_theta) or rope_theta <= 1):
            raise ValueError('rope_theta must be a finite number greater than one.')
        heads = config['heads']
        if width % heads:
            raise ValueError('width must be divisible by heads.')
        head_width = width // heads
        if position_encoding == 'rope' and head_width % 2:
            raise ValueError('RoPE requires an even attention head width.')

        # Initialize the unchanged backbone exactly as the supplied baseline.
        # New FFNs receive the same normal(std=.02) / zero-bias initialization.
        super().__init__(config)
        self.ffn_type = ffn_type
        self.ffn_hidden = hidden
        self.position_encoding = position_encoding
        self.tie_embeddings = tie_embeddings
        self.cache_weight = cache_weight
        self.cache_theta = cache_theta
        for block in self.blocks:
            if ffn_type == 'swiglu':
                block.mlp = SwiGLU(width, hidden)
                block.mlp.apply(self.initialize)
            elif ffn_type == 'silu':
                block.mlp[1] = nn.SiLU()
        if residual_init_scale != 1.:
            with torch.no_grad():
                for block in self.blocks:
                    block.proj.weight.mul_(residual_init_scale)
                    down = block.mlp.down if ffn_type == 'swiglu' else block.mlp[2]
                    down.weight.mul_(residual_init_scale)
        if norm_type == 'rmsnorm':
            for block in self.blocks:
                block.norm1 = RMSNorm(width)
                block.norm2 = RMSNorm(width)
            self.norm = RMSNorm(width)
        if not tie_embeddings:
            # Start from exactly the tied model's logits while allowing input
            # and output token representations to specialize during training.
            self.head = nn.Linear(width, config['vocab'], bias=False)
            with torch.no_grad():
                self.head.weight.copy_(self.token.weight)
        if position_encoding == 'rope':
            # Construct the original learned positions above so every shared
            # weight has the same seed-17 initialization as the control, then
            # remove them from the predictor and checkpoint.
            self.pos = None
            inverse_frequency = 1. / (float(rope_theta) **
                                      (torch.arange(0, head_width, 2).float() / head_width))
            angles = torch.arange(self.context).float()[:, None] * inverse_frequency[None, :]
            self.register_buffer('rope_cos', angles.cos(), persistent=False)
            self.register_buffer('rope_sin', angles.sin(), persistent=False)

    def predict_log_probs(self, ids):
        if self.cache_weight == 0:
            return super().predict_log_probs(ids)
        features = self.features(ids)
        neural = F.log_softmax(self.head(features).float(), dim=-1)
        cache = causal_cache_probs(features, ids, self.cache_theta)
        mixed = torch.logaddexp(neural + math.log1p(-self.cache_weight),
                                cache.clamp_min(1e-38).log() + math.log(self.cache_weight))
        mixed[:, 0] = neural[:, 0]
        return mixed

    def apply_rope(self, tensor):
        """Rotate adjacent Q/K feature pairs at their absolute positions."""
        length = tensor.shape[-2]
        if length > self.context:
            raise ValueError(f'Input length {length} exceeds context {self.context}.')
        cos = self.rope_cos[:length].to(dtype=tensor.dtype)[None, None]
        sin = self.rope_sin[:length].to(dtype=tensor.dtype)[None, None]
        even, odd = tensor[..., 0::2], tensor[..., 1::2]
        return torch.stack((even * cos - odd * sin,
                            even * sin + odd * cos), dim=-1).flatten(-2)

    def features(self, ids):
        if self.position_encoding == 'learned':
            return super().features(ids)
        x = self.token(ids)
        for block in self.blocks:
            batch, length, width = x.shape
            q, k, v = block.qkv(block.norm1(x)).view(
                batch, length, 3, block.heads, width // block.heads
            ).permute(2, 0, 3, 1, 4)
            q, k = self.apply_rope(q), self.apply_rope(k)
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + block.proj(attended.transpose(1, 2).reshape(batch, length, width))
            x = x + block.mlp(block.norm2(x))
        return self.norm(x)


def build_model(config):
    return StudentGPT(config)
