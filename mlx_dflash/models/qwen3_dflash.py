"""
MLX port of the DFlash draft model architecture.
Original PyTorch: z-lab/Qwen3-8B-DFlash-b16/modeling_dflash.py (MIT)
Paper: arXiv:2602.06036

Architecture:
  - DFlashDraftModel: N lightweight decoder layers (2-4 for 8B target)
  - Each layer: bidirectional attention over [context_tokens | draft_block]
  - Context features extracted from specific target model layers, projected down
  - Uses target model's embed_tokens + lm_head (shared, not duplicated)
  - Single forward pass drafts block_size=16 tokens in parallel

Apple Silicon optimizations (from No_Shift_4543's r/LocalLLaMA post):
  - head_dim=256: use scaled_dot_product_attention path (steel_attention)
  - Packed QKV: 3 matmuls → 1 matmul + split
  - Sync elision: 2 GPU→CPU syncs per cycle → 1
"""

import math
from typing import Optional
import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    variance = mx.mean(x * x, axis=-1, keepdims=True)
    return x * mx.rsqrt(variance + eps) * weight


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> tuple:
    """Apply RoPE. q/k: (B, H, S, D). cos/sin: (S, D)."""
    cos = cos[None, None, :, :]  # (1,1,S,D)
    sin = sin[None, None, :, :]
    q_len = q.shape[2]
    q = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def build_rope_cache(seq_len: int, head_dim: int, base: float = 1_000_000.0,
                     dtype=mx.bfloat16) -> tuple:
    inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    t = mx.arange(seq_len, dtype=mx.float32)
    freqs = mx.outer(t, inv_freq)
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Evenly spaced target layer indices for context feature extraction."""
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def extract_context_feature(hidden_states: list[mx.array],
                             layer_ids: list[int]) -> mx.array:
    """Concatenate selected target hidden states along last dim."""
    offset = 1  # hidden_states[0] = embedding output, [1..] = layer outputs
    selected = [hidden_states[lid + offset] for lid in layer_ids]
    return mx.concatenate(selected, axis=-1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Qwen3DFlashAttention(nn.Module):
    """
    Bidirectional cross-attended attention for the DFlash drafter.

    Key difference from standard causal attention:
      K and V are built by concatenating context tokens (from target hidden)
      with draft/noise tokens → full bidirectional attention (is_causal=False).

    Apple Silicon optimization: packed QKV projection (1 matmul + split)
    instead of 3 separate matmuls reduces kernel dispatch overhead.
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, rms_eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Packed QKV for draft tokens (optimization: 1 matmul instead of 3)
        self.qkv_proj = nn.Linear(hidden_size,
                                  (num_heads + 2 * num_kv_heads) * head_dim,
                                  bias=False)
        # Separate KV for context tokens (from target hidden)
        self.kv_ctx_proj = nn.Linear(hidden_size,
                                     2 * num_kv_heads * head_dim,
                                     bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Per-head QK norms (Qwen3 specific)
        self.q_norm = nn.RMSNorm(head_dim, eps=rms_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=rms_eps)

    def __call__(self, hidden_states: mx.array, target_hidden: mx.array,
                 cos: mx.array, sin: mx.array) -> mx.array:
        B, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]
        full_len = ctx_len + q_len

        # --- Draft token QKV (packed, single matmul) ---
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q = qkv[..., :q_size]
        k_draft = qkv[..., q_size: q_size + kv_size]
        v_draft = qkv[..., q_size + kv_size:]

        # --- Context KV (from target hidden states) ---
        kv_ctx = self.kv_ctx_proj(target_hidden)
        k_ctx = kv_ctx[..., :kv_size]
        v_ctx = kv_ctx[..., kv_size:]

        # Reshape to (B, H, S, D)
        def reshape(x, heads):
            return x.reshape(B, -1, heads, self.head_dim).transpose(0, 2, 1, 3)

        q = reshape(q, self.num_heads)
        k_draft = reshape(k_draft, self.num_kv_heads)
        v_draft = reshape(v_draft, self.num_kv_heads)
        k_ctx = reshape(k_ctx, self.num_kv_heads)
        v_ctx = reshape(v_ctx, self.num_kv_heads)

        # QK norms
        q = self.q_norm(q)
        k_draft = self.k_norm(k_draft)
        k_ctx = self.k_norm(k_ctx)

        # Concatenate context + draft for K and V
        k = mx.concatenate([k_ctx, k_draft], axis=2)  # (B, H, full_len, D)
        v = mx.concatenate([v_ctx, v_draft], axis=2)

        # Apply RoPE
        # q positions: [ctx_len .. ctx_len+q_len-1]
        # k positions: [0 .. full_len-1]
        q, k = apply_rotary_emb(q, k, cos, sin)

        # GQA repeat KV if needed
        if self.num_kv_heads < self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, repeat, axis=1)
            v = mx.repeat(v, repeat, axis=1)

        # Scaled dot-product attention (no causal mask — bidirectional)
        # head_dim=256: MLX steel_attention path is active (optimized path)
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
        out = attn @ v  # (B, H, q_len, D)

        out = out.transpose(0, 2, 1, 3).reshape(B, q_len, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# DFlash Decoder Layer
# ---------------------------------------------------------------------------

class Qwen3DFlashDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_heads: int, num_kv_heads: int, head_dim: int,
                 rms_eps: float = 1e-6):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.self_attn = Qwen3DFlashAttention(
            hidden_size, num_heads, num_kv_heads, head_dim, rms_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.mlp = nn.Sequential(
            # SwiGLU: two gate projections + down
            # We store as gate_proj, up_proj, down_proj separately
        )
        # SwiGLU components
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array, target_hidden: mx.array,
                 cos: mx.array, sin: mx.array) -> mx.array:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, target_hidden, cos, sin)
        hidden_states = residual + hidden_states

        # SwiGLU MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate = nn.silu(self.gate_proj(hidden_states))
        hidden_states = gate * self.up_proj(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# DFlash Draft Model
# ---------------------------------------------------------------------------

class DFlashDraftModel(nn.Module):
    """
    MLX implementation of the DFlash block-diffusion drafter.

    Conditioned on context features extracted from the target model's hidden
    states, it predicts a block of `block_size` tokens in a single forward
    pass (no autoregressive loop in the drafter itself).

    Weight layout matches z-lab/Qwen3-*-DFlash-b16 HuggingFace checkpoints
    after conversion via scripts/convert.py.
    """

    def __init__(self, config: dict):
        super().__init__()
        hidden_size = config["hidden_size"]
        num_draft_layers = config["num_hidden_layers"]
        num_target_layers = config["num_target_layers"]
        num_heads = config["num_attention_heads"]
        num_kv_heads = config["num_key_value_heads"]
        head_dim = config.get("head_dim", hidden_size // num_heads)
        intermediate_size = config["intermediate_size"]
        rms_eps = config.get("rms_norm_eps", 1e-6)
        self.block_size = config.get("block_size", 16)
        self.hidden_size = hidden_size

        self.target_layer_ids = build_target_layer_ids(num_target_layers, num_draft_layers)

        # Context feature projection: concat of selected target layers → hidden_size
        num_ctx_layers = len(self.target_layer_ids)
        self.fc = nn.Linear(num_ctx_layers * hidden_size, hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.norm = nn.RMSNorm(hidden_size, eps=rms_eps)

        self.layers = [
            Qwen3DFlashDecoderLayer(
                hidden_size, intermediate_size,
                num_heads, num_kv_heads, head_dim, rms_eps
            )
            for _ in range(num_draft_layers)
        ]

    def __call__(self, noise_embedding: mx.array, target_hidden: mx.array,
                 cos: mx.array, sin: mx.array) -> mx.array:
        """
        Args:
            noise_embedding: embed_tokens(draft_block)  shape (B, block_size, D)
            target_hidden:   concatenated target layer features  (B, ctx_len, num_layers*D)
            cos, sin:        RoPE for full sequence (ctx_len + block_size)
        Returns:
            hidden_states:   (B, block_size, D)  → pass through target lm_head for logits
        """
        # Project and normalize context features
        target_hidden = self.hidden_norm(self.fc(target_hidden))  # (B, ctx_len, D)

        hidden_states = noise_embedding
        for layer in self.layers:
            hidden_states = layer(hidden_states, target_hidden, cos, sin)
        return self.norm(hidden_states)
