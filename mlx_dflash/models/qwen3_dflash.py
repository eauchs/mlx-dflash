"""
MLX port of DFlash draft model — with draft KV cache.
Source: z-lab/Qwen3-8B-DFlash-b16/dflash.py (MIT)
Paper: arXiv:2602.06036
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List


def rotate_half(x):
    h = x.shape[-1] // 2
    return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    q: (B, H, q_len, D)
    k: (B, H, ctx_len+q_len, D)
    cos, sin: (1, 1, ctx_len+q_len, D)
    q gets last q_len positions, k gets all positions.
    """
    q_len = q.shape[2]
    q_out = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


def build_rope_cache(seq_len, head_dim, base=1_000_000.0, dtype=mx.bfloat16):
    inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    t = mx.arange(seq_len, dtype=mx.float32)
    freqs = mx.outer(t, inv_freq)
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


def build_target_layer_ids(num_target_layers, num_draft_layers):
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    span = end - start
    return [int(round(start + (i * span) / (num_draft_layers - 1)))
            for i in range(num_draft_layers)]


def extract_context_feature(hidden_states, layer_ids):
    offset = 1
    selected = [hidden_states[lid + offset] for lid in layer_ids]
    return mx.concatenate(selected, axis=-1)


class DraftKVCache:
    """Simple KV cache for the draft model."""
    def __init__(self):
        self.k: Optional[mx.array] = None
        self.v: Optional[mx.array] = None

    def update(self, new_k, new_v):
        if self.k is None:
            self.k, self.v = new_k, new_v
        else:
            self.k = mx.concatenate([self.k, new_k], axis=2)
            self.v = mx.concatenate([self.v, new_v], axis=2)
        return self.k, self.v

    def crop(self, seq_len):
        if self.k is not None:
            self.k = self.k[:, :, :seq_len, :]
            self.v = self.v[:, :, :seq_len, :]

    @property
    def seq_len(self):
        return 0 if self.k is None else self.k.shape[2]


class Qwen3DFlashAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, rms_eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv_proj = nn.Linear(hidden_size, (num_heads + 2 * num_kv_heads) * head_dim, bias=False)
        self.kv_ctx_proj = nn.Linear(hidden_size, 2 * num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=rms_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=rms_eps)

    def __call__(self, hidden_states, target_hidden, cos, sin, cache=None):
        B, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        qkv = self.qkv_proj(hidden_states)
        kv_ctx = self.kv_ctx_proj(target_hidden)

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        q, k_draft, v_draft = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)
        k_ctx, v_ctx = mx.split(kv_ctx, [kv_size], axis=-1)

        q = q.reshape(B, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.q_norm(q)

        k = mx.concatenate([k_ctx, k_draft], axis=1)
        v = mx.concatenate([v_ctx, v_draft], axis=1)

        k = k.reshape(B, ctx_len + q_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, ctx_len + q_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_norm(k)

        # RoPE on full k (ctx + draft)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Draft KV cache
        if cache is not None:
            k, v = cache.update(k, v)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, q_len, -1)
        return self.o_proj(out)


class Qwen3DFlashDecoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, num_kv_heads,
                 head_dim, rms_eps=1e-6):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.self_attn = Qwen3DFlashAttention(
            hidden_size, num_heads, num_kv_heads, head_dim, rms_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.mlp_gate_up = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.mlp_down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, hidden_states, target_hidden, cos, sin, cache=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, target_hidden, cos, sin, cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate_up = self.mlp_gate_up(hidden_states)
        gate, up = mx.split(gate_up, 2, axis=-1)
        hidden_states = self.mlp_down(nn.silu(gate) * up)
        return residual + hidden_states


class DFlashDraftModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size       = config["hidden_size"]
        num_draft_layers  = config["num_hidden_layers"]
        num_target_layers = config["num_target_layers"]
        num_heads         = config["num_attention_heads"]
        num_kv_heads      = config["num_key_value_heads"]
        head_dim          = config.get("head_dim", hidden_size // num_heads)
        intermediate_size = config["intermediate_size"]
        rms_eps           = config.get("rms_norm_eps", 1e-6)

        self.block_size    = config.get("block_size", 16)
        self.hidden_size   = hidden_size
        self.head_dim      = head_dim
        self.mask_token_id = 151669

        dflash_cfg = config.get("dflash_config", {})
        self.target_layer_ids = dflash_cfg.get(
            "target_layer_ids",
            build_target_layer_ids(num_target_layers, num_draft_layers))

        num_ctx = len(self.target_layer_ids)
        self.fc          = nn.Linear(num_ctx * hidden_size, hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.norm        = nn.RMSNorm(hidden_size, eps=rms_eps)

        self.layers = [
            Qwen3DFlashDecoderLayer(
                hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, rms_eps)
            for _ in range(num_draft_layers)
        ]

    def make_cache(self):
        return [DraftKVCache() for _ in self.layers]

    def load_weights_from_original(self, pt_weights):
        def to_mx(t):
            return mx.array(t.float().numpy(), dtype=mx.bfloat16)

        weights = {
            "fc.weight":          to_mx(pt_weights["fc.weight"]),
            "hidden_norm.weight": to_mx(pt_weights["hidden_norm.weight"]),
            "norm.weight":        to_mx(pt_weights["norm.weight"]),
        }
        for i in range(len(self.layers)):
            p = f"layers.{i}"
            weights[f"layers.{i}.input_layernorm.weight"]          = to_mx(pt_weights[f"{p}.input_layernorm.weight"])
            weights[f"layers.{i}.post_attention_layernorm.weight"] = to_mx(pt_weights[f"{p}.post_attention_layernorm.weight"])
            
            q = pt_weights[f"{p}.self_attn.q_proj.weight"]
            k = pt_weights[f"{p}.self_attn.k_proj.weight"]
            v = pt_weights[f"{p}.self_attn.v_proj.weight"]
            import torch
            weights[f"layers.{i}.self_attn.qkv_proj.weight"]       = to_mx(torch.cat([q, k, v], dim=0))
            weights[f"layers.{i}.self_attn.kv_ctx_proj.weight"]    = to_mx(torch.cat([k, v], dim=0))
            
            weights[f"layers.{i}.self_attn.o_proj.weight"]         = to_mx(pt_weights[f"{p}.self_attn.o_proj.weight"])
            weights[f"layers.{i}.self_attn.q_norm.weight"]         = to_mx(pt_weights[f"{p}.self_attn.q_norm.weight"])
            weights[f"layers.{i}.self_attn.k_norm.weight"]         = to_mx(pt_weights[f"{p}.self_attn.k_norm.weight"])
            
            gate = pt_weights[f"{p}.mlp.gate_proj.weight"]
            up = pt_weights[f"{p}.mlp.up_proj.weight"]
            weights[f"layers.{i}.mlp_gate_up.weight"]              = to_mx(torch.cat([gate, up], dim=0))
            weights[f"layers.{i}.mlp_down.weight"]                 = to_mx(pt_weights[f"{p}.mlp.down_proj.weight"])

        self.load_weights(list(weights.items()))

    def __call__(self, noise_embedding, target_hidden_concat, cos, sin, cache=None):
        ctx = self.hidden_norm(self.fc(target_hidden_concat))
        h = noise_embedding
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h = layer(h, ctx, cos, sin, layer_cache)
        return self.norm(h)
