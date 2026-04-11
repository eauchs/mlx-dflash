"""
Convert z-lab DFlash draft model weights (PyTorch safetensors, BF16)
to MLX-compatible format.

Usage:
    python -m mlx_dflash.convert \
        --hf-model z-lab/Qwen3-8B-DFlash-b16 \
        --output mlx_models/qwen3-8b-dflash \
        --dtype bfloat16

Weight remapping (PyTorch → MLX key conventions):

  PyTorch (modeling_dflash.py)          MLX (qwen3_dflash.py)
  ─────────────────────────────────     ────────────────────────────────
  layers.N.self_attn.q_proj.weight  →  layers.N.self_attn.qkv_proj (packed, split)
  layers.N.self_attn.k_proj.weight  →  (packed into qkv_proj)
  layers.N.self_attn.v_proj.weight  →  (packed into qkv_proj)
  layers.N.self_attn.o_proj.weight  →  layers.N.self_attn.o_proj.weight
  layers.N.self_attn.q_norm.weight  →  layers.N.self_attn.q_norm.weight
  layers.N.self_attn.k_norm.weight  →  layers.N.self_attn.k_norm.weight
  layers.N.mlp.gate_proj.weight     →  layers.N.gate_proj.weight
  layers.N.mlp.up_proj.weight       →  layers.N.up_proj.weight
  layers.N.mlp.down_proj.weight     →  layers.N.down_proj.weight
  layers.N.input_layernorm.weight   →  layers.N.input_layernorm.weight
  layers.N.post_attn_layernorm      →  layers.N.post_attention_layernorm.weight
  norm.weight                       →  norm.weight
  fc.weight                         →  fc.weight
  hidden_norm.weight                →  hidden_norm.weight

Notes:
  - The context KV projection (kv_ctx_proj) does NOT exist in the original
    PyTorch model. In the original, k_proj/v_proj are applied to both
    target_hidden AND noise_hidden with shared weights. We split them:
      qkv_proj   = [q_proj | k_proj | v_proj] applied to draft tokens
      kv_ctx_proj = [k_proj | v_proj] applied to context tokens (same weights!)
    So kv_ctx_proj.weight = concat([k_proj.weight, v_proj.weight])
    This is a zero-cost weight duplication at conversion time.

  - Linear weights in MLX are stored transposed vs PyTorch convention.
    PyTorch: (out, in)  →  MLX: (in, out). We transpose on load.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np


DTYPE_MAP = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
}


def load_safetensors(path: Path) -> dict:
    """Load safetensors using torch (handles bfloat16 natively)."""
    import torch
    from safetensors.torch import load_file
    weights = load_file(str(path))
    return {k: v.float().numpy() for k, v in weights.items()}


def convert_weights(pt_weights: dict,
                    config: dict,
                    dtype=None) -> dict:
    """
    Remap PyTorch weight names to MLX names, pack QKV, transpose linears.
    """
    num_layers = config["num_hidden_layers"]
    mlx_weights: dict = {}

    def to_mlx(arr: np.ndarray):
        try:
            import mlx.core as mx_
            mlx_dtype = getattr(mx_, dtype) if isinstance(dtype, str) else dtype
            return mx_.array(arr).astype(mlx_dtype)
        except (ImportError, AttributeError):
            return arr  # fallback: return numpy array

    def transpose_linear(arr: np.ndarray) -> np.ndarray:
        """MLX Linear weight shape = (out, in), same as PyTorch. No transpose needed."""
        return arr  # no-op: shapes already match

    for layer_idx in range(num_layers):
        pt_prefix = f"layers.{layer_idx}"
        mx_prefix = f"layers.{layer_idx}"

        # ------------------------------------------------------------------
        # Pack Q + K + V into single qkv_proj (applied to draft/noise tokens)
        # ------------------------------------------------------------------
        q = pt_weights[f"{pt_prefix}.self_attn.q_proj.weight"]
        k = pt_weights[f"{pt_prefix}.self_attn.k_proj.weight"]
        v = pt_weights[f"{pt_prefix}.self_attn.v_proj.weight"]

        # Stack along output dim (axis=0 in PT = rows), then transpose for MLX
        qkv = np.concatenate([q, k, v], axis=0)         # (Q+K+V out, in_dim)
        mlx_weights[f"{mx_prefix}.self_attn.qkv_proj.weight"] = to_mlx(
            transpose_linear(qkv))

        # ------------------------------------------------------------------
        # kv_ctx_proj: same k/v weights applied to context tokens
        # (zero-cost duplication — same parameters, different input)
        # ------------------------------------------------------------------
        kv_ctx = np.concatenate([k, v], axis=0)          # (K+V out, in_dim)
        mlx_weights[f"{mx_prefix}.self_attn.kv_ctx_proj.weight"] = to_mlx(
            transpose_linear(kv_ctx))

        # ------------------------------------------------------------------
        # Output projection
        # ------------------------------------------------------------------
        o = pt_weights[f"{pt_prefix}.self_attn.o_proj.weight"]
        mlx_weights[f"{mx_prefix}.self_attn.o_proj.weight"] = to_mlx(
            transpose_linear(o))

        # ------------------------------------------------------------------
        # QK norms
        # ------------------------------------------------------------------
        mlx_weights[f"{mx_prefix}.self_attn.q_norm.weight"] = to_mlx(
            pt_weights[f"{pt_prefix}.self_attn.q_norm.weight"])
        mlx_weights[f"{mx_prefix}.self_attn.k_norm.weight"] = to_mlx(
            pt_weights[f"{pt_prefix}.self_attn.k_norm.weight"])

        # ------------------------------------------------------------------
        # MLP (SwiGLU)
        # ------------------------------------------------------------------
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            w = pt_weights[f"{pt_prefix}.mlp.{proj}.weight"]
            mlx_weights[f"{mx_prefix}.{proj}.weight"] = to_mlx(
                transpose_linear(w))

        # ------------------------------------------------------------------
        # Layer norms
        # ------------------------------------------------------------------
        mlx_weights[f"{mx_prefix}.input_layernorm.weight"] = to_mlx(
            pt_weights[f"{pt_prefix}.input_layernorm.weight"])
        mlx_weights[f"{mx_prefix}.post_attention_layernorm.weight"] = to_mlx(
            pt_weights[f"{pt_prefix}.post_attention_layernorm.weight"])

    # ----------------------------------------------------------------------
    # Top-level weights
    # ----------------------------------------------------------------------
    mlx_weights["norm.weight"] = to_mlx(pt_weights["norm.weight"])
    mlx_weights["fc.weight"] = to_mlx(
        transpose_linear(pt_weights["fc.weight"]))
    mlx_weights["hidden_norm.weight"] = to_mlx(pt_weights["hidden_norm.weight"])

    return mlx_weights


def convert(hf_model: str, output_dir: str, dtype: str = "bfloat16",
            cache_dir: Optional[str] = None):
    """
    Download (if needed) and convert z-lab DFlash model to MLX format.
    """
    from huggingface_hub import snapshot_download

    print(f"Downloading {hf_model} ...")
    local_dir = snapshot_download(hf_model, cache_dir=cache_dir,
                                   ignore_patterns=["*.pt", "*.bin", "flax_*"])
    local_dir = Path(local_dir)

    # Load config
    with open(local_dir / "config.json") as f:
        config = json.load(f)

    # Find safetensors file(s)
    st_files = list(local_dir.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors in {local_dir}")

    print(f"Loading {len(st_files)} safetensors shard(s) ...")
    pt_weights: dict[str, np.ndarray] = {}
    for f in st_files:
        pt_weights.update(load_safetensors(f))

    print(f"Converting {len(pt_weights)} tensors → MLX ({dtype}) ...")
    mlx_dtype = DTYPE_MAP[dtype]
    mlx_weights = convert_weights(pt_weights, config, mlx_dtype)

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save weights as numpy npz (MLX can load these directly)
    npz_path = out / "weights.npz"
    # Save as safetensors (supports bfloat16 natively)
    import mlx.core as mx
    st_path = out / "weights.safetensors"
    mx.save_safetensors(str(st_path), mlx_weights)
    print(f"Weights saved → {st_path}")

    # Save config
    mlx_config = {
        "hidden_size": config["hidden_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config["num_key_value_heads"],
        "head_dim": config.get("head_dim",
                               config["hidden_size"] // config["num_attention_heads"]),
        "intermediate_size": config["intermediate_size"],
        "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
        "block_size": config.get("block_size", 16),
        "num_target_layers": config["num_target_layers"],
        "rope_theta": config.get("rope_theta", 1_000_000.0),
    }
    with open(out / "config.json", "w") as f:
        json.dump(mlx_config, f, indent=2)
    print(f"Config saved → {out / 'config.json'}")
    print("Done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", default="z-lab/Qwen3-8B-DFlash-b16")
    p.add_argument("--output", required=True)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args()
    convert(args.hf_model, args.output, args.dtype, args.cache_dir)


if __name__ == "__main__":
    main()
