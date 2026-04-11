"""
Convert z-lab DFlash draft model weights (PyTorch safetensors, BF16)
to MLX-compatible format.
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import numpy as np

DTYPE_MAP = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
}

def load_safetensors_bf16(path: Path) -> dict:
    """
    Load safetensors safely. 
    If it's BF16, we load raw bytes and reconstruct float32 for numpy.
    """
    from safetensors import safe_open
    result = {}
    with safe_open(str(path), framework="numpy") as f:
        for k in f.keys():
            # Get the tensor
            t = f.get_tensor(k)
            
            # If numpy interpreted it as uint16 (standard for BF16 in some envs)
            # or if it's explicitly BF16, we convert to float32 for processing
            if hasattr(t, "dtype") and (t.dtype == np.uint16 or "float" not in t.dtype.name):
                # Manual bit conversion from BF16 to F32
                # BF16 is just the top 16 bits of an IEEE 754 float32
                raw = t.view(np.uint16).astype(np.uint32) << 16
                t = raw.view(np.float32).reshape(t.shape)
            result[k] = t
    return result

def convert_weights(pt_weights: dict, config: dict, dtype_str="bfloat16") -> dict:
    """Remap PyTorch weight names to MLX names, pack QKV, transpose linears."""
    num_layers = config["num_hidden_layers"]
    mlx_weights: dict = {}

    def to_mlx(arr: np.ndarray):
        try:
            import mlx.core as mx_
            # If we have MLX, we can use the actual bfloat16 type
            target_dtype = getattr(mx_, dtype_str)
            return mx_.array(arr).astype(target_dtype)
        except ImportError:
            return arr

    def transpose_linear(arr: np.ndarray) -> np.ndarray:
        """PyTorch Linear: (out, in) -> MLX: (in, out)."""
        return arr.T

    for layer_idx in range(num_layers):
        pt_p = f"layers.{layer_idx}"
        mx_p = f"layers.{layer_idx}"

        # QKV Packing: 3 matmuls -> 1 matmul (Apple Silicon optimization)
        q = pt_weights[f"{pt_p}.self_attn.q_proj.weight"]
        k = pt_weights[f"{pt_p}.self_attn.k_proj.weight"]
        v = pt_weights[f"{pt_p}.self_attn.v_proj.weight"]
        qkv = np.concatenate([q, k, v], axis=0)
        mlx_weights[f"{mx_p}.self_attn.qkv_proj.weight"] = to_mlx(transpose_linear(qkv))

        # KV Context projection (duplicate of K/V weights for context tokens)
        kv_ctx = np.concatenate([k, v], axis=0)
        mlx_weights[f"{mx_p}.self_attn.kv_ctx_proj.weight"] = to_mlx(transpose_linear(kv_ctx))

        # Attention Output & Norms
        mlx_weights[f"{mx_p}.self_attn.o_proj.weight"] = to_mlx(transpose_linear(pt_weights[f"{pt_p}.self_attn.o_proj.weight"]))
        mlx_weights[f"{mx_p}.self_attn.q_norm.weight"] = to_mlx(pt_weights[f"{pt_p}.self_attn.q_norm.weight"])
        mlx_weights[f"{mx_p}.self_attn.k_norm.weight"] = to_mlx(pt_weights[f"{pt_p}.self_attn.k_norm.weight"])

        # MLP (SwiGLU)
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            w = pt_weights[f"{pt_p}.mlp.{proj}.weight"]
            mlx_weights[f"{mx_p}.{proj}.weight"] = to_mlx(transpose_linear(w))

        # Layer Norms
        mlx_weights[f"{mx_p}.input_layernorm.weight"] = to_mlx(pt_weights[f"{pt_p}.input_layernorm.weight"])
        mlx_weights[f"{mx_p}.post_attention_layernorm.weight"] = to_mlx(pt_weights[f"{pt_p}.post_attention_layernorm.weight"])

    # Top-level weights
    mlx_weights["norm.weight"] = to_mlx(pt_weights["norm.weight"])
    mlx_weights["fc.weight"] = to_mlx(transpose_linear(pt_weights["fc.weight"]))
    mlx_weights["hidden_norm.weight"] = to_mlx(pt_weights["hidden_norm.weight"])

    return mlx_weights

def convert(hf_model: str, output_dir: str, dtype: str = "bfloat16", cache_dir: Optional[str] = None):
    from huggingface_hub import snapshot_download
    print(f"[*] Downloading {hf_model}...")
    local_dir = Path(snapshot_download(hf_model, cache_dir=cache_dir, ignore_patterns=["*.pt", "*.bin", "flax_*"]))

    with open(local_dir / "config.json") as f:
        config = json.load(f)

    st_files = list(local_dir.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors in {local_dir}")

    print(f"[*] Loading {len(st_files)} shards...")
    pt_weights = {}
    for f in st_files:
        pt_weights.update(load_safetensors_bf16(f))

    print(f"[*] Converting to MLX ({dtype})...")
    mlx_weights = convert_weights(pt_weights, config, dtype)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Save weights (Prefer .safetensors if MLX is available)
    try:
        import mlx.core as mx
        # Convert all to MLX arrays if they aren't already
        final_weights = {k: mx.array(v) if not isinstance(v, mx.array) else v for k, v in mlx_weights.items()}
        mx.save_safetensors(str(out / "weights.safetensors"), final_weights)
        print(f"[*] Weights saved -> {out / 'weights.safetensors'}")
    except ImportError:
        # Fallback to .npz
        np.savez(str(out / "weights.npz"), **{k: np.array(v) for k, v in mlx_weights.items()})
        print(f"[*] Weights saved -> {out / 'weights.npz'}")

    mlx_config = {
        "hidden_size": config["hidden_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config["num_key_value_heads"],
        "head_dim": config.get("head_dim", config["hidden_size"] // config["num_attention_heads"]),
        "intermediate_size": config["intermediate_size"],
        "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
        "block_size": config.get("block_size", 16),
        "num_target_layers": config["num_target_layers"],
        "rope_theta": config.get("rope_theta", 1000000.0),
    }
    with open(out / "config.json", "w") as f:
        json.dump(mlx_config, f, indent=2)
    print(f"[+] Done. Model converted to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    convert(args.hf_model, args.output, args.dtype, args.cache_dir)

if __name__ == "__main__":
    main()
