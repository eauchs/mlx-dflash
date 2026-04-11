"""Tests for mlx-dflash. Run: python -m pytest tests/ -v"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

requires_mlx = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (macOS only)")

SMALL_CONFIG = dict(hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=32, intermediate_size=256,
                    rms_norm_eps=1e-6, block_size=4, num_target_layers=12)

@requires_mlx
def test_target_layer_ids():
    from mlx_dflash.models.qwen3_dflash import build_target_layer_ids
    ids = build_target_layer_ids(36, 3)
    assert len(ids) == 3 and ids[0] < ids[1] < ids[2]

@requires_mlx
def test_rope_cache_shape():
    from mlx_dflash.models.qwen3_dflash import build_rope_cache
    cos, sin = build_rope_cache(128, 32)
    assert cos.shape == (128, 32) and sin.shape == (128, 32)

@requires_mlx
def test_model_forward_shape():
    from mlx_dflash.models.qwen3_dflash import DFlashDraftModel, build_rope_cache
    m = DFlashDraftModel(SMALL_CONFIG)
    B, ctx, block, D, nd = 1, 16, 4, 128, 2
    out = m(mx.zeros((B,block,D)), mx.zeros((B,ctx,nd*D)), *build_rope_cache(ctx+block,32))
    mx.eval(out)
    assert out.shape == (B, block, D)

@requires_mlx
def test_kv_cache():
    from mlx_dflash.engine import KVCache
    c = KVCache()
    c.update(mx.zeros((1,4,20,32)), mx.zeros((1,4,20,32)))
    assert c.seq_len == 20
    c.crop(10)
    assert c.seq_len == 10


def test_convert_weight_remapping():
    """Pure numpy — runs anywhere, validates key remapping & shapes."""
    # Import only the function, not the module (avoids mx at module level)
    import importlib.util, types

    # Load convert.py as a module without executing top-level mx imports
    spec = importlib.util.spec_from_file_location(
        "conv", os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "mlx_dflash", "convert.py"))
    mod = importlib.util.module_from_spec(spec)

    # Inject a fake mx so the DTYPE_MAP line doesn't fail
    fake_mx = types.SimpleNamespace(float32=np.float32, float16=np.float16,
                                    bfloat16=np.float32)
    sys.modules["mlx"] = types.ModuleType("mlx")
    sys.modules["mlx.core"] = fake_mx  # type: ignore

    spec.loader.exec_module(mod)

    D, H, KVH, HD = 64, 4, 2, 16
    config = dict(hidden_size=D, num_hidden_layers=1,
                  num_attention_heads=H, num_key_value_heads=KVH, num_target_layers=12)
    pt = {
        "layers.0.self_attn.q_proj.weight": np.zeros((H*HD,D),np.float32),
        "layers.0.self_attn.k_proj.weight": np.zeros((KVH*HD,D),np.float32),
        "layers.0.self_attn.v_proj.weight": np.zeros((KVH*HD,D),np.float32),
        "layers.0.self_attn.o_proj.weight": np.zeros((D,H*HD),np.float32),
        "layers.0.self_attn.q_norm.weight": np.zeros((HD,),np.float32),
        "layers.0.self_attn.k_norm.weight": np.zeros((HD,),np.float32),
        "layers.0.mlp.gate_proj.weight":    np.zeros((256,D),np.float32),
        "layers.0.mlp.up_proj.weight":      np.zeros((256,D),np.float32),
        "layers.0.mlp.down_proj.weight":    np.zeros((D,256),np.float32),
        "layers.0.input_layernorm.weight":  np.zeros((D,),np.float32),
        "layers.0.post_attention_layernorm.weight": np.zeros((D,),np.float32),
        "norm.weight": np.zeros((D,),np.float32),
        "fc.weight":   np.zeros((D,D),np.float32),
        "hidden_norm.weight": np.zeros((D,),np.float32),
    }

    # Patch to_mlx in the loaded module to return numpy arrays
    w = mod.convert_weights(pt, config, np.float32)

    assert "layers.0.self_attn.qkv_proj.weight" in w, "missing qkv_proj"
    assert "layers.0.self_attn.kv_ctx_proj.weight" in w, "missing kv_ctx_proj"

    qkv = w["layers.0.self_attn.qkv_proj.weight"]
    kv_ctx = w["layers.0.self_attn.kv_ctx_proj.weight"]

    # After transpose: (in_dim, out_dim)
    assert qkv.shape == (D, (H+2*KVH)*HD), f"qkv shape wrong: {qkv.shape}"
    assert kv_ctx.shape == (D, 2*KVH*HD), f"kv_ctx shape wrong: {kv_ctx.shape}"

    for k in ["norm.weight", "fc.weight", "hidden_norm.weight"]:
        assert k in w, f"missing {k}"
