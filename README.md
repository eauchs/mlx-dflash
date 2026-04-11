# mlx-dflash

Native MLX implementation of [DFlash](https://arxiv.org/abs/2602.06036) speculative decoding for Apple Silicon.

> **DFlash**: Block Diffusion for Flash Speculative Decoding — Chen et al., 2026

A small block-diffusion draft model generates 16 tokens **in parallel** (one forward pass). The target verifies them in one pass. Output is bit-for-bit identical to greedy baseline.

No CUDA. No PyTorch. Pure MLX.

---

## Results (M3 Max, 128GB, mlx-lm)

> Benchmarks pending — hardware available, running this week.
> Community results welcome via Issues.

Reference from No_Shift_4543 (M5 Max, 64GB):

| Model | DFlash | Baseline | Speedup |
|---|---|---|---|
| Qwen3.5-9B bf16 | 85 tok/s | 26 tok/s | **3.3×** |
| Qwen3.5-4B bf16 | 109 tok/s | 41 tok/s | **2.7×** |
| Qwen3.5-27B 8bit | 35 tok/s | 14 tok/s | **2.5×** |

---

## Install

```bash
pip install git+https://github.com/eauchs/mlx-dflash
```

Or from source:

```bash
git clone https://github.com/eauchs/mlx-dflash
cd mlx-dflash
pip install -e .
```

---

## Quick Start

### 1. Convert draft model weights

```bash
mlx-dflash-convert \
  --hf-model z-lab/Qwen3-8B-DFlash-b16 \
  --output   mlx_models/qwen3-8b-dflash \
  --dtype    bfloat16
```

Available draft models from z-lab:
- `z-lab/Qwen3-8B-DFlash-b16`
- `z-lab/Qwen3-4B-DFlash-b16`
- `z-lab/Qwen3.5-27B-DFlash-b16`
- `z-lab/Qwen3.5-35B-A3B-DFlash` (MoE)

### 2. Generate

```bash
mlx-dflash \
  --target mlx-community/Qwen3-8B-4bit \
  --draft  mlx_models/qwen3-8b-dflash \
  --prompt "Explain the Riemann hypothesis." \
  --max-tokens 512
```

### 3. Python API

```python
from mlx_lm import load
from mlx_dflash import DFlashDraftModel, DFlashEngine
import mlx.core as mx
import json, numpy as np

# Load target
target, tokenizer = load("mlx-community/Qwen3-8B-4bit")

# Load draft
with open("mlx_models/qwen3-8b-dflash/config.json") as f:
    config = json.load(f)
draft = DFlashDraftModel(config)
weights = {k: mx.array(v) for k, v in np.load("mlx_models/qwen3-8b-dflash/weights.npz").items()}
draft.load_weights(list(weights.items()))

# Generate
engine = DFlashEngine(target, draft, tokenizer, block_size=16)
ids = mx.array([tokenizer.encode("Hello, world!")], dtype=mx.int32)
out = engine.generate(ids, max_new_tokens=256)
print(tokenizer.decode(out[0].tolist()))
```

---

## Architecture

```
Target model (e.g. Qwen3-8B)
│
├─ Prefill → KV cache + hidden states [L0..LN]
│
├─ Extract context features from layers [L2, L14, L27] → fc → (B, ctx, D)
│
└─ Decode loop:
   ┌─ Draft model (2-4 lightweight layers, ~1B params)
   │   input:  embed(draft_block)   ← noise tokens (current block + masks)
   │   cond:   context features from target hidden states
   │   attn:   bidirectional over [context | draft_block]  (no causal mask)
   │   output: hidden_states → target.lm_head → 16 logits
   │
   ├─ Verify: target(draft_block) → 16 posterior logits
   │
   └─ Accept: greedy cumprod match → advance by (accept_len + 1)
```

The drafter is conditioned on hidden states extracted from **multiple target layers**,
not just the last one. This gives the tiny drafter access to the target's reasoning
without needing to replicate its depth.

---

## Apple Silicon Notes

Lessons from implementing this on unified memory hardware:

**What works:**
- Packed QKV projection (1 matmul + split instead of 3) → fewer kernel dispatches
- Single `mx.eval()` per decode step (sync elision) → saves ~0.5ms at 80+ tok/s
- `head_dim=256` compatible with MLX's `steel_attention` fast path

**What doesn't:**
- Custom Metal kernels for GEMV/SiLU/SDPA came back 0.5-0.8× slower than stock MLX steel GEMM
- "Verify fewer tokens when confidence is low" doesn't help — weight loading dominates, not token count
- On quantized targets (int4), the bf16 draft becomes the bottleneck (opposite of bf16 case)

---

## Roadmap

- [ ] Benchmarks on M3 Max 128GB (in progress)
- [ ] LLaMA-3.1-8B support
- [ ] Draft model quantization (fix bf16 draft bottleneck on int4 targets)
- [ ] Long context stability (speedup degrades past 2K — KV cache growth)
- [ ] MoE: Qwen3.5-35B-A3B

---

## Citation

```bibtex
@misc{chen2026dflash,
  title         = {DFlash: Block Diffusion for Flash Speculative Decoding},
  author        = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  year          = {2026},
  eprint        = {2602.06036},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
}
```

---

## License

MIT. Draft model weights from z-lab are MIT licensed.
This repo is an independent MLX port — not affiliated with Z Lab.
