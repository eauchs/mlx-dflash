"""
DFlash benchmark: mesure tok/s DFlash vs baseline mlx-lm.

Usage:
    PYTHONPATH=. ~/.venv/bin/python3 scripts/benchmark.py \
        --target mlx-community/Qwen3-8B-4bit \
        --draft  mlx_models/qwen3-8b-dflash \
        --gen-lengths 512 1024
"""

import argparse
import json
import time
from pathlib import Path
import mlx.core as mx

# Patch MUST come before mlx_lm import
import mlx_dflash.patch_qwen3  # noqa: F401

from mlx_lm import load, generate as mlx_generate
from mlx_dflash.models.qwen3_dflash import DFlashDraftModel
from mlx_dflash.engine import DFlashEngine


def load_draft(draft_path: str) -> DFlashDraftModel:
    d = Path(draft_path)
    with open(d / "config.json") as f:
        config = json.load(f)
    model = DFlashDraftModel(config)
    # Load from original z-lab weights directly
    from safetensors.torch import load_file
    import glob, os
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    files = glob.glob(f"{cache}/**/models--z-lab--Qwen3-8B-DFlash-b16/**/model.safetensors", recursive=True)
    if not files:
        raise FileNotFoundError("z-lab weights not found in HF cache")
    pt_weights = load_file(files[0])
    model.load_weights_from_original(pt_weights)
    mx.eval(model.parameters())
    return model


def run_baseline(target_model, tokenizer, prompt: str, max_tokens: int) -> float:
    t0 = time.perf_counter()
    out = mlx_generate(target_model, tokenizer, prompt=prompt,
                       max_tokens=max_tokens, verbose=False)
    dt = time.perf_counter() - t0
    generated = len(tokenizer.encode(out)) 
    return generated / max(dt, 1e-6)


def run_dflash(engine: DFlashEngine, tokenizer, prompt: str,
               max_tokens: int) -> float:
    ids = mx.array([tokenizer.encode(prompt)], dtype=mx.int32)
    t0 = time.perf_counter()
    out = engine.generate(ids, max_new_tokens=max_tokens, verbose=True)
    dt = time.perf_counter() - t0
    generated = out.shape[1] - ids.shape[1]
    return generated / max(dt, 1e-6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--prompt", default=(
        "Explain in detail the key architectural differences between "
        "transformer-based and diffusion-based language models."))
    p.add_argument("--gen-lengths", nargs="+", type=int, default=[512, 1024])
    p.add_argument("--runs", type=int, default=2)
    args = p.parse_args()

    print(f"Loading target: {args.target}")
    target, tokenizer = load(args.target)

    print(f"Loading draft: {args.draft}")
    draft = load_draft(args.draft)

    engine = DFlashEngine(target, draft, tokenizer, block_size=16)

    print(f"\n{'Model':<30} {'Gen':<8} {'Baseline':>12} {'DFlash':>12} {'Speedup':>10}")
    print("-" * 76)

    model_name = Path(args.target).name

    for gen_len in args.gen_lengths:
        # Warm-up
        run_baseline(target, tokenizer, args.prompt, 32)
        run_dflash(engine, tokenizer, args.prompt, 32)

        baseline_speeds, dflash_speeds = [], []
        for _ in range(args.runs):
            baseline_speeds.append(run_baseline(target, tokenizer, args.prompt, gen_len))
            dflash_speeds.append(run_dflash(engine, tokenizer, args.prompt, gen_len))

        b = sum(baseline_speeds) / len(baseline_speeds)
        d = sum(dflash_speeds) / len(dflash_speeds)
        sp = d / max(b, 1e-6)

        print(f"{model_name:<30} {gen_len:<8} {b:>10.1f}t/s {d:>10.1f}t/s {sp:>9.2f}x")

    print("\nPaste these results into README.md")


if __name__ == "__main__":
    main()
