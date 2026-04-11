"""
mlx-dflash generate CLI

Usage:
    python -m mlx_dflash.generate \
        --target mlx-community/Qwen3-8B-4bit \
        --draft  mlx_models/qwen3-8b-dflash \
        --prompt "Explain the Riemann hypothesis." \
        --max-tokens 512

Requires:
    pip install mlx mlx-lm huggingface-hub safetensors
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_dflash.models.qwen3_dflash import DFlashDraftModel
from mlx_dflash.engine import DFlashEngine


def load_draft(draft_dir: str) -> DFlashDraftModel:
    draft_dir = Path(draft_dir)
    with open(draft_dir / "config.json") as f:
        config = json.load(f)

    model = DFlashDraftModel(config)

    # Load weights
    weights = dict(np.load(str(draft_dir / "weights.npz")))
    mlx_weights = {k: mx.array(v) for k, v in weights.items()}
    model.load_weights(list(mlx_weights.items()))
    mx.eval(model.parameters())
    return model


def main():
    p = argparse.ArgumentParser(description="DFlash speculative decoding on Apple Silicon (MLX)")
    p.add_argument("--target", required=True,
                   help="Target model: HF repo or local path (mlx-lm format)")
    p.add_argument("--draft", required=True,
                   help="Draft model directory (converted via mlx_dflash.convert)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--no-chat-template", action="store_true")
    args = p.parse_args()

    # Load target model via mlx-lm
    try:
        from mlx_lm import load as mlx_load
    except ImportError:
        raise ImportError("pip install mlx-lm")

    print(f"Loading target model: {args.target}")
    target_model, tokenizer = mlx_load(args.target)
    target_model.eval()

    print(f"Loading draft model: {args.draft}")
    draft_model = load_draft(args.draft)
    draft_model.eval()

    # Tokenize prompt
    if args.no_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        text = args.prompt
    else:
        messages = [{"role": "user", "content": args.prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

    input_ids = mx.array([tokenizer.encode(text)], dtype=mx.int32)
    print(f"Prompt: {input_ids.shape[1]} tokens")

    # Build engine
    engine = DFlashEngine(
        target=target_model,
        draft=draft_model,
        tokenizer=tokenizer,
        block_size=args.block_size,
    )

    # Generate
    t0 = time.time()
    output_ids = engine.generate(
        input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
        verbose=True,
    )
    total = time.time() - t0

    # Decode
    prompt_len = input_ids.shape[1]
    generated_ids = output_ids[0, prompt_len:].tolist()
    print("\n" + "─" * 60)
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))
    print("─" * 60)
    print(f"Total wall time: {total:.2f}s")


if __name__ == "__main__":
    main()
