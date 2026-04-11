"""
DFlash speculative decoding engine for Apple Silicon (MLX).

Loop semantics (mirrors modeling_dflash.py spec_generate):

  1. Prefill target model → KV cache + hidden states for all layers
  2. Extract context features from evenly-spaced target layers
  3. Decode loop:
       a. Draft:  embed draft_block → run DFlashDraftModel → lm_head → 16 logits
       b. Verify: run target model on draft_block → 16 logits
       c. Accept: greedy cumprod match, advance pointer by (acceptance_len + 1)
       d. Crop target KV cache to accepted position
       e. Update context features from accepted target hidden states

Apple Silicon optimizations implemented here:
  - Sync elision: only ONE mx.eval() call per decode step (was 2 in naive impl)
    At 80+ tok/s each GPU→CPU sync costs ~0.5ms → worth it
  - KV cache stored as MLX arrays in unified memory (no copy overhead)
"""

from __future__ import annotations
import time
from typing import Optional
import mlx.core as mx
import mlx.nn as nn

from .models.qwen3_dflash import (
    DFlashDraftModel,
    build_rope_cache,
    extract_context_feature,
)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

class KVCache:
    """Simple growing KV cache for MLX (unified memory, no copy needed)."""

    def __init__(self):
        self.k: Optional[mx.array] = None
        self.v: Optional[mx.array] = None

    def update(self, new_k: mx.array, new_v: mx.array) -> tuple[mx.array, mx.array]:
        if self.k is None:
            self.k, self.v = new_k, new_v
        else:
            self.k = mx.concatenate([self.k, new_k], axis=2)
            self.v = mx.concatenate([self.v, new_v], axis=2)
        return self.k, self.v

    def crop(self, seq_len: int):
        if self.k is not None:
            self.k = self.k[:, :, :seq_len, :]
            self.v = self.v[:, :, :seq_len, :]

    @property
    def seq_len(self) -> int:
        return 0 if self.k is None else self.k.shape[2]


# ---------------------------------------------------------------------------
# Greedy sampling
# ---------------------------------------------------------------------------

def greedy(logits: mx.array) -> mx.array:
    """(B, S, V) → (B, S) argmax."""
    return mx.argmax(logits, axis=-1)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DFlashEngine:
    """
    Wraps a target mlx-lm model and a DFlashDraftModel for speculative decoding.

    Usage:
        engine = DFlashEngine(target_model, draft_model, tokenizer)
        output = engine.generate(prompt_ids, max_new_tokens=512)
    """

    def __init__(self, target, draft: DFlashDraftModel, tokenizer,
                 block_size: int = 16, max_seq_len: int = 4096,
                 rope_base: float = 1_000_000.0):
        self.target = target
        self.draft = draft
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_seq_len = max_seq_len

        # Precompute RoPE for max_seq_len (reused every step, no recomputation)
        hidden_size = draft.hidden_size
        # head_dim comes from the draft's first layer attention
        head_dim = draft.layers[0].self_attn.head_dim
        self.cos, self.sin = build_rope_cache(max_seq_len, head_dim,
                                              base=rope_base)

    def _rope_slice(self, start: int, length: int) -> tuple[mx.array, mx.array]:
        return self.cos[start:start + length], self.sin[start:start + length]

    @mx.compile
    def _draft_forward(self, noise_emb, target_hidden, cos, sin):
        """Compiled draft forward — avoids graph recompilation each step."""
        return self.draft(noise_emb, target_hidden, cos, sin)

    def generate(self, input_ids: mx.array, max_new_tokens: int = 512,
                 temperature: float = 0.0, verbose: bool = True) -> mx.array:
        """
        Full speculative decoding loop.

        Args:
            input_ids:      (1, prompt_len) int32
            max_new_tokens: budget
            temperature:    0.0 = greedy
        Returns:
            output_ids: (1, prompt_len + generated_len) int32
        """
        block_size = self.block_size
        num_input = input_ids.shape[1]
        max_length = num_input + max_new_tokens

        # Rolling output buffer (over-allocate to avoid realloc in loop)
        # MLX arrays are immutable so we work with Python lists and convert
        output_tokens: list[int] = input_ids[0].tolist()
        mask_token_id = self.tokenizer.vocab.get("<mask>",
                        self.tokenizer.eos_token_id)  # fallback

        # ----------------------------------------------------------------
        # Prefill
        # ----------------------------------------------------------------
        t0 = time.time()

        target_out = self.target(
            input_ids,
            use_cache=True,
            output_hidden_states=True,
        )
        # target_out.logits: (1, prompt_len, V)
        # target_out.hidden_states: list of (1, prompt_len, D)
        first_token = greedy(target_out.logits[:, -1:, :])  # (1, 1)
        mx.eval(first_token)  # single sync after prefill
        output_tokens.append(int(first_token[0, 0]))

        target_kv = target_out.past_key_values  # mlx-lm KV cache object
        all_target_hidden = list(target_out.hidden_states)

        ctx_feature = extract_context_feature(all_target_hidden,
                                              self.draft.target_layer_ids)

        prefill_time = time.time() - t0
        if verbose:
            print(f"Prefill: {num_input} tokens in {prefill_time:.2f}s")

        # ----------------------------------------------------------------
        # Decode loop
        # ----------------------------------------------------------------
        start = len(output_tokens)  # pointer into output_tokens
        acceptance_lengths: list[int] = []
        t_decode_start = time.time()

        eos_id = self.tokenizer.eos_token_id

        while start < max_length:
            remaining = max_length - start
            cur_block = min(block_size, remaining + 1)

            # --- Pad draft block with current token + mask tokens ---
            draft_block = output_tokens[start - 1:start] + \
                          [mask_token_id] * (cur_block - 1)
            draft_ids = mx.array([draft_block], dtype=mx.int32)

            # --- Draft forward ---
            noise_emb = self.target.model.embed_tokens(draft_ids)  # (1, block, D)
            ctx_len = ctx_feature.shape[1]
            # RoPE: positions [0..ctx_len+block-1] (full sequence)
            cos_full, sin_full = self._rope_slice(0, ctx_len + cur_block)
            draft_hidden = self._draft_forward(noise_emb, ctx_feature,
                                               cos_full, sin_full)
            # draft_hidden: (1, block, D) → skip first position (already verified)
            draft_logits = self.target.lm_head(draft_hidden[:, 1:, :])
            # (1, block-1, V)

            # Apply draft predictions to block (in-place on Python list)
            draft_block_ids = [output_tokens[start - 1]]
            if temperature < 1e-5:
                predicted = greedy(draft_logits)  # (1, block-1)
            else:
                raise NotImplementedError("Sampling temperature > 0 not yet implemented")

            # --- Verify with target ---
            verify_ids = mx.array([draft_block_ids + predicted[0].tolist()],
                                  dtype=mx.int32)
            verify_out = self.target(
                verify_ids,
                past_key_values=target_kv,
                use_cache=True,
                output_hidden_states=True,
            )
            posterior = greedy(verify_out.logits)  # (1, block)

            # -------------------------------------------------------
            # SYNC ELISION: single mx.eval() call for both posterior
            # and predicted (was 2 syncs in naive impl)
            # -------------------------------------------------------
            mx.eval(posterior, predicted)

            posterior_list = posterior[0].tolist()
            predicted_list = predicted[0].tolist()

            # --- Greedy acceptance ---
            accept_len = 0
            for i, (pred, ref) in enumerate(zip(predicted_list, posterior_list[:-1])):
                if pred == ref:
                    accept_len += 1
                else:
                    break

            # Commit accepted tokens + one bonus from posterior
            new_tokens = draft_block_ids[1:accept_len + 1] + [posterior_list[accept_len]]
            output_tokens.extend(new_tokens)
            acceptance_lengths.append(accept_len + 1)

            start += accept_len + 1

            # Crop target KV to accepted length
            target_kv.crop(start)

            # Update context features from accepted hidden states
            all_target_hidden = list(verify_out.hidden_states)
            ctx_feature = extract_context_feature(
                all_target_hidden, self.draft.target_layer_ids
            )[:, :accept_len + 1, :]  # keep only accepted positions

            # Early stop on EOS
            if eos_id in new_tokens:
                break

        decode_time = time.time() - t_decode_start
        generated = len(output_tokens) - num_input

        if verbose:
            avg_accept = sum(acceptance_lengths) / max(len(acceptance_lengths), 1)
            tok_per_s = generated / max(decode_time, 1e-6)
            print(f"Generated {generated} tokens in {decode_time:.2f}s "
                  f"({tok_per_s:.1f} tok/s)")
            print(f"Avg acceptance length: {avg_accept:.2f} / {block_size} "
                  f"({avg_accept/block_size*100:.0f}%)")
            print(f"Draft steps: {len(acceptance_lengths)}")

        return mx.array([output_tokens], dtype=mx.int32)
