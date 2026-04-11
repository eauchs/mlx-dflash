"""
DFlash speculative decoding engine for Apple Silicon (MLX).
Requires patch_qwen3 to be imported first to expose hidden states.
"""

from __future__ import annotations
import time
import mlx.core as mx

from .models.qwen3_dflash import (
    DFlashDraftModel,
    build_rope_cache,
    extract_context_feature,
)


def greedy(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1)


class DFlashEngine:
    def __init__(self, target, draft: DFlashDraftModel, tokenizer,
                 block_size: int = 16, max_seq_len: int = 4096,
                 rope_base: float = 1_000_000.0):
        self.target = target
        self.draft = draft
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.cos, self.sin = build_rope_cache(max_seq_len, draft.head_dim,
                                              base=rope_base)
        self.mask_token_id = 151669  # from dflash_config

    def _rope_slice(self, start, length):
        return self.cos[start:start+length], self.sin[start:start+length]

    def _target_forward(self, input_ids, cache):
        logits = self.target(input_ids, cache=cache)
        hidden_states = self.target.model._last_hidden_states
        return logits, hidden_states

    def _lm_head(self, hidden):
        if self.target.args.tie_word_embeddings:
            return self.target.model.embed_tokens.as_linear(hidden)
        return self.target.lm_head(hidden)

    def generate(self, input_ids: mx.array, max_new_tokens: int = 512,
                 temperature: float = 0.0, verbose: bool = True) -> mx.array:
        block_size = self.block_size
        num_input = input_ids.shape[1]
        max_length = num_input + max_new_tokens
        output_tokens = input_ids[0].tolist()
        eos_id = self.tokenizer.eos_token_id
        mask_token_id = getattr(self, 'mask_token_id', eos_id)

        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
        cache = make_prompt_cache(self.target)

        # Prefill
        t0 = time.time()
        logits, all_hidden = self._target_forward(input_ids, cache)
        first_token = greedy(logits[:, -1:, :])
        mx.eval(first_token)
        output_tokens.append(int(first_token[0, 0]))
        ctx_feature = extract_context_feature(all_hidden, self.draft.target_layer_ids)
        if verbose:
            print(f"Prefill: {num_input} tokens in {time.time()-t0:.2f}s")

        # Decode
        start = len(output_tokens)
        acceptance_lengths = []
        t_decode = time.time()
        
        draft_cache = self.draft.make_cache()

        while start < max_length:
            cur_block = min(block_size, max_length - start + 1)
            draft_block = output_tokens[start-1:start] + [mask_token_id] * (cur_block - 1)
            draft_ids = mx.array([draft_block], dtype=mx.int32)

            # Draft forward
            noise_emb = self.target.model.embed_tokens(draft_ids)
            ctx_len = ctx_feature.shape[1]
            past_len = draft_cache[0].seq_len
            cos_f, sin_f = self._rope_slice(past_len, ctx_len + cur_block)
            draft_hidden = self.draft(noise_emb, ctx_feature, cos_f, sin_f, cache=draft_cache)
            
            for c in draft_cache:
                c.crop(start - 1)
                
            draft_logits = self._lm_head(draft_hidden[:, 1:, :])
            predicted = greedy(draft_logits)

            # Verify
            last_token = mx.array([output_tokens[start-1:start]], dtype=mx.int32)
            verify_ids = mx.concatenate([last_token, predicted], axis=1)
            
            verify_logits, verify_hidden = self._target_forward(verify_ids, cache)
            posterior = greedy(verify_logits)

            # Single sync
            eval_args = [posterior, predicted]
            if cache is not None:
                for c in cache:
                    if hasattr(c, 'keys') and c.keys is not None:
                        eval_args.extend([c.keys, c.values])
            for c in draft_cache:
                if c.k is not None:
                    eval_args.extend([c.k, c.v])
            mx.eval(*eval_args)

            p_list = posterior[0].tolist()
            pred_list = predicted[0].tolist()

            accept_len = 0
            for pred, ref in zip(pred_list, p_list[:-1]):
                if pred == ref:
                    accept_len += 1
                else:
                    break

            new_tokens = pred_list[:accept_len] + [p_list[accept_len]]
            output_tokens.extend(new_tokens)
            acceptance_lengths.append(accept_len + 1)
            
            valid_cache_len = start + accept_len
            start += accept_len + 1

            trim_amount = cache[0].offset - valid_cache_len
            if trim_amount > 0:
                trim_prompt_cache(cache, trim_amount)
                
            ctx_feature = extract_context_feature(
                verify_hidden, self.draft.target_layer_ids)[:, :accept_len+1, :]

            if eos_id in new_tokens:
                break

        decode_time = time.time() - t_decode
        generated = len(output_tokens) - num_input
        if verbose:
            avg = sum(acceptance_lengths) / max(len(acceptance_lengths), 1)
            tps = generated / max(decode_time, 1e-6)
            print(f"{generated} tokens | {tps:.1f} tok/s | "
                  f"avg accept {avg:.2f}/{block_size}")

        return mx.array([output_tokens], dtype=mx.int32)
