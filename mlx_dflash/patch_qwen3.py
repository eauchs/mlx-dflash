"""
Monkey-patch mlx-lm's Qwen3Model to expose intermediate hidden states.
Import this before loading the target model.

Usage:
    import mlx_dflash.patch_qwen3  # noqa
    from mlx_lm import load
    model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
    # model.model now has _last_hidden_states after each forward call
"""

import mlx.core as mx
from mlx_lm.models.qwen3 import Qwen3Model
from mlx_lm.models.base import create_attention_mask


def _patched_call(self, inputs, cache=None, input_embeddings=None):
    if input_embeddings is not None:
        h = input_embeddings
    else:
        h = self.embed_tokens(inputs)

    if cache is None:
        cache = [None] * len(self.layers)
    mask = create_attention_mask(h, cache[0])

    # Collect hidden states: [embed_output, layer0_out, layer1_out, ...]
    all_hidden = [h]
    for layer, c in zip(self.layers, cache):
        h = layer(h, mask, c)
        all_hidden.append(h)

    self._last_hidden_states = all_hidden  # store for DFlash engine
    return self.norm(h)


# Apply patch
Qwen3Model.__call__ = _patched_call
