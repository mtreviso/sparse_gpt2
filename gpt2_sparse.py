import math
from functools import partial

import torch
import torch.nn as nn
from entmax import entmax15, sparsemax, entmax_bisect
from transformers.models.gpt2 import GPT2Model, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


class SparseGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, alpha=1.0, save_qk=True):
        super().__init__(config)
        print('Using SparseGPT2LMHeadModel with: alpha={}, save_qk={}'.format(alpha, save_qk))
        for i, layer in enumerate(self.transformer.h):
            layer.attn = SparseGPT2Attention(config, alpha=alpha, save_qk=save_qk)
        self.init_weights()  # reinit weights (important if we want to train from scratch)


class SparseGPT2Model(GPT2Model):
    def __init__(self, config, alpha=1.5, save_qk=False):
        super().__init__(config)
        print('Using SparseGPT2Model with: alpha={}, save_qk={}'.format(alpha, save_qk))
        for i, layer in enumerate(self.h):
            layer.attn = SparseGPT2Attention(config, alpha=alpha, save_qk=save_qk)
        self.init_weights()  # reinit weights (important if we want to train from scratch)


class SparseGPT2Attention(GPT2Attention):
    def __init__(self, config, alpha=1.5, save_qk=False):
        super().__init__(config)
        if alpha == 1.0:
            self.transform_fn = torch.softmax
        elif alpha == 1.5:
            self.transform_fn = entmax15
        elif alpha == 2.0:
            self.transform_fn = sparsemax
        else:
            self.transform_fn = partial(entmax_bisect, alpha=alpha)
        self.alpha = alpha
        self.save_qk = save_qk
        self.q_vectors = None
        self.k_vectors = None

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # EDITED: save q, k vectors
        if self.save_qk:
            self.q_vectors = query
            self.k_vectors = key

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        # EDITED: Apply the entmax transformation
        attn_weights = self.transform_fn(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights