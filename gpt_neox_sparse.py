import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
from entmax import entmax15, sparsemax, entmax_bisect
from transformers.models.gpt_neox import GPTNeoXModel, GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, GPTNeoXFlashAttention2, \
    apply_rotary_pos_emb


class SparseGPTNeoXForCausalLM(GPTNeoXForCausalLM):
    def __init__(self, config, alpha=1.0, save_qk=True, reinit_weights=False):
        super().__init__(config)
        print('Using SparseGPTNeoXForCausalLM with: alpha={}, save_qk={}'.format(alpha, save_qk))
        for i, layer in enumerate(self.gpt_neox.layers):
            layer.attention = SparseGPTNeoSelfAttention(config, alpha=alpha, save_qk=save_qk)
        # reinit weights (important if we want to train from scratch)
        if reinit_weights:
            self.init_weights()


class SparseGPTNeoXModel(GPTNeoXModel):
    def __init__(self, config, alpha=1.0, save_qk=False, reinit_weights=False):
        super().__init__(config)
        print('Using SparseGPTNeoXModel with: alpha={}, save_qk={}'.format(alpha, save_qk))
        for i, layer in enumerate(self.h):
            layer.attention = SparseGPTNeoSelfAttention(config, alpha=alpha, save_qk=save_qk)
        # reinit weights (important if we want to train from scratch)
        if reinit_weights:
            self.init_weights()


class SparseGPTNeoSelfAttention(GPTNeoXAttention):
    def __init__(self, config, alpha=1.0, save_qk=False):
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
        if self.save_qk:
            self.q_vectors = query.detach().cpu()
            self.k_vectors = key.detach().cpu()

        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        # dynamically increase the causal mask with the key length, if needed.
        if key_length > self.bias.shape[-1]:
            self._init_bias(key_length, device=key.device)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=self.norm_factor,
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = self.transform_fn(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


class SparseGPTNeoFlashAttention2(GPTNeoXFlashAttention2):
    """
    This implementation is not really sparse, it just saves qk vectors for debugging purposes.
    """
    def __init__(self, config, attention_type, alpha=1.0, save_qk=False):
        super().__init__(config, attention_type)
        self.alpha = alpha
        self.save_qk = save_qk
        self.q_vectors = None
        self.k_vectors = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size: 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size:].permute(0, 2, 1, 3)

        query_length = query.shape[-2]

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims:]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims:]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # GPT-neo-X casts query and key in fp32 to apply rotary embedding in full precision
        target_dtype = value.dtype
        if query.dtype != target_dtype:
            query = query.to(target_dtype)
        if key.dtype != target_dtype:
            key = key.to(target_dtype)

        # Permute to get the expected shape for Flash Attention
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 / bfloat16 just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        input_dtype = query.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.query_key_value.weight.dtype

            print(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query = query.to(target_dtype)
            key = key.to(target_dtype)
            value = value.to(target_dtype)

        attention_dropout = self.config.attention_dropout if self.training else 0.0

        if self.save_qk:
            self.q_vectors = query.detach().cpu()
            self.k_vectors = key.detach().cpu()

        # Compute attention
        attn_weights = self._flash_attention_forward(
            query, key, value, attention_mask, query_length, dropout=attention_dropout, softmax_scale=self.norm_factor
        )

        # Reshape outputs
        attn_output = attn_weights.reshape(
            attn_weights.shape[0], attn_weights.shape[1], self.num_attention_heads * self.head_size
        )
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
