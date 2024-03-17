import argparse

import torch
from entmax import entmax15, sparsemax, entmax_bisect


# from Pythia
def get_causal_mask(n):
    mask = torch.tril(torch.ones((n, n), dtype=torch.bool)).view(1, 1, n, n)
    return mask


# from Pythia
def compute_attention(query, key, alpha=1.0):
    # q, k: [bs, num_attention_heads, seq_len, attn_head_size]
    batch_size, num_attention_heads, query_length, attn_head_size = query.size()
    key_length = key.size(-2)
    # compute causal mask from causal mask buffer
    causal_mask = get_causal_mask(key_length).to(key.device)
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
        alpha=attn_head_size**-0.5,
    )
    attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)
    mask_value = torch.finfo(attn_scores.dtype).min
    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
    attn_scores = torch.where(causal_mask, attn_scores, mask_value)
    # compute attention weights
    if alpha == 1.0:
        attn_weights = torch.softmax(attn_scores, dim=-1)
    elif alpha == 1.5:
        attn_weights = entmax15(attn_scores, dim=-1)
    elif alpha == 2.0:
        attn_weights = sparsemax(attn_scores, dim=-1)
    else:
        attn_weights = entmax_bisect(attn_scores, alpha=alpha, dim=-1)
    return attn_weights


def compute_sparsity_ratio(attn_weights, mask=None):
    """
    Compute the sparsity of the distribution `attn_weights`

    Args:
        attn_weights: float tensor (n, n), attention weights for a single head, padded with 0s
        mask: bool tensor  (n, n), indicating which elements are valid (True) vs padded (False)
    """
    total = mask.sum().float()
    num_selected = (attn_weights > 0).sum().float()
    positive_ratio = num_selected / total
    sparsity_ratio = 1 - positive_ratio.item()
    return sparsity_ratio


def get_sparsity_ratio_for_alpha(qk_list, alpha=1.5, device=None):
    sparsity = 0
    for i in range(len(qk_list)):
        q = qk_list[i][0].unsqueeze(0).unsqueeze(1)
        k = qk_list[i][1].unsqueeze(0).unsqueeze(1)
        if device is not None:
            q = q.to(device)
            k = k.to(device)
        attn_weights = compute_attention(q, k, alpha=alpha)
        attn_weights = attn_weights.squeeze(1).squeeze(0)
        causal_mask = get_causal_mask(k.size(-2)).to(k.device)
        sparsity += compute_sparsity_ratio(attn_weights, causal_mask)
    return sparsity / len(qk_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--alpha", type=float, default=1.5, help="Entmax alpha parameter")
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading q, k vectors from {args.input}")
    qk_vectors = torch.load(args.input, map_location="cpu")

    # Compute sparsity
    print(f"Computing sparsity for {len(qk_vectors)} layers")
    for layer_head, qk_list in qk_vectors.items():
        layer, head = layer_head.split("_")

        if len(qk_list) == 0:
            continue

        sparsity = get_sparsity_ratio_for_alpha(qk_list, alpha=1.5)
        print(f"Layer {layer} Head {head} Alpha 1.5 sparsity: {sparsity}")

        sparsity = get_sparsity_ratio_for_alpha(qk_list, alpha=2.0)
        print(f"Layer {layer} Head {head} Alpha 2.0 sparsity: {sparsity}")

        if args.alpha not in [1.5, 2.0]:
            sparsity = get_sparsity_ratio_for_alpha(qk_list, alpha=args.alpha)
            print(f"Layer {layer} Head {head} Alpha {args.alpha} sparsity: {sparsity}")
