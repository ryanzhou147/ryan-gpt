import torch
from einops import einsum

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:

    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):
    assert mask.max() <= 1 and mask.min() >= 0, "Mask tensor must be binary (0s and 1s)"
    assert value.size(-2) == key.size(-2), "Key and Value must have the same sequence length"
    d_k = query.size(-1)
    scores = einsum(query, key, '... i d, ... j d -> ... i j') / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = softmax(scores, dim=-1)
    
    outputs = einsum(attn_weights, value, '... i j, ... j d -> ... i d')
    return outputs
