import torch
import math
from collections.abc import Iterable 
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

def learning_rate_schedule(current_iteration: int, max_learning_rate: float, minimum_learning_rate: float, warmup_iterations: int, cosine_annealing_iterations: int) -> float:
    if current_iteration < warmup_iterations:
        lr = max_learning_rate * (current_iteration / warmup_iterations)
    elif current_iteration <= cosine_annealing_iterations: 
        progress = (current_iteration - warmup_iterations) / (cosine_annealing_iterations - warmup_iterations)
        lr = minimum_learning_rate + 0.5 * (1 + math.cos(progress * math.pi)) * (max_learning_rate - minimum_learning_rate)
    else: 
        lr = minimum_learning_rate
    
    return lr

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    total_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            param_norm = torch.linalg.vector_norm(parameter.grad.data, ord=2)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5

    if total_norm >= max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.data.mul_(scale)
    