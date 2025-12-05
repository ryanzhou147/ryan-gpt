import torch
import math
import os
import typing
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


def data_loading(x: torch.Tensor, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = x.size(0) # Total number of tokens in the dataset
    input_sequences = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
    target_sequences = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
    for i in range(batch_size):
        start_index = torch.randint(0, num_tokens - context_length, (1,)).item() # Random start index 
        input_sequences[i] = x[start_index : start_index + context_length]
        target_sequences[i] = x[start_index + 1 : start_index + context_length + 1]
    return input_sequences, target_sequences

# What if the dataset is too big to load into memory? We can use a Unix systemcall named mmap which
# maps a file on disk to virtual memory, and lazily loads the file contents when that memory location is
# accessed. Thus, you can “pretend” you have the entire dataset in memory. Numpy implements this through
# np.memmap (or the flag mmap_mode='r' to np.load, if you originally saved the array with np.save), which
# will return a numpy array-like object that loads the entries on-demand as you access them. When sampling
# from your dataset (i.e., a numpy array) during training, be sure load the dataset in memorymapped mode (via np.memmap or the flag mmap_mode='r' to np.load, depending on how you saved the
# array). Make sure you also specify a dtype that matches the array that you’re loading. It may be helpful
# to explicitly verify that the memory-mapped data looks correct (e.g., doesn’t contain values beyond the
# expected vocabulary size).

# A checkpoint should have all the states that we need to resume training. We of course want to be able
# to restore model weights at a minimum. If using a stateful optimizer (such as AdamW), we will also need
# to save the optimizer’s state (e.g., in the case of AdamW, the moment estimates). Finally, to resume the
# learning rate schedule, we will need to know the iteration number we stopped at. PyTorch makes it easy to
# save all of these: every nn.Module has a state_dict() method that returns a dictionary with all learnable
# weights; we can restore these weights later with the sister method load_state_dict(). The same goes
# for any nn.optim.Optimizer. Finally, torch.save(obj, dest) can dump an object (e.g., a dictionary
# containing tensors in some values, but also regular Python objects like integers) to a file (path) or file-like
# object, which can then be loaded back into memory with torch.load(src).

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> int:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']