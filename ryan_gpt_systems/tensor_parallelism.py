"""
Tensor Parallelism for distributed training.

- Column Parallel: Split weight matrix along columns (output dim)
- Row Parallel: Split weight matrix along rows (input dim)
- Attention Parallelism: Split attention heads across GPUs
- MLP Parallelism: Split FFN layers across GPUs
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class TensorParallelGroup:
    """
    Manages tensor parallel process group and utilities.
    """
    
    def __init__(
        self, 
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        process_group: Optional[dist.ProcessGroup] = None
    ):
        if process_group is not None:
            self.process_group = process_group
            self.world_size = dist.get_world_size(process_group)
            self.rank = dist.get_rank(process_group)
        elif dist.is_initialized():
            self.process_group = dist.group.WORLD
            self.world_size = world_size or dist.get_world_size()
            self.rank = rank if rank is not None else dist.get_rank()
        else:
            # Single GPU mode
            self.process_group = None
            self.world_size = world_size or 1
            self.rank = rank or 0
    
    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce sum across tensor parallel group."""
        if self.world_size == 1:
            return tensor
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.process_group)
        return tensor
    
    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """All-gather tensors along specified dimension."""
        if self.world_size == 1:
            return tensor
        
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor, group=self.process_group)
        return torch.cat(tensor_list, dim=dim)
    
    def reduce_scatter(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Reduce-scatter: sum-reduce then scatter along dimension."""
        if self.world_size == 1:
            return tensor
        
        # Split along dim
        chunks = tensor.chunk(self.world_size, dim=dim)
        output = torch.zeros_like(chunks[0])
        
        # reduce_scatter doesn't exist in all PyTorch versions, use all_reduce + split
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.process_group)
        
        # Take our chunk
        output = tensor.chunk(self.world_size, dim=dim)[self.rank].contiguous()
        return output
    
    def scatter(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Scatter tensor along dimension to each rank."""
        if self.world_size == 1:
            return tensor
        
        chunks = tensor.chunk(self.world_size, dim=dim)
        return chunks[self.rank].contiguous()

class _CopyToModelParallelRegion(torch.autograd.Function):
    """Copy input to model parallel region (identity in forward, all-reduce in backward)."""
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor, tp_group: TensorParallelGroup):
        ctx.tp_group = tp_group
        return input_
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # All-reduce gradients in backward
        ctx.tp_group.all_reduce(grad_output)
        return grad_output, None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce in forward, identity in backward."""
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor, tp_group: TensorParallelGroup):
        ctx.tp_group = tp_group
        return tp_group.all_reduce(input_.clone())
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """All-gather in forward, scatter in backward."""
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor, tp_group: TensorParallelGroup, dim: int):
        ctx.tp_group = tp_group
        ctx.dim = dim
        return tp_group.all_gather(input_, dim=dim)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.tp_group.scatter(grad_output, dim=ctx.dim), None, None


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Scatter in forward, all-gather in backward."""
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor, tp_group: TensorParallelGroup, dim: int):
        ctx.tp_group = tp_group
        ctx.dim = dim
        return tp_group.scatter(input_, dim=dim)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.tp_group.all_gather(grad_output, dim=ctx.dim), None, None


def copy_to_tensor_parallel_region(input_: torch.Tensor, tp_group: TensorParallelGroup) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(input_, tp_group)


def reduce_from_tensor_parallel_region(input_: torch.Tensor, tp_group: TensorParallelGroup) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(input_, tp_group)


def gather_from_tensor_parallel_region(input_: torch.Tensor, tp_group: TensorParallelGroup, dim: int = -1) -> torch.Tensor:
    return _GatherFromModelParallelRegion.apply(input_, tp_group, dim)


def scatter_to_tensor_parallel_region(input_: torch.Tensor, tp_group: TensorParallelGroup, dim: int = -1) -> torch.Tensor:
    return _ScatterToModelParallelRegion.apply(input_, tp_group, dim)


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.
    
    Splits the weight matrix along the output dimension.
    Y = XA where A is partitioned column-wise: A = [A1, A2, ..., An]
    Each GPU computes: Yi = X @ Ai
    
    Args:
        in_features: Input dimension
        out_features: Output dimension (will be divided by world_size)
        tp_group: Tensor parallel group
        bias: Whether to include bias
        gather_output: If True, gather output from all ranks
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_group: TensorParallelGroup,
        bias: bool = True,
        gather_output: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.tp_group = tp_group
        self.gather_output = gather_output
        
        # Divide output features among ranks
        assert out_features % tp_group.world_size == 0, \
            f"out_features ({out_features}) must be divisible by world_size ({tp_group.world_size})"
        
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_partition = out_features // tp_group.world_size
        
        # Each rank has a slice of the weight matrix
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_partition, device=device, dtype=dtype)
            )
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Copy input to tensor parallel region (identity forward, all-reduce backward)
        input_parallel = copy_to_tensor_parallel_region(input_, self.tp_group)
        
        # Local linear
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        
        if self.gather_output:
            # Gather outputs from all ranks
            output = gather_from_tensor_parallel_region(output_parallel, self.tp_group, dim=-1)
        else:
            output = output_parallel
        
        return output


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.
    
    Splits the weight matrix along the input dimension.
    Y = XA where A is partitioned row-wise and X is partitioned accordingly.
    Each GPU computes: Yi = Xi @ Ai, then output = sum(Yi)
    
    Args:
        in_features: Input dimension (will be divided by world_size)
        out_features: Output dimension
        tp_group: Tensor parallel group
        bias: Whether to include bias
        input_is_parallel: If True, input is already partitioned
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_group: TensorParallelGroup,
        bias: bool = True,
        input_is_parallel: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.tp_group = tp_group
        self.input_is_parallel = input_is_parallel
        
        assert in_features % tp_group.world_size == 0, \
            f"in_features ({in_features}) must be divisible by world_size ({tp_group.world_size})"
        
        self.in_features = in_features
        self.out_features = out_features
        self.in_features_per_partition = in_features // tp_group.world_size
        
        # Each rank has a slice of the weight matrix
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition, device=device, dtype=dtype)
        )
        if bias:
            # Bias is not partitioned (only rank 0 uses it, others have zeros)
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features  # Full fan_in for proper scaling
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            # Scatter input across tensor parallel ranks
            input_parallel = scatter_to_tensor_parallel_region(input_, self.tp_group, dim=-1)
        
        # Local linear (no bias, will add after reduce)
        output_parallel = F.linear(input_parallel, self.weight)
        
        # All-reduce to sum partial results
        output = reduce_from_tensor_parallel_region(output_parallel, self.tp_group)
        
        # Add bias (only once, not partitioned)
        if self.bias is not None:
            output = output + self.bias
        
        return output


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with vocabulary parallelism.
    
    Splits the embedding table across GPUs along the vocabulary dimension.
    Each GPU holds vocab_size/world_size embeddings.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_group: TensorParallelGroup,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.tp_group = tp_group
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        assert num_embeddings % tp_group.world_size == 0, \
            f"num_embeddings ({num_embeddings}) must be divisible by world_size ({tp_group.world_size})"
        
        self.num_embeddings_per_partition = num_embeddings // tp_group.world_size
        self.vocab_start_index = tp_group.rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition
        
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim, device=device, dtype=dtype)
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.weight)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Mask for tokens in this partition's vocabulary range
        input_mask = (input_ >= self.vocab_start_index) & (input_ < self.vocab_end_index)
        
        # Remap indices to local range
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[~input_mask] = 0  # Out-of-range indices get 0
        
        # Local embedding lookup
        output_parallel = F.embedding(masked_input, self.weight)
        
        # Zero out embeddings for out-of-range indices
        output_parallel[~input_mask] = 0.0
        
        # All-reduce to combine embeddings from all partitions
        output = reduce_from_tensor_parallel_region(output_parallel, self.tp_group)
        
        return output


class TensorParallelAttention(nn.Module):
    """
    Multi-head attention with tensor parallelism.
    
    Splits attention heads across GPUs. Each GPU processes num_heads/world_size heads.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        tp_group: TensorParallelGroup,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.tp_group = tp_group
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert num_heads % tp_group.world_size == 0, \
            f"num_heads ({num_heads}) must be divisible by world_size ({tp_group.world_size})"
        
        self.num_heads_per_partition = num_heads // tp_group.world_size
        self.head_dim = d_model // num_heads
        
        # QKV projection - column parallel (split output dim)
        self.qkv = ColumnParallelLinear(
            d_model, 
            3 * d_model,  # Q, K, V concatenated
            tp_group,
            gather_output=False,  # Keep partitioned
            device=device,
            dtype=dtype,
        )
        
        # Output projection - row parallel (split input dim)
        self.out_proj = RowParallelLinear(
            d_model,
            d_model,
            tp_group,
            input_is_parallel=True,  # Input comes from partitioned attention
            device=device,
            dtype=dtype,
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # QKV projection (column parallel)
        qkv = self.qkv(x)  # [B, S, 3 * d_model / world_size]
        
        # Reshape to [B, S, 3, num_heads_per_partition, head_dim]
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads_per_partition, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H_local, S, D]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, H_local, S, D]
        
        # Attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H_local, S, S]
        
        # Apply mask (causal or custom)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        else:
            # Default causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), 
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B, H_local, S, D]
        
        # Reshape back: [B, S, H_local * D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        # Output projection (row parallel with all-reduce)
        output = self.out_proj(attn_output)
        
        return output


class TensorParallelMLP(nn.Module):
    """
    MLP/FFN with tensor parallelism.
    
    Uses column-parallel for the first linear and row-parallel for the second.
    This is efficient because there's no communication between them.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        tp_group: TensorParallelGroup,
        activation: str = "gelu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.tp_group = tp_group
        
        # First linear: column parallel (no gather)
        self.fc1 = ColumnParallelLinear(
            d_model, d_ff, tp_group,
            gather_output=False,
            device=device, dtype=dtype,
        )
        
        # Second linear: row parallel (with reduce)
        self.fc2 = RowParallelLinear(
            d_ff, d_model, tp_group,
            input_is_parallel=True,
            device=device, dtype=dtype,
        )
        
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class TensorParallelTransformerBlock(nn.Module):
    """
    Transformer block with tensor parallelism.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        tp_group: TensorParallelGroup,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.tp_group = tp_group
        
        self.norm1 = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(d_model, device=device, dtype=dtype)
        
        self.attn = TensorParallelAttention(
            d_model, num_heads, tp_group,
            dropout=dropout, device=device, dtype=dtype,
        )
        
        self.mlp = TensorParallelMLP(
            d_model, d_ff, tp_group,
            device=device, dtype=dtype,
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        # Pre-norm MLP
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class TensorParallelTransformerLM(nn.Module):
    """
    Full transformer language model with tensor parallelism.
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        tp_group: TensorParallelGroup,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.tp_group = tp_group
        self.vocab_size = vocab_size
        self.context_length = context_length
        
        # Embedding (not parallelized for simplicity - can use ParallelEmbedding for very large vocab)
        self.token_embeddings = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # Transformer blocks with tensor parallelism
        self.layers = nn.ModuleList([
            TensorParallelTransformerBlock(
                d_model, num_heads, d_ff, tp_group,
                dropout=dropout, device=device, dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        
        # LM head: column parallel
        self.lm_head = ColumnParallelLinear(
            d_model, vocab_size, tp_group,
            gather_output=True,  # Gather for full vocabulary output
            device=device, dtype=dtype,
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits


# =============================================================================
# Utility Functions
# =============================================================================

def split_tensor(tensor: torch.Tensor, num_partitions: int, dim: int = 0) -> list:
    """Split a tensor into num_partitions along the given dimension."""
    return list(tensor.chunk(num_partitions, dim=dim))


def merge_tensor(tensors: list, dim: int = 0) -> torch.Tensor:
    """Merge a list of tensors along the given dimension."""
    return torch.cat(tensors, dim=dim)


def shard_model_for_tensor_parallelism(
    model: nn.Module,
    tp_group: TensorParallelGroup,
) -> nn.Module:
    """
    Shard an existing model's weights for tensor parallelism.
    
    This function modifies the model's weights in-place to keep only
    the partition belonging to this rank.
    """
    rank = tp_group.rank
    world_size = tp_group.world_size
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Determine parallelism strategy based on layer name/position
            out_features = module.out_features
            in_features = module.in_features
            
            # For now, use column parallelism by default
            if out_features % world_size == 0:
                # Column parallel: split output dimension
                new_out = out_features // world_size
                start = rank * new_out
                end = start + new_out
                
                module.weight.data = module.weight.data[start:end, :].contiguous()
                if module.bias is not None:
                    module.bias.data = module.bias.data[start:end].contiguous()
                
                module.out_features = new_out
    
    return model
