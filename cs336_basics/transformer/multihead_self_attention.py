import torch
import torch.nn as nn
from einops import einsum, rearrange
from cs336_basics.utility import scaled_dot_product_attention
from cs336_basics.transformer.rope import RotaryPositionalEmbedding
from cs336_basics.transformer.linear import Linear

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, rope_theta: float | None = None, max_seq_len: int | None = None, with_rope: bool = False,
                 device=None, dtype=None) -> None:

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.w_q = nn.Parameter(torch.empty((num_heads * self.d_k, d_model), device=device, dtype=dtype))
        self.w_k = nn.Parameter(torch.empty((num_heads * self.d_k, d_model), device=device, dtype=dtype))
        self.w_v = nn.Parameter(torch.empty((num_heads * self.d_v, d_model), device=device, dtype=dtype))
        self.w_o = nn.Parameter(torch.empty((d_model, num_heads * self.d_v), device=device, dtype=dtype))

        self.with_rope = with_rope

        if self.with_rope:
            self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input of shape (batch, seq_len, d_model)
            token_positions: Optional token positions for RoPE
            
        Returns:
            Output of shape (batch, seq_len, d_model)
        """

        # Project to Q, K, V using einsum: x @ W^T
        # x: (b s d_model), W: (h_dk d_model) -> result: (b s h_dk)
        Q = einsum(x, self.w_q, 'b s d_in, h_dk d_in -> b s h_dk')
        K = einsum(x, self.w_k, 'b s d_in, h_dk d_in -> b s h_dk')
        V = einsum(x, self.w_v, 'b s d_in, h_dv d_in -> b s h_dv')
        
        # Reshape to separate heads using rearrange
        # (batch, seq_len, num_heads * d_k) -> (batch, num_heads, seq_len, d_k)
        Q = rearrange(Q, 'b s (h d_k) -> b h s d_k', h=self.num_heads, d_k=self.d_k)
        K = rearrange(K, 'b s (h d_k) -> b h s d_k', h=self.num_heads, d_k=self.d_k)
        V = rearrange(V, 'b s (h d_v) -> b h s d_v', h=self.num_heads, d_v=self.d_v)
        
        _, seq_len, _ = x.size()
        
        if self.rope is not None and token_positions is not None:
            # Apply RoPE to Q and K per-head (not to V)
            # Q, K shape: (batch, num_heads, seq_len, d_k)
            # RoPE expects: (batch, seq_len, d_k)
            Q_rope = []
            K_rope = []
            for head_idx in range(self.num_heads):
                Q_head = Q[:, head_idx, :, :]  # (batch, seq_len, d_k)
                K_head = K[:, head_idx, :, :]  # (batch, seq_len, d_k)
                Q_rope.append(self.rope(Q_head, token_positions))
                K_rope.append(self.rope(K_head, token_positions))
            
            Q = torch.stack(Q_rope, dim=1)  # (batch, num_heads, seq_len, d_k)
            K = torch.stack(K_rope, dim=1)  # (batch, num_heads, seq_len, d_k)
        
        # Create causal mask: token i can attend to j <= i
        # Lower triangular (1s below diagonal, including diagonal) means ALLOW
        # mask[i,j] = 1 if j <= i (allow attending to past/current), 0 otherwise
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
        causal_mask = rearrange(causal_mask, 's1 s2 -> 1 1 s1 s2')  # Broadcast dimensions

        # Attention: (batch, num_heads, seq_len, d_k) -> (batch, num_heads, seq_len, d_v)
        attn_output = scaled_dot_product_attention(Q, K, V, causal_mask)

        # Concatenate heads using rearrange
        # (batch, num_heads, seq_len, d_v) -> (batch, seq_len, num_heads * d_v)
        attn_output = rearrange(attn_output, 'b h s d_v -> b s (h d_v)')
        
        # Output projection using einsum
        # (batch, seq_len, h·dv) @ W_o^T -> (batch, seq_len, d_model)
        output = einsum(attn_output, self.w_o, 'b s h_dv, d_out h_dv -> b s d_out')
        
        return output
