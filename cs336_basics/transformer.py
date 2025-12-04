import torch
import torch.nn as nn
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding
from einops import rearrange, einsum

class TransformerLM(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, with_rope: bool = False,
                 rope_theta: float | None = None, max_seq_len: int | None = None,
                 device=None, dtype=None) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.transformer_block = TransformerBlock(d_model, num_heads, d_ff, with_rope, rope_theta, max_seq_len, device, dtype)
        self.rmsnorm_final = nn.LayerNorm(d_model, elementwise_affine=True, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        Apply the Transformer language model to the input indices.
        
        in_indices: torch.Tensor Input tensor of shape (batch_size, sequence_length)
        """
        batch_size, seq_len = in_indices.shape
        assert seq_len <= self.context_length, f"Input sequence length {seq_len} exceeds context_length {self.context_length}"

        # Get token positions
        token_positions = torch.arange(seq_len, device=in_indices.device).unsqueeze(0).expand(batch_size, -1) # Shape: (batch_size, seq_len)
        # Embed input indices
        x = self.token_embeddings(in_indices) # Shape: (batch_size, seq_len, d_model)
        # Apply transformer blocks
        for _ in range(self.num_layers):
            x = self.transformer_block(x, token_positions)
        # Apply final RMSNorm
        x = self.rmsnorm_final(x)
        # Project to vocabulary logits
        logits = self.lm_head(x) # Shape: (batch_size, seq_len, vocab_size)
        return logits