# Problem (embedding): Implement the embedding module (1 point)
# Deliverable: Implement the Embedding class that inherits from torch.nn.Module and performs an
# embedding lookup. Your implementation should follow the interface of PyTorch’s built-in
# nn.Embedding module. We recommend the following interface:
# def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) Construct
# an embedding module. This function should accept the following parameters:
# num_embeddings: int Size of the vocabulary
# 19
# embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
# device: torch.device | None = None Device to store the parameters on
# dtype: torch.dtype | None = None Data type of the parameters
# def forward(self, token_ids: torch.Tensor) -> torch.Tensor Lookup the embedding vectors
# for the given token IDs.
# Make sure to:
# • subclass nn.Module
# • call the superclass constructor
# • initialize your embedding matrix as a nn.Parameter
# • store the embedding matrix with the d_model being the final dimension
# • of course, don’t use nn.Embedding or nn.functional.embedding
# Again, use the settings from above for initialization, and use torch.nn.init.trunc_normal_ to
# initialize the weights.
# To test your implementation, implement the test adapter at [adapters.run_embedding]. Then, run
# uv run pytest -k test_embedding
import torch
import torch.nn as nn
import torch.nn.init as init
class Embedding(nn.Module):
    
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None): 
        """Construct an embedding module. This function should accept the following parameters:
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """

        super().__init__()
        # Embedding parameter (num_embeddings, embedding_dim)
        self.embedding_matrix = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        # Initialize weights with truncated normal
        init.trunc_normal_(self.embedding_matrix, mean=0.0, std=0.02)

        return 


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""

        return self.embedding_matrix[token_ids]
