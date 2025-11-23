import torch
import torch.nn as nn

def __init__(self, in_features, out_features, device=None, dtype=None): 
    """Construct a
    linear transformation module. This function should accept the following parameters:
    in_features: int final dimension of the input
    out_features: int final dimension of the output
    device: torch.device | None = None Device to store the parameters on
    dtype: torch.dtype | None = None Data type of the parameters
    """

    return 


def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply the linear transformation to the input."""

    return 

x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print("Original x:")
print(x)

layer_norm = nn.LayerNorm(3)
x_norm = layer_norm(x)
print("\nAfter LayerNorm:")
print(x_norm)

# Check mean and variance along features
mean = x_norm.mean(dim=-1)
std = x_norm.std(dim=-1, unbiased=False)
print("\nMean per token:", mean)
print("Std per token:", std)