import torch
import torch.nn as nn
import torch.nn.init as init

class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        

# Problem (positionwise_feedforward): Implement the position-wise feed-forward network
# (2 points)
# Deliverable: Implement the SwiGLU feed-forward network, composed of a SiLU activation
# function and a GLU.
# Note: in this particular case, you should feel free to use torch.sigmoid in your implementation
# for numerical stability.
# You should set dff to approximately 8
# 3 × dmodel in your implementation, while ensuring that
# the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your
# hardware. To test your implementation against our provided tests, you will need to implement
# the test adapter at [adapters.run_swiglu]. Then, run uv run pytest -k test_swiglu to
# test your implementation.