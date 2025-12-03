import torch
import torch.nn as nn
import torch.nn.init as init


class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None) -> None:
        super().__init__()
        
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = ((d_ff + 63) // 64) * 64  # Round to nearest multiple of 64
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.w1 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.w2 = nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)
        self.w3 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        
        init.trunc_normal_(self.w1.weight)
        init.trunc_normal_(self.w2.weight)
        init.trunc_normal_(self.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        w1_out = self.w1(x)
        silu_out = w1_out * torch.sigmoid(w1_out)

        w3_out = self.w3(x)
        gated = silu_out * w3_out

        return self.w2(gated)


