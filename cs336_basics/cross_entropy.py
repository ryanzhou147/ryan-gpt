# Problem (cross_entropy): Implement Cross entropy
# Deliverable: Write a function to compute the cross entropy loss, which takes in predicted logits
# (oi) and targets (xi+1) and computes the cross entropy ℓi = − log softmax(oi)[xi+1]. Your function
# should handle the following:
# • Subtract the largest element for numerical stability.
# • Cancel out log and exp whenever possible.
# • Handle any additional batch dimensions and return the average across the batch. As with section 3.3, we assume batch-like dimensions always come first, before the vocabulary size dimension.
# Implement [adapters.run_cross_entropy], then run uv run pytest -k test_cross_entropy
# to test your implementation.

import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):

    def __init__(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.targets = targets
    
    def forward(self) -> torch.Tensor:
        
        # Subtract max for numerical stability
        x_max = torch.max(self.logits, dim=-1, keepdim=True).values
        self.logits = self.logits - x_max
        # Compute log probabilities
        log_probs = self.logits - torch.log(torch.sum(torch.exp(self.logits), dim=-1, keepdim=True))
        # Gather the log probabilities of the target classes
        nll_loss = -log_probs.gather(dim=-1, index=self.targets.unsqueeze(-1)).squeeze(-1)
        return nll_loss.mean()

    def perlexity(self) -> torch.Tensor:
        """Compute the perplexity from the cross-entropy loss."""
        ce_loss = self.forward()
        perplexity = torch.exp(ce_loss)
        return perplexity