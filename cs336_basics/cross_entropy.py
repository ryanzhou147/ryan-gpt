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
import torch.init.nn as nn
class cr