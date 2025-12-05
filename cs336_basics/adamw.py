# Problem (adamw): Implement AdamW (2 points)
# Deliverable: Implement the AdamW optimizer as a subclass of torch.optim.Optimizer. Your
# class should take the learning rate α in __init__, as well as the β, ϵ and λ hyperparameters. To help
# you keep state, the base Optimizer class gives you a dictionary self.state, which maps nn.Parameter
# objects to a dictionary that stores any information you need for that parameter (for AdamW, this would
# be the moment estimates). Implement [adapters.get_adamw_cls] and make sure it passes uv run
# pytest -k test_adamw

import torch
from typing import Optional
from collections.abc import Callable

class AdamW(torch.optim.Optimizer):
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

# init(θ) (Initialize learnable parameters)
# m ← 0 (Initial value of the first moment vector; same shape as θ)
# v ← 0 (Initial value of the second moment vector; same shape as θ)
# for t = 1, . . . , T do
# Sample batch of data Bt
# g ← ∇θℓ(θ; Bt) (Compute the gradient of the loss at the current time step)
# m ← β1m + (1 − β1)g (Update the first moment estimate)
# v ← β2v + (1 − β2)g^2 (Update the second moment estimate)
# αt ← α √1−(β2)^t / 1−(β1)^t (Compute adjusted α for iteration t)
# θ ← θ − αt √m / v+ϵ (Update the parameters)
# θ ← θ − αλθ (Apply weight decay)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected first moment estimate
                denom = (exp_avg_sq.sqrt() + group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                
                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

