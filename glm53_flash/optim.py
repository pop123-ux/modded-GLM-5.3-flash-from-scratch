"""Portable Muon-style matrix updates plus AdamW for embeddings/router/scalars.

Inspired by Keller Jordan's Muon; this is not the latest NanoGPT optimizer.
"""
import torch

@torch.no_grad()
def orthogonalize(g, steps=5):
    x = g.float()
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a = x @ x.T
        x = 3.4445 * x + (-4.7750 * a + 2.0315 * (a @ a)) @ x
    return (x.T if transpose else x).to(g.dtype)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps))
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError('Muon accepts only matrices')
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                m = state['momentum_buffer']
                m.lerp_(p.grad, 1-group['momentum'])
                update = orthogonalize(p.grad.lerp(m, group['momentum']), group['ns_steps'])
                update *= max(1, p.shape[0] / p.shape[1]) ** 0.5
                p.mul_(1-group['lr']*group['weight_decay'])
                p.add_(update, alpha=-group['lr'])
        return loss

class OptimizerBundle:
    def __init__(self, optimizers):
        self.optimizers = optimizers
    @property
    def param_groups(self):
        return [g for opt in self.optimizers for g in opt.param_groups]
    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)
    def step(self):
        for opt in self.optimizers:
            opt.step()
    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]
    def load_state_dict(self, state):
        if len(state) != len(self.optimizers):
            raise ValueError('optimizer bundle mismatch')
        for opt, data in zip(self.optimizers, state):
            opt.load_state_dict(data)

def build_optimizer(model, kind='adamw', lr=3e-4, muon_lr=0.02, weight_decay=0.1):
    if kind == 'adamw':
        return OptimizerBundle([torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95), weight_decay=weight_decay)])
    if kind != 'muon':
        raise ValueError(kind)
    matrices, other = [], []
    for name, p in model.named_parameters():
        # Embeddings/output and router keep adaptive updates. Tied weights appear once.
        target = matrices if p.ndim == 2 and not any(word in name for word in ('embedding','output','router')) else other
        target.append(p)
    return OptimizerBundle([Muon(matrices, lr=muon_lr, weight_decay=weight_decay),
        torch.optim.AdamW(other, lr=lr, betas=(0.9,0.95), weight_decay=weight_decay)])
