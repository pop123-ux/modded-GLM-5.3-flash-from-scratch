"""Chunked positive-feature causal attention with differentiable boundary state.

Checkpointing trades recomputation for smaller saved activations. Boundary
states still scale with the number of chunks; this is not constant-memory training.
"""
import torch
from torch.utils.checkpoint import checkpoint

def _chunk(q, k, v, state_k, state_kv):
    prefix_k = k.cumsum(1) + state_k[:, None]
    prefix_kv = torch.einsum('bthd,bthe->bthde', k, v).cumsum(1) + state_kv[:, None]
    numerator = torch.einsum('bthd,bthde->bthe', q, prefix_kv)
    denominator = torch.einsum('bthd,bthd->bth', q, prefix_k)[..., None]
    output = numerator / denominator.clamp_min(1e-6)
    # Clone compact boundary states: a view would retain the entire chunk storage.
    return output, prefix_k[:, -1].clone(), prefix_kv[:, -1].clone()

def chunked_linear_attention(q, k, v, chunk_size, checkpoint_chunks=False):
    if chunk_size < 1 or q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or not q.shape[1]:
        raise ValueError('expected nonempty matching [B,T,H,d] inputs and positive chunk size')
    b,t,h,d = q.shape
    state_k = k.new_zeros(b,h,d)
    state_kv = k.new_zeros(b,h,d,d)
    outputs = []
    for start in range(0,t,chunk_size):
        args = (q[:,start:start+chunk_size], k[:,start:start+chunk_size], v[:,start:start+chunk_size], state_k, state_kv)
        if checkpoint_chunks and torch.is_grad_enabled() and any(x.requires_grad for x in args):
            output,state_k,state_kv = checkpoint(_chunk,*args,use_reentrant=False)
        else:
            output,state_k,state_kv = _chunk(*args)
        outputs.append(output)
    return torch.cat(outputs,dim=1)
