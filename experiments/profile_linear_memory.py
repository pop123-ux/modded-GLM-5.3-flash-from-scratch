#!/usr/bin/env python3
"""Measure storage referenced by tensors saved for backward, not total peak memory."""
import argparse
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from glm53_flash.linear import chunked_linear_attention

def prefix_reference(q,k,v):
    state=torch.einsum('bthd,bthe->bthde',k,v).cumsum(1)
    numerator=torch.einsum('bthd,bthde->bthe',q,state)
    denominator=torch.einsum('bthd,bthd->bth',q,k.cumsum(1))[...,None]
    return numerator/denominator.clamp_min(1e-6)

def measure(fn,length,heads,head_dim,device):
    storages={};elements=[]
    def pack(t):
        storage=t.untyped_storage()
        key=(str(t.device),storage.data_ptr(),storage.nbytes())
        storages[key]=storage.nbytes();elements.append(t.numel())
        return t
    torch.manual_seed(42)
    q,k,v=[torch.rand(1,length,heads,head_dim,device=device,requires_grad=True) for _ in range(3)]
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):out=fn(q,k,v)
    out.square().mean().backward()
    return dict(saved_tensor_elements=sum(elements),unique_saved_storage_bytes=sum(storages.values()),
                finite_gradients=all(bool(torch.isfinite(t.grad).all()) for t in (q,k,v)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--length',type=int,default=256);p.add_argument('--heads',type=int,default=4)
    p.add_argument('--head-dim',type=int,default=16);p.add_argument('--chunk-size',type=int,default=32)
    p.add_argument('--device',default='cpu');a=p.parse_args()
    if min(a.length,a.heads,a.head_dim,a.chunk_size)<1:p.error('counts must be positive')
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(1);device=torch.device(a.device)
    result=dict(reference=measure(prefix_reference,a.length,a.heads,a.head_dim,device),
        chunked=measure(lambda q,k,v:chunked_linear_attention(q,k,v,a.chunk_size,True),a.length,a.heads,a.head_dim,device),
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},torch=torch.__version__,
        note='Storage referenced by saved tensors during this isolated forward. Includes inputs; deduplicates shared storage. This is not full-model or peak allocator memory and says nothing about speed/convergence.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
