#!/usr/bin/env python3
"""Profile forward+backward at multiple sequence lengths, without convergence claims."""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from glm53_flash.model import ModelConfig,LinearAttention,SparseAttention,CausalSelfAttention
from scripts.train_pretrain import synchronize

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--lengths',default='64,128,256,512')
 p.add_argument('--iterations',type=int,default=10);p.add_argument('--warmup',type=int,default=2)
 p.add_argument('--batch-size',type=int,default=4);p.add_argument('--dim',type=int,default=128)
 p.add_argument('--device',default='cpu');a=p.parse_args()
 lengths=[int(x) for x in a.lengths.split(',')]
 if min(lengths+[a.iterations,a.batch_size])<1 or a.warmup<0:p.error('invalid counts')
 if a.output.exists():raise FileExistsError(a.output)
 torch.set_num_threads(1);device=torch.device(a.device);rows=[]
 for length in lengths:
  cfg=ModelConfig(dim=a.dim,max_sequence_length=length,residual_dropout=0.)
  for name,cls,backend in [('linear',LinearAttention,'gather'),('gather_sparse',SparseAttention,'gather'),('masked_sdpa',SparseAttention,'sdpa'),('full_sdpa',CausalSelfAttention,'gather')]:
   torch.manual_seed(42);module=cls(replace(cfg,sparse_backend=backend)).to(device)
   x=torch.randn(a.batch_size,length,a.dim,device=device,requires_grad=True)
   def step():
    module.zero_grad(set_to_none=True);x.grad=None;module(x).square().mean().backward()
   for _ in range(a.warmup):step()
   synchronize(device)
   if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
   started=time.perf_counter()
   for _ in range(a.iterations):step()
   synchronize(device);seconds=(time.perf_counter()-started)/a.iterations
   rows.append(dict(attention=name,length=length,seconds_per_forward_backward=seconds,
    peak_vram_gib=torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else None))
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(dict(torch=torch.__version__,device=str(device),arguments=vars(a)|{'output':str(a.output)},results=rows),indent=2)+'\n')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
