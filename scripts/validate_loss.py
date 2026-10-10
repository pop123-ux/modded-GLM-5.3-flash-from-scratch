#!/usr/bin/env python3
"""Evaluate a checkpoint on a named frozen split; do not tune on confirm."""
import argparse
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from glm53_flash.runtime import load_checkpoint,sha256_file
from glm53_flash.tasks import frozen_tasks
from glm53_flash.data import task_digest
from scripts.train_pretrain import validation

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['dev','final','confirm'],default='confirm')
    p.add_argument('--per-family',type=int,default=8);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--sequence-length',type=int,default=320);p.add_argument('--device',default='cpu')
    p.add_argument('--trim-padding',action='store_true')
    p.add_argument('--loss-scope',choices=['full','completion'],default='full')
    a=p.parse_args()
    if min(a.per_family,a.batch_size,a.sequence_length)<1:p.error('counts must be positive')
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(1);device=torch.device(a.device);model=load_checkpoint(a.checkpoint,device)
    tasks=frozen_tasks(a.split,per_family=a.per_family)
    result=dict(validation(model,tasks,a.sequence_length,a.batch_size,device,a.loss_scope,trim_padding=a.trim_padding),split=a.split,
        loss_scope=a.loss_scope,task_sha256=task_digest(tasks),weights_sha256=sha256_file(a.checkpoint/'model.pt'))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
