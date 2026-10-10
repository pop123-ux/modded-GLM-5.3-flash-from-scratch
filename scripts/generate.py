#!/usr/bin/env python3
"""Generate bytes from a checkpoint, stopping only at EOS or the requested cap."""
import argparse
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from glm53_flash import ByteTokenizer
from glm53_flash.runtime import load_checkpoint

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--prompt',required=True)
    p.add_argument('--max-new-tokens',type=int,default=64);p.add_argument('--temperature',type=float,default=0.)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(a.seed)
    if a.max_new_tokens<1 or a.temperature<0: p.error('invalid generation limits')
    tok=ByteTokenizer();model=load_checkpoint(a.checkpoint,torch.device(a.device)).eval()
    ids=tok.encode(a.prompt,bos=True)
    if len(ids)+a.max_new_tokens>model.config.max_sequence_length: p.error('prompt plus cap exceeds context')
    generated=[]
    with torch.no_grad():
        for _ in range(a.max_new_tokens):
            logits,_=model(torch.tensor([ids+generated],device=a.device))
            values=logits[0,-1].float();values[[0,1,3]]=-torch.inf
            token=int(values.argmax()) if a.temperature==0 else int(torch.multinomial((values/a.temperature).softmax(-1),1))
            generated.append(token)
            if token==tok.eos_id: break
    print(a.prompt+tok.decode(generated))
if __name__=='__main__':main()
