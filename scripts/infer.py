#!/usr/bin/env python3
"""Infer from a checkpoint: prompt, UTF-8 file, or interactive independent prompts."""
import argparse,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from glm53_flash.inference import generate_text
from glm53_flash.runtime import load_checkpoint

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--prompt');inputs.add_argument('--prompt-file',type=Path);inputs.add_argument('--interactive',action='store_true')
    p.add_argument('--max-new-tokens',type=int,default=64);p.add_argument('--temperature',type=float,default=0.)
    p.add_argument('--top-k',type=int,default=0);p.add_argument('--top-p',type=float,default=1.);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto');p.add_argument('--json',action='store_true')
    p.add_argument('--output',type=Path,help='Save a single JSON result')
    a=p.parse_args()
    if a.interactive and a.output:p.error('--output cannot be used with --interactive')
    if a.output and a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(1);device=a.device
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model=load_checkpoint(a.checkpoint,torch.device(device)).eval()
    def run(prompt):
        result=generate_text(model,prompt,max_new_tokens=a.max_new_tokens,temperature=a.temperature,top_k=a.top_k,top_p=a.top_p,seed=a.seed)
        data=result.to_dict()|dict(checkpoint=str(a.checkpoint),device=device,seed=a.seed,temperature=a.temperature,top_k=a.top_k,top_p=a.top_p)
        print(json.dumps(data,ensure_ascii=True) if a.json else result.text)
        if a.output:
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    if a.interactive:
        print('Independent prompts (not chat history). /quit exits.',file=sys.stderr)
        while True:
            try:
                prompt=input('> ')
                if prompt=='/quit':break
                run(prompt)
            except (EOFError,KeyboardInterrupt):break
            except ValueError as exc:print(str(exc),file=sys.stderr)
    else:run(a.prompt if a.prompt is not None else a.prompt_file.read_text(encoding='utf-8'))
if __name__=='__main__':main()
