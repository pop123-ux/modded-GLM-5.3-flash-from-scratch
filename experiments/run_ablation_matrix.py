#!/usr/bin/env python3
"""Launch paired one-change ablations with shared data/validation and saved commands."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARMS={
 'vuk':({'residual_alpha':1.,'residual_dropout':0.},['--implementation','vuk']),
 'hybrid':({'residual_alpha':1.,'residual_dropout':0.},[]),
 'user_repaired':({'residual_alpha':0.01,'residual_dropout':0.1},[]),
 'small_alpha':({'residual_alpha':0.01,'residual_dropout':0.},[]),
 'dropout':({'residual_alpha':1.,'residual_dropout':0.1},[]),
 'streams1':({'streams':1},[]),'streams4':({'streams':4},[]),
 'standard_residual':({'residual_mode':'standard','streams':1},[]),
 'sparse_sdpa':({'sparse_backend':'sdpa'},[]),'full_sdpa':({'attention_mode':'sdpa'},[]),
 'qk_norm':({'qk_norm':True},[]),'embedding_skip':({'embedding_skip':True},[]),
 'softcap':({'logit_softcap':15.},[]),
 'balanced':({},['--balance-weight','0.01']),
 'router_z':({},['--z-weight','0.001']),
 'muon':({},['--optimizer','muon']),
 'cosine':({},['--schedule','cosine','--warmup-steps','5']),
 'real_tokens':({'padding_aware_routing':True},['--trim-padding']),
 'chunked_linear':({'linear_chunk_size':32,'linear_checkpoint_chunks':True},[]),
 'dense_ffn':({'ffn_mode':'dense'},[]),
 'bias_balanced':({'router_bias_rate':0.001},[]),
 'future_bytes':({'future_heads':2},['--future-weight','0.1']),
}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--base-config',type=Path,default=ROOT/'configs/hybrid_baseline.json')
 p.add_argument('--arms',default='vuk,hybrid,user_repaired,balanced,sparse_sdpa,full_sdpa,muon')
 p.add_argument('--seeds',default='42,43,44');p.add_argument('--steps',type=int,default=400)
 p.add_argument('--batch-size',type=int,default=16);p.add_argument('--sequence-length',type=int,default=320)
 p.add_argument('--device',default='auto');p.add_argument('--target-loss',type=float);p.add_argument('--dry-run',action='store_true')
 a=p.parse_args();arms=a.arms.split(',');seeds=[int(s) for s in a.seeds.split(',')]
 if any(x not in ARMS for x in arms):p.error('unknown arm; see ARMS in this file')
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);base=json.loads(a.base_config.read_text());commands=[]
 for arm in arms:
  changes,extra=ARMS[arm];path=a.output/f'{arm}.json';path.write_text(json.dumps(dict(base,**changes),indent=2)+'\n')
  for seed in seeds:
   cmd=[sys.executable,str(ROOT/'scripts/train_pretrain.py'),'--config',str(path.resolve()),'--output',str((a.output/f'{arm}-seed{seed}').resolve()),'--seed',str(seed),'--data-seed','42','--steps',str(a.steps),'--batch-size',str(a.batch_size),'--sequence-length',str(a.sequence_length),'--device',a.device]+extra
   if a.target_loss is not None:cmd+=['--target-loss',str(a.target_loss)]
   commands.append(dict(arm=arm,seed=seed,argv=cmd))
 (a.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 if not a.dry_run:
  for row in commands:
   subprocess.run(row['argv'],check=True,cwd=ROOT)
 print(f'{len(commands)} commands '+('prepared' if a.dry_run else 'completed'))
if __name__=='__main__':main()
