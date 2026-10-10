#!/usr/bin/env python3
"""Summarize compatible run receipts; incomplete target runs remain failures."""
import argparse
import json
import statistics
from pathlib import Path

def compatibility_key(r):
 a=r['arguments']
 return (r['validation_sha256'],a['data_seed'],a['sequence_length'],a['batch_size'],a['loss_scope'],
         a.get('families',''),a.get('target_loss'),r['device_name'],r['torch'],a['steps'],a['eval_every'],a['eval_per_family'],a['threads'],a['amp'])

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();paths=sorted(a.root.rglob('training-receipt.json'))
 if not paths:p.error('no completed receipts')
 rows=[];groups={}
 for path in paths:
  r=json.loads(path.read_text());key=compatibility_key(r);group=groups.setdefault(key,len(groups))
  rows.append(dict(run=str(path.parent),comparison_group=group,seed=r['seed'],implementation=r['implementation'],
   final_val_loss=r['validation_curve'][-1]['val_loss'],completion_val_loss=r['validation_curve'][-1]['completion_val_loss'],
   training_seconds=r['training_seconds'],target_reached=r['first_target_hit'] is not None,
   time_to_target=r['first_target_hit']['training_seconds'] if r['first_target_hit'] else None,
   tokens_per_second=r['tokens_seen']/r['training_seconds'],
   trim_padding=r['arguments'].get('trim_padding',False),ffn_mode=r['config'].get('ffn_mode','moe'),
   future_heads=r['config'].get('future_heads',0)))
 summaries=[]
 for group in set(x['comparison_group'] for x in rows):
  subset=[x for x in rows if x['comparison_group']==group]
  by_arm={}
  for row in subset:
   arm=Path(row['run']).name.rsplit('-seed',1)[0];by_arm.setdefault(arm,[]).append(row)
  for arm,values in by_arm.items():
   losses=[v['final_val_loss'] for v in values]
   summaries.append(dict(comparison_group=group,arm=arm,runs=len(values),
    mean_loss=statistics.mean(losses),stdev_loss=statistics.stdev(losses) if len(losses)>1 else None,
    mean_training_seconds=statistics.mean(v['training_seconds'] for v in values),
    target_successes=sum(v['target_reached'] for v in values),
    mean_time_to_target=statistics.mean(v['time_to_target'] for v in values) if all(v['target_reached'] for v in values) else None))
 if a.output.exists():raise FileExistsError(a.output)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(rows=rows,summaries=summaries,note='Descriptive statistics only; no statistical significance or GPU speedup claim.'),indent=2)+'\n')
 print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
