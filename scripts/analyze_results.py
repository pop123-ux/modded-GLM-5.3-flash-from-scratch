#!/usr/bin/env python3
"""Summarize supplied training/evaluation receipts; do not infer historical filenames."""
import argparse
import json
from pathlib import Path

def summarize(path):
    d=json.loads(path.read_text());row={'source':str(path),'status':d.get('status')}
    if d.get('stage')=='coding_pretraining':
        row.update(stage=d['stage'],tokens_seen=d['tokens_seen'],elapsed_seconds=d['elapsed_seconds'],parameter_counts=d['parameter_counts'])
        if 'validation_curve' in d:row['last_validation']=d['validation_curve'][-1]
    elif d.get('stage')=='executable_reward_rl':
        row.update(stage=d['stage'],groups=d['groups'],elapsed_seconds=d['elapsed_seconds'])
    elif 'summary' in d:
        row.update(summary=d['summary'],split=d.get('split'))
    else:row['note']='No supported summary fields; inspect the source receipt.'
    return row

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipts',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result={'receipts':[summarize(path) for path in a.receipts],'note':'No cross-run causal or significance claim is inferred.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
