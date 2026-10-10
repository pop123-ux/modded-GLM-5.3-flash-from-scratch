#!/usr/bin/env python3
"""Describe only the supplied curriculum/diversity receipts, without canned conclusions."""
import argparse
import itertools
import json
import statistics
from pathlib import Path

def analyze(path):
    receipt=json.loads(path.read_text());groups={}
    for run in receipt['runs']:
        name=str(run.get('condition',run.get('diversity')))
        if run['seed'] in groups.setdefault(name,{}):raise ValueError('duplicate condition/seed')
        groups[name][run['seed']]=run
    summaries=[];paired=[]
    for name,runs in groups.items():
        values=list(runs.values())
        summaries.append(dict(condition=name,seeds=sorted(runs),runs=len(values),
            mean_target_loss=statistics.mean(v['after']['target_loss'] for v in values),
            mean_byte_accuracy=statistics.mean(v['after']['target_byte_accuracy'] for v in values),
            mean_exact_accuracy=statistics.mean(v['after']['target_exact_accuracy'] for v in values),
            mean_elapsed_seconds=statistics.mean(v['elapsed_seconds'] for v in values),
            parameter_counts=values[0]['parameter_counts']))
    for left,right in itertools.combinations(groups,2):
        seeds=sorted(set(groups[left])&set(groups[right]))
        if seeds:
            deltas=[groups[right][s]['after']['target_byte_accuracy']-groups[left][s]['after']['target_byte_accuracy'] for s in seeds]
            paired.append(dict(left=left,right=right,paired_seeds=seeds,mean_byte_accuracy_delta=statistics.mean(deltas),
                               stdev_delta=statistics.stdev(deltas) if len(deltas)>1 else None))
    return dict(source=str(path),steps=receipt['steps'],batch_size=receipt['batch_size'],summaries=summaries,paired_differences=paired)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipts',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    data=dict(studies=[analyze(path) for path in a.receipts],note='Descriptive results only; no invented historical results or significance claims.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
if __name__=='__main__':main()
