#!/usr/bin/env python3
"""Plot raw/smoothed training loss, including interrupted JSONL runs."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_rows(path):
    lines=path.read_text(encoding='utf-8').splitlines();rows=[]
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:row=json.loads(line)
        except json.JSONDecodeError:
            if i==len(lines)-1:
                print('Ignoring incomplete final JSONL line',file=sys.stderr);break
            raise
        if 'step' not in row or 'loss' not in row:raise ValueError('Expected coding-pretraining rows with step and loss')
        if not np.isfinite(row['loss']):raise ValueError('Non-finite training loss')
        rows.append(row)
    if not rows:raise ValueError('No logged training steps yet')
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--window',type=int,default=10)
    a=p.parse_args()
    if a.window<1:p.error('--window must be positive')
    if a.output.exists():raise FileExistsError(a.output)
    path=a.run/'train.jsonl' if a.run.is_dir() else a.run;rows=read_rows(path)
    receipt=path.parent/'training-receipt.json';validation=[]
    if receipt.exists():validation=json.loads(receipt.read_text(encoding='utf-8')).get('validation_curve',[])
    a.output.mkdir(parents=True)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));steps=[r['step'] for r in rows];loss=np.array([r['loss'] for r in rows])
    axes[0].plot(steps,loss,color='#2563eb',alpha=.3,label='Raw next-byte training CE')
    window=min(a.window,len(rows));smooth=np.convolve(loss,np.ones(window)/window,mode='valid')
    axes[0].plot(steps[window-1:],smooth,color='#2563eb',label=f'Trailing mean ({window} steps)')
    if validation:axes[0].plot([r['step'] for r in validation],[r['val_loss'] for r in validation],color='#ea580c',marker='o',label='Dev next-byte CE')
    axes[0].set(xlabel='Optimizer step',ylabel='Cross entropy (nats)',title='Primary losses (training and dev differ)');axes[0].legend(fontsize=8)
    present=[]
    for key,color in [('balance_loss','#0d9488'),('z_loss','#7c3aed'),('future_loss','#0891b2')]:
        if any(key in r for r in rows):
            axes[1].plot(steps,[r.get(key,np.nan) for r in rows],color=color,label=key);present.append(key)
    axes[1].set(xlabel='Optimizer step',ylabel='Unweighted diagnostic loss',title='Auxiliary objectives (separate scales)')
    if present:axes[1].legend(fontsize=8)
    else:axes[1].text(.5,.5,'No auxiliary loss diagnostics logged',ha='center',transform=axes[1].transAxes)
    for ax in axes:ax.grid(alpha=.15)
    fig.text(.01,.01,'Source: '+str(path)+' | '+('completed receipt present' if receipt.exists() else 'partial run; no final receipt'),fontsize=7)
    fig.tight_layout(rect=(0,.05,1,1))
    for ext in ['png','svg']:fig.savefig(a.output/f'loss.{ext}',dpi=180,facecolor='white')
    plt.close(fig);(a.output/'loss-data.json').write_text(json.dumps(dict(source=str(path),rows=rows,validation_curve=validation,window=window),indent=2)+'\n')
    print(f'Plotted {len(rows)} training steps into {a.output}')
if __name__=='__main__':main()
