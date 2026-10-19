#!/usr/bin/env python3
"""Separate descriptive figures for attention, structural, RL, vision and code studies."""
import argparse,json,re,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    paths=[a.root] if a.root.is_file() else sorted(a.root.rglob('*.json'))
    a.output.mkdir(parents=True);plotted=[];skipped=[]
    def save(fig,path,label):
        stem=re.sub(r'[^a-zA-Z0-9_-]','-',str(path.relative_to(a.root) if a.root.is_dir() else path.name))+'-'+label
        fig.suptitle(label.replace('-',' '));fig.text(.01,.01,str(path),fontsize=7)
        fig.tight_layout(rect=(0,.06,1,.94))
        for ext in ['png','svg']:fig.savefig(a.output/f'{stem}.{ext}',dpi=180,facecolor='white')
        plt.close(fig);plotted.append(dict(source=str(path),figure=stem))
    for path in paths:
        try:r=json.loads(path.read_text(encoding='utf-8'))
        except (ValueError,UnicodeError):skipped.append(str(path));continue
        if not isinstance(r,dict):continue
        if isinstance(r.get('results'),list) and r['results'] and 'seconds_per_forward_backward' in r['results'][0]:
            fig,ax=plt.subplots(figsize=(9,4))
            for name in sorted({x['attention'] for x in r['results']}):
                rows=[x for x in r['results'] if x['attention']==name];ax.plot([x['length'] for x in rows],[x['seconds_per_forward_backward']*1000 for x in rows],marker='o',label=name)
            ax.set(xlabel='Sequence positions',ylabel='Milliseconds / forward + backward');ax.legend();save(fig,path,'attention-time')
        elif all(k in r for k in ['reference','chunked','settings']):
            fig,ax=plt.subplots(figsize=(7,4));ax.bar(['Prefix reference','Checkpointed chunks'],[r[x]['unique_saved_storage_bytes']/1024 for x in ['reference','chunked']],color=['#2563eb','#7c3aed'])
            ax.set(ylabel='KiB of unique saved-tensor storage',title='Isolated attention core; NOT peak/full-model memory');save(fig,path,'saved-storage')
        elif isinstance(r.get('runs'),list) and r['runs'] and 'after' in r['runs'][0]:
            groups={}
            for run in r['runs']:groups.setdefault(str(run.get('condition',run.get('diversity'))),[]).append(run)
            fig,axes=plt.subplots(1,2,figsize=(12,4));names=list(groups)
            for i,name in enumerate(names):
                vals=[x['after']['target_byte_accuracy'] for x in groups[name]]
                axes[0].bar(i,statistics.mean(vals),yerr=statistics.stdev(vals) if len(vals)>1 else None,color='#0d9488',capsize=4)
                for run in groups[name]:
                    curve=run.get('training_curve',[])
                    if curve:axes[1].plot([x['step'] for x in curve],[x['training_loss'] for x in curve],alpha=.6,label=f'{name} seed{run["seed"]}')
            axes[0].set_xticks(range(len(names)),names,rotation=20,ha='right');axes[0].set(ylabel='Structural-holdout byte accuracy (fraction)',ylim=(0,1))
            axes[1].set(xlabel='Optimizer step',ylabel='Training CE');axes[1].legend(fontsize=6);save(fig,path,'structural-results')
        elif r.get('stage')=='executable_reward_rl':
            rows=r.get('groups_detail',[])
            if not rows:continue
            fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].plot([x['group'] for x in rows],[statistics.mean(x['rewards']) for x in rows],marker='o')
            axes[0].set(xlabel='Rollout group',ylabel='Mean sampled reward')
            axes[1].plot([x['group'] for x in rows],[sum(bool(y.get('updated')) for y in rows[:i+1]) for i,x in enumerate(rows)])
            axes[1].set(xlabel='Rollout group',ylabel='Cumulative optimizer updates');save(fig,path,'rl-rewards')
        elif isinstance(r.get('results'),list) and r['results'] and 'architecture' in r['results'][0]:
            fig,ax=plt.subplots(figsize=(8,4));rows=r['results'];x=np.arange(len(rows))
            ax.bar(x-.17,[v['before']['accuracy'] for v in rows],.34,label='Before');ax.bar(x+.17,[v['after']['accuracy'] for v in rows],.34,label='After')
            ax.set_xticks(x,[v['architecture'] for v in rows]);ax.set(ylabel='Held-out digit accuracy (fraction)',ylim=(0,1));ax.legend();save(fig,path,'vision-accuracy')
        elif all(isinstance(r.get(k),dict) and 'accuracy' in r[k] for k in ['before','after']):
            fig,ax=plt.subplots(figsize=(7,4));ax.bar(['Before','After'],[r[k]['accuracy'] for k in ['before','after']]);ax.set(ylabel='Digit accuracy (fraction)',ylim=(0,1));save(fig,path,'digit-pilot')
        elif isinstance(r.get('by_family'),dict):
            values=[];names=[]
            for name,v in r['by_family'].items():
                numerator=v.get('solved',v.get('tasks_solved_at_k'));denominator=v.get('tasks',0)
                if numerator is not None and denominator:names.append(name);values.append(numerator/denominator)
            if not names:continue
            fig,ax=plt.subplots(figsize=(9,4));ax.bar(names,values);ax.tick_params(axis='x',rotation=25);ax.set(ylabel='Tasks solved / tasks evaluated',ylim=(0,1));save(fig,path,'executable-success-observed-k' if 'tasks_solved_at_k' in next(iter(r['by_family'].values())) else 'greedy-success')
        else:skipped.append(str(path))
    (a.output/'sources.json').write_text(json.dumps(dict(plotted=plotted,skipped=skipped,note='Separate tasks/schemas. Bars with multiple seeds use sample SD; no significance or cross-task quality claims.'),indent=2)+'\n')
    print(f'Created {len(plotted)} figures; {len(skipped)} JSON files did not match an auxiliary-study schema')
if __name__=='__main__':main()
