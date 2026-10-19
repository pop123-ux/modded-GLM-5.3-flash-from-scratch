#!/usr/bin/env python3
"""Plot completed, compatible training receipts; never invent missing results."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_benchmarks import compatibility_key
from scripts.plot_receipts import collect_receipts

COLORS = ['#2563eb', '#0d9488', '#7c3aed', '#ea580c', '#dc2626', '#0891b2', '#64748b']
ARM_COLORS = dict(hybrid='#2563eb', real_tokens='#0d9488', chunked_linear='#7c3aed',
                  dense_ffn='#ea580c', bias_balanced='#dc2626', future_bytes='#0891b2')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} already exists; choose a fresh output directory')
    runs, skipped = collect_receipts(args.root)
    for item in skipped:
        print(f"Skipped {item['path']}: {item['reason']}", file=sys.stderr)
    if not runs:
        parser.error('No complete coding-pretraining receipts found. Check CAMPAIGN, training completion, and train.jsonl; use plot_loss.py for unfinished runs.')
    groups = {}
    for path, receipt in runs:
        groups.setdefault(compatibility_key(receipt), []).append((path, receipt))
    args.output.mkdir(parents=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .15, 'svg.fonttype': 'none'})
    (args.output/'skipped-receipts.json').write_text(json.dumps(skipped, indent=2)+'\n', encoding='utf-8')
    rows, metadata = [], []
    for index, (key, runs) in enumerate(groups.items()):
        dest = args.output / f'group-{index}'
        dest.mkdir()
        arms = sorted({p.parent.name.rsplit('-seed', 1)[0] for p, _ in runs})
        palette = {arm: ARM_COLORS.get(arm, COLORS[i % len(COLORS)]) for i, arm in enumerate(arms)}
        device = runs[0][1]['device_name']
        counts = {arm: sum(p.parent.name.rsplit('-seed', 1)[0] == arm for p, _ in runs) for arm in arms}
        foot = f'{device} | completed runs per arm: ' + ', '.join(f'{a}={counts[a]}' for a in arms)
        def save(fig, name):
            fig.text(.01, .01, foot, fontsize=7, wrap=True)
            fig.tight_layout(rect=(0, .07, 1, 1))
            for extension in ('png', 'svg'):
                fig.savefig(dest / f'{name}.{extension}', dpi=180, facecolor='white')
            plt.close(fig)
        for axis_key, xlabel, name in [('step', 'Optimizer step', 'validation-vs-step'),
                                       ('training_seconds', 'Cumulative timed training (seconds)', 'validation-vs-time')]:
            fig, ax = plt.subplots(figsize=(10, 5))
            seen = set()
            for path, r in runs:
                arm = path.parent.name.rsplit('-seed', 1)[0]
                curve = r['validation_curve']
                ax.plot([x[axis_key] for x in curve], [x['val_loss'] for x in curve],
                        color=palette[arm], alpha=.75, linewidth=1.5,
                        label=arm if arm not in seen else None)
                seen.add(arm)
            ax.set(xlabel=xlabel, ylabel='Dev next-byte cross entropy (nats)',
                   title='Primary validation loss — each line is one seed')
            ax.legend(fontsize=8)
            save(fig, name)
        metrics = [('final_val_loss', 'Final dev next-byte CE (nats)', 'final-validation'),
                   ('tokens_per_second', 'Supervised main-task bytes / timed second', 'throughput'),
                   ('peak_vram_gib', 'Peak allocated GPU memory (GiB)', 'peak-memory')]
        for path, r in runs:
            rows.append(dict(group=index, arm=path.parent.name.rsplit('-seed', 1)[0], seed=r['seed'],
                run=str(path), final_val_loss=r['validation_curve'][-1]['val_loss'],
                tokens_per_second=r['tokens_seen']/r['training_seconds'], peak_vram_gib=r['peak_vram_gib'],
                training_seconds=r['training_seconds'], total_parameters=r['parameter_counts']['total']))
        local = [row for row in rows if row['group'] == index]
        for metric, ylabel, name in metrics:
            present = [a for a in arms if any(row[metric] is not None for row in local if row['arm'] == a)]
            if not present:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            for position, arm in enumerate(present):
                values = [row[metric] for row in local if row['arm'] == arm and row[metric] is not None]
                ax.bar(position, statistics.mean(values), color=palette[arm], alpha=.65,
                       yerr=statistics.stdev(values) if len(values) > 1 else None, capsize=4)
                ax.scatter(position + np.linspace(-.12, .12, len(values)), values, s=20, color=palette[arm])
            ax.set_xticks(range(len(present)), present, rotation=25, ha='right')
            ax.set(ylabel=ylabel, title='Mean, individual seeds and sample SD where n > 1')
            save(fig, name)
        routed = []
        for arm in arms:
            usages = [np.asarray(r['training_curve'][-1]['expert_usage'], dtype=float)
                      for p, r in runs if p.parent.name.rsplit('-seed', 1)[0] == arm
                      and r['config'].get('ffn_mode', 'moe') == 'moe']
            if usages:
                # Layers can differ across arms: average within each model first.
                vectors = [u.mean(axis=0) if u.ndim == 2 else u for u in usages]
                routed.append((arm, np.mean(vectors, axis=0)))
        if routed and len({len(v) for _, v in routed}) == 1:
            fig, ax = plt.subplots(figsize=(8, 4))
            im = ax.imshow(np.stack([v for _, v in routed]), cmap='viridis', aspect='auto')
            ax.set_yticks(range(len(routed)), [a for a, _ in routed])
            ax.set(xlabel='Routed expert index', title='Last training batch — assignment fraction averaged over layers/seeds')
            fig.colorbar(im, ax=ax, label='Assignment fraction')
            save(fig, 'expert-load')
        targets = [r.get('target_loss') for _, r in runs]
        if targets[0] is not None:
            fig, ax = plt.subplots(figsize=(10, 5))
            for pos, (path, r) in enumerate(runs):
                arm = path.parent.name.rsplit('-seed', 1)[0]
                hit = r.get('first_target_hit')
                if hit is None:
                    # Mark censored runs at their budget, not as successful target hits.
                    ax.scatter(r['training_seconds'], pos, marker='x', color='#dc2626')
                    ax.annotate('not reached', (r['training_seconds'], pos), xytext=(4, 0), textcoords='offset points', fontsize=7)
                else:
                    ax.scatter(hit['training_seconds'], pos, color=palette[arm])
            ax.set_yticks(range(len(runs)), [p.parent.name for p, _ in runs], fontsize=7)
            ax.set(xlabel='Timed training seconds', title=f'First evaluated dev CE ≤ {targets[0]} (red × = censored)')
            save(fig, 'time-to-target')
        metadata.append(dict(group=index, compatibility_key=list(key), arms=arms, runs=len(runs),
            note='Primary CE only. Timing excludes evaluation/saves. Error bars are sample SD, not confidence intervals. Expert load is a final-batch snapshot. Peak memory is allocated CUDA memory.'))
    with (args.output/'runs.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output/'plot-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(f'Plotted {len(rows)} completed runs in {len(groups)} compatibility groups into {args.output}')

if __name__ == '__main__':
    main()
