"""Select complete coding-pretraining receipts from mixed experiment folders."""
import json,math
from pathlib import Path
from scripts.analyze_benchmarks import compatibility_key

def collect_receipts(root):
    root=Path(root);paths=[root] if root.is_file() else sorted(root.rglob('training-receipt.json'))
    valid=[];skipped=[]
    for path in paths:
        try:
            r=json.loads(path.read_text(encoding='utf-8'))
            if r.get('status')!='complete':raise ValueError('not complete')
            if r.get('stage','coding_pretraining')!='coding_pretraining':raise ValueError('different experiment stage (e.g. RL); use its own analysis')
            compatibility_key(r)
            for key in ('device_name','config','parameter_counts','training_curve','validation_curve','tokens_seen','seed'):r[key]
            if not r['training_curve'] or not r['validation_curve']:raise ValueError('empty learning curve')
            if not math.isfinite(r['training_seconds']) or r['training_seconds']<=0:raise ValueError('invalid elapsed training time')
            if not math.isfinite(r['validation_curve'][-1]['val_loss']):raise ValueError('invalid final validation loss')
            if 'total' not in r['parameter_counts']:raise ValueError('missing total parameter count')
            valid.append((path,r))
        except (KeyError,ValueError,TypeError) as exc:skipped.append(dict(path=str(path),reason=str(exc)))
    return valid,skipped
