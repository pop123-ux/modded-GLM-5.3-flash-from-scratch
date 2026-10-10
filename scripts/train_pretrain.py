#!/usr/bin/env python3
"""Train matched Vuk/modded runs and record token-weighted held-out losses."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm53_flash import ByteTokenizer, GLM53FlashFromScratch, ModelConfig
from glm53_flash.baseline import build_baseline
from glm53_flash.data import training_tasks, batch_from_tasks, task_digest
from glm53_flash.optim import build_optimizer
from glm53_flash.runtime import save_checkpoint, load_checkpoint
from glm53_flash.tasks import frozen_tasks

# Retain the helper imported by upstream curriculum/diversity experiments.
def batch_for(tokenizer, *, step, batch_size, sequence_length, seed, device):
    x, labels, _ = batch_from_tasks(list(training_tasks(step*batch_size,batch_size,seed)),sequence_length,device)
    return x, labels, int((labels != -100).sum())

def synchronize(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)
    elif device.type == 'mps': torch.mps.synchronize()

@torch.no_grad()
def validation(model, tasks, sequence_length, batch_size, device, loss_scope='full', amp=False, trim_padding=False):
    was_training = model.training
    model.eval()
    totals = [0.,0.,0,0]
    for start in range(0,len(tasks),batch_size):
        x, labels, completion = batch_from_tasks(tasks[start:start+batch_size],sequence_length,device,loss_scope,trim_padding=trim_padding)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):
            logits, _ = model(x)
        for j, y in enumerate((labels,completion)):
            totals[j] += float(F.cross_entropy(logits.float().flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum'))
            totals[j+2] += int((y != -100).sum())
    model.train(was_training)
    return {'val_loss':totals[0]/totals[2], 'completion_val_loss':totals[1]/totals[3],
            'val_tokens':totals[2], 'completion_val_tokens':totals[3]}

def cpu_name():
    value = platform.processor()
    path = Path('/proc/cpuinfo')
    if not value and path.exists():
        value = next((line.split(':',1)[1].strip() for line in path.read_text().splitlines() if line.startswith('model name')), '')
    return value or platform.machine()

def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('glm53_flash','scripts','baselines/upstream/glm53_flash') for p in sorted((root/folder).glob('*.py'))}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--implementation',choices=['modded','vuk'],default='modded')
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--checkpoints',default='')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--sequence-length',type=int,default=320)
    p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--muon-lr',type=float,default=0.02)
    p.add_argument('--optimizer',choices=['adamw','muon'],default='adamw')
    p.add_argument('--weight-decay',type=float,default=0.1)
    p.add_argument('--balance-weight',type=float,default=0.)
    p.add_argument('--z-weight',type=float,default=0.)
    p.add_argument('--future-weight',type=float,default=0.1)
    p.add_argument('--trim-padding',action='store_true',help='Crop right-padded batches to the last supervised target')
    p.add_argument('--loss-scope',choices=['full','completion'],default='full')
    p.add_argument('--schedule',choices=['constant','cosine'],default='constant')
    p.add_argument('--warmup-steps',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--data-seed',type=int,default=42)
    p.add_argument('--families',default='')
    p.add_argument('--dim',type=int)
    p.add_argument('--layers',type=int)
    p.add_argument('--eval-every',type=int,default=20)
    p.add_argument('--eval-per-family',type=int,default=4)
    p.add_argument('--target-loss',type=float)
    p.add_argument('--device',choices=['auto','cuda','mps','cpu'],default='auto')
    p.add_argument('--amp',action='store_true',help='CUDA BF16 only; fails on unsupported devices')
    p.add_argument('--threads',type=int,default=1)
    args = p.parse_args()
    if min(args.steps,args.batch_size,args.eval_every,args.eval_per_family,args.threads,args.sequence_length) <= 0:
        p.error('counts must be positive')
    if args.warmup_steps < 0 or args.balance_weight < 0 or args.z_weight < 0 or args.future_weight < 0:
        p.error('weights/warmup must be nonnegative')
    if args.implementation == 'vuk' and (args.balance_weight or args.z_weight):
        p.error('pinned Vuk model has no differentiable router loss API')
    cfg = json.loads(args.config.read_text()) if args.config else {}
    for name in ('dim','layers'):
        if getattr(args,name) is not None: cfg[name]=getattr(args,name)
    cfg['max_sequence_length'] = max(cfg.get('max_sequence_length',512),args.sequence_length)
    config = ModelConfig(**cfg)
    if args.implementation == 'vuk' and (config.padding_aware_routing or config.linear_chunk_size or config.ffn_mode != 'moe' or config.router_bias_rate or config.future_heads):
        p.error('pinned Vuk implementation does not support the new model options')
    torch.set_num_threads(args.threads)
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    name = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if args.device=='auto' else args.device
    device=torch.device(name)
    if args.amp and (name != 'cuda' or not torch.cuda.is_bf16_supported()):
        p.error('--amp requires CUDA BF16 support')
    selected={x for x in args.families.split(',') if x} or None
    tasks=[t for t in frozen_tasks('dev',per_family=args.eval_per_family) if not selected or t.family in selected]
    if not tasks or selected and selected != {t.family for t in tasks}: p.error('unknown family')
    # Fail before creating the output directory for oversized examples.
    batch_from_tasks(tasks,args.sequence_length,device,args.loss_scope,trim_padding=args.trim_padding)
    batch_from_tasks(list(training_tasks(0,args.batch_size,args.data_seed,selected)),args.sequence_length,device,args.loss_scope,trim_padding=args.trim_padding)
    model=(build_baseline(config) if args.implementation=='vuk' else GLM53FlashFromScratch(config)).to(device)
    optimizer=build_optimizer(model,args.optimizer,args.learning_rate,args.muon_lr,args.weight_decay)
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    marks={int(x) for x in args.checkpoints.split(',') if x} | {args.steps}
    if any(x < 1 or x > args.steps for x in marks): raise ValueError('checkpoint outside training range')
    base_lrs=[g['lr'] for g in optimizer.param_groups]
    counts=model.parameter_counts()
    meta={'implementation':args.implementation,'seed':args.seed,'data_seed':args.data_seed,'parameter_counts':counts}
    saved=[dict(step=0,**save_checkpoint(model,args.output/'checkpoint-0000',meta))]
    initial=validation(model,tasks,args.sequence_length,args.batch_size,device,args.loss_scope,args.amp,args.trim_padding)
    history=[dict(step=0,training_seconds=0.,**initial)]
    curve=[]; training_seconds=0.; wall_started=time.perf_counter(); first_hit=None
    if args.target_loss is not None and initial['val_loss'] <= args.target_loss:
        first_hit={'step':0,'training_seconds':0.,'val_loss':initial['val_loss']}
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    model.train()
    with (args.output/'train.jsonl').open('w') as log:
        for step in range(1,args.steps+1):
            synchronize(device); started=time.perf_counter()
            x,y,_=batch_from_tasks(list(training_tasks((step-1)*args.batch_size,args.batch_size,args.data_seed,selected)),args.sequence_length,device,args.loss_scope,trim_padding=args.trim_padding)
            factor=1.
            if step <= args.warmup_steps: factor=step/max(1,args.warmup_steps)
            elif args.schedule=='cosine': factor=0.1+0.9*0.5*(1+math.cos(math.pi*(step-args.warmup_steps)/max(1,args.steps-args.warmup_steps)))
            for g,lr in zip(optimizer.param_groups,base_lrs): g['lr']=lr*factor
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=args.amp):
                logits,usage=model(x)
                ce=F.cross_entropy(logits.float().flatten(0,1),y.flatten(),ignore_index=-100)
                balance,z = model.router_losses() if hasattr(model,'router_losses') else (ce.new_zeros(()),ce.new_zeros(()))
                future=model.future_prediction_loss(y) if config.future_heads and hasattr(model,'future_prediction_loss') else ce.new_zeros(())
                loss=ce+args.balance_weight*balance+args.z_weight*z+args.future_weight*future
            if not bool(torch.isfinite(loss)): raise FloatingPointError('non-finite loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            if hasattr(model, "update_router_biases"): model.update_router_biases()
            synchronize(device)
            elapsed=time.perf_counter()-started; training_seconds+=elapsed
            row={'step':step,'loss':float(ce.detach()),'balance_loss':float(balance.detach()),'z_loss':float(z.detach()),'future_loss':float(future.detach()),
                 'gradient_norm':float(norm),'tokens':int((y!=-100).sum()),'seconds':elapsed,
                 'training_seconds':training_seconds,'lr_factor':factor,'expert_usage':usage.detach().cpu().tolist(),
                 'input_positions':x.numel(),'nonpad_input_positions':int((x!=0).sum()),'effective_sequence_length':x.shape[1]}
            curve.append(row); log.write(json.dumps(row)+'\n'); log.flush()
            if step==1 or step%args.eval_every==0: print(json.dumps({k:row[k] for k in ('step','loss','training_seconds')}),flush=True)
            if step%args.eval_every==0 or step==args.steps:
                values=validation(model,tasks,args.sequence_length,args.batch_size,device,args.loss_scope,args.amp,args.trim_padding)
                history.append(dict(step=step,training_seconds=training_seconds,**values))
                if first_hit is None and args.target_loss is not None and values['val_loss']<=args.target_loss:
                    first_hit=dict(step=step,training_seconds=training_seconds,val_loss=values['val_loss'])
            if step in marks:
                checkpoint=args.output/f'checkpoint-{step:04d}'
                saved.append(dict(step=step,**save_checkpoint(model,checkpoint,dict(meta,step=step))))
                # Optimizer state is separate and never loaded by inference.
                torch.save(optimizer.state_dict(),checkpoint/'optimizer.pt')
    synchronize(device)
    receipt={'schema_version':'2.0','status':'complete','stage':'coding_pretraining',
        'config':model.config.to_dict(),'parameter_counts':counts,'implementation':args.implementation,
        'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'seed':args.seed,'steps':args.steps,'batch_size':args.batch_size,'sequence_length':args.sequence_length,
        'learning_rate':args.learning_rate,'tokens_seen':sum(r['tokens'] for r in curve),'training_seconds':training_seconds,
        'elapsed_seconds':time.perf_counter()-wall_started,'device':str(device),'torch':torch.__version__,
        'python':platform.python_version(),'platform':platform.platform(),
        'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else cpu_name(),
        'peak_vram_gib':torch.cuda.max_memory_allocated()/1024**3 if device.type=='cuda' else None,
        'validation_sha256':task_digest(tasks),'source_sha256':source_hashes(),
        'input_positions_seen':sum(r['input_positions'] for r in curve),
        'nonpad_input_positions_seen':sum(r['nonpad_input_positions'] for r in curve),
        'target_loss':args.target_loss,'first_target_hit':first_hit,'validation_curve':history,
        'checkpoints':saved,'training_curve':curve,
        'timing_policy':'Synchronized training includes data construction, forward, backward, optimizer; excludes validation, logs and saves. No untimed optimization steps.'}
    (args.output/'training-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':'complete','final_validation':history[-1],'training_seconds':training_seconds},indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
