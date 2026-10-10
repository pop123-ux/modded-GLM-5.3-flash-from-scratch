"""Deterministic examples and explicit full-sequence/completion-only masks."""
import hashlib
import json
import torch
from .tasks import FAMILIES, make_task
from .tokenizer import ByteTokenizer

def training_tasks(start, count, seed=42, families=None):
    import random
    pool = [f for f in FAMILIES if not families or f.name in families]
    if not pool:
        raise ValueError('no families selected')
    for index in range(start, start+count):
        rng = random.Random(seed * 10_100_019 + index)
        family = pool[rng.randrange(len(pool))]
        yield make_task(family, split='pretrain', index=index, seed=seed)

def batch_from_tasks(tasks, sequence_length, device, loss_scope='full', trim_padding=False):
    tokenizer = ByteTokenizer()
    inputs, full_labels, completion_labels = [], [], []
    for task in tasks:
        ids = tokenizer.encode(task.reference_source, bos=True, eos=True)
        if len(ids) > sequence_length + 1:
            raise ValueError(f'{task.task_id}: {len(ids)} bytes exceed context {sequence_length}; increase --sequence-length (no silent truncation)')
        prompt_length = len(tokenizer.encode(task.prompt, bos=True))
        row = ids + [tokenizer.pad_id] * (sequence_length+1-len(ids))
        labels = [x if x != tokenizer.pad_id else -100 for x in row[1:]]
        completion = labels.copy()
        completion[:prompt_length-1] = [-100] * (prompt_length-1)
        inputs.append(row[:-1]); full_labels.append(labels); completion_labels.append(completion)
    x = torch.tensor(inputs, dtype=torch.long, device=device)
    full = torch.tensor(full_labels, dtype=torch.long, device=device)
    completion = torch.tensor(completion_labels, dtype=torch.long, device=device)
    if trim_padding:
        # Right-padding only: preserve every supervised next-byte target, including EOS.
        valid_columns = (full != -100).any(dim=0).nonzero(as_tuple=True)[0]
        width = int(valid_columns[-1]) + 1
        x, full, completion = x[:, :width], full[:, :width], completion[:, :width]
    return x, full if loss_scope == 'full' else completion, completion

def task_digest(tasks):
    content = [{'id':t.task_id,'prompt':t.prompt,'completion':t.reference_completion,'cases':t.cases} for t in tasks]
    return hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()
