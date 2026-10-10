"""Auxiliary future-byte prediction with explicit offsets and ignored targets."""
import torch
import torch.nn.functional as F

def future_byte_loss(head_logits, labels, anchor):
    losses = []
    for shift, logits in enumerate(head_logits, start=1):
        if labels.shape != logits.shape[:2]:
            raise ValueError('labels must match head batch/time dimensions')
        if shift >= labels.shape[1]:
            continue
        # labels[t] already means byte t+1; shifting by one means byte t+2.
        targets = labels[:,shift:]
        values = logits[:,:-shift]
        valid = targets != -100
        if bool(valid.any()):
            losses.append(F.cross_entropy(values.float().flatten(0,1),targets.flatten(),ignore_index=-100))
    return torch.stack(losses).mean() if losses else anchor.sum() * 0
