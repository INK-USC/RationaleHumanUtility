import torch
import torch.nn.functional as F


def calc_task_loss(logits, targets, reduction='mean', class_weights=None):
    assert len(logits) == len(targets)
    return F.cross_entropy(logits, targets, weight=class_weights, reduction=reduction)