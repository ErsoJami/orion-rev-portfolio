"""Trainable continuation of the released Rev adapters, without merging into base weights."""

import torch
from torch import nn
from torch.nn import functional as F


class RevLoRA(nn.Module):
    def __init__(self, base, weights, scale=2.0):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Parameter(
            weights["a"].to(device=base.weight.device, dtype=torch.float32).clone()
        )
        self.b = nn.Parameter(
            weights["b"].to(device=base.weight.device, dtype=torch.float32).clone()
        )
        self.scale = scale

    def forward(self, x):
        original = self.base(x)
        # Match upstream training: FP32 adapter products, BF16 residual to
        # backbone.
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = F.linear(F.linear(x.float(), self.a), self.b) * self.scale
        return original + delta.to(original.dtype)


def pointer_logits(backbone, headq, headk, encoded, pad_id, temperature):
    device = next(backbone.parameters()).device
    length = ((max(len(ids) for ids, _ in encoded) + 15) // 16) * 16
    inputs = torch.full((len(encoded), length), pad_id, dtype=torch.long, device=device)
    for i, (ids, _) in enumerate(encoded):
        inputs[i, : len(ids)] = torch.tensor(ids, device=device)
    hidden = backbone(input_ids=inputs, attention_mask=None, use_cache=False).last_hidden_state
    return torch.stack(
        [
            (headk(hidden[i, positions].float()) * headq(hidden[i, len(ids) - 1].float())).sum(-1)
            / 16
            / temperature
            for i, (ids, positions) in enumerate(encoded)
        ]
    )


def paired_loss(logits, targets, orders, weight=0.1):
    """Two adjacent views of each command; align probability axes before KL."""
    ce = F.cross_entropy(logits, targets)
    logp = logits.log_softmax(-1)
    aligned = torch.stack(
        [
            logp[i, torch.argsort(torch.tensor(order, device=logits.device))]
            for i, order in enumerate(orders)
        ]
    )
    left, right = aligned[0::2], aligned[1::2]
    kl = 0.5 * (
        (left.exp() * (left - right)).sum(-1).mean() + (right.exp() * (right - left)).sum(-1).mean()
    )
    return ce + weight * kl, ce, kl
