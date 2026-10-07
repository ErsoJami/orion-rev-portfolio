"""Independent typed slots and separately normalized LM/action objectives."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .tasks import FIELDS, MAX_ACTIONS, TOOL_FIELD, TOOLS, make_plan


class ActionHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.count = nn.Linear(hidden_size, MAX_ACTIONS + 1)
        self.tool = nn.Linear(hidden_size, MAX_ACTIONS * len(TOOLS))
        self.fields = nn.ModuleDict(
            {"arg_" + f: nn.Linear(hidden_size, MAX_ACTIONS * len(v)) for f, v in FIELDS.items()}
        )

    def forward(self, hidden: torch.Tensor) -> dict:
        # FP32 head avoids AMP indexed-write dtype mismatches and tiny-logit
        # underflow.
        hidden = hidden.float()
        return {
            "count": self.count(hidden),
            "tool": self.tool(hidden).reshape(-1, MAX_ACTIONS, len(TOOLS)),
            **{
                f: self.fields["arg_" + f](hidden).reshape(-1, MAX_ACTIONS, len(values))
                for f, values in FIELDS.items()
            },
        }


def action_loss(logits: dict, target: dict) -> tuple[torch.Tensor, dict]:
    terms = {"count": F.cross_entropy(logits["count"], target["count"])}
    for field in ("tool", *FIELDS):
        active = target[field] != -100
        if active.any():
            terms[field] = F.cross_entropy(logits[field][active], target[field][active])
    # Count/tool group and argument group have separate normalization.
    base = terms["count"] + terms.get("tool", terms["count"] * 0)
    args = [terms[f] for f in FIELDS if f in terms]
    loss = base + (torch.stack(args).mean() if args else 0)
    return loss, {f"loss/action_{k}": float(v.detach()) for k, v in terms.items()}


def lm_loss(backbone, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    active = labels[:, 1:] != -100
    if not active.any():
        return hidden.sum() * 0
    # Only answer token logits: [number_of_labelled_tokens, vocabulary].
    logits = backbone.lm_head(hidden[:, :-1][active]).float()
    return F.cross_entropy(logits, labels[:, 1:][active])


def predict_plans(logits: dict, rows: list[dict]) -> list[dict]:
    decisions = {f: value.argmax(-1).detach().cpu().tolist() for f, value in logits.items()}
    plans = []
    for b, row in enumerate(rows):
        actions = []
        for i in range(decisions["count"][b]):
            tool = TOOLS[decisions["tool"][b][i]]
            field = TOOL_FIELD.get(tool)
            args = {field: FIELDS[field][decisions[field][b][i]]} if field else {}
            actions.append({"tool": tool, "arguments": args})
        plans.append(make_plan(row["command"], actions))
    return plans
