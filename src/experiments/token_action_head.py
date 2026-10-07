"""Six learned queries: one global decision/count query and five ordered slots."""

import math

import torch
from torch import nn

from .action_data import DECISIONS
from .tasks import FIELDS, MAX_ACTIONS, TOOLS


class TokenActionHead(nn.Module):
    def __init__(self, hidden_size, width=192, query_std=1.0):
        super().__init__()
        if width % 4 or width % 2:
            raise ValueError("attention head width must be divisible by four")
        self.width = width
        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_projection = nn.Linear(hidden_size, width)
        self.context = nn.Linear(hidden_size, width)
        # Distinct slot queries must not be drowned out by the shared context.
        self.queries = nn.Parameter(torch.randn(MAX_ACTIONS + 1, width) * query_std)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True, dropout=0.0)
        self.norm1 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
        )
        self.norm2 = nn.LayerNorm(width)
        self.count = nn.Linear(width, MAX_ACTIONS + 1)
        self.decision = nn.Linear(width, len(DECISIONS))
        self.tool = nn.Linear(width, len(TOOLS))
        self.fields = nn.ModuleDict(
            {"arg_" + f: nn.Linear(width, len(v)) for f, v in FIELDS.items()}
        )

    def forward(self, hidden, attention_mask):
        hidden = hidden.float()
        mask = attention_mask.bool()
        if not mask.any(dim=1).all():
            raise ValueError("empty token sequence")
        tokens = self.input_projection(self.input_norm(hidden))
        # Positions distinguish ordered commands. Padding never participates in
        # attention.
        positions = torch.arange(hidden.shape[1], device=hidden.device).float()[:, None]
        frequencies = torch.exp(
            torch.arange(0, self.width, 2, device=hidden.device).float()
            * (-math.log(10000) / self.width)
        )
        encoding = torch.zeros(hidden.shape[1], self.width, device=hidden.device)
        encoding[:, 0::2], encoding[:, 1::2] = torch.sin(positions * frequencies), torch.cos(
            positions * frequencies
        )
        tokens = tokens + encoding[None]
        last = mask.long().sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        queries = self.queries[None] + self.context(pooled)[:, None]
        attended, _ = self.attention(
            queries, tokens, tokens, key_padding_mask=~mask, need_weights=False
        )
        slots = self.norm1(queries + attended)
        slots = self.norm2(slots + self.ffn(slots))
        global_state, action_slots = slots[:, 0], slots[:, 1:]
        return {
            "count": self.count(global_state),
            "decision": self.decision(global_state),
            "tool": self.tool(action_slots),
            **{f: self.fields["arg_" + f](action_slots) for f in FIELDS},
        }
