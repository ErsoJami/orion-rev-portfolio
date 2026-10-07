"""Small, controlled Action-vs-dynamic-choice experiment; not a Rev reproduction."""

from __future__ import annotations

import hashlib
import json
import random

import torch
from torch import nn

from .action_data import REASONS, row_from
from .tasks import FIELDS, TOOL_FIELD, TOOLS, labels, make_plan


class ChoiceActionHead(nn.Module):
    """Bilinear pointer readout as in Rev's public serve_local.py.

    No output parameter is tied to an option index or number of options.
    The causal backbone sees command + options; no LM logits are computed.
    """

    def __init__(self, hidden_size, width=256):
        super().__init__()
        self.width = width
        self.headq = nn.Linear(hidden_size, width, bias=False)
        self.headk = nn.Linear(hidden_size, width, bias=False)

    def forward(self, hidden, attention_mask, option_positions, option_mask):
        hidden = hidden.float()
        if not option_mask.any(-1).all():
            raise ValueError("every question needs at least one option")
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        options = hidden[batch[:, None], option_positions.clamp_min(0)]
        last = hidden[batch, attention_mask.long().sum(-1) - 1]
        scores = (self.headk(options) * self.headq(last)[:, None]).sum(-1) / self.width**0.5
        return scores.masked_fill(~option_mask.bool(), -torch.inf)


class MeanChoiceActionHead(ChoiceActionHead):
    """Same bilinear parameters; pool description tokens instead of line endpoint."""

    def forward(
        self, hidden, attention_mask, option_positions, option_mask, span_starts, span_ends
    ):
        hidden = hidden.float()
        positions = torch.arange(hidden.shape[1], device=hidden.device)[None, None]
        span_mask = (
            (positions >= span_starts[:, :, None])
            & (positions <= span_ends[:, :, None])
            & attention_mask.bool()[:, None]
            & option_mask.bool()[:, :, None]
        )
        sizes = span_mask.sum(-1)
        if not option_mask.any(-1).all() or (sizes[option_mask.bool()] == 0).any():
            raise ValueError("empty valid option span")
        options = (
            torch.einsum("bkt,bth->bkh", span_mask.float(), hidden) / sizes.clamp_min(1)[:, :, None]
        )
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        last = hidden[batch, attention_mask.long().sum(-1) - 1]
        scores = (self.headk(options) * self.headq(last)[:, None]).sum(-1) / self.width**0.5
        return scores.masked_fill(~option_mask.bool(), -torch.inf)


def signature(decision, actions):
    return json.dumps(
        [decision, [{"tool": a["tool"], "arguments": a["arguments"]} for a in actions]],
        ensure_ascii=False,
        sort_keys=True,
    )


def candidates():
    result = []
    for tool in TOOLS:
        field = TOOL_FIELD.get(tool)
        for value in FIELDS[field] if field else (None,):
            action = {"tool": tool, "arguments": {field: value} if field else {}}
            result.append(
                {
                    "decision": "EXECUTE",
                    "actions": [action],
                    "description": tool + " " + json.dumps(action["arguments"], ensure_ascii=False),
                }
            )
    for decision in ("CLARIFY", "UNSUPPORTED", "NO_ACTION"):
        result.append(
            {
                "decision": decision,
                "actions": [],
                "description": decision + ": " + REASONS[decision],
            }
        )
    return result


def dataset():
    # Twelve human-written source messages; the first alone is stage one.
    specs = [
        (
            "Поставь таймер на 60 секунд",
            "timer.create",
            60,
            "EXECUTE",
            ("Отсчитай одну минуту", "Мне нужен минутный таймер"),
        ),
        (
            "Поставь музыку на паузу",
            "music.pause",
            None,
            "EXECUTE",
            ("Приостанови воспроизведение песни", "Музыку пока останови"),
        ),
        (
            "Продолжи музыку",
            "music.resume",
            None,
            "EXECUTE",
            ("Возобнови воспроизведение трека", "Сними музыку с паузы"),
        ),
        (
            "Поставь громкость 40 процентов",
            "music.volume.set",
            40,
            "EXECUTE",
            ("Уровень звука установи на сорок процентов", "Сделай звук 40%"),
        ),
        (
            "Включи режим сон",
            "mode.activate",
            "sleep",
            "EXECUTE",
            ("Переведи часы в режим сна", "Активируй сон как режим часов"),
        ),
        (
            "Начни тренировку бег",
            "workout.start",
            "run",
            "EXECUTE",
            ("Начни записывать беговую тренировку", "Запусти тренировку по бегу"),
        ),
        (
            "Поставь таймер на 300 секунд",
            "timer.create",
            300,
            "EXECUTE",
            ("Поставь таймер на пять минут", "Мне нужен отсчёт на 5 минут"),
        ),
        (
            "Установи громкость 80 процентов",
            "music.volume.set",
            80,
            "EXECUTE",
            ("Громкость сделай восемьдесят процентов", "Выставь звук на 80%"),
        ),
        (
            "Включи режим работа",
            "mode.activate",
            "work",
            "EXECUTE",
            ("Переключи часы в рабочий режим", "Активируй режим работы"),
        ),
        (
            "Поставь таймер",
            None,
            None,
            "CLARIFY",
            ("Запусти таймер, длительность пока не выбрана", "Нужен таймер, время сообщу позже"),
        ),
        (
            "Запусти ядерную бомбу на меня",
            None,
            None,
            "UNSUPPORTED",
            ("Закажи такси", "Отправь письмо другу"),
        ),
        (
            "Не ставь музыку на паузу",
            None,
            None,
            "NO_ACTION",
            ("Я не прошу приостанавливать музыку", "Спасибо, ничего менять не нужно"),
        ),
    ]
    train, probes = [], []
    for command, tool, value, decision, variants in specs:
        field = TOOL_FIELD.get(tool)
        actions = [{"tool": tool, "arguments": {field: value} if field else {}}] if tool else []
        train.append(row_from(command, actions, decision, "train"))
        for variant in variants:
            row = row_from(variant, actions, decision, "probe")
            row["source_train_id"] = train[-1]["id"]
            probes.append(row)
    assert not {r["command"] for r in train} & {r["command"] for r in probes}
    payload = {"train": train, "probe": probes, "candidates": candidates()}
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    return payload


def encode_choices(
    tokenizer,
    rows,
    orders,
    device="cuda",
    max_tokens=1024,
    *,
    catalog=None,
    language="ru",
    option_spans=False,
    instructions=None,
):
    sequences, positions, texts, targets, span_starts, span_ends = [], [], [], [], [], []
    catalog = candidates() if catalog is None else catalog
    if language not in ("ru", "en"):
        raise ValueError("unsupported prompt language")
    for row, order in zip(rows, orders, strict=True):
        if (
            len(order) != len(set(order))
            or not order
            or any(i < 0 or i >= len(catalog) for i in order)
        ):
            raise ValueError("options must be nonempty and unique")
        prefix = (
            (
                "Выбери подходящий план для локальной команды часов.\nКоманда: "
                + row["command"]
                + "\nВарианты:\n"
            )
            if language == "ru"
            else (
                "Choose the appropriate plan for a local watch command.\nCommand: "
                + row["command"]
                + "\nOptions:\n"
            )
        )
        if instructions is not None:
            prefix = (
                instructions
                + ("\nКоманда: " if language == "ru" else "\nCommand: ")
                + row["command"]
                + ("\nВарианты:\n" if language == "ru" else "\nOptions:\n")
            )
        sequence = tokenizer.encode(prefix, add_special_tokens=False)
        text, ends, starts, content_ends = prefix, [], [], []
        for i, option_id in enumerate(order):
            part = f"{i + 1}. {catalog[option_id]['description']}\n"
            part_ids = tokenizer.encode(part, add_special_tokens=False)
            if option_spans:
                mapped = tokenizer(part, add_special_tokens=False, return_offsets_mapping=True)
                if mapped["input_ids"] != part_ids:
                    raise ValueError("offset tokenization changed input IDs")
                prefix_length = len(f"{i + 1}. ")
                active = [
                    j
                    for j, (start, end) in enumerate(mapped["offset_mapping"])
                    if end > prefix_length and start < len(part) - 1
                ]
                if not active:
                    raise ValueError("option description has no tokens")
                starts.append(len(sequence) + active[0])
                content_ends.append(len(sequence) + active[-1])
            sequence.extend(part_ids)
            ends.append(len(sequence) - 1)
            text += part
        suffix = "Выбери один вариант." if language == "ru" else "Choose one option."
        sequence.extend(tokenizer.encode(suffix, add_special_tokens=False))
        text += suffix
        if len(sequence) > max_tokens:
            raise ValueError(f"{len(sequence)} tokens exceed {max_tokens}; no truncation permitted")
        wanted = signature(row["decision"], row["plan"]["actions"])
        index = next(
            i
            for i, option_id in enumerate(order)
            if signature(catalog[option_id]["decision"], catalog[option_id]["actions"]) == wanted
        )
        sequences.append(sequence)
        positions.append(ends)
        span_starts.append(starts if option_spans else ends)
        span_ends.append(content_ends if option_spans else ends)
        texts.append(text)
        targets.append(index)
    width, options = max(map(len, sequences)), max(map(len, positions))
    ids = torch.full((len(rows), width), tokenizer.pad_token_id, device=device, dtype=torch.long)
    mask, ends = torch.zeros_like(ids), torch.zeros(
        (len(rows), options), device=device, dtype=torch.long
    )
    available = torch.zeros_like(ends, dtype=torch.bool)
    starts_tensor, content_ends_tensor = torch.zeros_like(ends), torch.zeros_like(ends)
    for b, (sequence, end) in enumerate(zip(sequences, positions, strict=True)):
        ids[b, : len(sequence)] = torch.tensor(sequence, device=device)
        mask[b, : len(sequence)] = 1
        ends[b, : len(end)] = torch.tensor(end, device=device)
        available[b, : len(end)] = True
        starts_tensor[b, : len(end)] = torch.tensor(span_starts[b], device=device)
        content_ends_tensor[b, : len(end)] = torch.tensor(span_ends[b], device=device)
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "option_positions": ends,
        "option_mask": available,
        "targets": torch.tensor(targets, device=device),
        "option_span_starts": starts_tensor,
        "option_span_ends": content_ends_tensor,
        "orders": orders,
        "texts": texts,
        "action_labels": labels(rows, device),
    }


def decode_choices(logits, rows, orders, *, catalog=None):
    catalog = candidates() if catalog is None else catalog
    output = []
    for index, row, order in zip(
        logits.detach().argmax(-1).cpu().tolist(), rows, orders, strict=True
    ):
        choice = catalog[order[index]]
        plan = make_plan(row["command"], choice["actions"])
        if not choice["actions"]:
            plan["clarification"] = REASONS[choice["decision"]]
        output.append(
            {
                "decision": choice["decision"],
                "plan": plan,
                "raw_count": len(choice["actions"]),
                "raw_status_count_consistent": True,
                "option_index": index,
                "candidate_id": order[index],
            }
        )
    return output


def shuffled_orders(rows, seed):
    rng = random.Random(seed)
    output = []
    for _ in rows:
        order = list(range(len(candidates())))
        rng.shuffle(order)
        output.append(order)
    return output
