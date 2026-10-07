"""Finite typed-decision benchmark; generated labels are not a production corpus.

Templates are split before generation; exact prompts cannot cross splits.
The finite candidate vocabularies are public and include held-out argument values.
"""

from __future__ import annotations

import hashlib
import json
import random
from uuid import NAMESPACE_URL, uuid5

import torch

from data.contracts.agent import deserialize_target, serialize_target, tool_registry, validate_plan

TOOLS = (
    "timer.create",
    "music.pause",
    "music.resume",
    "music.volume.set",
    "mode.activate",
    "workout.start",
)
FIELDS = {
    "durationSeconds": (10, 30, 60, 120, 300, 600, 1200, 1800, 3600),
    "level": (0, 20, 40, 60, 80, 100),
    "mode": ("sleep", "work", "workout"),
    "type": ("walk", "run", "general"),
}
TOOL_FIELD = {
    "timer.create": "durationSeconds",
    "music.volume.set": "level",
    "mode.activate": "mode",
    "workout.start": "type",
}
MAX_ACTIONS = 5
CLARIFICATION = "Уточните команду: этот эксперимент поддерживает только шесть инструментов."
TEMPLATES = {
    "timer.create": (
        "Поставь таймер на {v} секунд",
        "Таймер на {v} секунд",
        "Отсчитай {v} секунд",
        "Создай таймер длительностью {v} секунд",
        "Запусти отсчёт на {v} секунд",
        "Мне нужен таймер на {v} секунд",
    ),
    "music.pause": (
        "Поставь музыку на паузу",
        "Приостанови музыку",
        "Пауза для музыки",
        "Приостанови воспроизведение",
        "Пожалуйста, останови воспроизведение музыки",
        "Музыку пока приостанови",
    ),
    "music.resume": (
        "Продолжи музыку",
        "Возобнови музыку",
        "Сними музыку с паузы",
        "Продолжи воспроизведение",
        "Пожалуйста, возобнови воспроизведение музыки",
        "Пусть музыка снова играет",
    ),
    "music.volume.set": (
        "Громкость {v} процентов",
        "Поставь громкость {v} процентов",
        "Установи громкость {v} процентов",
        "Сделай громкость {v} процентов",
        "Измени громкость на {v} процентов",
        "Хочу громкость {v} процентов",
    ),
    "mode.activate": (
        "Включи режим {v}",
        "Активируй режим {v}",
        "Переключись в режим {v}",
        "Поставь режим {v}",
        "Выбери режим {v}",
        "Мне нужен режим {v}",
    ),
    "workout.start": (
        "Начни тренировку {v}",
        "Запусти тренировку {v}",
        "Включи тренировку {v}",
        "Начни упражнение {v}",
        "Запиши начало тренировки {v}",
        "Хочу начать тренировку {v}",
    ),
}
DISPLAY = {
    "sleep": "сон",
    "work": "работа",
    "workout": "тренировка",
    "walk": "ходьба",
    "run": "бег",
    "general": "общая",
}
UNSUPPORTED = {
    "train": ("Закажи такси", "Купи билеты", "Отправь письмо", "Открой браузер"),
    "validation": ("Вызови машину через интернет", "Оплати покупку онлайн"),
    "test": ("Забронируй гостиницу", "Переведи деньги другу"),
}
TIMER_VALUES = {"train": (10, 30, 60, 300, 1200), "validation": (120, 600), "test": (1800, 3600)}


def make_plan(command: str, actions: list[dict]) -> dict:
    plan = {
        "schemaVersion": 1,
        "utteranceId": str(uuid5(NAMESPACE_URL, command)),
        "actions": [
            {"id": f"a{i + 1}", "tool": a["tool"], "arguments": a["arguments"], "dependsOn": []}
            for i, a in enumerate(actions)
        ],
        "clarification": CLARIFICATION if not actions else None,
        "confirmation": (
            "REQUIRED"
            if any(tool_registry()[a["tool"]]["requiresConfirmation"] for a in actions)
            else "NONE"
        ),
    }
    validate_plan(plan)
    return plan


def prompt(command: str) -> str:
    return "Разбери локальную команду для часов.\nКоманда: " + command + "\nОтвет:"


def text_target(actions: list[dict]) -> str:
    if not actions:
        return CLARIFICATION
    # Describes a proposed plan, never claims that an action has executed.
    return (
        "Предлагаю план: "
        + "; ".join(
            a["tool"]
            + (
                " " + json.dumps(a["arguments"], ensure_ascii=False, sort_keys=True)
                if a["arguments"]
                else ""
            )
            for a in actions
        )
        + "."
    )


def build_suite(seed: int = 17, sizes=(192, 40, 48)) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    splits = {}
    for split, size in zip(("train", "validation", "test"), sizes, strict=True):
        rows, seen, attempts = [], set(), 0
        while len(rows) < size:
            attempts += 1
            if attempts > size * 2000:
                raise ValueError("requested suite exceeds distinct synthetic combinations")
            i = len(rows)
            count = (1, 2, 3, 4, 5)[i % 5] if split == "test" else (1, 2, 3)[i % 3]
            if i < len(UNSUPPORTED[split]):
                command, actions = UNSUPPORTED[split][i], []
            else:
                phrases, actions = [], []
                for tool in rng.sample(TOOLS, count):
                    field = TOOL_FIELD.get(tool)
                    candidates = (
                        TIMER_VALUES[split]
                        if field == "durationSeconds"
                        else FIELDS.get(field, (None,))
                    )
                    value = rng.choice(candidates)
                    template = (
                        rng.randrange(4) if split == "train" else 4 if split == "validation" else 5
                    )
                    phrases.append(TEMPLATES[tool][template].format(v=DISPLAY.get(value, value)))
                    actions.append({"tool": tool, "arguments": {field: value} if field else {}})
                command = " и ".join(phrases)
            if command in seen:
                continue
            seen.add(command)
            plan = make_plan(command, actions)
            rows.append(
                {
                    "id": hashlib.sha256(command.encode()).hexdigest()[:16],
                    "split": split,
                    "command": command,
                    "prompt": prompt(command),
                    "plan": plan,
                    "json_target": serialize_target(plan),
                    "text_target": text_target(actions),
                }
            )
        splits[split] = rows
    audit_suite(splits)
    return splits


def audit_suite(suite: dict) -> dict:
    seen = set()
    for split, rows in suite.items():
        for row in rows:
            assert row["split"] == split
            assert row["command"] not in seen, "cross-split prompt leakage"
            seen.add(row["command"])
            validate_plan(row["plan"])
    digest = hashlib.sha256(
        json.dumps(suite, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    return {
        "sha256": digest,
        "sizes": {s: len(r) for s, r in suite.items()},
        "tools": list(TOOLS),
        "max_actions": MAX_ACTIONS,
        "split_policy": "disjoint template families; disjoint timer values; exact prompt audit",
        "is_production_quality_holdout": False,
    }


def labels(rows: list[dict], device) -> dict[str, torch.Tensor]:
    count = torch.tensor([len(r["plan"]["actions"]) for r in rows], device=device)
    tools = torch.full((len(rows), MAX_ACTIONS), -100, dtype=torch.long, device=device)
    fields = {f: torch.full_like(tools, -100) for f in FIELDS}
    for b, row in enumerate(rows):
        for i, action in enumerate(row["plan"]["actions"]):
            tools[b, i] = TOOLS.index(action["tool"])
            for field, value in action["arguments"].items():
                fields[field][b, i] = FIELDS[field].index(value)
    return {"count": count, "tool": tools, **fields}


def encode_batch(
    tokenizer, rows: list[dict], objective: str, device, max_tokens: int = 512
) -> dict:
    sequences, lengths, targets = [], [], []
    for row in rows:
        prefix = tokenizer.encode(row["prompt"], add_special_tokens=False)
        answer = (
            []
            if objective == "action"
            else tokenizer.encode(
                row["json_target"] if objective == "json" else row["text_target"],
                add_special_tokens=False,
            )
            + [tokenizer.eos_token_id]
        )
        if len(prefix) + len(answer) > max_tokens:
            raise ValueError(
                "token budget exceeded; increase max_tokens instead of truncating labels"
            )
        sequences.append(prefix + answer)
        lengths.append(len(prefix))
        targets.append([-100] * len(prefix) + answer)
    width = max(map(len, sequences))
    ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    lm_labels = torch.full_like(ids, -100)
    for b, sequence in enumerate(sequences):
        ids[b, : len(sequence)] = torch.tensor(sequence, device=device)
        mask[b, : len(sequence)] = 1
        lm_labels[b, : len(sequence)] = torch.tensor(targets[b], device=device)
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "lm_labels": lm_labels,
        "prompt_lengths": torch.tensor(lengths, device=device),
        "action_labels": labels(rows, device),
    }


def parse_json_prediction(text: str, row: dict) -> dict | None:
    try:
        # No repair or extraction of a convenient substring: invalid output
        # stays invalid.
        return deserialize_target(text.strip(), utterance_id=row["plan"]["utteranceId"])
    except (ValueError, TypeError, KeyError):
        return None


def plan_match(prediction: dict | None, expected: dict) -> bool:
    return prediction is not None and serialize_target(prediction) == serialize_target(expected)
