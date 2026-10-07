"""Controlled v2 corpus for action-only training, independent of the old smoke suite."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter

from data.contracts.agent import serialize_target, validate_plan

from .tasks import DISPLAY, FIELDS, TEMPLATES, TOOL_FIELD, TOOLS, make_plan, prompt

DECISIONS = ("EXECUTE", "CLARIFY", "UNSUPPORTED", "NO_ACTION")
REASONS = {
    "CLARIFY": "Уточните недостающие параметры команды.",
    "UNSUPPORTED": "Запрос содержит действие, которое этот эксперимент не поддерживает.",
    "NO_ACTION": "Действий не требуется.",
}
PREFIXES = (
    "",
    "Пожалуйста, ",
    "Сейчас: ",
    "На часах: ",
    "Моя просьба: ",
    "Для меня: ",
    "Если можно, ",
    "Команда: ",
    "Прошу: ",
    "Будь добр, ",
)
SUFFIXES = ("", ".", ", пожалуйста", ". Спасибо", ". Это моя просьба")
UNKNOWN = {
    "train": (
        "Закажи такси",
        "Купи билет на самолёт",
        "Отправь письмо",
        "Открой браузер",
        "Запусти ядерную бомбу на меня",
        "Переведи деньги",
        "Включи кофеварку",
        "Скачай фильм",
        "Распечатай документ",
        "Сделай фотографию",
    ),
    "validation": (
        "Вызови машину через интернет",
        "Оплати покупку онлайн",
        "Открой банковский счёт",
        "Управляй атомной ракетой",
        "Включи пылесос",
        "Отправь посылку",
    ),
    "test": (
        "Забронируй гостиницу",
        "Закажи еду с доставкой",
        "Запусти ракету",
        "Установи приложение",
        "Вызови лифт",
        "Купи акции",
    ),
}
INCOMPLETE = {
    "train": (
        "Поставь таймер",
        "Установи громкость",
        "Включи режим",
        "Начни тренировку",
        "Поставь таймер на несколько секунд",
        "Сделай громкость сколько-нибудь процентов",
    ),
    "validation": (
        "Запусти отсчёт без указанной длительности",
        "Измени громкость, значение не указано",
        "Выбери режим, название не указано",
        "Запиши начало тренировки, тип не указан",
    ),
    "test": (
        "Мне нужен таймер, но длительность пока не выбрана",
        "Хочу громкость, уровень ещё не выбран",
        "Мне нужен режим, но какой именно пока не знаю",
        "Хочу начать тренировку, тип ещё не выбран",
    ),
}
CHAT = {
    "train": ("Привет", "Спасибо", "Доброе утро", "Как дела", "Ты здесь", "Хорошего дня"),
    "validation": ("Здравствуй", "Благодарю за помощь", "Рад тебя видеть", "Добрый вечер"),
    "test": ("Приветствую тебя", "Спасибо тебе большое", "До встречи", "Доброй ночи"),
}


def row_from(command, actions, decision, split):
    plan = make_plan(command, actions)
    if not actions:
        # v1 requires non-null clarification for an empty plan. Decision remains an
        # experimental sidecar, rather than adding illegal properties to
        # AgentPlan.
        plan["clarification"] = REASONS[decision]
    validate_plan(plan)
    return {
        "id": hashlib.sha256(command.encode()).hexdigest()[:16],
        "split": split,
        "command": command,
        "prompt": prompt(command),
        "decision": decision,
        "plan": plan,
        "json_target": serialize_target(plan),
        "text_target": plan["clarification"] or "Предлагаю выполнить указанные действия.",
    }


def build_action_suite(seed=41, sizes=(2000, 300, 300)):
    rng = random.Random(seed)
    suite, global_seen = {}, set()
    for split, size in zip(("train", "validation", "test"), sizes, strict=True):
        if size < 32 or size % 4:
            raise ValueError("split size must be >=32 and divisible by four")
        rows = []
        indices = range(4) if split == "train" else (4,) if split == "validation" else (5,)

        def positive(count):
            actions, phrases = [], []
            for tool in rng.sample(TOOLS, count):
                field = TOOL_FIELD.get(tool)
                value = rng.choice(FIELDS[field]) if field else None
                phrases.append(
                    TEMPLATES[tool][rng.choice(tuple(indices))].format(v=DISPLAY.get(value, value))
                )
                actions.append({"tool": tool, "arguments": {field: value} if field else {}})
            return " и ".join(phrases), actions

        for i in range(size):
            negative = i < size // 4
            decision = ("UNSUPPORTED", "CLARIFY", "NO_ACTION")[i % 3] if negative else "EXECUTE"
            for attempt in range(10000):
                if decision == "EXECUTE":
                    core, actions = positive(1 + (i - size // 4) % 5)
                elif decision == "UNSUPPORTED":
                    core, actions = rng.choice(UNKNOWN[split]), []
                    # Mixed requests are deliberately all-or-nothing in this
                    # benchmark.
                    if i % 2:
                        supported, _ = positive(1)
                        core = supported + " и " + core
                elif decision == "CLARIFY":
                    core, actions = rng.choice(INCOMPLETE[split]), []
                else:
                    supported, _ = positive(1)
                    kind = i % 4
                    if kind == 0:
                        core = rng.choice(CHAT[split])
                    elif kind == 1:
                        core = (
                            "Не выполняй команду: "
                            if split == "train"
                            else (
                                "Отмени просьбу: "
                                if split == "validation"
                                else "Не нужно выполнять: "
                            )
                        ) + supported
                    elif kind == 2:
                        core = (
                            "Объясни команду: "
                            if split == "train"
                            else (
                                "Что означает команда: "
                                if split == "validation"
                                else "Расскажи о команде: "
                            )
                        ) + supported
                    else:
                        core = (
                            "Вчера я просил: "
                            if split == "train"
                            else (
                                "Раньше я говорил: "
                                if split == "validation"
                                else "Это была прошлая просьба: "
                            )
                        ) + supported
                    actions = []
                command = rng.choice(PREFIXES) + core + rng.choice(SUFFIXES)
                if command not in global_seen:
                    break
            else:
                raise ValueError("synthetic template capacity exhausted")
            global_seen.add(command)
            rows.append(row_from(command, actions, decision, split))
        rng.shuffle(rows)
        suite[split] = rows
    audit_action_suite(suite)
    return suite


def audit_action_suite(suite):
    seen, summary, train_values = set(), {}, {f: set() for f in FIELDS}
    for split, rows in suite.items():
        for row in rows:
            assert row["split"] == split
            assert row["command"] not in seen, "exact prompt leakage"
            seen.add(row["command"])
            validate_plan(row["plan"])
            assert bool(row["plan"]["actions"]) == (row["decision"] == "EXECUTE")
            if split == "train":
                for action in row["plan"]["actions"]:
                    for field, value in action["arguments"].items():
                        train_values[field].add(value)
        summary[split] = {
            "rows": len(rows),
            "decisions": dict(Counter(r["decision"] for r in rows)),
            "action_counts": dict(Counter(len(r["plan"]["actions"]) for r in rows)),
        }
    for field, values in FIELDS.items():
        assert train_values[field] == set(values), f"untrained classifier labels: {field}"
    return {
        "version": 2,
        "seed_policy": "deterministic controlled templates with randomized parameters",
        "sha256": hashlib.sha256(
            json.dumps(suite, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "splits": summary,
        "candidate_values": FIELDS,
        "split_policy": "disjoint core template families; shared known argument vocabulary; exact prompt audit",
        "limitations": [
            "six tools; closed argument vocabulary",
            "synthetic holdout, not real user evaluation",
            "empty-plan reason is a decision sidecar adapted to the unchanged AgentPlan v1",
            "mixed unsupported requests produce zero actions; independent actions only",
        ],
    }


def tiny_subset(rows, size=32):
    # Exercise every decision, tool and count before spending a full pilot
    # budget.
    chosen = []
    keys = [lambda r, d=d: r["decision"] == d for d in DECISIONS]
    keys += [lambda r, t=t: any(a["tool"] == t for a in r["plan"]["actions"]) for t in TOOLS]
    keys += [lambda r, n=n: len(r["plan"]["actions"]) == n for n in range(1, 6)]
    for predicate in keys:
        candidate = next(r for r in rows if predicate(r) and r not in chosen)
        chosen.append(candidate)
    chosen.extend(r for r in rows if r not in chosen and len(chosen) < size)
    return chosen[:size]
