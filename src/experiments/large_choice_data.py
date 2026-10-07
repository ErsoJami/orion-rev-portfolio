"""1500-row controlled RU corpus for the original 26-choice Smol pointer."""

import hashlib
import json
import random
from collections import Counter
from itertools import product

from .action_data import row_from
from .choice_action import candidates
from .choice_action import dataset as original_dataset
from .choice_action import signature
from .tasks import DISPLAY, FIELDS, TOOL_FIELD

INSTRUCTIONS = """Выбери ровно один подходящий план для запроса пользователя к часам.
Доступны только перечисленные варианты и значения аргументов. EXECUTE означает выполнить одну явно запрошенную команду.
CLARIFY: пользователь просит поддерживаемую команду, но не указал обязательный параметр.
UNSUPPORTED: пользователь просит действие, которого нет среди доступных инструментов.
NO_ACTION: разговор, просьба объяснить команду или явный запрет её выполнять. Запрет одного действия сам по себе не разрешает противоположное.
Если одно действие явно запрошено, а другое запрещено, выбери запрошенное."""
PREFIXES = (
    "",
    "Пожалуйста, ",
    "На часах: ",
    "Моя просьба: ",
    "Если можно, ",
    "Сейчас: ",
    "Прошу: ",
    "Для меня: ",
)
SUFFIXES = ("", ".", ", пожалуйста", ". Спасибо", ". Это моя просьба")
OBJECTS = (
    "музыку",
    "песню",
    "трек",
    "воспроизведение",
    "проигрывание",
    "прослушивание",
    "текущую композицию",
    "текущий трек",
)
POSITIVE = {
    "timer.create": (
        (
            "Поставь таймер на {v}",
            "Создай таймер длительностью {v}",
            "Запусти отсчёт на {v}",
            "Отсчитай {v} таймером",
            "Мне нужен таймер на {v}",
            "Установи таймер: {v}",
        ),
        ("Таймер должен закончить отсчёт через {v}", "Засеки интервал {v} с помощью таймера"),
        ("Хочу отсчёт времени длительностью {v}", "Начни измерять {v} таймером"),
    ),
    "music.volume.set": (
        (
            "Установи громкость на {v}",
            "Сделай уровень звука {v}",
            "Для громкости выбери {v}",
            "Поставь громкость: {v}",
            "Громкость должна быть {v}",
            "Нужен уровень громкости {v}",
        ),
        ("Выставь для звука уровень {v}", "Пусть громкость станет равна {v}"),
        ("Отрегулируй громкость до значения {v}", "Хочу слышать музыку с громкостью {v}"),
    ),
    "mode.activate": (
        (
            "Включи режим «{v}»",
            "Активируй режим «{v}»",
            "Переключи часы в режим «{v}»",
            "Выбери режим «{v}»",
            "Установи режим «{v}»",
            "Сделай режим «{v}» активным",
        ),
        ("Пусть часы перейдут в режим «{v}»", "Сейчас мне подходит режим «{v}», включи его"),
        ("Хочу пользоваться режимом «{v}», активируй его", "Назначь часам режим «{v}»"),
    ),
    "workout.start": (
        (
            "Начни тренировку типа «{v}»",
            "Запусти тренировку «{v}»",
            "Включи запись тренировки типа «{v}»",
            "Начинай записывать тренировку «{v}»",
            "Стартуй тренировку типа «{v}»",
            "Прошу начать тренировку «{v}»",
        ),
        (
            "Пора записывать тренировку типа «{v}», начинай",
            "С этого момента записывай тренировку «{v}»",
        ),
        (
            "Я приступаю к тренировке типа «{v}», начни запись",
            "Открой новую запись тренировки типа «{v}»",
        ),
    ),
    "music.pause": (
        (
            "Поставь {v} на паузу",
            "Приостанови {v}",
            "Временно прерви {v}",
            "Мне нужна пауза: приостанови {v}",
            "Прошу сделать паузу: приостанови {v}",
            "Сейчас приостанови {v}",
        ),
        (
            "Сделай перерыв в музыке: поставь {v} на паузу",
            "Нужно временно остановить {v}, сделай паузу",
        ),
        ("Я отвлёкся: поставь {v} на паузу", "До следующей команды приостанови {v}"),
    ),
    "music.resume": (
        (
            "Продолжи {v}",
            "Возобнови {v}",
            "Сними {v} с паузы",
            "Верни {v} после паузы",
            "Я снова хочу слушать: продолжи {v}",
            "Музыкальная пауза закончена, возобнови {v}",
        ),
        ("Перерыв в прослушивании завершён: продолжи {v}", "Хочу снова слушать, возобнови {v}"),
        ("Дай музыке снова звучать: продолжи {v}", "Вернись к прослушиванию: возобнови {v}"),
    ),
}
SCOPE_TEMPLATES = (
    (
        ("Надо {a} {o}, не надо {b} {o}", "Не надо {a} {o}, надо {b} {o}"),
        ("Я хочу {a} {o}, но не хочу {b} {o}", "Я не хочу {a} {o}, но хочу {b} {o}"),
        ("Требуется {a} {o}; не требуется {b} {o}", "Не требуется {a} {o}; требуется {b} {o}"),
        ("Следует {a} {o}, а не {b} {o}", "Следует не {a} {o}, а {b} {o}"),
        ("Прошу {a} {o}; {b} {o} не прошу", "{a} {o} не прошу; прошу {b} {o}"),
    ),
    (
        ("Мне нужно {a} {o}, а {b} {o} не нужно", "{a} {o} мне не нужно, а {b} {o} нужно"),
        ("Выбираю {a} {o}, вариант {b} {o} отклоняю", "Вариант {a} {o} отклоняю, выбираю {b} {o}"),
    ),
    (
        ("{a} {o} разрешаю, {b} {o} запрещаю", "{a} {o} запрещаю, {b} {o} разрешаю"),
        (
            "Хочу, чтобы ты стал {a} {o}, но не стал {b} {o}",
            "Хочу, чтобы ты не стал {a} {o}, но стал {b} {o}",
        ),
    ),
)
SCOPE_WORDS = (
    ("приостанавливать", "продолжать", "музыку"),
    ("ставить на паузу", "возобновлять", "трек"),
    ("приостанавливать", "возобновлять", "песню"),
    ("прерывать", "продолжать", "воспроизведение"),
    ("ставить на паузу", "продолжать", "композицию"),
    ("приостанавливать", "возобновлять", "проигрывание"),
)
MISSING = (
    (
        "Поставь таймер, длительность я пока не указал",
        "Установи громкость, число процентов не выбрано",
        "Включи режим, его название сообщу потом",
        "Начни тренировку, её тип ещё не выбран",
    ),
    (
        "Нужен отсчёт, но я ещё не решил на сколько",
        "Хочу поменять уровень звука, значение неизвестно",
        "Переключи режим, какой именно ещё не знаю",
        "Запиши тренировку, но сначала надо выбрать тип",
    ),
    (
        "Создай таймер, время уточню следующим сообщением",
        "Измени громкость, конкретное число скажу позже",
        "Активируй какой-нибудь режим, название не определено",
        "Хочу тренировку, не определился с её видом",
    ),
)
UNSUPPORTED = (
    (
        "Закажи такси",
        "Купи билет на самолёт",
        "Отправь письмо другу",
        "Открой браузер",
        "Включи кофеварку",
        "Скачай фильм",
        "Распечатай документ",
        "Сделай фотографию",
        "Переведи деньги",
        "Закажи доставку еды",
        "Открой банковский счёт",
        "Запусти ядерную бомбу на меня",
    ),
    (
        "Забронируй гостиницу",
        "Оплати покупку онлайн",
        "Управляй атомной ракетой",
        "Отправь посылку",
        "Установи приложение",
        "Включи пылесос",
        "Найди маршрут к аэропорту",
        "Купи акции",
        "Открой электронную почту",
        "Вызови лифт",
    ),
    (
        "Забронируй столик в ресторане",
        "Опубликуй фотографию",
        "Позвони врачу",
        "Запусти видеозвонок",
        "Закажи новые наушники",
        "Измени пароль аккаунта",
        "Открой электронную книгу",
        "Проверь банковский баланс",
        "Загрузи файл на сервер",
        "Купи абонемент",
    ),
)
CHAT = (
    (
        "Привет",
        "Спасибо",
        "Доброе утро",
        "Как дела",
        "Ты здесь",
        "Хорошего дня",
        "Я рад",
        "Всё понятно",
    ),
    (
        "Здравствуй",
        "Благодарю за помощь",
        "Добрый вечер",
        "Рад тебя видеть",
        "Мне всё ясно",
        "Приятно поговорить",
    ),
    (
        "Приветствую тебя",
        "Спасибо тебе большое",
        "До встречи",
        "Доброй ночи",
        "Хорошо получилось",
        "Я доволен",
    ),
)
CANCEL = (
    ("Не выполняй просьбу: {v}", "Не нужно выполнять команду: {v}", "Запрещаю выполнять: {v}"),
    ("Отмени эту просьбу: {v}", "Воздержись от выполнения: {v}"),
    ("Не меняй ничего по запросу: {v}", "Оставь просьбу без выполнения: {v}"),
)
INFORMATION = (
    ("Объясни, что означает команда «{v}»", "Расскажи о команде «{v}», выполнять её не надо"),
    ("Как понимать команду «{v}»? Только объясни", "Опиши смысл команды «{v}» без выполнения"),
    (
        "Что произойдёт от команды «{v}»? Я прошу объяснение",
        "Мне нужно описание команды «{v}», а не её выполнение",
    ),
)


def normalize(text):
    return " ".join(text.casefold().replace("ё", "е").split())


def value_forms(tool, value):
    if tool == "timer.create":
        forms = [f"{value} секунд"]
        aliases = {
            10: "десять секунд",
            30: "полминуты",
            60: "одну минуту",
            120: "две минуты",
            300: "пять минут",
            600: "десять минут",
            1200: "двадцать минут",
            1800: "полчаса",
            3600: "один час",
        }
        forms.append(aliases[value])
        if value % 60 == 0:
            forms.append(f"{value // 60} мин")
        return forms
    if tool == "music.volume.set":
        words = {
            0: "ноль",
            20: "двадцать",
            40: "сорок",
            60: "шестьдесят",
            80: "восемьдесят",
            100: "сто",
        }
        return [f"{value} процентов", f"{value}%", words[value] + " процентов"]
    return [DISPLAY[value]]


def build_large_choice_dataset(seed=41):
    rng = random.Random(seed)
    known_challenge = original_dataset()["probe"]
    from .binary_scope_data import build_scope_dataset
    from .single_action_data import build_single_challenge

    challenge = build_single_challenge() + build_scope_dataset()["rows"] + known_challenge
    blocked = {normalize(r["command"]) for r in challenge}
    original = original_dataset()["train"]
    catalog = candidates()
    result = {
        "candidates": catalog,
        "instructions": INSTRUCTIONS,
        "splits": {},
        "seed": seed,
        "source": "deterministic synthetic templates with human-authored grammar and rule-derived labels",
        "limitations": [
            "1500 rows are synthetic, with shared politeness/context wrappers; not1500 real-user utterances.",
            "Core grammar families split before generation; argument values are known and finite in all splits.",
            "Each EXECUTE candidate contains one action; no multi-action requests or arbitrary argument values.",
            "Legacy challenge has informed earlier hypotheses and is diagnostic, not independent final test.",
        ],
    }
    globally_seen = set(blocked)
    for split_index, (split, multiplier) in enumerate(
        (("train", 5), ("validation", 1), ("test", 1))
    ):
        rows = []
        retained = original if split == "train" else []
        for old in retained:
            row = row_from(old["command"], old["plan"]["actions"], old["decision"], split)
            row.update(
                family="legacy-train12",
                data_kind="legacy",
                target_signature=signature(row["decision"], row["plan"]["actions"]),
            )
            rows.append(row)
            globally_seen.add(normalize(row["command"]))

        def wrap_pool(cores, actions, decision, family):
            output = []
            for core, prefix, suffix in product(cores, PREFIXES, SUFFIXES):
                command = prefix + (core[:1].lower() + core[1:] if prefix else core) + suffix
                if normalize(command) in globally_seen:
                    continue
                row = row_from(command, actions, decision, split)
                row.update(
                    family=family,
                    data_kind="template",
                    target_signature=signature(decision, actions),
                )
                output.append(row)
            return output

        def add(pool, target, wanted):
            current = sum(r["target_signature"] == target for r in rows)
            required = wanted - current
            rng.shuffle(pool)
            # Cover different core templates first through round-robin family
            # groups.
            groups = {}
            for row in pool:
                groups.setdefault(row["family"], []).append(row)
            while required > 0:
                progress = False
                for group in groups.values():
                    while group and normalize(group[-1]["command"]) in globally_seen:
                        group.pop()
                    if not group or required == 0:
                        continue
                    row = group.pop()
                    rows.append(row)
                    globally_seen.add(normalize(row["command"]))
                    required -= 1
                    progress = True
                if not progress:
                    raise ValueError("template capacity exhausted")

        # Timers9*7, volume6*8, modes3*11, workouts3*11 =177 positives in each
        # small split.
        for tool, per_value in (
            ("timer.create", 7),
            ("music.volume.set", 8),
            ("mode.activate", 11),
            ("workout.start", 11),
        ):
            field = TOOL_FIELD[tool]
            for value in FIELDS[field]:
                actions = [{"tool": tool, "arguments": {field: value}}]
                pool = []
                for family, template in enumerate(POSITIVE[tool][split_index]):
                    cores = [template.format(v=v) for v in value_forms(tool, value)]
                    pool.extend(
                        wrap_pool(cores, actions, "EXECUTE", f"{split}:{tool}:grammar{family}")
                    )
                add(pool, signature("EXECUTE", actions), per_value * multiplier)

        # Complete semantic quartets: both tools first, both scopes, same word
        # multisets.
        scope_words = SCOPE_WORDS if split == "train" else SCOPE_WORDS[:3]
        for family, patterns in enumerate(SCOPE_TEMPLATES[split_index]):
            for word_index, (pause, resume, obj) in enumerate(scope_words):
                group_id = f"{split}:scope:f{family}:v{word_index}"
                for first_tool, a, b in (
                    ("music.pause", pause, resume),
                    ("music.resume", resume, pause),
                ):
                    second_tool = "music.resume" if first_tool == "music.pause" else "music.pause"
                    for scope_index, (pattern, tool) in enumerate(
                        zip(patterns, (first_tool, second_tool), strict=True)
                    ):
                        command = pattern.format(a=a, b=b, o=obj)
                        assert normalize(command) not in globally_seen
                        row = row_from(command, [{"tool": tool, "arguments": {}}], "EXECUTE", split)
                        row.update(
                            family=f"{split}:scope:grammar{family}",
                            data_kind="scope_quartet",
                            scope_group=group_id,
                            first_tool=first_tool,
                            scope_index=scope_index,
                            target_signature=signature("EXECUTE", row["plan"]["actions"]),
                        )
                        rows.append(row)
                        globally_seen.add(normalize(command))
        for tool in ("music.pause", "music.resume"):
            actions = [{"tool": tool, "arguments": {}}]
            pool = []
            for family, template in enumerate(POSITIVE[tool][split_index]):
                pool.extend(
                    wrap_pool(
                        [template.format(v=o) for o in OBJECTS],
                        actions,
                        "EXECUTE",
                        f"{split}:{tool}:grammar{family}",
                    )
                )
            add(pool, signature("EXECUTE", actions), 24 * multiplier)
        for decision, cores in (
            ("CLARIFY", MISSING[split_index]),
            ("UNSUPPORTED", UNSUPPORTED[split_index]),
        ):
            pool = []
            for i, core in enumerate(cores):
                pool.extend(wrap_pool([core], [], decision, f"{split}:{decision}:core{i}"))
            add(pool, signature(decision, []), 25 * multiplier)
        no_action = wrap_pool(CHAT[split_index], [], "NO_ACTION", f"{split}:NO_ACTION:chat")
        exemplar = [
            POSITIVE[t][split_index][0].format(v=value_forms(t, v)[0])
            for t in ("timer.create", "music.volume.set", "mode.activate", "workout.start")
            for v in FIELDS[TOOL_FIELD[t]]
        ]
        exemplar += [
            POSITIVE[t][split_index][0].format(v="музыку") for t in ("music.pause", "music.resume")
        ]
        for kind, patterns in (
            ("cancel", CANCEL[split_index]),
            ("information", INFORMATION[split_index]),
        ):
            for i, template in enumerate(patterns):
                no_action.extend(
                    wrap_pool(
                        [template.format(v=e) for e in exemplar],
                        [],
                        "NO_ACTION",
                        f"{split}:NO_ACTION:{kind}{i}",
                    )
                )
        add(no_action, signature("NO_ACTION", []), 25 * multiplier)
        rng.shuffle(rows)
        assert len(rows) == 300 * multiplier
        assert Counter(r["decision"] for r in rows) == {
            "EXECUTE": 225 * multiplier,
            "CLARIFY": 25 * multiplier,
            "UNSUPPORTED": 25 * multiplier,
            "NO_ACTION": 25 * multiplier,
        }
        result["splits"][split] = rows
    # Deduplicate diagnostics themselves; keep source and previous visibility
    # explicit.
    unique = {}
    for row in challenge:
        row = row_from(row["command"], row["plan"]["actions"], row["decision"], "legacy_challenge")
        row.update(
            family="legacy-challenge",
            data_kind="previously_visible",
            target_signature=signature(row["decision"], row["plan"]["actions"]),
        )
        unique.setdefault(normalize(row["command"]), row)
    result["splits"]["legacy_challenge"] = list(unique.values())
    result["sha256"] = hashlib.sha256(
        json.dumps(result, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    return result
