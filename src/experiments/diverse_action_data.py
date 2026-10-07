"""Family-held-out synthetic Russian action corpus with explicit label derivations.

This expands semantic constructions, not the finite action/argument catalogue.
It is not a collection of real user messages or an independent author evaluation.
"""

import hashlib
import json
import random
from collections import Counter, defaultdict

from .action_data import row_from
from .choice_action import signature
from .large_choice_data import normalize, value_forms
from .tasks import DISPLAY, FIELDS, TOOL_FIELD

# Each tuple contains train / development / sealed-test constructions, assigned
# before rendering arguments or selecting rows. Outer semantic operations are
# shared; the complete sentence grammars are held out.
POSITIVE = {
    "timer.create": (
        (
            "Поставь таймер на {v}",
            "Запусти отсчёт на {v}",
            "Мне нужен таймер длительностью {v}",
            "Отмерь таймером {v}",
            "Длительность таймера — {v}, запускай",
        ),
        (
            "Прошу включить таймер с интервалом {v}",
            "Начинай обратный отсчёт: его продолжительность {v}",
            "Нужен работающий таймер; отведи ему {v}",
        ),
        (
            "Для отсчёта выбрана длительность {v}; создай таймер",
            "Запуск таймера нужен сейчас, время отсчёта составляет {v}",
            "Время до сигнала установи на {v} и запусти таймер",
        ),
    ),
    "music.volume.set": (
        (
            "Установи громкость на {v}",
            "Поставь уровень звука {v}",
            "Громкость музыки должна быть {v}",
            "Задай громкость: {v}",
            "Сделай музыку громкостью {v}",
        ),
        (
            "Выбранный уровень громкости — {v}; примени его",
            "Для музыки задай значение громкости {v}",
            "Измени уровень громкости: нужен {v}",
        ),
        (
            "Требуемая громкость равна {v}, выстави её",
            "Настрой звук музыки так, чтобы громкость была {v}",
            "Значение {v} выбрано для громкости, установи его",
        ),
    ),
    "mode.activate": (
        (
            "Включи режим «{v}»",
            "Переключи часы в режим «{v}»",
            "Активируй режим «{v}»",
            "Нужен режим «{v}», включи его",
            "Сделай активным режим «{v}»",
        ),
        (
            "Мой выбор режима — «{v}», примени его",
            "Перейди к использованию режима «{v}»",
            "На часах требуется включённый режим «{v}»",
        ),
        (
            "Режим с названием «{v}» должен стать активным",
            "Прошу сменить режим часов на «{v}»",
            "Среди режимов выбираю «{v}», переключись на него",
        ),
    ),
    "workout.start": (
        (
            "Начни тренировку типа «{v}»",
            "Запусти запись тренировки «{v}»",
            "Включи тренировку типа «{v}»",
            "Начинай записывать тренировку «{v}»",
            "Старт тренировки «{v}» нужен сейчас",
        ),
        (
            "Тренировка выбрана: «{v}», начни её запись",
            "Приступай к записи тренировки с типом «{v}»",
            "Пора начать тренировку «{v}» на часах",
        ),
        (
            "Тип новой тренировки — «{v}»; запускай запись",
            "Запись активности «{v}» должна начаться сейчас",
            "Создай новую запись тренировки типа «{v}» и начни её",
        ),
    ),
    "music.pause": (
        (
            "Поставь {v} на паузу",
            "Приостанови {v}",
            "Прерви воспроизведение: поставь {v} на паузу",
            "Сделай паузу для {gen}",
        ),
        (
            "Прошу временно приостановить {v}",
            "Для {gen} сейчас требуется пауза",
            "Останови {v} временно, до возобновления",
        ),
        (
            "Прослушивание надо прервать: приостанови {v}",
            "Сейчас {nom} должна быть на паузе",
            "Переходи к паузе в воспроизведении {gen}",
        ),
    ),
    "music.resume": (
        ("Продолжи {v}", "Возобнови {v}", "Сними {v} с паузы", "Пауза закончена: продолжи {v}"),
        (
            "Прошу снова воспроизводить {v}",
            "Для {gen} пауза окончена, возобнови её",
            "Включи продолжение {gen} после паузы",
        ),
        (
            "Прослушивание надо продолжить: возобнови {v}",
            "Сейчас верни {v} к воспроизведению после паузы",
            "Переходи от паузы к продолжению {gen}",
        ),
    ),
}

TRANSFORMS = {
    "direct": (("{x}",), ("{x}",), ("{x}",)),
    "question_request": (
        (
            "Можешь выполнить мою просьбу: {x}?",
            "Сделай, пожалуйста, вот что: {x}",
            "Выполни сейчас: {x}",
            "Я прошу выполнить: {x}",
        ),
        ("Сможешь сделать для меня следующее: {x}?", "Обращаюсь с просьбой выполнить сейчас: {x}"),
        ("Будешь добр выполнить действие: {x}?", "Прошу приступить к выполнению: {x}"),
    ),
    "cancel": (
        (
            "Не выполняй: {x}",
            "Запрещаю выполнять просьбу «{x}»",
            "Отмени эту просьбу: {x}",
            "Воздержись от выполнения: {x}",
            "Выполнять «{x}» не нужно",
            "Я не прошу выполнить: {x}",
        ),
        (
            "Просьба «{x}» отменена; ничего по ней не выполняй",
            "Отказываюсь от команды «{x}» — оставь всё как есть",
        ),
        (
            "Снимать запрет на «{x}» я не разрешаю; не выполняй",
            "Я передумал насчёт «{x}»; выполнять её запрещено",
        ),
    ),
    "explain": (
        (
            "Объясни смысл команды «{x}», не выполняя её",
            "Как понимать «{x}»? Только объясни",
            "Опиши команду «{x}» без выполнения",
            "Мне нужно описание «{x}», а не выполнение",
        ),
        (
            "Дай пояснение к словам «{x}»; запускать ничего не надо",
            "Расшифруй смысл «{x}», это вопрос об описании",
        ),
        (
            "Интересует лишь значение фразы «{x}»; никаких действий",
            "Нужен словесный разбор «{x}» без её исполнения",
        ),
    ),
    "quotation": (
        (
            "В тексте написано «{x}». Это цитата, а не моя команда",
            "Кто-то сказал «{x}». Я не прошу это выполнять",
            "Пример команды: «{x}». Ничего не делай",
        ),
        (
            "Слова «{x}» приведены как образец; выполнять образец не требуется",
            "Пересказываю чужое «{x}», сам такого действия не запрашиваю",
        ),
        (
            "В сообщении друга было «{x}»; это пересказ, без поручения часам",
            "Взял фразу «{x}» из инструкции; она здесь только для чтения",
        ),
    ),
    "past_report": (
        (
            "Я вчера выполнил команду «{x}». Сейчас ничего не меняй",
            "Раньше я просил «{x}», это рассказ о прошлом",
            "«{x}» уже выполнено; повторять не нужно",
        ),
        (
            "Сообщаю о вчерашней просьбе «{x}»; новой команды нет",
            "Упоминаю прошедшее выполнение «{x}», без повторного запуска",
        ),
        (
            "В прошлый раз была команда «{x}». Теперь лишь рассказываю об этом",
            "Речь о ранее исполненной просьбе «{x}»; сейчас поручений нет",
        ),
    ),
    "hypothetical": (
        (
            "Если бы я сказал «{x}», что бы произошло? Не выполняй",
            "Представим команду «{x}», но пока ничего не запускай",
            "Возможно, позже попрошу «{x}». Сейчас выполнять не надо",
        ),
        (
            "Обсуждаем возможное «{x}», а не отдаём команду сейчас",
            "Планирую когда-нибудь попросить «{x}»; сейчас просьба не действует",
        ),
        (
            "Допустим, в будущем прозвучит «{x}». Это пока только предположение",
            "Теоретический вариант — «{x}»; исполнения сейчас не требуется",
        ),
    ),
}

MISSING = {
    "timer.create": (
        (
            "Поставь таймер",
            "Запусти отсчёт, но длительность пока не выбрана",
            "Нужен таймер; сколько времени — ещё не решил",
            "Создай таймер без указанной длительности",
        ),
        (
            "Хочу новый таймер, интервал предстоит уточнить",
            "Запрос на запуск таймера есть, а время я ещё не назвал",
        ),
        (
            "Начать отсчёт нужно, продолжительность сообщу отдельно",
            "Прошу создать таймер; его длительности в этой просьбе нет",
        ),
    ),
    "music.volume.set": (
        (
            "Установи громкость",
            "Измени громкость, значение пока не указано",
            "Нужна другая громкость, но число процентов не выбрано",
            "Поставь уровень звука, число скажу потом",
        ),
        (
            "Прошу настроить громкость; нужное значение ещё неизвестно",
            "Громкость надо поменять, процент предстоит уточнить",
        ),
        (
            "Запрос на изменение громкости есть, конкретного уровня ещё нет",
            "Сделай настройку уровня звука; сам уровень сообщу отдельно",
        ),
    ),
    "mode.activate": (
        (
            "Включи режим",
            "Переключи режим, но название не выбрано",
            "Нужен режим; какой именно — сообщу потом",
            "Активируй режим без указанного названия",
        ),
        (
            "Хочу другой режим на часах, пока не определился с названием",
            "Режим нужно сменить, выбор ещё предстоит уточнить",
        ),
        (
            "Прошу активировать режим; какой именно, в просьбе не сказано",
            "Есть запрос на смену режима, а его название пока отсутствует",
        ),
    ),
    "workout.start": (
        (
            "Начни тренировку",
            "Запусти тренировку, тип пока не выбран",
            "Нужна запись тренировки без указанного типа",
            "Начинай тренировку, вид активности скажу позже",
        ),
        (
            "Хочу начать запись тренировки; вид ещё предстоит уточнить",
            "Запрос на тренировку есть, конкретную активность не назвал",
        ),
        (
            "Нужно приступить к тренировке, её тип сообщу отдельно",
            "Прошу включить запись тренировки, но вид занятия пока неизвестен",
        ),
    ),
}

UNSUPPORTED = (
    (
        "Закажи такси",
        "Отправь письмо другу",
        "Открой браузер",
        "Купи билет на самолёт",
        "Запусти ядерную бомбу на меня",
        "Установи приложение",
        "Распечатай документ",
        "Забронируй гостиницу",
        "Включи пылесос",
        "Переведи деньги",
        "Сделай фотографию",
        "Закажи доставку еды",
        "Позвони врачу",
        "Открой банковский счёт",
        "Скачай фильм",
        "Отправь посылку",
        "Полей цветы",
        "Закрой дверь",
        "Включи телевизор",
        "Нарисуй картину",
    ),
    (
        "Закажи товар из магазина",
        "Запиши встречу в календарь",
        "Удали файл на компьютере",
        "Поменяй пароль почты",
        "Отправь сообщение сестре",
        "Оплати коммунальные услуги",
        "Забронируй экскурсию",
        "Установи будильник",
        "Найди адрес школы",
        "Открой электронную книгу",
    ),
    (
        "Оформи страховку",
        "Запиши меня к стоматологу",
        "Покажи карту метро",
        "Закажи печать фотографий",
        "Перезагрузи роутер",
        "Открой гараж",
        "Создай папку на диске",
        "Отправь факс",
        "Купи абонемент в бассейн",
        "Настрой термостат",
    ),
)

SCOPE = (
    (
        ("Нужно {a} {o}, но не нужно {b} {o}", "Не нужно {a} {o}, но нужно {b} {o}"),
        ("Выбираю {a} {o}, вариант {b} {o} отклоняю", "Вариант {a} {o} отклоняю, выбираю {b} {o}"),
        ("Прошу {a} {o}; запрещаю {b} {o}", "Запрещаю {a} {o}; прошу {b} {o}"),
        ("Хочу {a} {o}, а не {b} {o}", "Хочу не {a} {o}, а {b} {o}"),
    ),
    (
        (
            "Моё поручение — {a} {o}; {b} {o} не поручаю",
            "{a} {o} не поручаю; моё поручение — {b} {o}",
        ),
        (
            "Следующее действие: {a} {o}. {b} {o} запрещено",
            "{a} {o} запрещено. Следующее действие: {b} {o}",
        ),
    ),
    (
        (
            "Разрешаю и прошу {a} {o}, а {b} {o} делать нельзя",
            "{a} {o} делать нельзя, а {b} {o} разрешаю и прошу",
        ),
        (
            "Запрос относится к тому, чтобы {a} {o}; просьбы {b} {o} нет",
            "Просьбы {a} {o} нет; запрос относится к тому, чтобы {b} {o}",
        ),
    ),
)

CORRECTION = (
    (
        "Для {noun} выбери {new}, а не {old}",
        "Исправляю просьбу: {x}. Значение {old} не используй",
        "Вместо {old} нужно {new} для {noun}",
        "Значение для {noun}: не {old}, а {new}",
    ),
    ("Новое значение для {noun} — {new}; прежнее {old} отменяю", "Отказываюсь от {old}: {x}"),
    (
        "Окончательный выбор для {noun} — {new}; вариант {old} отвергнут",
        "{x}. Уточняю: {old} было ошибкой, нужно {new}",
    ),
)
NOUN = {
    "timer.create": "длительности таймера",
    "music.volume.set": "громкости",
    "mode.activate": "режима часов",
    "workout.start": "типа тренировки",
}
OBJECTS = ("музыку", "песню", "композицию", "мелодию")
OBJECT_CASES = {
    "музыку": ("музыка", "музыки"),
    "песню": ("песня", "песни"),
    "композицию": ("композиция", "композиции"),
    "мелодию": ("мелодия", "мелодии"),
}
SCOPE_OBJECTS = ("музыку", "трек", "песню", "воспроизведение")
# These neutral additions are shared between splits and never determine a
# label.
PREFIX = (
    "",
    "Я сейчас дома. ",
    "Обращаюсь к часам. ",
    "Передаю запрос: ",
    "Пожалуйста, ",
    "Сейчас моя просьба такая: ",
    "Скажу ещё раз: ",
    "Для меня: ",
)
SUFFIX = ("", ". Спасибо", ", пожалуйста", ". Это всё", ". Запрос закончен")


def build(old, fresh_old=(), seed=781):
    """Return dev corpus, sealed test and reproducible quality audit."""
    rng = random.Random(seed)
    blocked = {normalize(r["command"]) for rows in old["splits"].values() for r in rows}
    blocked.update(normalize(r["command"]) for r in fresh_old)
    used = set(blocked)
    result = {
        "candidates": old["candidates"],
        "instructions": old["instructions"],
        "splits": {},
        "seed": seed,
        "source": "engineer-authored synthetic grammars; deterministic rule-derived labels",
        "parent_dataset_sha256": old["sha256"],
        "limitations": [
            "Finite 26 plans, single action, known enum arguments; no general agent or production claim.",
            "New complete sentence grammar families held out before rendering; same author/transform types and shared values/context.",
            "Rows are not independent real user utterances: context wrappers repeat cores.",
            "Old train is retained, old validation/test are never inserted into new train.",
        ],
    }
    audit = {}
    sealed = []
    for si, split in enumerate(("train", "validation", "test")):
        groups = defaultdict(list)
        seen_pool = {}

        def add(core, decision, actions, category, grammar, tool=None, evidence=None):
            family = f"diverse:{split}:{category}:{
                tool or decision}:g{grammar}"
            contexts = [(p, s) for p in PREFIX for s in SUFFIX]
            rng.shuffle(contexts)
            budget = (
                20
                if category in ("missing_argument", "unavailable_tool")
                else 12 if category == "negation_scope" else 4
            )
            for prefix, suffix in contexts[:budget]:
                text = prefix + core + suffix
                norm = normalize(text)
                if norm in used:
                    continue
                target = signature(decision, actions)
                if norm in seen_pool:
                    assert seen_pool[norm] == target, ("contradictory label", text)
                    continue
                seen_pool[norm] = target
                row = row_from(text, actions, decision, split)
                row.update(
                    family=family,
                    data_kind="synthetic_semantic",
                    semantic_category=category,
                    core=core,
                    target_signature=target,
                    derivation={
                        "operation": category,
                        "template_index": grammar,
                        "tool": tool,
                        "arguments": actions[0]["arguments"] if actions else {},
                        "evidence": evidence or category,
                    },
                )
                groups[(decision, category, tool or decision)].append(row)

        for tool, templates in POSITIVE.items():
            field = TOOL_FIELD.get(tool)
            for value in FIELDS[field] if field else OBJECTS:
                forms = value_forms(tool, value) if field else (value,)
                actions = [{"tool": tool, "arguments": {field: value} if field else {}}]
                for g, template in enumerate(templates[si]):
                    for form in forms:
                        nom, gen = OBJECT_CASES.get(form, (form, form))
                        command = template.format(v=form, nom=nom, gen=gen)
                        for category, outer in TRANSFORMS.items():
                            for j, pattern in enumerate(outer[si]):
                                add(
                                    pattern.format(x=command),
                                    (
                                        "EXECUTE"
                                        if category in ("direct", "question_request")
                                        else "NO_ACTION"
                                    ),
                                    actions if category in ("direct", "question_request") else [],
                                    category,
                                    f"{g}.{j}",
                                    tool,
                                )
                        if field:
                            for other in FIELDS[field]:
                                if other == value:
                                    continue
                                old_form = value_forms(tool, other)[0]
                                for j, pattern in enumerate(CORRECTION[si]):
                                    add(
                                        pattern.format(
                                            x=command, noun=NOUN[tool], new=form, old=old_form
                                        ),
                                        "EXECUTE",
                                        actions,
                                        "argument_correction",
                                        f"{g}.{j}",
                                        tool,
                                        {"selected": value, "rejected": other},
                                    )
        for tool, templates in MISSING.items():
            for g, template in enumerate(templates[si]):
                for j, outer in enumerate(TRANSFORMS["question_request"][si] + ("{x}",)):
                    add(
                        outer.format(x=template),
                        "CLARIFY",
                        [],
                        "missing_argument",
                        f"{g}.{j}",
                        tool,
                    )
        for g, command in enumerate(UNSUPPORTED[si]):
            for j, outer in enumerate(TRANSFORMS["question_request"][si] + ("{x}",)):
                add(outer.format(x=command), "UNSUPPORTED", [], "unavailable_tool", f"{g}.{j}")
        for g, pair in enumerate(SCOPE[si]):
            for obj in SCOPE_OBJECTS:
                for first, a, b in (
                    ("music.pause", "приостановить", "возобновить"),
                    ("music.resume", "возобновить", "приостановить"),
                ):
                    second = "music.resume" if first == "music.pause" else "music.pause"
                    for j, (pattern, chosen) in enumerate(zip(pair, (first, second), strict=True)):
                        add(
                            pattern.format(a=a, b=b, o=obj),
                            "EXECUTE",
                            [{"tool": chosen, "arguments": {}}],
                            "negation_scope",
                            f"{g}.{j}",
                            chosen,
                            {"requested": chosen, "forbidden": second if j == 0 else first},
                        )
        # Class proportions intentionally increase the no-action mass. Round-robin
        # categories/tools prevent the large timer cross-product dominating.
        quotas = (
            {"EXECUTE": 3000, "NO_ACTION": 2000, "CLARIFY": 750, "UNSUPPORTED": 750}
            if split == "train"
            else (
                {"EXECUTE": 500, "NO_ACTION": 250, "CLARIFY": 125, "UNSUPPORTED": 125}
                if split == "validation"
                else {"EXECUTE": 1000, "NO_ACTION": 500, "CLARIFY": 250, "UNSUPPORTED": 250}
            )
        )
        rows = [dict(r) for r in old["splits"]["train"]] if split == "train" else []
        for row in rows:
            row.update(semantic_category="retained_old_train", core=row["command"])
        for pool in groups.values():
            rng.shuffle(pool)
        for decision, count in quotas.items():
            pools = [v for k, v in groups.items() if k[0] == decision]
            for i in range(count):
                active = [p for p in pools if p]
                if not active:
                    raise ValueError(f"Insufficient {split}/{decision} capacity")
                pool = active[i % len(active)]
                row = pool.pop()
                rows.append(row)
                used.add(normalize(row["command"]))
        rng.shuffle(rows)
        # A command alone may not distinguish templates across reused inner
        # requests, but complete generated sentence families are disjoint.
        audit[split] = {
            "rows": len(rows),
            "decisions": dict(Counter(r["decision"] for r in rows)),
            "categories": dict(Counter(r["semantic_category"] for r in rows)),
            "unique_cores": len({normalize(r["core"]) for r in rows}),
            "families": sorted({r["family"] for r in rows}),
        }
        if split == "test":
            sealed = rows
        else:
            result["splits"][split] = rows
    result["sha256"] = hashlib.sha256(
        json.dumps(result, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    test = {
        "rows": sealed,
        "parent_dataset_sha256": result["sha256"],
        "selection_allowed": False,
        "limitations": result["limitations"],
    }
    test["sha256"] = hashlib.sha256(
        json.dumps(test, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    all_rows = [r for rows in result["splits"].values() for r in rows] + sealed
    assert len({normalize(r["command"]) for r in all_rows}) == len(all_rows)
    allowed = {signature(c["decision"], c["actions"]) for c in old["candidates"]}
    assert all(r["target_signature"] in allowed for r in all_rows)
    audit.update(
        dataset_sha256=result["sha256"],
        test_sha256=test["sha256"],
        cross_split_normalized_duplicates=0,
        conflicting_generated_labels=0,
        reused_old_rows=1500,
        new_rows_exclude_all_old_splits=True,
    )
    return result, test, audit
