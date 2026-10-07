"""New training grammars for surface diversity and explicit semantic contrasts."""

import hashlib
import json
import random
from collections import Counter, defaultdict

from .action_data import row_from
from .choice_action import signature
from .diverse_action_data import OBJECT_CASES, OBJECTS
from .large_choice_data import normalize, value_forms
from .tasks import DISPLAY, FIELDS, TOOL_FIELD

POSITIVE = {
    "timer.create": (
        "Дай старт таймеру с длительностью {v}",
        "Пускай таймер отсчитывает {v}",
        "Приступи к отсчёту длительностью {v}",
        "Для таймера задай продолжительность {v} и начинай",
        "Нужен запущенный таймер с длительностью {v}",
        "Таймеру требуется интервал {v}; создай его",
        "Отсчёт должен работать в течение {v}",
        "Запуск отсчёта разрешаю и прошу: время {v}",
        "Для текущего таймера выбираю {v} и прошу запуск",
        "Моя команда таймеру: отмерять {v}",
        "Сейчас создай работающий таймер; ему отведено {v}",
        "Хочу включённый обратный отсчёт на {v}",
    ),
    "music.volume.set": (
        "Сделай значение громкости равным {v}",
        "Звук музыки настрой на {v}",
        "Примени для громкости музыки {v}",
        "Сейчас громкости требуется значение {v}; задай его",
        "Громкость на уровне {v} — мой запрос",
        "Музыку сделай с уровнем громкости {v}",
        "Прошу выставить громкость музыки ровно {v}",
        "Мне требуется громкость {v}; поменяй уровень",
        "Для уровня звука утверждаю {v}; примени настройку",
        "Нужное значение громкости — {v}, устанавливай",
        "В параметр громкости поставь {v}",
        "Перейди на громкость {v} для музыки",
    ),
    "mode.activate": (
        "Сделай текущим режим с названием «{v}»",
        "Применяй режим часов «{v}»",
        "Мой запрос — переход в режим «{v}»",
        "Прошу сделать активным режим «{v}» на часах",
        "В настройке режима выбираю «{v}» и прошу включение",
        "Часы сейчас переведи на режим «{v}»",
        "Сменить режим на «{v}» требуется сейчас",
        "Нужна активация режима часов «{v}»",
        "Для часов действующим режимом сделай «{v}»",
        "Запускай режим с именем «{v}»",
        "Пусть включится режим «{v}» на часах",
        "Моё поручение по режиму: включить «{v}»",
    ),
    "workout.start": (
        "Начинай регистрацию тренировки «{v}»",
        "Пусть начнётся запись тренировки типа «{v}»",
        "Запись занятия «{v}» нужно запустить",
        "Моё поручение — старт записи тренировки «{v}»",
        "Для нового занятия выбираю тип «{v}»; начни тренировку",
        "Приступи к тренировке вида «{v}»",
        "Нужна работающая запись тренировки «{v}»",
        "Создай и запусти тренировку типа «{v}»",
        "Сейчас пора записывать тренировку «{v}»",
        "Включение записи занятия «{v}» — моя просьба",
        "Пусть часы начнут тренировку с типом «{v}»",
        "Дай старт записи активности «{v}»",
    ),
    "music.pause": (
        "{nom} должна временно перестать играть",
        "Сейчас нужна пауза для {gen}; сделай её",
        "Прошу сделать перерыв в проигрывании {gen}",
        "Пусть {nom} перестанет звучать на время",
        "Перейди к временному прерыванию {gen}",
        "До дальнейшей просьбы приостанови {v}",
        "Сейчас мой запрос — временно остановить {v}",
        "Продолжение {gen} отложи, сейчас поставь её на паузу",
        "Нужен перерыв для {gen}; приостанови её",
        "Пауза для {gen} нужна прямо сейчас",
        "Поставить {v} на паузу — моё текущее поручение",
        "Я прошу временное прекращение звучания {gen}",
    ),
    "music.resume": (
        "{nom} должна снова начать играть",
        "Сейчас нужно продолжение {gen}; возобнови её",
        "Прошу закончить перерыв в проигрывании {gen}",
        "Пусть {nom} вернётся к звучанию",
        "Перейди к возобновлению {gen}",
        "После перерыва продолжай {v}",
        "Сейчас мой запрос — снова воспроизводить {v}",
        "Пауза для {gen} больше не нужна, возобнови её",
        "Нужен возврат к проигрыванию {gen}; продолжи её",
        "Продолжение {gen} нужно прямо сейчас",
        "Снять {v} с паузы — моё текущее поручение",
        "Я прошу возобновить звучание {gen}",
    ),
}
REQUEST = (
    "Выполни именно эту команду: {x}",
    "Моя актуальная просьба звучит так: {x}",
    "Для меня необходимо следующее: {x}",
    "Это поручение часам: {x}",
    "Текущий запрос: {x}. Выполняй",
    "Команда «{x}» адресована тебе, исполни её",
    "Я действительно хочу выполнения «{x}»",
    "Повтори сейчас действие «{x}»",
    "Запрос подтверждён: {x}",
    "Сейчас исполни фразу «{x}» как команду",
    "Это не пример, а просьба: {x}",
    "Не ограничивайся описанием; выполни: {x}",
    "Прошу сделать «{x}» прямо сейчас",
    "Снова поручаю: {x}",
    "После разговора перехожу к реальной просьбе: {x}",
    "Пожалуйста, выполни действие «{x}»",
)
NEGATIVE = {
    "cancel": (
        "Больше не прошу «{x}», ничего не выполняй",
        "Отозвал просьбу «{x}»; она недействительна",
        "На выполнение «{x}» наложен запрет",
        "Оставь без исполнения команду «{x}»",
        "Просьба «{x}» больше не актуальна; не исполняй",
        "Никакого выполнения «{x}» сейчас не хочу",
    ),
    "explain": (
        "Меня интересует пояснение к «{x}», действий не требуется",
        "Нужен только текст о смысле «{x}», без исполнения",
        "Разбери значение команды «{x}» словами",
        "Действовать не надо; объясни фразу «{x}»",
        "Хочу узнать смысл «{x}», а запуск не запрашиваю",
        "Речь только о толковании «{x}», не о выполнении",
    ),
    "quotation": (
        "Чужая фраза «{x}» приведена для чтения, не как запрос",
        "Я цитирую инструкцию «{x}»; действовать по цитате не нужно",
        "Цитата из примера: «{x}». Команду я не отдаю",
        "Привёл чужой текст «{x}» без поручения его исполнить",
        "В книге встречается «{x}»; просто сообщаю об этой строке",
        "Слова «{x}» принадлежат другому человеку; я такого действия не прошу",
    ),
    "past_report": (
        "Вчера прозвучало «{x}», сегодня я лишь вспоминаю",
        "Давно просил «{x}»; повторять сейчас не прошу",
        "Я рассказываю о прошлом «{x}», без нового запроса",
        "Просьба «{x}» уже осталась в прошлом, ничего не делай",
        "У нас раньше было «{x}». Это историческое сообщение",
        "Напомнил о старом «{x}», но запускать заново не надо",
    ),
    "hypothetical": (
        "Когда-нибудь, возможно, попрошу «{x}», сейчас не действуй",
        "Сейчас «{x}» только гипотеза, а не поручение",
        "Мысленно рассматриваю «{x}», ничего не запускай",
        "Это теория: «{x}». Реального запроса на действие нет",
        "Пока планируем возможное «{x}», выполнять рано",
        "В воображении звучит «{x}»; исполнение не требуется",
    ),
}
CORRECTION = (
    "{old_x}. Стоп, поправка: {x}. Первую просьбу отменяю",
    "{x}. Прежнее значение {old} для этой команды не используй",
    "Не используй старое {old}; сейчас {x}",
    "Новый выбор для {noun}: {new}. {old} больше не нужно",
    "Раньше думал про {old}, а для {noun} теперь прошу {new}",
    "{x}; вариант с параметром {old} считать недействительным",
    "Новому значению «{new}» для {noun} даю ход, прежний вариант «{old}» отклоняю",
    "Вместо прежнего {old} для {noun} прошу новое {new}",
    "Для {noun} {old} названо по ошибке. Моя правильная команда: {x}",
    "Моя команда: {x}. Про {old} забудь, этот вариант отозван",
    "Старый параметр {old} отменён; требуется {new} для {noun}",
    "Для {noun} выбираю новое «{new}»; старое «{old}» не действует",
)
NOUN = {
    "timer.create": "длительности таймера",
    "music.volume.set": "громкости",
    "mode.activate": "режима часов",
    "workout.start": "типа тренировки",
}
SCOPE = (
    "Моя команда на сейчас: {want}. {block} выполнять нельзя",
    "Есть просьба {want}. Просьбу {block} не даю",
    "Запрашиваю {want}; {block} не запрашиваю",
    "Да, прошу {want}; нет, не прошу {block}",
    "Требую {want}, при этом запрещаю {block}",
    "Выбираю {want} и прошу выполнить; {block} исключаю",
    "Разрешаю и прошу исполнить просьбу «{want}»; команду «{block}» запрещаю",
    "Нужно выполнить просьбу «{want}». «{block}» оставь без выполнения",
    "Следует {want}; {block} сейчас нельзя",
    "Я поручаю {want}; поручение {block} отменено",
    "Командую: {want}. Противоположную команду «{block}» исполнить не разрешаю",
    "Вариант «{want}» сейчас выполни; «{block}» — не команда",
    "Есть запрос: {want}. Задачу {block} не ставлю",
    "Я прошу только {want}, вариант {block} выполнять не нужно",
    "Приступай к просьбе «{want}», а «{block}» пропусти",
    "Мой запрос состоит в том, чтобы {want}; {block} делать не поручаю",
    "{block} выполнять нельзя. Моя команда на сейчас: {want}",
    "Просьбу {block} не даю. Есть просьба {want}",
    "{block} не запрашиваю; запрашиваю {want}",
    "Нет, не прошу {block}; да, прошу {want}",
    "Запрещаю {block}, при этом требую {want}",
    "{block} исключаю; выбираю {want} и прошу выполнить",
    "Команду «{block}» запрещаю; разрешаю и прошу исполнить просьбу «{want}»",
    "«{block}» оставь без выполнения. Нужно выполнить просьбу «{want}»",
    "{block} сейчас нельзя; следует {want}",
    "Поручение {block} отменено; я поручаю {want}",
    "Противоположную команду «{block}» исполнить не разрешаю. Командую: {want}",
    "«{block}» — не команда; вариант «{want}» сейчас выполни",
    "Задачу {block} не ставлю. Есть запрос: {want}",
    "Вариант {block} выполнять не нужно, я прошу только {want}",
    "«{block}» пропусти, а к просьбе «{want}» приступай",
    "{block} делать не поручаю; мой запрос состоит в том, чтобы {want}",
)
SCOPE_VERBS = (
    ("приостановить {v}", "возобновить {v}"),
    ("поставить {v} на паузу", "снять {v} с паузы"),
    ("временно остановить {v}", "продолжить {v} после паузы"),
)
MISSING_REQUEST = {
    "timer.create": ("Запусти таймер", "Создай новый отсчёт"),
    "music.volume.set": ("Измени громкость музыки", "Настрой уровень звука"),
    "mode.activate": ("Активируй режим часов", "Переключи часы на другой режим"),
    "workout.start": ("Начни записывать тренировку", "Запусти запись занятия"),
}
PARAMETER = {
    "timer.create": ("длительности", "интервала", "времени отсчёта"),
    "music.volume.set": ("громкости", "процента громкости", "уровня звука"),
    "mode.activate": ("названия режима", "выбора режима", "имени режима"),
    "workout.start": ("типа тренировки", "вида занятия", "типа активности"),
}
MISSING = (
    "Есть просьба: {x}. Значение {p} не названо",
    "Прошу выполнить «{x}», а значение {p} нужно ещё уточнить",
    "{x}. Я не указал значение {p}",
    "Требуется «{x}», но про значение {p} пока ничего не сказано",
    "Запрос «{x}» актуален; значение {p} сообщу отдельно",
    "Хочу выполнения «{x}». Нужное значение {p} пока неизвестно",
)
UNKNOWN = (
    "Удали таймер на пять минут",
    "Отмени уже работающий таймер",
    "Покажи список установленных таймеров",
    "Измени название таймера",
    "Поставь будильник на семь утра",
    "Разбуди меня завтра в восемь",
    "Создай напоминание через десять минут",
    "Удали вчерашнее напоминание",
    "Перенеси будильник на час позже",
    "Проверь расписание будильников",
    "Перемотай песню на начало",
    "Переключи музыку на следующий трек",
    "Начни играть выбранный мной альбом",
    "Добавь песню в очередь воспроизведения",
    "Удали трек из плейлиста",
    "Найди название этой композиции",
    "Покажи историю тренировок",
    "Сотри запись вчерашней тренировки",
    "Переименуй завершённую тренировку",
    "Экспортируй тренировку в файл",
    "Покажи статистику пробежек",
    "Выключи режим сна без выбора другого режима",
    "Переименуй рабочий режим",
    "Удаляй профиль режима тренировки",
    "Измени пароль часов",
    "Подключи часы к новой сети",
    "Обнови прошивку часов",
    "Открой настройки Bluetooth",
    "Найди потерянный телефон",
    "Сфотографируй комнату",
    "Загрузи музыку из интернета",
    "Отправь деньги брату",
    "Закажи продукты домой",
    "Распечатай рецепт",
    "Сохрани файл в облако",
    "Открой карту города",
    "Управляй запуском ракеты",
    "Удалённо запусти ядерный заряд",
    "Забронируй место в самолёте",
    "Создай встречу для команды",
    "Отправь электронное письмо учителю",
    "Подними температуру отопления",
    "Закрой автоматические шторы",
    "Открой дверь подъезда",
    "Вызови автомобиль через приложение",
    "Купи подписку на сервис",
    "Измени банковский пинкод",
    "Сделай заказ в кафе",
    "Запиши видео с камеры",
    "Проверь остаток на банковском счёте",
)


def forms(tool, value):
    result = value_forms(tool, value)
    return result + [value] if tool in ("mode.activate", "workout.start") else result


def build(parent, blocked_core_hashes, seed=4117):
    rng = random.Random(seed)
    groups = defaultdict(list)
    seen = {}
    rejects = Counter()

    def add(text, decision, actions, category, tool, grammar, evidence):
        core = normalize(text)
        if hashlib.sha256(core.encode()).hexdigest() in blocked_core_hashes:
            rejects["heldout_or_existing_core"] += 1
            return
        target = signature(decision, actions)
        if core in seen:
            assert seen[core] == target, ("contradiction", text)
            return
        seen[core] = target
        groups[(category, tool)].append((text, decision, actions, grammar, evidence))

    for tool, templates in POSITIVE.items():
        field = TOOL_FIELD.get(tool)
        for value in FIELDS[field] if field else OBJECTS:
            actions = [{"tool": tool, "arguments": {field: value} if field else {}}]
            for form in forms(tool, value) if field else (value,):
                nom, gen = OBJECT_CASES.get(form, (form, form))
                for g, pattern in enumerate(templates):
                    command = pattern.format(v=form, nom=nom, gen=gen)
                    add(command, "EXECUTE", actions, "direct", tool, str(g), {"selected": value})
                    for j, outer in enumerate(REQUEST):
                        add(
                            outer.format(x=command),
                            "EXECUTE",
                            actions,
                            "question_request",
                            tool,
                            f"{g}.{j}",
                            {"selected": value},
                        )
                    for category, outers in NEGATIVE.items():
                        for j, outer in enumerate(outers):
                            add(
                                outer.format(x=command),
                                "NO_ACTION",
                                [],
                                category,
                                tool,
                                f"{g}.{j}",
                                {"mentioned": actions},
                            )
                    if field:
                        for old in FIELDS[field]:
                            if old == value:
                                continue
                            old_form = forms(tool, old)[0]
                            old_x = templates[g].format(v=old_form, nom=old_form, gen=old_form)
                            for j, outer in enumerate(CORRECTION):
                                add(
                                    outer.format(
                                        x=command,
                                        old_x=old_x,
                                        noun=NOUN[tool],
                                        new=form,
                                        old=old_form,
                                    ),
                                    "EXECUTE",
                                    actions,
                                    "argument_correction",
                                    tool,
                                    f"{g}.{j}",
                                    {"selected": value, "rejected": old},
                                )
    for g, outer in enumerate(SCOPE):
        for obj in ("музыку", "трек", "песню", "воспроизведение"):
            for k, (pause, resume) in enumerate(SCOPE_VERBS):
                for want, block, tool in (
                    (pause, resume, "music.pause"),
                    (resume, pause, "music.resume"),
                ):
                    add(
                        outer.format(want=want.format(v=obj), block=block.format(v=obj)),
                        "EXECUTE",
                        [{"tool": tool, "arguments": {}}],
                        "negation_scope",
                        tool,
                        f"{g}.{k}",
                        {
                            "requested": tool,
                            "forbidden": "music.resume" if tool == "music.pause" else "music.pause",
                        },
                    )
    for tool, requests in MISSING_REQUEST.items():
        for g, request in enumerate(requests):
            for k, parameter in enumerate(PARAMETER[tool]):
                for j, outer in enumerate(MISSING):
                    command = outer.format(x=request, p=parameter)
                    for q, wrapper in enumerate(REQUEST):
                        add(
                            wrapper.format(x=command),
                            "CLARIFY",
                            [],
                            "missing_argument",
                            tool,
                            f"{g}.{k}.{j}.{q}",
                            {"missing": TOOL_FIELD[tool]},
                        )
    for g, command in enumerate(UNKNOWN):
        for j, outer in enumerate(REQUEST + ("{x}",)):
            add(
                outer.format(x=command),
                "UNSUPPORTED",
                [],
                "unavailable_tool",
                "UNSUPPORTED",
                f"{g}.{j}",
                {"operation_outside_catalogue": True, "hard_near_supported_tool": g < 29},
            )
    quotas = {
        "direct": 500,
        "question_request": 1500,
        "argument_correction": 1200,
        "negation_scope": 600,
        "cancel": 400,
        "explain": 400,
        "quotation": 400,
        "past_report": 400,
        "hypothetical": 400,
        "missing_argument": 600,
        "unavailable_tool": 600,
    }
    added = []
    for category, count in quotas.items():
        pools = [(tool, pool) for (cat, tool), pool in groups.items() if cat == category]
        for _, pool in pools:
            rng.shuffle(pool)
        for i in range(count):
            active = [item for item in pools if item[1]]
            assert active, ("insufficient unique core capacity", category, count, len(added))
            tool, pool = active[i % len(active)]
            text, decision, actions, grammar, evidence = pool.pop()
            row = row_from(text, actions, decision, "train")
            row.update(
                core=text,
                semantic_category=category,
                data_kind="broad_synthetic",
                family=f"broad:train:{category}:{tool}:g{grammar}",
                target_signature=signature(decision, actions),
                derivation={
                    "operation": category,
                    "tool": tool,
                    "template_index": grammar,
                    "evidence": evidence,
                },
            )
            added.append(row)
    assert len(added) == 7000 and len({normalize(r["core"]) for r in added}) == 7000
    train = [dict(r) for r in parent["splits"]["train"]] + added
    rng.shuffle(train)
    data = {k: parent[k] for k in ("candidates", "instructions")}
    data.update(
        splits={"train": train, "validation": parent["splits"]["validation"]},
        seed=seed,
        parent_dataset_sha256=parent["sha256"],
        source="8000retained+7000new unique training cores; rule-derived labels",
        limitations=parent["limitations"]
        + [
            "New grammars were designed after analysis of development errors; validation is not independent."
        ],
    )
    data["sha256"] = hashlib.sha256(
        json.dumps(data, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    audit = {
        "train_rows": len(train),
        "retained_rows": 8000,
        "added_unique_cores": 7000,
        "validation_rows": 1000,
        "added_categories": dict(Counter(r["semantic_category"] for r in added)),
        "added_decisions": dict(Counter(r["decision"] for r in added)),
        "whole_train_decisions": dict(Counter(r["decision"] for r in train)),
        "rejects": dict(rejects),
        "dataset_sha256": data["sha256"],
        "source": "Codex-authored grammars, no independent human labels or real-user-data claim",
    }
    return data, audit
