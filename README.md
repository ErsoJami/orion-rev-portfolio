# Orion Rev action selection

Воспроизводимый портфолио-пакет обучения SmolLM2-135M с REV-inspired pointer head и сравнения с опубликованным Original Rev на Qwen3.5-4B.

## Содержимое

- `data/train_validation.json` — 15 000 синтетических train-строк и 1 000 validation-строк. Sealed test сюда не включён.
- `weights/` — начальный, финальный resumable и исходный REV checkpoints. GitHub хранит их через Git LFS.
- `src/` — обучение, scorer, правила генерации данных, метрики и сравнение.
- `results/` — отчёты, метрики, TensorBoard scalar-история и manifest с SHA-256.
- `DATASET.md` и `RESULTS.md` — методика данных, результаты и ограничения эксперимента.

Пакет занимает около 2,12 ГиБ. Для получения checkpoint-файлов установите Git LFS до клонирования. На GitHub они представлены LFS-объектами; для обычного клона используйте:

```powershell
git lfs install
git clone https://github.com/ErsoJami/orion-rev-portfolio.git
```

## Требования

Нужны Python 3.12+, CUDA PyTorch, зависимости из `src/requirements.txt` и GPU с памятью для модели. SmolLM2 загружается с закреплённой ревизии `12fd25f77366fa6b3b4b768ec3050bf629380bac`. Для Original Rev используется Qwen3.5-4B `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` и опубликованный adapter из `weights/original_rev/`.

## Повторить обучение

Из корня клонированного репозитория:

```powershell
python .\src\scripts\train_smol_diverse_rev.py `
  --dataset .\data\train_validation.json `
  --initial-checkpoint .\weights\initial.pt `
  --max-updates 1875 --body-lr 0.000002 --head-lr 0.00001 `
  --seed 4118 --eval-every 250 --risk-weight 0 `
  --final-train-evaluation full --run-name portfolio-reproduction `
  --artifact-prefix portfolio_rev
```

Команда создаёт новый training run. В нём используются train и validation части корпуса; sealed test не передаётся trainer-у. Optimizer создаётся заново, даже если начальные веса уже были обучены.

## Повторить сравнение

```powershell
python .\src\compare.py
```

Скрипт оценивает обе модели на одинаковых 1 000 validation-командах и записывает exact match, false-action rate и batch-1 latency. Qwen3.5-4B в этом сравнении не дообучался.

## TensorBoard

```powershell
python .\src\build_tensorboard.py
python -m tensorboard.main --logdir .\results\tensorboard --host 127.0.0.1 --port 6021
```

Откройте `http://127.0.0.1:6021`. Dashboard строится из агрегированных логов и содержит кривые обучения, метрики датасета, сравнение точности и latency. Сырые тексты команд в TensorBoard не копируются.

## Результаты и целостность

Краткая сводка и границы интерпретации находятся в [RESULTS.md](RESULTS.md), описание датасета — в [DATASET.md](DATASET.md). Хэши корпуса и checkpoint-файлов указаны в `results/manifest.json`.
