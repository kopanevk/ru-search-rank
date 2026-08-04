# RuSearchRank

RuSearchRank — проект двухэтапного поиска по русскоязычной части MIRACL с проверяемым экспериментальным протоколом:

```text
запрос → BM25 top-100 → reranking с cross-encoder → итоговая выдача
```

Cross-encoder — модель, которая совместно обрабатывает запрос и документ и оценивает их релевантность. В проекте используется [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) в зафиксированной ревизии `1427fd652930e4ba29e8149678df786c240d8825`.

Проект охватывает построение кандидатов BM25, zero-shot reranking, fine-tuning на русскоязычных данных, выбор контрольной точки до доступа к dev, итоговое оценивание с NIST `trec_eval` и проверяемую упаковку результатов.

## Подтверждённый результат

Официальный запуск от 3 августа 2026 года дал следующие метрики на 1 252 запросах MIRACL-RU dev:

| Система | nDCG@10 | MRR@10 | Recall@100 |
|---|---:|---:|---:|
| BM25 | 0.3342 | 0.4079 | 0.6614 |
| Zero-shot cross-encoder | 0.5365 | 0.6462 | 0.6614 |
| Cross-encoder после fine-tuning | **0.5676** | **0.6762** | **0.6614** |

Точная разница между моделью после fine-tuning и zero-shot моделью:

```text
Δ nDCG@10 = 0.031123562300319492
95% CI     = [0.024844556709265178; 0.037390425319488815]
```

Интервал получен методом paired bootstrap — парной повторной выборки запросов. Его нижняя граница выше нуля и заранее заданного практически значимого порога 0.010. В рамках этого протокола итог классифицирован как `improvement_confirmed`.

Одинаковый Recall@100 ожидаем: все три системы работают с одним зафиксированным множеством top-100 BM25-кандидатов и меняют только их порядок.

> Статус воспроизводимости: метрики и контрольные суммы официальных ZIP независимо сверены с сохранёнными артефактами. Исправленная схема происхождения артефактов v1.0.1 прошла сквозной тест на временном наборе данных, но новый полный GPU-запуск ещё не выполнен. Поэтому проект не заявляет, что ZIP v1.0.1 уже воспроизведён побайтно.

## Что реализовано

- детерминированное построение top-100 кандидатов BM25;
- zero-shot reranking с сохранением исходных оценок в `float32` Parquet;
- разделение обучающих запросов на `train_fit` и `train_validation` без пересечения;
- попарное обучение с экспертно оценёнными отрицательными примерами;
- weak negatives — документы без экспертной оценки, используемые как слабые отрицательные примеры;
- контроль C1 с перемешанными метками;
- выбор скорости обучения, эпохи и контрольной точки только по обучающей и контрольной выборкам;
- неизменяемая запись выбора до первого разрешённого чтения dev qrels;
- журнал доступа к dev в виде цепочки контрольных сумм;
- расчёт nDCG@10, MRR@10 и Recall@100 официальным `trec_eval`;
- paired bootstrap и диагностика разреженной разметки;
- манифест происхождения каждого файла, CRC и SHA-256 для итоговых ZIP.

Подробности: [экспериментальный протокол](docs/experimental_protocol.md), [воспроизводимость](docs/reproducibility.md), [результаты и ограничения](docs/results.md).

## Данные и схема эксперимента

Используется русскоязычная часть [MIRACL](https://huggingface.co/datasets/miracl/miracl):

| Часть | Размер | Назначение |
|---|---:|---|
| MIRACL-RU train | 4 683 запроса | обучение и внутренний выбор |
| `train_fit` | 3 976 запросов | обучение C1, A1, A2 и B1 |
| `train_validation` | 707 запросов | выбор режима, скорости обучения и эпохи |
| MIRACL-RU dev | 1 252 запроса | однократная итоговая оценка |
| Кандидаты | 100 документов на запрос | неизменяемый вход для reranking |
| Пары на dev | 125 200 | применение cross-encoder |

Результат относится к фиксированному top-100 BM25. Проект не утверждает, что fine-tuning улучшает полноту первого этапа поиска.

## Архитектура

```text
MIRACL-RU
  ├─ train → BM25 top-100 → train_fit / train_validation
  │                         ├─ C1: перемешанные метки
  │                         ├─ A1: judged_only, LR 7e-6
  │                         ├─ A2: judged_only, LR 2e-5
  │                         └─ B1: weak_negatives, LR лучшего A1/A2
  │
  └─ dev → BM25 top-100 ───────────────┐
                                       ├─ BM25
выбор контрольной точки до dev ────────┼─ zero-shot cross-encoder
                                       └─ cross-encoder после fine-tuning
```

В официальном запуске выбран B1, эпоха 1, скорость обучения `2e-5`; внутренний nDCG@10 равен `0.5186686078167897`. Решение записано в `checkpoint_selection.json` в 20:27:20 UTC, первое событие доступа к dev — в 20:30:05 UTC.

Сравнение режимов `judged_only` и `weak_negatives` является исследовательским анализом после выбора, а не чистым причинным экспериментом: режимы отличаются и составом отрицательных примеров, и числом пригодных запросов.

## Быстрый локальный старт

Поддерживаются Python 3.12–3.14. Локальные тесты не выполняют дорогое обучение и не требуют GPU:

```bash
git clone https://github.com/kopanevk/ru-search-rank.git
cd ru-search-rank

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

python -m pytest -q
python scripts/validate_phase3_notebook.py
python -m compileall -q src
```

Для первого этапа с Pyserini:

```bash
python -m pip install -e '.[retrieval]'
```

## Полный запуск этапа 3 в Kaggle

Канонический файл: [scripts/run_finetune_kaggle.ipynb](scripts/run_finetune_kaggle.ipynb).

1. Создайте выпуск с неизменяемым тегом `phase3-v1.0.1`.
2. Подключите к Kaggle закрытый набор данных с официальными ZIP этапов 1 и 2.
3. Включите GPU.
4. Откройте notebook и выполните `Restart Kernel and Run All`.
5. Заберите два проверенных ZIP из рабочего каталога Kaggle.

Notebook содержит 24 стадии и не меняет `src/` или `configs/` во время запуска. Он проверяет:

- точную фиксацию Git, на которую указывает тег выпуска;
- Linux, GPU, RAM и свободное место;
- `trec_eval v9.0.8`, собранный из официальной фиксации Git;
- Python 3.12.13, torch 2.13.0, transformers 5.14.1 и tokenizers 0.22.2;
- контрольные суммы токенизатора и train qrels;
- основной набор pytest и отдельный отрицательный CUDA-тест;
- CRC и SHA-256 восстановленных и итоговых ZIP;
- полную первичную проверку перед C1;
- порядок C1 → A1 → A2 → B1 → выбор → dev.

Точное окружение задано в [requirements/kaggle.lock](requirements/kaggle.lock).

## Разрешение trec_eval

Один общий механизм используется во всех фазах. Приоритет:

1. `--trec-eval-path`;
2. `trec_eval_executable` в конфигурации;
3. переменная `TREC_EVAL_PATH`;
4. поиск `trec_eval` в `PATH`.

Пример:

```bash
export TREC_EVAL_PATH="$PWD/tools/trec_eval"
python -m rusearchrank.cli preflight   --stage rerank   --config configs/rerank.yaml
```

Перед официальным оцениванием проверяются обычный тип файла, право на выполнение, версия, SHA-256 и происхождение сборки.

## Порядок команд этапа 3

После подготовки входов команды выполняются строго последовательно:

```bash
python -m rusearchrank.cli validate-checkpoint --config configs/finetune.yaml --checkpoint base
python -m rusearchrank.cli smoke-finetune --config configs/finetune.yaml --limit-pairs 64
python -m rusearchrank.cli finetune --config configs/finetune.yaml --run-id C1
python -m rusearchrank.cli finetune --config configs/finetune.yaml --run-id A1
python -m rusearchrank.cli finetune --config configs/finetune.yaml --run-id A2
python -m rusearchrank.cli finetune --config configs/finetune.yaml --run-id B1
python -m rusearchrank.cli select-checkpoint --config configs/finetune.yaml
python -m rusearchrank.cli prepare-dev-evaluation --config configs/finetune.yaml
python -m rusearchrank.cli score-finetuned --config configs/finetune.yaml
python -m rusearchrank.cli evaluate-phase3 --config configs/finetune.yaml
python -m rusearchrank.cli package-phase3 --config configs/finetune.yaml
```

Эти команды рассчитаны на подготовленные артефакты этапов 1 и 2. Для полного запуска используйте Kaggle notebook: он восстанавливает входы и проверяет их контрольные суммы.

## Официальные архивы 2026-08-03

Старые ZIP сохранены без изменения:

```text
rusearchrank_phase3_results.zip
SHA-256 f0d413bf5c9bcf48aa60c8d21f3bfa9ab49d09b83db827b143bf5c89dfc50352

rusearchrank_phase3_model_B1.zip
SHA-256 ac570ff41a1c6a147b2208ea0b0a28ce2ed5000fad3a61b4ba11c838eaeef823
```

В старом `training_manifest.json` сведения B1 ошибочно применялись к историям A1 и A2. Это не меняет сохранённые метрики или веса, но делает происхождение этих двух записей некорректным. Старые ZIP намеренно не перепаковываются.

Новая схема v2 связывает `A1_history.json`, `A2_history.json` и `B1_history.json` только с соответствующими запусками. Новые имена:

```text
rusearchrank_phase3_results_v1.0.1.zip
rusearchrank_phase3_model_<run_id>_v1.0.1.zip
```

Контрольные суммы появятся только после полного чистого GPU-запуска. Проект не подставляет в новый манифест сведения, которых нет в старом архиве.

## Структура репозитория

```text
configs/                  зафиксированные протоколы трёх этапов
docs/                     методология, воспроизводимость и результаты
requirements/             окружение полного запуска Kaggle
scripts/                  notebook и статические валидаторы
src/rusearchrank/         конвейер, обучение, оценивание и CLI
tests/                    модульные и сквозные проверки на временных данных
artifacts/                создаваемые крупные артефакты, не хранятся в Git
reports/                  создаваемые метрики и проверочные отчёты
```

## Ограничения

- Выполнен один полный запуск обучения с одним начальным значением генератора; paired bootstrap оценивает изменчивость по запросам, а не по повторным запускам обучения.
- Итоговая оценка относится к одному MIRACL-RU dev и не заменяет проверку на независимом наборе.
- Разметка разрежена: отсутствие экспертной оценки не доказывает нерелевантность документа.
- B1 не является чистым сравнительным экспериментом одного фактора.
- Новый выпуск v1.0.1 требует полного GPU-запуска; локальные тесты подтверждают код и контракты, но не численное воспроизведение обучения.
- Данные MIRACL, исходные веса и крупные производные файлы не включены в Git.

## Лицензия и цитирование

Код распространяется по [лицензии MIT](LICENSE). Условия сторонних данных, модели и `trec_eval` сохраняются; подробности — в [NOTICE](NOTICE).

Сведения для цитирования находятся в [CITATION.cff](CITATION.cff). Основные внешние материалы:

- [MIRACL: Multilingual Information Retrieval Across a Continuum of Languages](https://arxiv.org/abs/2210.09984);
- [модель cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1);
- [NIST trec_eval](https://github.com/usnistgov/trec_eval).
