# RuSearchRank

RuSearchRank проверяет двухэтапный поиск по русской части MIRACL:

```text
полный MIRACL Russian → BM25 top-100 → cached candidates
→ cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 → reranking
```

Репозиторий содержит воспроизводимые протоколы Phase 1 (BM25), Phase 2
(zero-shot reranking) и Phase 3 (supervised fine-tuning). Тяжёлые production
этапы выполняются только в зафиксированных Colab-ноутбуках; локальный набор
проверяет алгоритмы, контракты артефактов и failure paths без полного обучения.

Локальный gate проверяет unit-, fixture-E2E- и notebook-контракты; реальные
Russian MIRACL corpus/topics/qrels ранее проверены. Тяжёлый Phase 3 smoke,
production training и официальный dev evaluation требуют Colab CUDA и здесь не
выдаются за выполненные.

## Установка

Основное окружение поддерживает Python 3.12–3.14:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Границы зависимостей намеренно широкие, но ограничены следующими major-релизами:
Transformers начинается с 4.41 (минимум, заявленный Sentence Transformers),
PyTorch — с 2.4 (текущий минимум Transformers), а верхние границы защищают от
непроверенных breaking changes. Диапазоны охватывают актуальные на момент аудита
Datasets 5.x, Transformers 5.x, Sentence Transformers 5.x, PyTorch 2.x,
pandas 3.x, NumPy 2.x и PyArrow 25.x. Pyserini вынесен в optional group, потому
что его официальный стек требует Python 3.12 и Java 21:

```bash
python -m pip install -e '.[retrieval]'
```

## Smoke audit

```bash
python -m rusearchrank.cli environment-report
python -m rusearchrank.cli inspect-data
python -m rusearchrank.cli inspect-checkpoint
python -m rusearchrank.cli inspect-checkpoint --with-model
python -m rusearchrank.cli validate-candidates artifacts/candidates/example.parquet
pytest
```

`inspect-data` читает только небольшой corpus sample через официальный Dataset
Viewer и небольшие topics/qrels; весь корпус не скачивается. Команда
`inspect-checkpoint` по умолчанию загружает только config/tokenizer. Флаг
`--with-model` явно разрешает загрузку весов и CPU/MPS inference.

Audit-файлы находятся в `reports/audit/`. Полный retrieval отдельно описан в
`reports/audit/retrieval_plan.md` и в Phase 0 не запускается.

## Candidate schema

Обязательные поля:

| Поле | Назначение |
|---|---|
| `query_id`, `docid` | непустые идентификаторы без дублей пары |
| `bm25_rank` | положительный уникальный rank внутри query |
| `bm25_score` | исходный BM25 score |
| `judgment` | `relevant`, `non_relevant` или `unjudged` |
| `relevance` | 1, 0 или null соответственно |
| `is_judged` | true, true или false соответственно |

Контракт состояний:

```text
relevant     → relevance=1,    is_judged=true
non_relevant → relevance=0,    is_judged=true
unjudged     → relevance=null, is_judged=false
```

**Unjudged никогда нельзя трактовать как non-relevant.** Отсутствие пары в
qrels означает лишь отсутствие суждения.

Следующий gate — воспроизвести официальный Russian MIRACL BM25 на полном готовом
индексе, проверить nDCG@10 и Recall@100, затем сохранить детерминированный
top-100 candidate cache.

## Phase 2: zero-shot reranking

Phase 1 теперь завершена внешним Colab-прогоном; её BM25-раны и candidate cache
служат неизменяемым входом для Phase 2. Реализован zero-shot cross-encoder
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` на pinned revision
`1427fd652930e4ba29e8149678df786c240d8825`. Полный inference выполняется только
на Colab GPU через `scripts/run_zeroshot_rerank.ipynb`; локальные тесты используют
инъецируемый детерминированный scorer и не скачивают модель.

Основной порядок команд:

```bash
python -m rusearchrank.cli preflight --stage rerank --config configs/rerank.yaml
python -m rusearchrank.cli smoke-rerank --config configs/rerank.yaml --limit 64
python -m rusearchrank.cli rerank-score --config configs/rerank.yaml --split dev
python -m rusearchrank.cli build-rerank-run --config configs/rerank.yaml --split dev --depth 100
python -m rusearchrank.cli build-rerank-run --config configs/rerank.yaml --split dev --depth 10
python -m rusearchrank.cli build-rerank-run --config configs/rerank.yaml --split dev --depth 20
python -m rusearchrank.cli build-rerank-run --config configs/rerank.yaml --split dev --depth 50
python -m rusearchrank.cli evaluate-rerank --config configs/rerank.yaml --split dev
python -m rusearchrank.cli package-phase2 --config configs/rerank.yaml
```

`rerank-score` заблокирован на уровне CLI до успешного real-model smoke с
совпадающими config/source/input/model fingerprints. Скоринг возобновляется по
query-shards; несовместимые шарды сохраняются как `*.stale.<UTC>`. Сырые logits
хранятся без округления в float32 Parquet, а TREC-файл получает отдельный
строго убывающий score `1000000 - rank`, поэтому tie-break `trec_eval` не может
изменить порядок.

Официальная метрика Phase 2 — только standard nDCG@10 из NIST `trec_eval`
v9.0.8. Recall@100, MRR@10, condensed nDCG, paired bootstrap, sparse-judgment
диагностика и depth profile K=10/20/50 являются диагностическими. Финальный ZIP
содержит score Parquet, официальный K=100 run, четыре metrics JSON, byte-exact
snapshot протокола и non-self-referential manifest.

## Phase 3: supervised fine-tuning

Phase 3 обучает тот же pinned cross-encoder на MIRACL Russian train и полностью
изолирует финальный evaluation split до атомарной записи
`reports/audit/checkpoint_selection.json`. Production training и полный scoring
на Mac запрещены; точный порядок запуска находится в
`scripts/run_finetune.ipynb` и статически проверяется
`scripts/validate_phase3_notebook.py`.

Основной Colab-порядок:

```bash
python -m rusearchrank.cli build-training-split --config configs/finetune.yaml
python -m rusearchrank.cli build-training-pairs --config configs/finetune.yaml --regime judged_only
python -m rusearchrank.cli build-training-pairs --config configs/finetune.yaml --regime weak_negatives
python -m rusearchrank.cli build-training-pairs --config configs/finetune.yaml --regime control_c1
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

Все режимы используют только positives внутри BM25 top-100. `judged_only` и
`weak_negatives` обучаются на разных популяциях запросов: второй режим включает
positive-запросы без judged negative, поэтому их описательная дельта смешивает
большее число негативов на запрос и большее число пригодных запросов. Оба
`usable_query_count` публикуются в `pairs_manifest.json`.

Вес 0.5 уменьшает вклад отдельной weak-пары, но не фиксирует их совокупную долю:
при 8 judged + 8 weak она равна 33%, при 2 judged + 8 weak — 67%, без judged —
100%. Манифест публикует полное распределение по запросам. Ранги 26–100, до
8 weak документов, вес 0.5 и cap 8/8/16 — консервативные preregistered
эвристики, зафиксированные до финальной оценки, а не найденные оптимумы.

Validation A/B является `exploratory_post_selection`: тот же holdout выбирает
learning rate, epoch, лучший judged-only run и итоговый fine-tuned checkpoint.
Поэтому сравнение не является confirmatory и не поддерживает причинную
интерпретацию. Финальная оценка выполняется один раз для содержимого
`artifacts/models/best_finetuned/`; `production_system` выбирается отдельно и
может остаться zero-shot.
