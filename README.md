# RuSearchRank

RuSearchRank проверяет двухэтапный поиск по русской части MIRACL:

```text
полный MIRACL Russian → BM25 top-100 → cached candidates
→ cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 → reranking
```

Сейчас реализован только **Phase 0**: контракты данных и кандидатов, безопасные
smoke-команды, проверка checkpoint и план воспроизведения BM25. Обучения,
финального evaluation и полной загрузки/индексации корпуса в репозитории нет.

Статус gate на 2026-08-02: 25 unit-тестов проходят; реальные Russian MIRACL
corpus/topics/qrels проверены; checkpoint smoke inference проходит на CPU и
MPS. Готовый официальный BM25 index подтверждён, но retrieval не запускался:
локально отсутствует Java 21, поэтому полный gate запланирован в Linux-среде.

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
