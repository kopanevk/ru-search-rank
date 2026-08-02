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
