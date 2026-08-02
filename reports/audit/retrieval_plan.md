# MIRACL Russian BM25 retrieval gate

Status: **planned, not executed**.

## Decision

Use **C. a separate Linux environment** with Python 3.12 and Java 21. The
current Mac has sufficient free disk (about 333 GiB), but has no Java runtime
and only Python 3.14; Pyserini 2.x documents Python 3.12 as its build target and
Java 21 as required by Anserini. Colab remains a fallback, but its ephemeral
cache is less suitable for a 6.9 GB index.

## Official index and stack

- A prebuilt official index exists: `miracl-v1.0-ru`.
- Current manifest: Lucene 10.4.0, 9,543,918 documents.
- Compressed archive: 6,895,124,480 bytes (about 6.42 GiB), MD5
  `b0a57963ccfe52ec7edb89cff1fb8c33`.
- Conservative working-space budget: **15 GiB (UNVERIFIED)** to allow the
  archive, extracted cache, run file, and temporary overhead to coexist.
- Retrieval dependency: `pyserini>=2.3,<3.0` from the optional `retrieval`
  dependency group; Java 21; Python 3.12.

Sources:

- https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md
- https://github.com/castorini/pyserini/blob/master/pyserini/resources/prebuilt-indexes/lucene/miracl-inverted.json
- https://castorini.github.io/pyserini/2cr/miracl.html

## Exact next commands

After creating a Python 3.12 environment, installing Java 21, and running
`python -m pip install -e '.[retrieval]'`:

```bash
python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --language ru \
  --topics miracl-v1.0-ru-dev \
  --index miracl-v1.0-ru \
  --output run.miracl.bm25.ru.dev.txt \
  --bm25 --hits 1000
```

The 1,000-hit run is retained for official reproduction. The project candidate
cache is then deterministically cut to top-100 per query.

```bash
trec_eval -c -M 100 -m ndcg_cut.10 \
  artifacts/raw/miracl-ru/qrels.miracl-v1.0-ru-dev.tsv \
  artifacts/runs/dev_bm25_top1000.trec

trec_eval -c -m recall.100 \
  artifacts/raw/miracl-ru/qrels.miracl-v1.0-ru-dev.tsv \
  artifacts/runs/dev_bm25_top1000.trec
```

## Numerical gate

Official Pyserini MIRACL Russian BM25 values are nDCG@10 = **0.334** and
Recall@100 = **0.661**. Both must reproduce within an absolute tolerance of
**0.001**, matching the precision published by the official 2CR table.

No index download, full retrieval, or local indexing was started in Phase 0.
