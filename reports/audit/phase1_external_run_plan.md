# Phase 1A external Linux/Colab run plan

Status: **BLOCKED_EXTERNAL_RUN** until the notebook completes on Linux.

## Confirmed protocol

- Corpus: [`miracl/miracl-corpus`](https://huggingface.co/datasets/miracl/miracl-corpus),
  revision `d921ec7e349ce0d28daf30b2da9da5ee698bef0d`.
- Topics and qrels: [`miracl/miracl`](https://huggingface.co/datasets/miracl/miracl),
  revision `5be20db9509754dadad47689368639fcec739c00`.
- Prebuilt Lucene index: `miracl-v1.0-ru`, listed by the official
  [Pyserini index catalog](https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md).
- Runtime pin: Python 3.12, Java 21, `pyserini==2.3.0`. The
  [Pyserini 2.3.0 package page](https://pypi.org/project/pyserini/2.3.0/)
  states Python >=3.12 and the Python 3.12/Java 21 build target.
- Official Russian BM25 dev protocol and published values come from the
  [Pyserini MIRACL 2CR page](https://castorini.github.io/pyserini/2cr/miracl.html):
  nDCG@10 `0.334`, Recall@100 `0.661`.

Official retrieval command:

```bash
python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --language ru \
  --topics miracl-v1.0-ru-dev \
  --index miracl-v1.0-ru \
  --output run.miracl.bm25.ru.dev.txt \
  --bm25 --hits 1000
```

Official evaluation commands:

```bash
python -m pyserini.eval.trec_eval \
  -c -M 100 -m ndcg_cut.10 miracl-v1.0-ru-dev \
  run.miracl.bm25.ru.dev.txt

python -m pyserini.eval.trec_eval \
  -c -m recall.100 miracl-v1.0-ru-dev \
  run.miracl.bm25.ru.dev.txt
```

The project runner retrieves train top-100 and runs the official dev command
with `--hits 1000`, the same official index, Russian analyzer, and BM25
defaults. It evaluates the untouched raw dev top-1000 run with the official
tool first. Only after the reproduction gate passes does it deterministically
sort by score descending/docid ascending and truncate to the project top-100
candidate cache. The raw run's SHA-256 binds the gate to the cache build.

Pyserini does not register `miracl-v1.0-ru-train` as a topic id. The runner
therefore downloads and validates the revision-pinned official train TSV before
retrieval, then runs train with:

```bash
python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --language ru \
  --topics artifacts/raw/miracl-ru/topics.miracl-v1.0-ru-train.tsv \
  --index miracl-v1.0-ru \
  --output artifacts/runs/train_bm25_top100.trec \
  --bm25 --hits 100
```

## Expected external downloads

- Pyserini 2.3.0 source distribution: 112.7 MB as reported by PyPI, plus its
  Python dependencies (exact resolved size is environment-dependent).
- Official compressed Lucene index: 6,895,124,480 bytes (about 6.42 GiB), plus
  extracted index/cache space.
- Official Russian corpus stream: up to 20 revision-pinned gzip shards totaling
  1,575,287,888 bytes (about 1.47 GiB compressed). Shards are streamed until all
  candidate docids are found; only candidate passages are written to Parquet.
- Topics and qrels are small TSV files. The notebook requires at least 30 GiB
  free before the index smoke/download begins.

## Send the code to Colab

1. Review `git status --short` and `git diff --check` locally.
2. Commit the small Phase 1A source/config/test/notebook/audit-plan files on the
   current feature branch. Do not add anything under `artifacts/`.
3. Push that branch to a GitHub remote you control.
4. Open `scripts/run_full_bm25_retrieval.ipynb` in Google Colab (GitHub tab or
   `https://colab.research.google.com/github/<owner>/<repo>/blob/<branch>/scripts/run_full_bm25_retrieval.ipynb`).
5. In notebook cell 3, set the public/authorized `REPO_URL` and exact `BRANCH`.
   Never paste a GitHub token into the notebook.
6. Run cells 1–14 in order. The Python 3.12 test gate, index smoke, train
   retrieval, dev retrieval, official evaluation, corpus streaming, and
   packaging are separated. Cell 10 must pass before cell 11 can build cache.

## Retrieve the result

The final file is:

```text
artifacts/rusearchrank_phase1_results.zip
```

Its exact payload is:

```text
artifacts/candidates/train_top100.parquet
artifacts/candidates/dev_top100.parquet
artifacts/candidates/queries.parquet
artifacts/candidates/passages.parquet
artifacts/runs/train_bm25_top100.trec
artifacts/runs/dev_bm25_top1000.trec
artifacts/runs/dev_bm25_top100.trec
reports/audit/bm25_reproduction.json
reports/audit/qrels_audit.json
reports/audit/candidate_cache_manifest.json
```

The Lucene index, full corpus/cache, Python environment, `.git`, and temporary
work files are not included.

Download the ZIP and its SHA-256 shown by cell 14. On the Mac, place the ZIP at
`artifacts/rusearchrank_phase1_results.zip`, inspect it, verify the digest, and
only then overwrite the blocked audit placeholders:

```bash
shasum -a 256 artifacts/rusearchrank_phase1_results.zip
unzip -l artifacts/rusearchrank_phase1_results.zip
unzip -o artifacts/rusearchrank_phase1_results.zip -d .
```

The relative archive paths place results directly into:

```text
artifacts/candidates/
artifacts/runs/
reports/audit/
```

## Validate after unpacking on the Mac

These checks do not need Java or Pyserini:

```bash
.venv/bin/python -m rusearchrank.cli validate-candidates \
  artifacts/candidates/train_top100.parquet \
  --config configs/retrieval.yaml

.venv/bin/python -m rusearchrank.cli validate-candidates \
  artifacts/candidates/dev_top100.parquet \
  --config configs/retrieval.yaml

.venv/bin/python -m pytest -q
git diff --check
```

Compare every `size_bytes` and `sha256` entry in
`reports/audit/candidate_cache_manifest.json` with the extracted file before
accepting Phase 1. Candidate Parquet, TREC runs, the ZIP, corpus, index, and all
caches remain ignored and must not be added to Git. Only the small real audit
JSON files may be reviewed for a later commit.
