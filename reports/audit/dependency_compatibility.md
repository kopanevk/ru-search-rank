# Phase 1 dependency compatibility matrix

Why this file exists: the Colab run failed at Cell 11 with
`Dataset scripts are no longer supported, but found miracl-corpus.py`, and 93
passing tests did not catch it. The corpus loader was the only module whose
behaviour depended on the installed `datasets` major version, and it had no
test that executed it. This matrix records, for every dependency, which Phase 1
function depends on it and how that path is actually verified.

Verification levels:

- **real** — executed against the live Hugging Face Hub and real gzip bytes.
- **fixture** — executed against real local `.jsonl.gz`/Parquet/ZIP files, no network.
- **colab-required** — cannot run outside Linux + Python 3.12 + Java 21 + Pyserini.

| Dependency | Allowed version | Verified version | Phase 1 function that depends on it | Verified path |
| --- | --- | --- | --- | --- |
| `huggingface-hub` | `>=0.24,<2.0` | 1.26.0 (macOS probe) | `corpus.resolve_hub_shards`, `corpus.HubShardSource.local_path` (`HfApi.dataset_info`, `HfApi.get_paths_info`, `hf_hub_download`) | **real** — pinned revision resolved, 20 shards listed, `docs-0.jsonl.gz` downloaded and parsed; failure paths (hub down, wrong revision, missing shard, interrupted download) covered by stubs |
| `pyarrow` | `>=17,<26` | 25.0.0 | `corpus._PassageBatchWriter` (`ParquetWriter`), `corpus._validate_temporary_passages` (`ParquetFile.iter_batches`), `cli._build_candidate_cache` (`pq.read_table`) | **fixture** + **real** (smoke writes and re-reads Parquet) |
| `pandas` | `>=2.2,<4.0` | 3.0.5 | `retrieval.*`, `data.attach_qrels`, `evaluation.build_qrels_split_audit`, all Parquet candidate tables | **fixture** — full Cells 12–14 equivalent end to end |
| `numpy` | `>=1.26,<3.0` | 2.5.1 | `retrieval.validate_top_k`, `data.validate_candidate_schema`, `evaluation.*` | **fixture** |
| `PyYAML` | `>=6.0,<7.0` | 6.0.3 | `cli._load_retrieval_config` | **fixture** |
| `pytest` | `>=8.0,<10.0` | 9.1.1 | test suite only | **fixture** |
| `datasets` | `>=4.0,<6.0` | 5.0.1 | **none** — Phase 1 has no import of `datasets`; enforced by `scripts/validate_phase1_notebook.py` (AST check over `src/`) and by `test_production_extraction_never_imports_datasets` | **fixture** (a clean subprocess where importing `datasets` raises the exact Colab error still extracts passages) |
| `transformers`, `sentence-transformers`, `torch` | `>=4.41,<6.0`, `>=4.0,<6.0`, `>=2.4,<3.0` | not exercised in Phase 1 | `cli._inspect_checkpoint` (Phase 0/2 only) | not on the Phase 1 path |
| `pyserini` | `==2.3.0` (extra) | — | `retrieval.run_pyserini_bm25`, `retrieval.smoke_prebuilt_index`, `cli._preflight_retrieval` (`get_topics`) | **colab-required** |
| NIST `trec_eval` | `v9.0.8` (built from source in Cell 4) | — | `cli._run_trec_eval_binary` — the Python wrapper `pyserini.eval.trec_eval` is banned by the notebook validator | **colab-required** |
| Java | `21` | — | Pyserini/Lucene | **colab-required** |
| Python | `3.12` in Colab; `>=3.12,<3.15` allowed | 3.12 (Colab), 3.14.6 (local probe) | everything | **fixture** on 3.14, **colab-required** on 3.12 |

## Version probes

```bash
python -c "import datasets; print(datasets.__version__)"
python -c "import pandas; print(pandas.__version__)"
python -c "import pyarrow; print(pyarrow.__version__)"
python -c "import huggingface_hub; print(huggingface_hub.__version__)"
```

Cell 6 of `scripts/run_full_bm25_retrieval.ipynb` runs the same probe inside the
isolated Python 3.12 environment and stores the result in its command log, so
every Colab run records the exact resolved versions.

## Why the old tests missed `datasets==5.0.1`

`data.stream_candidate_passages` was the only caller of `load_dataset`, and the
only test that reached the candidate cache replaced it wholesale with
`monkeypatch.setattr(cli_module, "stream_candidate_passages", fake_passages)`.
The remaining passage tests called `extract_passages_from_rows` with a Python
list. No test ever imported `datasets`, so no test could observe that
script-backed loading had been removed from the library.

The replacements are:

1. `tests/test_corpus_shards.py` and `tests/test_corpus_fixture_e2e.py` never
   mock the reader; they write and read real `.jsonl.gz` shards.
2. `test_production_extraction_never_imports_datasets` runs the extraction in a
   clean subprocess where `import datasets` raises
   `Dataset scripts are no longer supported, but found miracl-corpus.py`. The
   old implementation would fail there; the current one passes.
3. `scripts/validate_phase1_notebook.py` parses every module under `src/` and
   fails on any `load_dataset(...)` call, any `trust_remote_code=` keyword, and
   any `import datasets`.
