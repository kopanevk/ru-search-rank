from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rusearchrank.training import (
    StageError,
    TokenCache,
    TRAINING_FINGERPRINT_FIELDS,
    accumulation_windows,
    build_adamw_optimizer,
    build_token_cache,
    build_training_fingerprint,
    disk_preflight,
    epoch_query_order,
    finalize_run,
    load_resume_state,
    pairwise_logistic_loss,
    publish_epoch_generation,
    publish_resume_state,
    query_pairwise_loss,
    run_accumulation_epoch,
    run_finetune,
    run_learning_rate,
    score_query_group_once,
    update_latest_checkpoint,
    validate_epoch_generation,
    weighted_query_loss,
)
from rusearchrank.training_data import (
    atomic_write_json,
    load_finetune_config,
    read_json,
    source_set_sha256,
)


def test_pairwise_logistic_matches_manual_cases_and_is_stable() -> None:
    positive = torch.tensor([1.0, 0.0, -1.0, 50.0, -50.0])
    negative = torch.zeros(5)
    losses = pairwise_logistic_loss(positive, negative)
    expected = torch.logaddexp(torch.zeros(5), -(positive - negative))
    assert torch.allclose(losses, expected)
    assert bool(torch.isfinite(losses).all())
    assert losses[0] < losses[1] < losses[2]


def test_weighted_query_loss_is_sum_over_weight_sum_not_mean_weighted_loss() -> None:
    losses = torch.tensor([1.0, 3.0])
    weights = torch.tensor([1.0, 0.5])
    observed = weighted_query_loss(losses, weights)
    assert torch.isclose(observed, torch.tensor(2.5 / 1.5))
    assert not torch.isclose(observed, (weights * losses).mean())
    repeated = weighted_query_loss(torch.ones(20), torch.full((20,), 0.5))
    assert repeated == 1.0


class CapturingSGD(torch.optim.SGD):
    def __init__(self, params: list[torch.nn.Parameter]) -> None:
        super().__init__(params, lr=0.0)
        self.gradients: list[float] = []

    def step(self, closure=None):  # type: ignore[no-untyped-def]
        parameter = self.param_groups[0]["params"][0]
        self.gradients.append(float(parameter.grad.detach()))
        return super().step(closure)


def test_gradient_accumulation_uses_actual_16_and_1_query_window_means() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = CapturingSGD([parameter])
    query_ids = [f"q{index:02d}" for index in range(17)]
    coefficients = {query_id: (index + 1) / 100.0 for index, query_id in enumerate(query_ids)}
    report = run_accumulation_epoch(
        query_ids,
        loss_for_query=lambda query_id: parameter * coefficients[query_id],
        optimizer=optimizer,
        scheduler=None,
        parameters=[parameter],
        accumulation=16,
        max_grad_norm=1.0,
    )
    assert optimizer.gradients == pytest.approx(
        [np.mean(list(coefficients.values())[:16]), coefficients["q16"]]
    )
    assert report["last_window_query_count"] == 1
    assert report["seen_query_ids"] == query_ids
    assert [len(window) for window in accumulation_windows(query_ids, 16)] == [16, 1]


def test_16_separate_backward_calls_equal_one_summed_objective() -> None:
    coefficients = torch.linspace(0.01, 0.16, 16)
    separate = torch.nn.Parameter(torch.tensor(2.0))
    for coefficient in coefficients:
        (separate * coefficient / 16).backward()
    combined = torch.nn.Parameter(torch.tensor(2.0))
    (combined * coefficients.mean()).backward()
    assert torch.allclose(separate.grad, combined.grad, atol=1e-7, rtol=1e-7)


def test_dropout_path_is_reproducible_for_same_seed_and_execution_path() -> None:
    def gradient(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.5), torch.nn.Linear(4, 1))
        model.train()
        model(torch.ones(3, 4)).sum().backward()
        return torch.cat([parameter.grad.reshape(-1) for parameter in model.parameters()])

    first = gradient(7)
    second = gradient(7)
    assert bool(torch.isfinite(first).all())
    assert torch.equal(first, second)


class CountingTokenizer:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        return {"input_ids": [0, 1, 2, 2], "attention_mask": [1, 1, 1, 1]}


def test_token_cache_ram_is_checked_before_any_tokenizer_allocation() -> None:
    pairs = pd.DataFrame(
        {
            "query_id": ["q"],
            "positive_docid": ["p"],
            "negative_docid": ["n"],
        }
    )
    queries = pd.DataFrame({"split": ["train"], "query_id": ["q"], "query_text": ["query"]})
    passages = pd.DataFrame(
        {"docid": ["p", "n"], "title": ["", ""], "text": ["positive", "negative"]}
    )
    tokenizer = CountingTokenizer()
    with pytest.raises(StageError, match="insufficient RAM"):
        build_token_cache(
            pairs,
            queries,
            passages,
            tokenizer,
            available_bytes=1,
            minimum_free_ram_gib=4,
        )
    assert tokenizer.calls == 0


class WeightDecayFixture(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(2, 2)
        self.LayerNorm = torch.nn.LayerNorm(2)


def test_adamw_no_decay_groups_match_name_patterns() -> None:
    model = WeightDecayFixture()
    optimizer = build_adamw_optimizer(model, learning_rate=1e-5)
    names = optimizer._rusearchrank_parameter_names  # type: ignore[attr-defined]
    assert all("bias" not in name and "LayerNorm.weight" not in name for name in names["decay"])
    assert all("bias" in name or "LayerNorm.weight" in name for name in names["no_decay"])
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.01, 0.0]


def fingerprint_fixture() -> dict[str, object]:
    return {
        "training_source_sha256": "a" * 64,
        "training_config_sha256": "b" * 64,
        "split_manifest_sha256": "c" * 64,
        "pair_file_sha256": "d" * 64,
        "pair_manifest_section_sha256": "e" * 64,
        "validation_groups_sha256": "f" * 64,
        "model_id": "model",
        "model_revision": "1" * 40,
        "tokenizer_revision": "2" * 40,
        "run_id": "A1",
        "regime": "judged_only",
        "learning_rate": 7e-6,
        "epochs": 3,
        "implementation_version": "3.0.0",
        "python_version": "3.12.0",
        "torch_version": "2.8.0",
        "transformers_version": "5.0.0",
        "tokenizers_version": "0.22.0",
    }


@pytest.mark.parametrize("field", sorted(TRAINING_FINGERPRINT_FIELDS))
def test_every_training_fingerprint_component_is_load_bearing(field: str) -> None:
    base = fingerprint_fixture()
    changed = copy.deepcopy(base)
    value = changed[field]
    changed[field] = not value if isinstance(value, bool) else f"{value}-changed"
    assert build_training_fingerprint(base) != build_training_fingerprint(changed)


def test_evaluation_source_is_not_a_training_fingerprint_component() -> None:
    base = fingerprint_fixture()
    assert "evaluation_source_sha256" not in base
    assert set(base) == set(TRAINING_FINGERPRINT_FIELDS)


def test_training_source_bytes_invalidate_even_when_git_provenance_is_unchanged(
    tmp_path: Path,
) -> None:
    training_files = (
        "src/rusearchrank/training.py",
        "src/rusearchrank/training_data.py",
        "src/rusearchrank/pair_encoding.py",
    )
    evaluation_file = tmp_path / "src/rusearchrank/evaluation.py"
    for relative in training_files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    evaluation_file.write_text("# evaluation v1\n", encoding="utf-8")
    components = fingerprint_fixture()
    components["training_source_sha256"] = source_set_sha256(
        tmp_path, training_files
    )[0]
    before = build_training_fingerprint(components)

    # Neither git_commit/git_dirty nor evaluation source are fingerprint fields.
    evaluation_file.write_text("# evaluation v2\n", encoding="utf-8")
    assert build_training_fingerprint(components) == before

    training_path = tmp_path / training_files[0]
    training_path.write_text("# changed training bytes\n", encoding="utf-8")
    changed = copy.deepcopy(components)
    changed["training_source_sha256"] = source_set_sha256(
        tmp_path, training_files
    )[0]
    assert build_training_fingerprint(changed) != before


def test_epoch_permutation_is_a_pure_seed_epoch_function() -> None:
    query_ids = ["q3", "q1", "q2"]
    assert epoch_query_order(query_ids, seed=20260803, epoch=2) == epoch_query_order(
        list(reversed(query_ids)), seed=20260803, epoch=2
    )
    assert epoch_query_order(query_ids, seed=20260803, epoch=1) != epoch_query_order(
        query_ids, seed=20260803, epoch=2
    )


def test_b1_requires_both_judged_runs_and_tie_uses_lower_lr() -> None:
    config = load_finetune_config()
    with pytest.raises(StageError, match="A1 is not complete"):
        run_learning_rate(config, "B1", {"runs": {}})
    metrics = {
        "runs": {
            run_id: {
                f"epoch_{epoch}": {"epoch": epoch, "ndcg_at_10": 0.4}
                for epoch in (1, 2, 3)
            }
            for run_id in ("A1", "A2")
        }
    }
    assert run_learning_rate(config, "B1", metrics) == 7e-6


def test_unknown_run_and_non_cuda_production_are_blocked() -> None:
    config = load_finetune_config()
    with pytest.raises(ValueError, match="allowed"):
        run_learning_rate(config, "unknown")
    with pytest.raises(StageError, match="requires CUDA"):
        run_finetune(config, run_id="A1")
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_finetune(config, run_id="A1", resume=True, overwrite=True)


def test_disk_preflight_fails_before_training_when_space_is_insufficient() -> None:
    with pytest.raises(StageError, match="insufficient free disk"):
        disk_preflight(
            ".",
            weight_size_bytes=1000,
            resume_state_size_bytes=3000,
            free_bytes=1,
        )


class SaveOnlyTokenizer:
    def save_pretrained(self, directory: str | Path) -> None:
        path = Path(directory)
        (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


def test_checkpoint_generation_resume_and_finalize_lifecycle(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        num_labels=1,
    )
    model = transformers.BertForSequenceClassification(config)
    tokenizer = SaveOnlyTokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    forward_batch = {
        "input_ids": torch.ones((8, 5), dtype=torch.long),
        "attention_mask": torch.ones((8, 5), dtype=torch.long),
    }
    fingerprint = "9" * 64
    sidecar = publish_epoch_generation(
        tmp_path,
        run_id="A1",
        epoch=1,
        model=model,
        tokenizer=tokenizer,
        training_fingerprint=fingerprint,
        fingerprint_components={"fixture": True},
        validation={"ndcg_at_10": 0.5},
        optimizer_steps=2,
        last_window_query_count=1,
        throughput={"pairs_per_second": 1.0},
        peak_gpu_memory_bytes=0,
        peak_rss=1,
        forward_batch=forward_batch,
    )
    assert "resume" not in json_text(tmp_path / "epoch_1" / "epoch_sidecar.json")
    publish_resume_state(
        tmp_path,
        run_id="A1",
        completed_epoch=1,
        training_fingerprint=fingerprint,
        model_generation_sha256=sidecar["model_generation_sha256"],
        optimizer=optimizer,
        scheduler=scheduler,
    )
    update_latest_checkpoint(
        tmp_path,
        epoch=1,
        model_generation_sha256=sidecar["model_generation_sha256"],
        resume_state_present=True,
    )
    assert load_resume_state(
        tmp_path,
        expected_fingerprint=fingerprint,
        optimizer=optimizer,
        scheduler=scheduler,
    )[0] == 1
    latest = read_json(tmp_path / "latest_checkpoint.json")
    assert latest["resume_state_present"] is True

    second = publish_epoch_generation(
        tmp_path,
        run_id="A1",
        epoch=2,
        model=model,
        tokenizer=tokenizer,
        training_fingerprint=fingerprint,
        fingerprint_components={"fixture": True},
        validation={"ndcg_at_10": 0.6},
        optimizer_steps=2,
        last_window_query_count=1,
        throughput={"pairs_per_second": 1.0},
        peak_gpu_memory_bytes=0,
        peak_rss=1,
        forward_batch=forward_batch,
    )
    # A generation-only interruption leaves the previous complete pointer valid.
    assert read_json(tmp_path / "latest_checkpoint.json")["epoch"] == 1
    publish_resume_state(
        tmp_path,
        run_id="A1",
        completed_epoch=2,
        training_fingerprint=fingerprint,
        model_generation_sha256=second["model_generation_sha256"],
        optimizer=optimizer,
        scheduler=scheduler,
    )
    # Replacing the singleton resume directory first invalidates the old pointer;
    # no JSON pointer is safer than one that refers to a mixed epoch boundary.
    assert not (tmp_path / "latest_checkpoint.json").exists()
    resume_sidecar_path = tmp_path / "resume_state/resume_sidecar.json"
    incompatible = read_json(resume_sidecar_path)
    incompatible["model_generation_sha256"] = "0" * 64
    atomic_write_json(resume_sidecar_path, incompatible)
    with pytest.raises(ValueError, match="model_generation_sha256 mismatch"):
        load_resume_state(
            tmp_path,
            expected_fingerprint=fingerprint,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    publish_resume_state(
        tmp_path,
        run_id="A1",
        completed_epoch=2,
        training_fingerprint=fingerprint,
        model_generation_sha256=second["model_generation_sha256"],
        optimizer=optimizer,
        scheduler=scheduler,
    )
    update_latest_checkpoint(
        tmp_path,
        epoch=2,
        model_generation_sha256=second["model_generation_sha256"],
        resume_state_present=True,
    )
    assert load_resume_state(
        tmp_path,
        expected_fingerprint=fingerprint,
        optimizer=optimizer,
        scheduler=scheduler,
    )[0] == 2
    finalized = finalize_run(
        tmp_path,
        {"run_id": "A1", "finalized": False, "resume_available": True},
        final_epoch=2,
        model_generation_sha256=second["model_generation_sha256"],
    )
    assert finalized["finalized"] is True
    assert not (tmp_path / "resume_state").exists()
    assert read_json(tmp_path / "latest_checkpoint.json")["resume_state_present"] is False
    assert validate_epoch_generation(tmp_path / "epoch_1")["epoch"] == 1
    assert validate_epoch_generation(tmp_path / "epoch_2")["epoch"] == 2


def test_epoch_generation_rejects_an_unhashed_extra_file(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        num_labels=1,
    )
    model = transformers.BertForSequenceClassification(config)
    batch = {
        "input_ids": torch.ones((8, 4), dtype=torch.long),
        "attention_mask": torch.ones((8, 4), dtype=torch.long),
    }
    publish_epoch_generation(
        tmp_path,
        run_id="A1",
        epoch=1,
        model=model,
        tokenizer=SaveOnlyTokenizer(),
        training_fingerprint="8" * 64,
        fingerprint_components={"fixture": True},
        validation={"ndcg_at_10": 0.1},
        optimizer_steps=1,
        last_window_query_count=1,
        throughput={"pairs_per_second": 1.0},
        peak_gpu_memory_bytes=0,
        peak_rss=1,
        forward_batch=batch,
    )
    (tmp_path / "epoch_1/unhashed.bin").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="allowlist mismatch"):
        validate_epoch_generation(tmp_path / "epoch_1")
    recovered = publish_epoch_generation(
        tmp_path,
        run_id="A1",
        epoch=1,
        model=model,
        tokenizer=SaveOnlyTokenizer(),
        training_fingerprint="8" * 64,
        fingerprint_components={"fixture": True},
        validation={"ndcg_at_10": 0.1},
        optimizer_steps=1,
        last_window_query_count=1,
        throughput={"pairs_per_second": 1.0},
        peak_gpu_memory_bytes=0,
        peak_rss=1,
        forward_batch=batch,
    )
    assert recovered["epoch"] == 1
    stale = list(tmp_path.glob("epoch_1.stale.*"))
    assert len(stale) == 1
    assert (stale[0] / "unhashed.bin").read_bytes() == b"unexpected"


class PaddingTokenizer:
    def pad(self, rows, *, padding, return_tensors):  # type: ignore[no-untyped-def]
        assert padding is True and return_tensors == "pt"
        maximum = max(len(row["input_ids"]) for row in rows)
        ids = [row["input_ids"] + [0] * (maximum - len(row["input_ids"])) for row in rows]
        masks = [
            row["attention_mask"] + [0] * (maximum - len(row["attention_mask"]))
            for row in rows
        ]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class OneForwardModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.batch_sizes: list[int] = []

    def forward(self, input_ids, attention_mask):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.batch_sizes.append(int(input_ids.shape[0]))
        logits = (input_ids * attention_mask).sum(dim=1, keepdim=True).float()
        return type("Output", (), {"logits": logits})()


def test_query_group_scores_each_unique_document_once_and_never_splits() -> None:
    keys = [("q", "d1"), ("q", "d2"), ("q", "d3")]
    cache = TokenCache(
        ids_buffer=np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int32),
        offsets=np.asarray([0, 2, 4, 6], dtype=np.int64),
        lookup={key: index for index, key in enumerate(keys)},
        tokens_before=np.asarray([2, 2, 2], dtype=np.int32),
        tokens_after=np.asarray([2, 2, 2], dtype=np.int32),
        truncated=np.asarray([False, False, False]),
    )
    group = pd.DataFrame(
        {
            "query_id": ["q", "q"],
            "positive_docid": ["d1", "d1"],
            "negative_docid": ["d2", "d3"],
            "pair_weight": [1.0, 1.0],
        }
    )
    model = OneForwardModel()
    scores, indices = score_query_group_once(
        model,
        PaddingTokenizer(),
        cache,
        group,
        device="cpu",
        max_sequences_per_microbatch=3,
    )
    assert model.calls == 1
    assert model.batch_sizes == [3]
    assert len(scores) == 3 and set(indices) == {"d1", "d2", "d3"}
    with pytest.raises(StageError, match="splitting is forbidden"):
        score_query_group_once(
            model,
            PaddingTokenizer(),
            cache,
            group,
            device="cpu",
            max_sequences_per_microbatch=2,
        )
    assert model.calls == 1


def json_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")
