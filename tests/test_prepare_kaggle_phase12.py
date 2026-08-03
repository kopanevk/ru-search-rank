from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts import prepare_kaggle_phase12 as helper


def write_zip(path: Path, *members: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in members:
            archive.writestr(member, b'{"status":"PASS"}\n')


def make_phase_archives(repository: Path) -> tuple[Path, Path]:
    phase1 = repository / "nested" / "not-named-phase1.zip"
    phase2 = repository / "elsewhere" / "not-named-phase2.zip"
    write_zip(phase1, "reports/audit/candidate_cache_manifest.json")
    write_zip(phase2, "reports/audit/rerank_manifest.json")
    return phase1, phase2


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_distinguishes_phase_archives_by_manifest(tmp_path: Path) -> None:
    phase1, phase2 = make_phase_archives(tmp_path)
    assert helper.find_phase_archives(tmp_path) == {
        "phase1": phase1.resolve(),
        "phase2": phase2.resolve(),
    }


def test_rejects_corrupt_zip(tmp_path: Path) -> None:
    make_phase_archives(tmp_path)
    (tmp_path / "corrupt.zip").write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="invalid ZIP"):
        helper.find_phase_archives(tmp_path)


def test_rejects_two_phase1_archives(tmp_path: Path) -> None:
    make_phase_archives(tmp_path)
    write_zip(tmp_path / "duplicate.zip", "candidate_cache_manifest.json")
    with pytest.raises(ValueError, match="exactly one phase1 ZIP, found 2"):
        helper.find_phase_archives(tmp_path)


def test_rejects_missing_phase2_archive(tmp_path: Path) -> None:
    write_zip(tmp_path / "only.zip", "candidate_cache_manifest.json")
    with pytest.raises(ValueError, match="exactly one phase2 ZIP, found 0"):
        helper.find_phase_archives(tmp_path)


def test_prepares_checksums_without_modifying_sources(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    output = tmp_path / "output"
    phase1, phase2 = make_phase_archives(repository)
    before = {phase1: phase1.read_bytes(), phase2: phase2.read_bytes()}

    report = helper.prepare_transfer(repository, output)

    checksums = json.loads((output / "sha256.json").read_text())
    assert checksums == {
        "rusearchrank_phase1_results.zip": file_sha256(phase1),
        "rusearchrank_phase2_results.zip": file_sha256(phase2),
    }
    assert report["sha256"] == checksums
    assert json.loads((output / "dataset-metadata.json.template").read_text()) == {
        "title": helper.DATASET_TITLE,
        "id": "USERNAME/rusearchrank-phase12",
        "licenses": [{"name": "CC0-1.0"}],
    }
    assert (output / "rusearchrank_phase1_results.zip").read_bytes() == before[phase1]
    assert (output / "rusearchrank_phase2_results.zip").read_bytes() == before[phase2]
    assert {path: path.read_bytes() for path in before} == before


def test_credentials_do_not_appear_in_output_or_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repository = tmp_path / "repository"
    output = tmp_path / "output"
    make_phase_archives(repository)
    fake_home = tmp_path / "home"
    credentials = fake_home / ".kaggle" / "kaggle.json"
    credentials.parent.mkdir(parents=True)
    secret = "DO-NOT-LOG-THIS-SECRET"
    credentials.write_text(json.dumps({"username": "private-user", "key": secret}))
    monkeypatch.setattr(helper.Path, "home", classmethod(lambda cls: fake_home))

    calls: list[list[str]] = []

    def fake_run(command: list[str]) -> object:
        calls.append(command)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(helper, "run_kaggle", fake_run)
    assert helper.main(
        [
            "--repository", str(repository),
            "--output", str(output),
            "--kaggle-dataset-id", "private-user/rusearchrank-phase12",
        ]
    ) == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert secret not in combined
    assert secret not in "\n".join(
        path.read_text(errors="ignore")
        for path in output.iterdir()
        if path.is_file() and path.suffix != ".zip"
    )
    assert calls[0][:3] == ["kaggle", "datasets", "status"]
    assert calls[1][:3] == ["kaggle", "datasets", "version"]
