#!/usr/bin/env python3
"""Prepare validated Phase 1/2 archives for a private Kaggle Dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile


PHASE_MARKERS = {
    "phase1": "candidate_cache_manifest.json",
    "phase2": "rerank_manifest.json",
}
DESTINATIONS = {
    "phase1": "rusearchrank_phase1_results.zip",
    "phase2": "rusearchrank_phase2_results.zip",
}
DEFAULT_DATASET_ID = "USERNAME/rusearchrank-phase12"
DATASET_TITLE = "RuSearchRank Phase 1 and Phase 2 artifacts"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_archive(path: Path) -> str | None:
    """Return phase1/phase2 based exclusively on manifest members."""
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"CRC failed for {path}: {bad_member}")
            basenames = {PurePosixPath(name).name for name in archive.namelist()}
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid ZIP archive: {path}") from exc
    matches = [
        phase for phase, marker in PHASE_MARKERS.items() if marker in basenames
    ]
    if len(matches) > 1:
        raise ValueError(f"archive contains both Phase 1 and Phase 2 manifests: {path}")
    return matches[0] if matches else None


def find_phase_archives(repository: Path) -> dict[str, Path]:
    if not repository.is_dir():
        raise ValueError(f"repository directory does not exist: {repository}")
    found: dict[str, list[Path]] = {phase: [] for phase in PHASE_MARKERS}
    for path in sorted(repository.rglob("*.zip")):
        phase = classify_archive(path)
        if phase is not None:
            found[phase].append(path.resolve())
    problems = []
    for phase, paths in found.items():
        if len(paths) != 1:
            rendered = ", ".join(map(str, paths)) if paths else "none"
            problems.append(f"expected exactly one {phase} ZIP, found {len(paths)}: {rendered}")
    if problems:
        raise ValueError("; ".join(problems))
    return {phase: paths[0] for phase, paths in found.items()}


def dataset_metadata(dataset_id: str) -> dict[str, object]:
    if dataset_id.count("/") != 1 or any(not part for part in dataset_id.split("/")):
        raise ValueError("Kaggle dataset id must have the form USERNAME/dataset-slug")
    return {
        "title": DATASET_TITLE,
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }


def copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != expected_sha256:
            raise ValueError(f"refusing to overwrite a different output file: {destination}")
        return
    shutil.copy2(source, destination)
    if sha256_file(destination) != expected_sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"copy verification failed: {destination}")


def prepare_transfer(repository: Path, output: Path) -> dict[str, object]:
    archives = find_phase_archives(repository.resolve())
    output.mkdir(parents=True, exist_ok=True)
    file_report: dict[str, dict[str, object]] = {}
    source_before = {
        phase: (path.stat().st_size, sha256_file(path))
        for phase, path in archives.items()
    }
    for phase in ("phase1", "phase2"):
        source = archives[phase]
        destination = output / DESTINATIONS[phase]
        size, digest = source_before[phase]
        copy_verified(source, destination, digest)
        file_report[destination.name] = {
            "size_bytes": size,
            "sha256": digest,
            "source": str(source),
        }
    for phase, source in archives.items():
        if (source.stat().st_size, sha256_file(source)) != source_before[phase]:
            raise RuntimeError(f"source archive changed while copying: {source}")

    checksum_document = {name: report["sha256"] for name, report in file_report.items()}
    (output / "sha256.json").write_text(
        json.dumps(checksum_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "dataset-metadata.json.template").write_text(
        json.dumps(dataset_metadata(DEFAULT_DATASET_ID), indent=2) + "\n",
        encoding="utf-8",
    )
    return {"files": file_report, "sha256": checksum_document}


def run_kaggle(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Kaggle CLI is not installed or is not on PATH") from exc


def publish_private_dataset(output: Path, dataset_id: str) -> str:
    credentials = Path.home() / ".kaggle" / "kaggle.json"
    if not credentials.is_file():
        raise RuntimeError(f"Kaggle credentials are missing at {credentials}")
    metadata_path = output / "dataset-metadata.json"
    metadata_path.write_text(
        json.dumps(dataset_metadata(dataset_id), indent=2) + "\n",
        encoding="utf-8",
    )

    status = run_kaggle(["kaggle", "datasets", "status", dataset_id])
    if status.returncode == 0:
        action = "updated"
        command = [
            "kaggle", "datasets", "version", "-p", str(output),
            "-m", "Validated RuSearchRank Phase 1/2 artifacts",
        ]
    else:
        action = "created"
        # Kaggle Dataset creation is private by default; the CLI exposes only
        # the inverse --public switch, which this helper intentionally omits.
        command = ["kaggle", "datasets", "create", "-p", str(output)]
    result = run_kaggle(command)
    if result.returncode != 0:
        # Kaggle CLI output is intentionally not echoed: credentials and request
        # details must never be copied into logs produced by this helper.
        raise RuntimeError(f"Kaggle CLI {action} command failed with return code {result.returncode}")
    return action


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kaggle-dataset-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = prepare_transfer(args.repository, args.output.resolve())
    for name, details in report["files"].items():
        print(
            json.dumps(
                {
                    "file": name,
                    "size_bytes": details["size_bytes"],
                    "sha256": details["sha256"],
                },
                sort_keys=True,
            )
        )
    if args.kaggle_dataset_id:
        action = publish_private_dataset(args.output.resolve(), args.kaggle_dataset_id)
        print(f"Kaggle Dataset {action}: {args.kaggle_dataset_id}")
        print("Uploaded files:")
        for path in sorted(args.output.iterdir()):
            if path.is_file() and path.name != "dataset-metadata.json":
                print(f"- {path.name}")
    else:
        print(f"Transfer directory ready: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"prepare_kaggle_phase12 failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
