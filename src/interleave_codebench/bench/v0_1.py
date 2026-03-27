from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from ..utils import ensure_directory, read_json, slugify, write_json


VERIFY_SCRIPT = """from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main() -> int:
    spec_path = Path(".interleave-task/verify_spec.json")
    spec = json.loads(spec_path.read_text())
    target = Path(spec["path"])
    if not target.exists():
        print(f"missing target file: {target}")
        return 1
    text = target.read_text()
    failures: list[str] = []
    if target.suffix == ".py":
        try:
            compile(text, str(target), "exec")
        except SyntaxError as exc:
            failures.append(f"syntax error in {target}: {exc}")
    for snippet in spec.get("must_contain", []):
        if snippet not in text:
            failures.append(f"missing required snippet: {snippet!r}")
    for snippet in spec.get("must_not_contain", []):
        if snippet in text:
            failures.append(f"unexpected snippet still present: {snippet!r}")
    for pattern in spec.get("must_match_regex", []):
        if re.search(pattern, text, flags=re.MULTILINE | re.DOTALL) is None:
            failures.append(f"missing required regex: {pattern}")
    if failures:
        print("verification failed")
        for item in failures:
            print(f"- {item}")
        return 1
    print("verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_manifest_path() -> Path:
    return repo_root() / "benchmark_samples" / "v0_1_realistic.json"


def default_output_root() -> Path:
    return repo_root() / ".cache" / "interleave_codebench" / "v0_1"


def prepare_v0_1_dataset(
    *, manifest_path: str | Path | None = None, output_root: str | Path | None = None
) -> Path:
    manifest_path = Path(manifest_path or default_manifest_path()).resolve()
    output_root = ensure_directory(Path(output_root or default_output_root()).resolve())
    manifest = read_json(manifest_path)
    cache_dir = ensure_directory(output_root / "_repo_cache")
    snapshot_dir = output_root / "snapshots"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    ensure_directory(snapshot_dir)

    swe_bench_records = []
    swe_ci_records = []

    for record in manifest.get("swe_bench_verified", []):
        snapshot_path = _materialize_snapshot(cache_dir, snapshot_dir, record)
        swe_bench_records.append(_build_swe_bench_record(record, snapshot_path))
    for record in manifest.get("swe_ci", []):
        snapshot_path = _materialize_snapshot(cache_dir, snapshot_dir, record)
        swe_ci_records.append(_build_swe_ci_record(record, snapshot_path))

    swe_bench_dir = ensure_directory(output_root / "swe_bench_verified")
    swe_ci_dir = ensure_directory(output_root / "swe_ci")
    write_json(swe_bench_dir / "tasks.json", swe_bench_records)
    _write_csv(swe_ci_dir / "tasks.csv", swe_ci_records)
    write_json(output_root / "manifest.json", manifest)
    return output_root


def _materialize_snapshot(cache_dir: Path, snapshot_root: Path, record: dict[str, Any]) -> Path:
    clone_dir = _ensure_repo_clone(cache_dir, record["repo_id"], record["repo_url"])
    checkout = record["repo_checkout"]
    _ensure_commit_present(clone_dir, checkout)
    snapshot_path = snapshot_root / record["task_id"]
    if snapshot_path.exists():
        shutil.rmtree(snapshot_path)
    ensure_directory(snapshot_path)
    _export_commit(clone_dir, checkout, snapshot_path)
    _inject_verifier(snapshot_path, record["verify_spec"])
    return snapshot_path


def _ensure_repo_clone(cache_dir: Path, repo_id: str, repo_url: str) -> Path:
    clone_dir = cache_dir / slugify(repo_id)
    if not clone_dir.exists():
        source = repo_url
        repo_path = Path(repo_url)
        if repo_path.exists():
            source = repo_path.resolve().as_uri()
        subprocess.run(
            ["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", source, str(clone_dir)],
            check=True,
        )
    return clone_dir


def _ensure_commit_present(clone_dir: Path, commit: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(clone_dir), "cat-file", "-e", f"{commit}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return
    subprocess.run(["git", "-C", str(clone_dir), "fetch", "--quiet", "origin", commit], check=True)


def _export_commit(clone_dir: Path, commit: str, snapshot_path: Path) -> None:
    archive_bytes = subprocess.check_output(
        ["git", "-C", str(clone_dir), "archive", "--format=tar", commit]
    )
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as handle:
        handle.extractall(snapshot_path)


def _inject_verifier(snapshot_path: Path, verify_spec: dict[str, Any]) -> None:
    verify_dir = ensure_directory(snapshot_path / ".interleave-task")
    write_json(verify_dir / "verify_spec.json", verify_spec)
    (snapshot_path / "verify.py").write_text(VERIFY_SCRIPT)


def _build_swe_bench_record(record: dict[str, Any], snapshot_path: Path) -> dict[str, Any]:
    payload = {
        "task_id": record["task_id"],
        "instance_id": record["task_id"],
        "repo": record["repo_id"],
        "repo_checkout": record["repo_checkout"],
        "base_commit": record["repo_checkout"],
        "problem_statement": record["task_prompt"],
        "task_prompt": record["task_prompt"],
        "language": "python",
        "eval_command": "python verify.py",
        "setup_cmds": list(record.get("setup_cmds", [])),
        "success_type": "explicit_finish_and_eval",
        "split": record["split"],
        "repo_source_path": str(snapshot_path),
        "pilot_ready": True,
        "target_file": record["target_file"],
        "target_symbol": record.get("target_symbol", ""),
        "repo_url": record["repo_url"],
        "change_summary": record.get("change_summary", ""),
        "verify_spec_path": ".interleave-task/verify_spec.json",
    }
    if record.get("upstream_task_id"):
        payload["upstream_task_id"] = record["upstream_task_id"]
    return payload


def _build_swe_ci_record(record: dict[str, Any], snapshot_path: Path) -> dict[str, Any]:
    payload = {
        "task_id": record["task_id"],
        "repo_name": record["repo_id"],
        "current_sha": record["repo_checkout"],
        "target_sha": record.get("target_sha", ""),
        "task_prompt": record["task_prompt"],
        "language": "python",
        "eval_command": "python verify.py",
        "setup_cmds": json.dumps(list(record.get("setup_cmds", []))),
        "success_type": "explicit_finish_and_eval",
        "split": record["split"],
        "repo_source_path": str(snapshot_path),
        "pilot_ready": "true",
        "target_file": record["target_file"],
        "target_symbol": record.get("target_symbol", ""),
        "repo_url": record["repo_url"],
        "change_summary": record.get("change_summary", ""),
        "verify_spec_path": ".interleave-task/verify_spec.json",
    }
    if record.get("upstream_task_id"):
        payload["upstream_task_id"] = record["upstream_task_id"]
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task_id",
        "repo_name",
        "current_sha",
        "target_sha",
        "task_prompt",
        "language",
        "eval_command",
        "setup_cmds",
        "success_type",
        "split",
        "repo_source_path",
        "pilot_ready",
        "target_file",
        "target_symbol",
        "repo_url",
        "change_summary",
        "verify_spec_path",
        "upstream_task_id",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(payload)
