"""Run and validate one phase of a configured Trust-Elo campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from trust_elo.experiment import (
    REPO_ROOT,
    ExperimentConfig,
    RunSpec,
    file_sha256,
    load_experiment,
)
from utils.local_vllm_server import local_vllm_server, local_vllm_server_pool


MAIN = Path(__file__).resolve().with_name("main.py")
GENERATION_CODE_FILES = (
    Path(__file__).resolve(),
    Path(__file__).resolve().with_name("experiment.py"),
    MAIN,
    Path(__file__).resolve().parents[1] / "utils/run_id.py",
    Path(__file__).resolve().parents[1] / "utils/steering.py",
    Path(__file__).resolve().parents[1] / "utils/steering_policy.py",
    Path(__file__).resolve().parents[1] / "utils/vllm_common.py",
    Path(__file__).resolve().parents[1] / "utils/vllm_lens_wrapper.py",
)
GRADING_CODE_FILES = (
    Path(__file__).resolve(),
    Path(__file__).resolve().with_name("experiment.py"),
    MAIN,
    MAIN.with_name("judging.py"),
    Path(__file__).resolve().parents[1] / "utils/local_vllm_server.py",
    Path(__file__).resolve().parents[1] / "utils/run_id.py",
)
GRADING_IDENTITY_FIELDS = (
    "experiment_config_sha256",
    "code_sha256",
    "expected_judge_model_type",
    "judge_config_sha256",
    "source_git_commit",
    "runtime_versions",
    "judge_devices",
    "judge_gpu_memory_utilization",
    "judge_max_model_len",
    "reasoning_parser",
)
RUNTIME_PACKAGES = ("litellm", "torch", "transformers", "vllm", "vllm-lens")


def _json_lines(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _dataset_rows(run: RunSpec, experiment: ExperimentConfig) -> list[dict[str, Any]]:
    dataset = load_from_disk(str(run.data_dir / "contrastive"))
    if experiment.generation.split not in dataset:
        raise ValueError(
            f"{run.data_dir}/contrastive has no {experiment.generation.split!r} split"
        )
    return list(dataset[experiment.generation.split])


def validate_model_checkpoint(model_path: Path, expected_model_type: str) -> str:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"model checkpoint has no config.json: {model_path}")
    config = json.loads(config_path.read_text())
    actual_model_type = config.get("model_type")
    if actual_model_type != expected_model_type:
        raise ValueError(
            f"model checkpoint {model_path} has model_type={actual_model_type!r}, "
            f"expected {expected_model_type!r}"
        )
    return file_sha256(config_path)


def validate_generations(
    path: Path,
    run: RunSpec,
    experiment: ExperimentConfig,
    *,
    allow_missing_target: bool = False,
) -> dict[str, Any]:
    rows = _json_lines(path)
    expected = _dataset_rows(run, experiment)
    if len(rows) != run.expected_rows or len(expected) != run.expected_rows:
        raise ValueError(
            f"{run.run_id} expected {run.expected_rows} rows, got "
            f"generations={len(rows)} dataset={len(expected)}"
        )

    ids = [str(row.get("id")) for row in rows]
    expected_ids = [str(row.get("id")) for row in expected]
    if len(set(ids)) != len(ids) or ids != expected_ids:
        raise ValueError(f"{run.run_id} generation IDs do not match the dataset")

    expected_strengths = {f"{value:g}" for value in experiment.generation.strengths}
    empty_draws = {strength: 0 for strength in expected_strengths}
    for index, (row, source) in enumerate(zip(rows, expected)):
        if row.get("context") != source.get("context"):
            raise ValueError(f"{run.run_id} row {index} context differs from dataset")
        if row.get("reference_trust") != source["messages_trust"][-1]["content"]:
            raise ValueError(f"{run.run_id} row {index} trust pole differs")
        if row.get("reference_distrust") != source["messages_distrust"][-1]["content"]:
            raise ValueError(f"{run.run_id} row {index} distrust pole differs")
        if row.get("vector_run_id") != run.run_id:
            raise ValueError(
                f"{run.run_id} row {index} records vector_run_id="
                f"{row.get('vector_run_id')!r}"
            )
        target = row.get("steering_target")
        if target is None and allow_missing_target:
            pass
        elif target != run.steering_target:
            raise ValueError(
                f"{run.run_id} row {index} records steering_target={target!r}"
            )
        generations = row.get("generations")
        if not isinstance(generations, dict) or set(generations) != expected_strengths:
            raise ValueError(f"{run.run_id} row {index} has wrong strength keys")
        for strength, draws in generations.items():
            if not isinstance(draws, list) or len(draws) != experiment.generation.draws:
                raise ValueError(
                    f"{run.run_id} row {index} strength {strength} has wrong draw count"
                )
            if not all(isinstance(draw, str) for draw in draws):
                raise ValueError(
                    f"{run.run_id} row {index} strength {strength} has a non-string draw"
                )
            empty_draws[strength] += sum(not draw.strip() for draw in draws)
        recorded_vector = row.get("vector")
        if recorded_vector and Path(str(recorded_vector)).name != run.vector.name:
            raise ValueError(
                f"{run.run_id} row {index} records vector {recorded_vector!r}"
            )
    return {
        "rows": len(rows),
        "strengths": sorted(expected_strengths),
        "draws": experiment.generation.draws,
        "empty_draws": empty_draws,
        "sha256": file_sha256(path),
    }


def validate_grading(
    directory: Path,
    run: RunSpec,
    *,
    expected_rows: int,
    generation_sha256: str,
    allowed_failures: int = 0,
) -> dict[str, Any]:
    results_path = directory / "results.json"
    scores_path = directory / "scores.jsonl"
    verdicts_path = directory / "verdicts.jsonl"
    for path in (results_path, scores_path, verdicts_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing grading artifact: {path}")
    results = json.loads(results_path.read_text())
    run_metadata = results.get("run") or {}
    if run_metadata.get("vector_run_id") != run.run_id:
        raise ValueError(f"grading result has wrong run ID: {run_metadata}")
    if int(run_metadata.get("n_rows", -1)) != expected_rows:
        raise ValueError(
            f"{run.run_id} grading expected {expected_rows} rows, got "
            f"{run_metadata.get('n_rows')}"
        )
    grading = ((results.get("trust_score") or {}).get("grading") or {})
    graded = int(grading.get("graded", -1))
    requested = int(grading.get("requested", -1))
    failed = int(grading.get("failed", -1))
    if min(graded, requested, failed) < 0 or graded + failed != requested:
        raise ValueError(f"{run.run_id} has inconsistent solo grading: {grading}")
    if failed > allowed_failures:
        raise ValueError(
            f"{run.run_id} has {failed} failed solo judge calls "
            f"(allowed {allowed_failures})"
        )
    score_rows = _json_lines(scores_path)
    verdict_rows = _json_lines(verdicts_path)
    if len(score_rows) != graded:
        raise ValueError(
            f"{run.run_id} has {len(score_rows)} score rows for {graded} grades"
        )
    games_in_fit = int(results.get("games_in_fit", -1))
    judge_failures = int(results.get("judge_failures", -1))
    if games_in_fit < 1 or judge_failures < 0 or len(verdict_rows) != games_in_fit:
        raise ValueError(
            f"{run.run_id} has inconsistent pairwise outputs: "
            f"verdicts={len(verdict_rows)} games={games_in_fit} "
            f"failures={judge_failures}"
        )
    if judge_failures > allowed_failures:
        raise ValueError(
            f"{run.run_id} has {judge_failures} failed pairwise judge calls "
            f"(allowed {allowed_failures})"
        )
    expected_conditions = {
        "dataset_distrust_pole",
        "baseline",
        "steer-2",
        "steer-1",
        "steer+1",
        "steer+2",
        "dataset_trust_pole",
    }
    score_table = ((results.get("trust_score") or {}).get("by_condition") or {})
    score_conditions = set(score_table)
    if score_conditions != expected_conditions:
        raise ValueError(f"{run.run_id} grading result has wrong score conditions")
    elo_conditions = set(results.get("elo") or {})
    unexpected_elo = elo_conditions - expected_conditions
    if unexpected_elo:
        raise ValueError(
            f"{run.run_id} grading result has unexpected Elo conditions: "
            f"{sorted(unexpected_elo)}"
        )
    for condition in expected_conditions - elo_conditions:
        cell = score_table[condition]
        coherent = cell.get("n_coherent", cell.get("n_responses"))
        if coherent != 0:
            raise ValueError(
                f"{run.run_id} omitted playable Elo condition {condition}: {cell}"
            )
    return {
        "rows": expected_rows,
        "generation_sha256": generation_sha256,
        "scores_sha256": file_sha256(scores_path),
        "verdicts_sha256": file_sha256(verdicts_path),
        "results_sha256": file_sha256(results_path),
        "grading_requested": requested,
        "grading_failed": failed,
        "judge_failures": judge_failures,
    }


def _code_sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _runtime_versions() -> dict[str, str]:
    """The installed versions of the packages a phase's outputs depend on."""
    return {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}


def _git_commit() -> str:
    """The checked-out commit, refusing a checkout with uncommitted changes.

    Only a phase's own code files are hashed, so the commit is what records
    the rest of the code it runs; uncommitted tracked changes would make it
    describe code that did not run.
    """
    changes = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    if changes.strip():
        raise ValueError("commit tracked changes before running a campaign phase")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _manifest_base(
    experiment: ExperimentConfig,
    run: RunSpec,
    phase: str,
    command: list[str],
    source_git_commit: str,
) -> dict[str, Any]:
    return {
        "campaign": experiment.name,
        "experiment_config": str(experiment.path),
        "experiment_config_sha256": experiment.sha256,
        "run_number": run.number,
        "run_id": run.run_id,
        "phase": phase,
        "model": run.model,
        "model_key": run.model_key,
        "vector": str(run.vector),
        "vector_sha256": file_sha256(run.vector),
        "data_dir": str(run.data_dir),
        "steering_target": run.steering_target,
        "git_commit": source_git_commit,
        "command": command,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def _validated_final(directory: Path, fingerprint: dict[str, Any]) -> bool:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    actual = {key: manifest.get(key) for key in fingerprint}
    if actual != fingerprint:
        raise ValueError(
            f"refusing to overwrite {directory}: existing fingerprint {actual} "
            f"does not match {fingerprint}"
        )
    return True


def _validate_recorded_outputs(
    directory: Path, validation: dict[str, Any]
) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    recorded = manifest.get("validation")
    if recorded != validation:
        raise ValueError(
            f"{directory} outputs no longer match their manifest: "
            f"recorded={recorded} actual={validation}"
        )


def _attempt(parent: Path, phase: str) -> Path:
    attempts = parent / ".attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f"{phase}-", dir=attempts))
    # mkdtemp creates the directory private to its owner, but a published
    # attempt is an ordinary output directory.
    attempt.chmod(0o755)
    return attempt


def _publish(attempt: Path, final: Path) -> None:
    try:
        os.replace(attempt, final)
        return
    except OSError:
        if not final.is_dir():
            raise
    attempt_manifest = json.loads((attempt / "manifest.json").read_text())
    final_manifest = json.loads((final / "manifest.json").read_text())
    stable_fields = (
        "campaign",
        "experiment_config_sha256",
        "run_id",
        "phase",
        "vector_sha256",
        "generation_sha256",
        "code_sha256",
        "expected_model_type",
        "model_config_sha256",
        "model_path",
        "tensor_parallel_size",
        "expected_judge_model_type",
        "judge_config_sha256",
        "judge_model_path",
        "judge_devices",
        "judge_gpu_memory_utilization",
        "judge_max_model_len",
        "reasoning_parser",
        "runtime_versions",
        "source_git_commit",
        "validation",
    )
    attempt_identity = {key: attempt_manifest.get(key) for key in stable_fields}
    final_identity = {key: final_manifest.get(key) for key in stable_fields}
    if attempt_identity != final_identity:
        raise FileExistsError(
            f"a conflicting concurrent job published {final}: {final_identity}"
        )
    shutil.rmtree(attempt)
    print(f"Equivalent concurrent output already published: {final}")


def require_calibration_approval(
    experiment: ExperimentConfig,
    output_root: Path,
    grading_identity: dict[str, Any],
) -> dict[str, Any]:
    directory = (
        output_root
        / "runs"
        / experiment.calibration_run_id
        / "calibration"
    )
    manifest_path = directory / "manifest.json"
    approval_path = output_root / "approvals" / f"{experiment.calibration_run_id}.json"
    results_path = directory / "results.json"
    if (
        not approval_path.is_file()
        or not results_path.is_file()
        or not manifest_path.is_file()
    ):
        raise ValueError(
            "full grading requires an approved calibration; run calibration, "
            "review it, then use python -m trust_elo.approve_calibration"
        )
    approval = json.loads(approval_path.read_text())
    expected = {
        "campaign": experiment.name,
        "experiment_config_sha256": experiment.sha256,
        "run_id": experiment.calibration_run_id,
        "results_sha256": file_sha256(results_path),
        "calibration_manifest_sha256": file_sha256(manifest_path),
        "approved": True,
        "grading_identity": grading_identity,
    }
    actual = {key: approval.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"calibration approval does not match results: {actual}")
    return approval


def _run(command: list[str], *, cwd: Path) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_generation(
    experiment: ExperimentConfig,
    run: RunSpec,
    *,
    model_path: str,
    expected_model_type: str,
    source_git_commit: str,
    runtime_versions: dict[str, str],
    tensor_parallel_size: int,
    output_root: Path,
) -> Path:
    if not run.needs_generation:
        raise ValueError(f"{run.run_id} is configured to reuse existing generations")
    model_path_value = Path(model_path)
    model_config_sha = validate_model_checkpoint(
        model_path_value, expected_model_type
    )
    run_root = output_root / "runs" / run.run_id
    final = run_root / "generation"
    fingerprint = {
        "campaign": experiment.name,
        "experiment_config_sha256": experiment.sha256,
        "run_id": run.run_id,
        "phase": "generation",
        "vector_sha256": file_sha256(run.vector),
        "code_sha256": _code_sha256(GENERATION_CODE_FILES),
        "model_path": model_path,
        "expected_model_type": expected_model_type,
        "model_config_sha256": model_config_sha,
        "source_git_commit": source_git_commit,
        "runtime_versions": runtime_versions,
        "tensor_parallel_size": tensor_parallel_size,
    }
    if _validated_final(final, fingerprint):
        validation = validate_generations(
            final / "generations.jsonl", run, experiment
        )
        _validate_recorded_outputs(final, validation)
        print(f"Generation already complete: {final}")
        return final

    attempt = _attempt(run_root, "generation")
    output = attempt / "generations.jsonl"
    protocol = experiment.generation
    command = [
        sys.executable,
        str(MAIN),
        "--vector",
        str(run.vector),
        "--model",
        model_path,
        "--data_dir",
        str(run.data_dir),
        "--split",
        protocol.split,
        "--strengths",
        *(f"{value:g}" for value in protocol.strengths),
        "--steering-target",
        run.steering_target,
        "--max_new_tokens",
        str(protocol.max_new_tokens),
        "--num_generations",
        str(protocol.draws),
        "--temperature",
        str(protocol.temperature),
        "--seed",
        str(protocol.seed),
        "--tensor_parallel_size",
        str(tensor_parallel_size),
        "--gpu_memory_utilization",
        str(protocol.gpu_memory_utilization),
        "--max_model_len",
        str(protocol.max_model_len),
        "--out",
        str(output),
    ]
    if protocol.think:
        command.append("--think")
    _run(command, cwd=MAIN.parent)
    validation = validate_generations(output, run, experiment)
    manifest = {
        **_manifest_base(
            experiment, run, "generation", command, source_git_commit
        ),
        **fingerprint,
        "validation": validation,
    }
    (attempt / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _publish(attempt, final)
    return final


def run_full_node_generation(
    experiment: ExperimentConfig,
    run: RunSpec,
    *,
    model_path: str,
    expected_model_type: str,
    source_git_commit: str,
    runtime_versions: dict[str, str],
    sampler_replicas: int,
    output_root: Path,
) -> Path:
    """Generate one dataset shard per GPU, then merge and validate atomically."""
    if not run.needs_generation:
        raise ValueError(f"{run.run_id} is configured to reuse existing generations")
    if sampler_replicas < 2:
        raise ValueError("sampler_replicas must be at least 2")
    model_path_value = Path(model_path)
    model_config_sha = validate_model_checkpoint(
        model_path_value, expected_model_type
    )
    dataset_rows = _dataset_rows(run, experiment)
    if sampler_replicas > len(dataset_rows):
        raise ValueError("sampler_replicas cannot exceed dataset rows")

    run_root = output_root / "runs" / run.run_id
    final = run_root / "generation"
    fingerprint = {
        "campaign": experiment.name,
        "experiment_config_sha256": experiment.sha256,
        "run_id": run.run_id,
        "phase": "generation",
        "vector_sha256": file_sha256(run.vector),
        "code_sha256": _code_sha256(GENERATION_CODE_FILES),
        "model_path": model_path,
        "expected_model_type": expected_model_type,
        "model_config_sha256": model_config_sha,
        "source_git_commit": source_git_commit,
        "runtime_versions": runtime_versions,
        "tensor_parallel_size": 1,
        "sampler_replicas": sampler_replicas,
    }
    if _validated_final(final, fingerprint):
        validation = validate_generations(
            final / "generations.jsonl", run, experiment
        )
        _validate_recorded_outputs(final, validation)
        print(f"Full-node generation already complete: {final}")
        return final

    attempt = _attempt(run_root, "generation")
    protocol = experiment.generation
    base_size, remainder = divmod(len(dataset_rows), sampler_replicas)
    processes: list[tuple[int, Path, Any, Any]] = []
    commands: list[list[str]] = []
    offset = 0
    for replica in range(sampler_replicas):
        count = base_size + (1 if replica < remainder else 0)
        shard_path = attempt / f"shard-{replica:02d}.jsonl"
        command = [
            sys.executable,
            str(MAIN),
            "--vector",
            str(run.vector),
            "--model",
            model_path,
            "--data_dir",
            str(run.data_dir),
            "--split",
            protocol.split,
            "--strengths",
            *(f"{value:g}" for value in protocol.strengths),
            "--steering-target",
            run.steering_target,
            "--max_new_tokens",
            str(protocol.max_new_tokens),
            "--num_generations",
            str(protocol.draws),
            "--temperature",
            str(protocol.temperature),
            "--seed",
            str(protocol.seed + offset * len(protocol.strengths) * protocol.draws),
            "--sample_offset",
            str(offset),
            "--num_samples",
            str(count),
            "--tensor_parallel_size",
            "1",
            "--gpu_memory_utilization",
            str(protocol.gpu_memory_utilization),
            "--max_model_len",
            str(protocol.max_model_len),
            "--out",
            str(shard_path),
        ]
        if protocol.think:
            command.append("--think")
        log_path = attempt / f"shard-{replica:02d}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(replica)
        process = subprocess.Popen(
            command,
            cwd=MAIN.parent,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((replica, shard_path, process, log_handle))
        commands.append(command)
        offset += count

    failures = []
    for replica, _shard_path, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            failures.append(f"replica {replica}: exit {return_code}")
    if failures:
        raise RuntimeError("full-node generation failed: " + ", ".join(failures))

    output = attempt / "generations.jsonl"
    with output.open("w", encoding="utf-8") as destination:
        for _replica, shard_path, _process, _log_handle in processes:
            with shard_path.open(encoding="utf-8") as source:
                shutil.copyfileobj(source, destination)
    validation = validate_generations(output, run, experiment)
    manifest = {
        **_manifest_base(
            experiment,
            run,
            "generation",
            ["full-node", *[" ".join(command) for command in commands]],
            source_git_commit,
        ),
        **fingerprint,
        "shards": [
            {
                "replica": replica,
                "path": shard_path.name,
                "sha256": file_sha256(shard_path),
            }
            for replica, shard_path, _process, _log_handle in processes
        ],
        "validation": validation,
    }
    (attempt / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _publish(attempt, final)
    return final


def materialize_generations(
    experiment: ExperimentConfig,
    run: RunSpec,
    *,
    output_root: Path,
    source_git_commit: str,
) -> Path:
    run_root = output_root / "runs" / run.run_id
    final = run_root / "generation"
    vector_sha = file_sha256(run.vector)
    fingerprint = {
        "campaign": experiment.name,
        "experiment_config_sha256": experiment.sha256,
        "run_id": run.run_id,
        "phase": "generation",
        "vector_sha256": vector_sha,
    }
    if final.is_dir():
        if not _validated_final(final, fingerprint):
            raise ValueError(f"incomplete generation directory: {final}")
        validation = validate_generations(
            final / "generations.jsonl",
            run,
            experiment,
            allow_missing_target=not run.needs_generation,
        )
        _validate_recorded_outputs(final, validation)
        return final
    if run.needs_generation:
        raise FileNotFoundError(f"generation has not completed: {final}")
    assert run.source_generations is not None
    attempt = _attempt(run_root, "generation")
    destination = attempt / "generations.jsonl"
    shutil.copyfile(run.source_generations, destination)
    validation = validate_generations(
        destination, run, experiment, allow_missing_target=True
    )
    command = ["copy", str(run.source_generations), str(destination)]
    manifest = {
        **_manifest_base(
            experiment, run, "generation", command, source_git_commit
        ),
        **fingerprint,
        "source_generations": str(run.source_generations),
        "source_generations_sha256": file_sha256(run.source_generations),
        "backend_note": (
            "Reused generation predates the vllm-lens backend. The campaign plan "
            "asserts matching sampling protocol; the JSONL does not encode backend, "
            "temperature, seed, thinking, or model revision."
        ),
        "validation": validation,
    }
    (attempt / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _publish(attempt, final)
    return final


def adopt_generation(
    experiment: ExperimentConfig,
    run: RunSpec,
    *,
    source_output_root: Path,
    output_root: Path,
    source_git_commit: str,
) -> Path:
    """Copy a validated generation into a new immutable judging campaign."""
    source = source_output_root / "runs" / run.run_id / "generation"
    source_generations = source / "generations.jsonl"
    source_manifest = source / "manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"source generation manifest not found: {source_manifest}")
    allow_missing_target = not run.needs_generation
    source_validation = validate_generations(
        source_generations,
        run,
        experiment,
        allow_missing_target=allow_missing_target,
    )

    run_root = output_root / "runs" / run.run_id
    final = run_root / "generation"
    fingerprint = {
        "campaign": experiment.name,
        "experiment_config_sha256": experiment.sha256,
        "run_id": run.run_id,
        "phase": "generation",
        "vector_sha256": file_sha256(run.vector),
        "source_generation_sha256": file_sha256(source_generations),
        "source_generation_manifest_sha256": file_sha256(source_manifest),
    }
    if _validated_final(final, fingerprint):
        validation = validate_generations(
            final / "generations.jsonl",
            run,
            experiment,
            allow_missing_target=allow_missing_target,
        )
        _validate_recorded_outputs(final, validation)
        print(f"Adopted generation already complete: {final}")
        return final

    attempt = _attempt(run_root, "generation")
    destination = attempt / "generations.jsonl"
    shutil.copyfile(source_generations, destination)
    validation = validate_generations(
        destination,
        run,
        experiment,
        allow_missing_target=allow_missing_target,
    )
    if validation != source_validation:
        raise ValueError(
            f"adopted generation changed during copy: {source_generations}"
        )
    command = ["adopt-generation", str(source_generations), str(destination)]
    manifest = {
        **_manifest_base(
            experiment, run, "generation", command, source_git_commit
        ),
        **fingerprint,
        "source_campaign": json.loads(source_manifest.read_text()).get("campaign"),
        "source_generation": str(source_generations),
        "source_generation_manifest": str(source_manifest),
        "validation": validation,
    }
    (attempt / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _publish(attempt, final)
    return final


def run_grading(
    experiment: ExperimentConfig,
    run: RunSpec,
    *,
    judge_model_path: Path,
    expected_judge_model_type: str,
    source_git_commit: str,
    runtime_versions: dict[str, str],
    judge_devices: list[str],
    output_root: Path,
    calibration: bool,
    judge_port: int,
    startup_timeout: int,
    judge_gpu_memory_utilization: float,
    judge_max_model_len: int,
    reasoning_parser: str,
) -> Path:
    judge_config_sha = validate_model_checkpoint(
        judge_model_path, expected_judge_model_type
    )
    generation = materialize_generations(
        experiment,
        run,
        output_root=output_root,
        source_git_commit=source_git_commit,
    )
    generation_path = generation / "generations.jsonl"
    generation_sha = file_sha256(generation_path)
    phase = "calibration" if calibration else "grading"
    run_root = output_root / "runs" / run.run_id
    final = run_root / phase
    fingerprint = {
        "campaign": experiment.name,
        "experiment_config_sha256": experiment.sha256,
        "run_id": run.run_id,
        "phase": phase,
        "vector_sha256": file_sha256(run.vector),
        "generation_sha256": generation_sha,
        "code_sha256": _code_sha256(GRADING_CODE_FILES),
        "judge_model_path": str(judge_model_path),
        "expected_judge_model_type": expected_judge_model_type,
        "judge_config_sha256": judge_config_sha,
        "source_git_commit": source_git_commit,
        "runtime_versions": runtime_versions,
        "judge_devices": judge_devices,
        "judge_gpu_memory_utilization": judge_gpu_memory_utilization,
        "judge_max_model_len": judge_max_model_len,
        "reasoning_parser": reasoning_parser,
    }
    grading_identity = {
        key: fingerprint.get(key) for key in GRADING_IDENTITY_FIELDS
    }
    if not calibration:
        require_calibration_approval(experiment, output_root, grading_identity)
    expected_rows = (
        min(experiment.judging.calibration_rows, run.expected_rows)
        if calibration
        else run.expected_rows
    )
    if _validated_final(final, fingerprint):
        validation = validate_grading(
            final, run, expected_rows=expected_rows, generation_sha256=generation_sha
        )
        _validate_recorded_outputs(final, validation)
        print(f"Grading already complete: {final}")
        return final

    attempt = _attempt(run_root, phase)
    protocol = experiment.judging
    command = [
        sys.executable,
        str(MAIN),
        "--from",
        str(generation_path),
        "--judge_model",
        protocol.model,
        "--max_tokens",
        str(protocol.max_tokens),
        "--request_timeout",
        str(protocol.request_timeout),
        "--concurrency",
        str(protocol.concurrency),
        "--bootstrap",
        str(protocol.bootstrap),
        "--prior",
        str(protocol.prior),
        "--score_repeats",
        str(protocol.score_repeats),
        "--score_temperature",
        str(protocol.score_temperature),
        "--verdicts",
        str(attempt / "verdicts.jsonl"),
        "--scores_out",
        str(attempt / "scores.jsonl"),
        "--results",
        str(attempt / "results.json"),
    ]
    command.append("--anchors" if protocol.anchors else "--no-anchors")
    command.append(
        "--require-coherent" if protocol.require_coherent else "--no-require-coherent"
    )
    if calibration:
        command.extend(["--num_samples", str(protocol.calibration_rows)])

    server_log = attempt / "judge-server.log"
    with local_vllm_server(
        model_path=judge_model_path,
        served_model_name=protocol.served_model_name,
        devices=judge_devices,
        log_path=server_log,
        port=judge_port,
        startup_timeout=startup_timeout,
        gpu_memory_utilization=judge_gpu_memory_utilization,
        max_model_len=judge_max_model_len,
        reasoning_parser=reasoning_parser,
    ) as api_base:
        command.extend(["--api_base", api_base])
        _run(command, cwd=MAIN.parent)
    validation = validate_grading(
        attempt, run, expected_rows=expected_rows, generation_sha256=generation_sha
    )
    manifest = {
        **_manifest_base(experiment, run, phase, command, source_git_commit),
        **fingerprint,
        "judge_model_path": str(judge_model_path),
        "judge_devices": judge_devices,
        "validation": validation,
    }
    (attempt / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _publish(attempt, final)
    return final


def run_full_node_grading(
    experiment: ExperimentConfig,
    run: RunSpec,
    *,
    judge_model_path: Path,
    expected_judge_model_type: str,
    source_git_commit: str,
    runtime_versions: dict[str, str],
    judge_devices: list[str],
    output_root: Path,
    judge_port: int,
    startup_timeout: int,
    judge_gpu_memory_utilization: float,
    judge_max_model_len: int,
    reasoning_parser: str,
    allowed_failures: int,
    approval_override: str,
) -> Path:
    """Grade one run through independent per-GPU judge replicas."""
    if len(judge_devices) < 2:
        raise ValueError("full-node grading requires at least two judge replicas")
    if allowed_failures < 0:
        raise ValueError("allowed_failures must be non-negative")
    if not approval_override:
        raise ValueError("full-node grading requires an explicit approval override")
    judge_config_sha = validate_model_checkpoint(
        judge_model_path, expected_judge_model_type
    )
    generation = materialize_generations(
        experiment,
        run,
        output_root=output_root,
        source_git_commit=source_git_commit,
    )
    generation_path = generation / "generations.jsonl"
    generation_sha = file_sha256(generation_path)
    run_root = output_root / "runs" / run.run_id
    final = run_root / "grading"
    fingerprint = {
        "campaign": experiment.name,
        "experiment_config_sha256": experiment.sha256,
        "run_id": run.run_id,
        "phase": "grading",
        "vector_sha256": file_sha256(run.vector),
        "generation_sha256": generation_sha,
        "code_sha256": _code_sha256(GRADING_CODE_FILES),
        "judge_model_path": str(judge_model_path),
        "expected_judge_model_type": expected_judge_model_type,
        "judge_config_sha256": judge_config_sha,
        "source_git_commit": source_git_commit,
        "runtime_versions": runtime_versions,
        "judge_devices": judge_devices,
        "judge_replicas": len(judge_devices),
        "judge_gpu_memory_utilization": judge_gpu_memory_utilization,
        "judge_max_model_len": judge_max_model_len,
        "reasoning_parser": reasoning_parser,
        "allowed_failures": allowed_failures,
        "approval_override": approval_override,
    }
    if _validated_final(final, fingerprint):
        validation = validate_grading(
            final,
            run,
            expected_rows=run.expected_rows,
            generation_sha256=generation_sha,
            allowed_failures=allowed_failures,
        )
        _validate_recorded_outputs(final, validation)
        print(f"Full-node grading already complete: {final}")
        return final

    attempt = _attempt(run_root, "grading")
    protocol = experiment.judging
    command = [
        sys.executable,
        str(MAIN),
        "--from",
        str(generation_path),
        "--judge_model",
        protocol.model,
        "--max_tokens",
        str(protocol.max_tokens),
        "--request_timeout",
        str(protocol.request_timeout),
        "--concurrency",
        str(protocol.concurrency * len(judge_devices)),
        "--bootstrap",
        str(protocol.bootstrap),
        "--prior",
        str(protocol.prior),
        "--score_repeats",
        str(protocol.score_repeats),
        "--score_temperature",
        str(protocol.score_temperature),
        "--verdicts",
        str(attempt / "verdicts.jsonl"),
        "--scores_out",
        str(attempt / "scores.jsonl"),
        "--results",
        str(attempt / "results.json"),
    ]
    command.append("--anchors" if protocol.anchors else "--no-anchors")
    command.append(
        "--require-coherent" if protocol.require_coherent else "--no-require-coherent"
    )
    with local_vllm_server_pool(
        model_path=judge_model_path,
        served_model_name=protocol.served_model_name,
        devices=judge_devices,
        log_dir=attempt / "judge-servers",
        base_port=judge_port,
        startup_timeout=startup_timeout,
        gpu_memory_utilization=judge_gpu_memory_utilization,
        max_model_len=judge_max_model_len,
        reasoning_parser=reasoning_parser,
    ) as api_bases:
        command.extend(["--api_base", ",".join(api_bases)])
        _run(command, cwd=MAIN.parent)
    validation = validate_grading(
        attempt,
        run,
        expected_rows=run.expected_rows,
        generation_sha256=generation_sha,
        allowed_failures=allowed_failures,
    )
    manifest = {
        **_manifest_base(experiment, run, "grading", command, source_git_commit),
        **fingerprint,
        "validation": validation,
    }
    (attempt / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _publish(attempt, final)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "generate",
            "generate-full-node",
            "adopt-generation",
            "calibrate",
            "grade",
            "grade-full-node",
        ),
    )
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--expected-model-type")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--sampler-replicas", type=int, default=8)
    parser.add_argument("--allowed-failures", type=int, default=10)
    parser.add_argument("--approval-override")
    parser.add_argument("--source-output-root", type=Path)
    parser.add_argument("--judge-model-path", type=Path)
    parser.add_argument("--expected-judge-model-type")
    parser.add_argument("--judge-devices", nargs="+", default=[])
    parser.add_argument("--judge-port", type=int, default=8001)
    parser.add_argument("--judge-startup-timeout", type=int, default=900)
    parser.add_argument("--judge-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--judge-max-model-len", type=int, default=32768)
    parser.add_argument("--reasoning-parser", default="openai_gptoss")
    return parser


def main() -> None:
    args = _parser().parse_args()
    experiment = load_experiment(args.experiment_config)
    run = experiment.run(args.run_id)
    source_git_commit = _git_commit()
    if args.phase == "adopt-generation":
        if args.source_output_root is None:
            raise ValueError("adopt-generation requires --source-output-root")
        adopt_generation(
            experiment,
            run,
            source_output_root=args.source_output_root,
            output_root=args.output_root,
            source_git_commit=source_git_commit,
        )
        return
    runtime_versions = _runtime_versions()
    if args.phase in ("generate", "generate-full-node"):
        if not args.model_path or not args.expected_model_type:
            raise ValueError("generate requires model path and expected model type")
        if args.phase == "generate-full-node":
            run_full_node_generation(
                experiment,
                run,
                model_path=args.model_path,
                expected_model_type=args.expected_model_type,
                source_git_commit=source_git_commit,
                runtime_versions=runtime_versions,
                sampler_replicas=args.sampler_replicas,
                output_root=args.output_root,
            )
        else:
            run_generation(
                experiment,
                run,
                model_path=args.model_path,
                expected_model_type=args.expected_model_type,
                source_git_commit=source_git_commit,
                runtime_versions=runtime_versions,
                tensor_parallel_size=args.tensor_parallel_size,
                output_root=args.output_root,
            )
        return
    if (
        args.judge_model_path is None
        or not args.expected_judge_model_type
        or not args.judge_devices
    ):
        raise ValueError(
            "grading requires judge model path, expected model type, and devices"
        )
    if args.phase == "calibrate" and run.run_id != experiment.calibration_run_id:
        raise ValueError(f"calibration is fixed to {experiment.calibration_run_id}")
    if args.phase == "grade-full-node":
        run_full_node_grading(
            experiment,
            run,
            judge_model_path=args.judge_model_path,
            expected_judge_model_type=args.expected_judge_model_type,
            source_git_commit=source_git_commit,
            runtime_versions=runtime_versions,
            judge_devices=args.judge_devices,
            output_root=args.output_root,
            judge_port=args.judge_port,
            startup_timeout=args.judge_startup_timeout,
            judge_gpu_memory_utilization=args.judge_gpu_memory_utilization,
            judge_max_model_len=args.judge_max_model_len,
            reasoning_parser=args.reasoning_parser,
            allowed_failures=args.allowed_failures,
            approval_override=args.approval_override or "",
        )
        return
    run_grading(
        experiment,
        run,
        judge_model_path=args.judge_model_path,
        expected_judge_model_type=args.expected_judge_model_type,
        source_git_commit=source_git_commit,
        runtime_versions=runtime_versions,
        judge_devices=args.judge_devices,
        output_root=args.output_root,
        calibration=args.phase == "calibrate",
        judge_port=args.judge_port,
        startup_timeout=args.judge_startup_timeout,
        judge_gpu_memory_utilization=args.judge_gpu_memory_utilization,
        judge_max_model_len=args.judge_max_model_len,
        reasoning_parser=args.reasoning_parser,
    )


if __name__ == "__main__":
    main()
