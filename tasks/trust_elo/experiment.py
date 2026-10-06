"""Machine-readable Trust-Elo campaign configuration and validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.steering_policy import training_span_for_target, validate_steering_target


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = re.compile(r"^\d{8}-\d{6}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GenerationProtocol:
    split: str
    strengths: tuple[float, ...]
    draws: int
    temperature: float
    seed: int
    think: bool
    max_new_tokens: int
    max_model_len: int
    gpu_memory_utilization: float


@dataclass(frozen=True)
class JudgingProtocol:
    model: str
    served_model_name: str
    max_tokens: int
    concurrency: int
    bootstrap: int
    prior: float
    anchors: bool
    require_coherent: bool
    score_repeats: int
    score_temperature: float
    calibration_rows: int
    request_timeout: float


@dataclass(frozen=True)
class RunSpec:
    number: int
    run_id: str
    model_key: str
    model: str
    vector: Path
    data_dir: Path
    expected_rows: int
    steering_target: str


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    generation: GenerationProtocol
    judging: JudgingProtocol
    calibration_run_id: str
    runs: tuple[RunSpec, ...]
    path: Path
    sha256: str

    def select(self, run_ids: list[str] | None = None) -> list[RunSpec]:
        requested = set(run_ids or [])
        known = {run.run_id for run in self.runs}
        unknown = requested - known
        if unknown:
            raise ValueError(f"unknown run IDs: {', '.join(sorted(unknown))}")
        selected = [
            run for run in self.runs if not requested or run.run_id in requested
        ]
        if not selected:
            raise ValueError("no runs selected")
        return selected

    def run(self, run_id: str) -> RunSpec:
        return self.select([run_id])[0]


def _dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _path(value: Any, field: str, workspace: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def _training_span(path: Path) -> str | None:
    name = path.name
    if "_Lall_user_" in name:
        return "user"
    if "_Lall_assistant_" in name:
        return "assistant"
    return None


def _generation(raw: Any) -> GenerationProtocol:
    value = _dict(raw, "generation")
    strengths_raw = value.get("strengths")
    if not isinstance(strengths_raw, list) or not strengths_raw:
        raise ValueError("generation.strengths must be a non-empty list")
    strengths = tuple(float(item) for item in strengths_raw)
    if len(set(strengths)) != len(strengths) or 0.0 not in strengths:
        raise ValueError("generation.strengths must be unique and include 0")
    protocol = GenerationProtocol(
        split=str(value.get("split", "test")),
        strengths=strengths,
        draws=int(value.get("draws", 1)),
        temperature=float(value.get("temperature", 0.0)),
        seed=int(value.get("seed", 0)),
        think=bool(value.get("think", False)),
        max_new_tokens=int(value.get("max_new_tokens", 2048)),
        max_model_len=int(value.get("max_model_len", 32768)),
        gpu_memory_utilization=float(value.get("gpu_memory_utilization", 0.9)),
    )
    if protocol.split not in {"train", "test"}:
        raise ValueError("generation.split must be train or test")
    if protocol.draws < 1:
        raise ValueError("generation.draws must be positive")
    if protocol.draws > 1 and protocol.temperature <= 0:
        raise ValueError("multiple generation draws require positive temperature")
    if protocol.max_new_tokens < 1 or protocol.max_model_len < 1:
        raise ValueError("generation token limits must be positive")
    if not 0 < protocol.gpu_memory_utilization <= 1:
        raise ValueError("generation.gpu_memory_utilization must be in (0, 1]")
    return protocol


def _judging(raw: Any) -> JudgingProtocol:
    value = _dict(raw, "judging")
    protocol = JudgingProtocol(
        model=str(value.get("model", "")),
        served_model_name=str(value.get("served_model_name", "")),
        max_tokens=int(value.get("max_tokens", 2048)),
        concurrency=int(value.get("concurrency", 64)),
        bootstrap=int(value.get("bootstrap", 200)),
        prior=float(value.get("prior", 0.5)),
        anchors=bool(value.get("anchors", True)),
        require_coherent=bool(value.get("require_coherent", True)),
        score_repeats=int(value.get("score_repeats", 1)),
        score_temperature=float(value.get("score_temperature", 0.0)),
        calibration_rows=int(value.get("calibration_rows", 40)),
        request_timeout=float(value.get("request_timeout", 300.0)),
    )
    if not protocol.model or not protocol.served_model_name:
        raise ValueError("judging model names must be non-empty")
    for field, number in (
        ("max_tokens", protocol.max_tokens),
        ("concurrency", protocol.concurrency),
        ("bootstrap", protocol.bootstrap),
        ("score_repeats", protocol.score_repeats),
        ("calibration_rows", protocol.calibration_rows),
    ):
        if number < 1:
            raise ValueError(f"judging.{field} must be positive")
    if protocol.score_repeats > 1 and protocol.score_temperature <= 0:
        raise ValueError("multiple score repeats require positive score temperature")
    if protocol.request_timeout <= 0:
        raise ValueError("judging.request_timeout must be positive")
    return protocol


def load_experiment(path: Path, *, workspace: Path = REPO_ROOT) -> ExperimentConfig:
    path = path.resolve()
    raw_bytes = path.read_bytes()
    raw = _dict(json.loads(raw_bytes), "experiment")
    runs_raw = raw.get("runs")
    if not isinstance(runs_raw, list) or not runs_raw:
        raise ValueError("runs must be a non-empty list")

    runs = []
    for index, item in enumerate(runs_raw):
        value = _dict(item, f"runs[{index}]")
        run_id = str(value.get("run_id", ""))
        if not RUN_ID.fullmatch(run_id):
            raise ValueError(f"runs[{index}].run_id must be YYYYMMDD-HHMMSS")
        vector = _path(value.get("vector"), f"runs[{index}].vector", workspace)
        data_dir = _path(value.get("data_dir"), f"runs[{index}].data_dir", workspace)
        target = validate_steering_target(str(value.get("steering_target", "")))
        run = RunSpec(
            number=int(value.get("number", index + 1)),
            run_id=run_id,
            model_key=str(value.get("model_key", "")),
            model=str(value.get("model", "")),
            vector=vector,
            data_dir=data_dir,
            expected_rows=int(value.get("expected_rows", 0)),
            steering_target=target,
        )
        if not run.model_key or not run.model:
            raise ValueError(f"runs[{index}] requires model_key and model")
        if run.expected_rows < 1:
            raise ValueError(f"runs[{index}].expected_rows must be positive")
        if not vector.is_file():
            raise ValueError(f"vector does not exist: {vector}")
        if not data_dir.is_dir():
            raise ValueError(f"dataset does not exist: {data_dir}")
        if not vector.parent.name.startswith(run_id + "_"):
            raise ValueError(
                f"{run.run_id} does not match vector directory {vector.parent.name!r}"
            )
        trained_span = _training_span(vector)
        required_span = training_span_for_target(target)
        if trained_span != required_span:
            raise ValueError(
                f"{run.run_id} vector is trained on {trained_span!r}, but "
                f"{target!r} requires {required_span!r}"
            )
        runs.append(run)

    numbers = [run.number for run in runs]
    run_ids = [run.run_id for run in runs]
    if len(set(numbers)) != len(numbers):
        raise ValueError("run numbers must be unique")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("run IDs must be unique")
    calibration_run_id = str(raw.get("calibration_run_id", ""))
    if calibration_run_id not in set(run_ids):
        raise ValueError("calibration_run_id must name a configured run")

    return ExperimentConfig(
        name=str(raw.get("name", "")),
        description=str(raw.get("description", "")),
        generation=_generation(raw.get("generation")),
        judging=_judging(raw.get("judging")),
        calibration_run_id=calibration_run_id,
        runs=tuple(runs),
        path=path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
