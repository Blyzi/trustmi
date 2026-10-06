"""Configuration and task loading for TrustMI Inspect evaluation suites."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from inspect_ai import Task
from inspect_ai.model import Model


DIRECTIONS = {"higher", "lower"}
RESERVED_EVAL_OPTIONS = {"log_dir", "metadata", "model", "model_roles", "tasks"}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: str
    label: str
    score: str | None = None


@dataclass(frozen=True)
class EvalSpec:
    name: str
    task: str
    description: str
    task_args: dict[str, Any] = field(default_factory=dict)
    judge_args: tuple[str, ...] = ()
    model_roles: tuple[str, ...] = ()
    metrics: tuple[MetricSpec, ...] = ()
    tags: tuple[str, ...] = ()
    eval_options: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_auxiliary_model(self) -> bool:
        return bool(self.judge_args or self.model_roles)


@dataclass(frozen=True)
class AuxiliaryModelSpec:
    model: str
    reasoning_effort: str
    max_tokens: int


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    description: str
    strengths: tuple[float, ...]
    eval_options: dict[str, Any]
    auxiliary_model: AuxiliaryModelSpec | None
    evals: tuple[EvalSpec, ...]
    path: Path
    sha256: str

    def select(
        self,
        names: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> list[EvalSpec]:
        requested_names = set(names or [])
        requested_tags = set(tags or [])
        known_names = {spec.name for spec in self.evals}
        unknown = requested_names - known_names
        if unknown:
            raise ValueError(f"unknown eval names: {', '.join(sorted(unknown))}")

        selected = []
        for spec in self.evals:
            if requested_names and spec.name not in requested_names:
                continue
            if requested_tags and not requested_tags.intersection(spec.tags):
                continue
            selected.append(spec)
        if not selected:
            raise ValueError("no evaluations match the requested filters")
        return selected

    def eval(self, name: str) -> EvalSpec:
        selected = self.select(names=[name])
        return selected[0]


def _require_type(value: Any, expected: type, field_name: str) -> Any:
    if not isinstance(value, expected):
        raise ValueError(f"{field_name} must be {expected.__name__}")
    return value


def _strings(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return tuple(value)


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(_require_type(value, dict, field_name))


def _metric(raw: Any, eval_name: str) -> MetricSpec:
    value = _require_type(raw, dict, f"{eval_name}.metrics[]")
    name = _require_type(value.get("name"), str, f"{eval_name}.metrics[].name")
    direction = _require_type(
        value.get("direction"), str, f"{eval_name}.metrics[{name}].direction"
    )
    if direction not in DIRECTIONS:
        raise ValueError(
            f"{eval_name}.metrics[{name}].direction must be one of "
            f"{sorted(DIRECTIONS)}"
        )
    label = value.get("label", name.replace("_", " ").title())
    score = value.get("score")
    if score is not None and not isinstance(score, str):
        raise ValueError(f"{eval_name}.metrics[{name}].score must be a string")
    return MetricSpec(name=name, direction=direction, label=str(label), score=score)


def _eval_spec(raw: Any) -> EvalSpec:
    value = _require_type(raw, dict, "evals[]")
    name = _require_type(value.get("name"), str, "evals[].name")
    task = _require_type(value.get("task"), str, f"{name}.task")
    if task.count(":") != 1:
        raise ValueError(f"{name}.task must use the format 'module:function'")
    eval_options = _mapping(value.get("eval_options"), f"{name}.eval_options")
    reserved = RESERVED_EVAL_OPTIONS.intersection(eval_options)
    if reserved:
        raise ValueError(
            f"{name}.eval_options cannot override reserved fields: "
            f"{', '.join(sorted(reserved))}"
        )
    return EvalSpec(
        name=name,
        task=task,
        description=str(value.get("description", "")),
        task_args=_mapping(value.get("task_args"), f"{name}.task_args"),
        judge_args=_strings(value.get("judge_args"), f"{name}.judge_args"),
        model_roles=_strings(value.get("model_roles"), f"{name}.model_roles"),
        metrics=tuple(_metric(metric, name) for metric in value.get("metrics", [])),
        tags=_strings(value.get("tags"), f"{name}.tags"),
        eval_options=eval_options,
    )


def load_suite(path: Path) -> SuiteConfig:
    path = path.resolve()
    raw_bytes = path.read_bytes()
    raw = _require_type(json.loads(raw_bytes), dict, "suite")
    strengths_raw = raw.get("strengths")
    if not isinstance(strengths_raw, list) or not strengths_raw:
        raise ValueError("strengths must be a non-empty list")
    strengths = tuple(float(value) for value in strengths_raw)
    if len(set(strengths)) != len(strengths):
        raise ValueError("strengths must not contain duplicates")

    evals_raw = raw.get("evals")
    if not isinstance(evals_raw, list) or not evals_raw:
        raise ValueError("evals must be a non-empty list")
    evals = tuple(_eval_spec(value) for value in evals_raw)
    names = [spec.name for spec in evals]
    if len(set(names)) != len(names):
        raise ValueError("eval names must be unique")

    eval_options = _mapping(raw.get("eval_options"), "eval_options")
    reserved = RESERVED_EVAL_OPTIONS.intersection(eval_options)
    if reserved:
        raise ValueError(
            "eval_options cannot override reserved fields: "
            f"{', '.join(sorted(reserved))}"
        )
    auxiliary_raw = raw.get("auxiliary_model")
    auxiliary_model = None
    if auxiliary_raw is not None:
        auxiliary = _require_type(auxiliary_raw, dict, "auxiliary_model")
        auxiliary_model = AuxiliaryModelSpec(
            model=_require_type(auxiliary.get("model"), str, "auxiliary_model.model"),
            reasoning_effort=_require_type(
                auxiliary.get("reasoning_effort"),
                str,
                "auxiliary_model.reasoning_effort",
            ),
            max_tokens=_require_type(
                auxiliary.get("max_tokens"), int, "auxiliary_model.max_tokens"
            ),
        )
        if auxiliary_model.max_tokens < 1:
            raise ValueError("auxiliary_model.max_tokens must be positive")

    if any(spec.needs_auxiliary_model for spec in evals) and auxiliary_model is None:
        raise ValueError(
            "auxiliary_model is required when an evaluation uses judge_args or "
            "model_roles"
        )

    return SuiteConfig(
        name=_require_type(raw.get("name"), str, "name"),
        description=str(raw.get("description", "")),
        strengths=strengths,
        eval_options=eval_options,
        auxiliary_model=auxiliary_model,
        evals=evals,
        path=path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def task_factory(reference: str) -> Callable[..., Task]:
    module_name, function_name = reference.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise ValueError(f"task factory is not callable: {reference}")
    return factory


def validate_task(spec: EvalSpec) -> None:
    factory = task_factory(spec.task)
    signature = inspect.signature(factory)
    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    supplied = set(spec.task_args).union(spec.judge_args)
    unknown = supplied - set(parameters)
    if unknown and not accepts_kwargs:
        raise ValueError(
            f"{spec.name} passes unsupported task arguments to {spec.task}: "
            f"{', '.join(sorted(unknown))}"
        )


def build_task(spec: EvalSpec, judge: Model | None) -> Task:
    if (spec.judge_args or spec.model_roles) and judge is None:
        raise ValueError(f"{spec.name} requires a judge or simulator model")
    kwargs = dict(spec.task_args)
    for name in spec.judge_args:
        kwargs[name] = judge
    return task_factory(spec.task)(**kwargs)


def merged_eval_options(suite: SuiteConfig, spec: EvalSpec) -> dict[str, Any]:
    options = dict(suite.eval_options)
    options.update(spec.eval_options)
    return options


def normalize_eval_options(options: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Split merged options into the pair a run manifest records.

    The seed is stored as its own manifest field rather than inside
    `eval_options`, and two defaults are filled in. Both the writer and any
    reader comparing a manifest against a suite must apply this identically,
    so it lives here rather than being repeated at each call site.
    """
    options = dict(options)
    seed = int(options.pop("seed", 0))
    options.setdefault("display", "plain")
    options.setdefault("fail_on_error", True)
    return options, seed
