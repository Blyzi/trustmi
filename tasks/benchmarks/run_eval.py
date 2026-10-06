"""Run one configured Inspect evaluation across steering strengths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog
from inspect_ai.model import GenerateConfig, get_model

from utils.steering_policy import DEFAULT_STEERING_TARGET, STEERING_TARGETS

from benchmarks import auxiliary_model as _auxiliary_registration  # noqa: F401
from benchmarks import model as _model_registration  # noqa: F401
from benchmarks.suite import (
    build_task,
    load_suite,
    merged_eval_options,
    normalize_eval_options,
    validate_task,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def condition_name(strength: float) -> str:
    if strength == 0:
        return "baseline"
    value = f"{abs(strength):g}".replace(".", "p")
    return f"strength_{'m' if strength < 0 else 'p'}{value}"


def extract_metrics(log: EvalLog) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if log.results is None:
        return rows
    for score in log.results.scores:
        for key, metric in score.metrics.items():
            rows.append(
                {
                    "scorer": score.scorer,
                    "score": score.name,
                    "metric": key,
                    "name": metric.name,
                    "group": metric.group,
                    "value": metric.value,
                }
            )
    return rows


def metric_value(result: dict[str, Any], spec: dict[str, Any]) -> float | None:
    for metric in result["metrics"]:
        if spec.get("score") is not None and metric["score"] != spec["score"]:
            continue
        if metric["metric"] == spec["name"] or metric["name"] == spec["name"]:
            return float(metric["value"])
    return None


def write_summary(root: Path, manifest: dict[str, Any]) -> None:
    results = manifest["results"]
    (root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")

    with (root / "metrics.csv").open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            [
                "eval",
                "condition",
                "strength",
                "status",
                "scorer",
                "score",
                "metric",
                "group",
                "value",
            ]
        )
        for result in results:
            for metric in result["metrics"]:
                writer.writerow(
                    [
                        manifest["eval_name"],
                        result["condition"],
                        result["strength"],
                        result["status"],
                        metric["scorer"],
                        metric["score"],
                        metric["metric"],
                        metric["group"] or "",
                        metric["value"],
                    ]
                )

    metric_specs = manifest["metrics"]
    header = ["Strength", "Samples"] + [
        f"{metric['label']} ({metric['direction']} is better)"
        for metric in metric_specs
    ]
    table = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---:"] * len(header)) + " |",
    ]
    for result in sorted(results, key=lambda item: item["strength"]):
        values = []
        for metric in metric_specs:
            value = metric_value(result, metric)
            values.append("—" if value is None else f"{value:.4f}")
        table.append(
            "| "
            + " | ".join(
                [
                    f"{result['strength']:g}",
                    f"{result['completed_samples']}/{result['total_samples']}",
                    *values,
                ]
            )
            + " |"
        )

    auxiliary = manifest["auxiliary_model"]
    auxiliary_lines = []
    if auxiliary is not None:
        auxiliary_lines = [
            f"- Judge/simulator model: `{auxiliary['model']}`",
            (
                f"- Judge/simulator reasoning effort: "
                f"`{auxiliary['reasoning_effort']}`; maximum output tokens: "
                f"`{auxiliary['max_tokens']}`"
            ),
        ]
    lines = [
        f"# {manifest['eval_name']} steering sweep",
        "",
        manifest["description"],
        "",
        *table,
        "",
        "## Method",
        "",
        f"- Suite: `{manifest['suite_name']}`",
        f"- Inspect task: `{manifest['task']}`",
        f"- Model: `{manifest['model']}`",
        f"- Steering vector SHA-256: `{manifest['vector_sha256']}`",
        f"- Steering target: `{manifest['steering_target']}`",
        f"- Seed: `{manifest['seed']}`; thinking: `{manifest['think']}`",
        *auxiliary_lines,
        (
            f"- Inspect AI: `{manifest['inspect_ai_version']}`; "
            f"Inspect Evals: `{manifest['inspect_evals_version']}`"
        ),
        "- Strength 0 is a matched baseline with no activation hook attached.",
        "",
        (
            "Machine-specific manifests and raw Inspect logs should not be "
            "committed; they may contain local paths or benchmark transcripts."
        ),
        "",
    ]
    report = "\n".join(lines)
    (root / "report.md").write_text(report)
    print("\n===== EVALUATION REPORT =====\n", flush=True)
    print(report, flush=True)


def _override(options: dict[str, Any], name: str, value: Any) -> None:
    if value is not None:
        options[name] = value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-config", type=Path, required=True)
    parser.add_argument("--eval-name", required=True)
    parser.add_argument("--model", required=True, help="Hugging Face model id or path")
    parser.add_argument("--vector", type=Path, required=True)
    parser.add_argument("--strengths", type=float, nargs="+")
    parser.add_argument(
        "--steering-target",
        choices=STEERING_TARGETS,
        default=DEFAULT_STEERING_TARGET,
    )
    parser.add_argument("--layers", help="Comma-separated decoder layer indices")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--message-limit", type=int)
    parser.add_argument("--time-limit", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--auxiliary-base-url")
    parser.add_argument("--auxiliary-api-key", default="inspectai")
    parser.add_argument("--output", type=Path, default=Path("inspect_logs"))
    parser.add_argument("--run-id")
    return parser


def run(args: argparse.Namespace) -> Path:
    started_at = datetime.now(timezone.utc)
    suite = load_suite(args.suite_config)
    spec = suite.eval(args.eval_name)
    validate_task(spec)

    vector = args.vector.resolve()
    if not vector.is_file():
        raise SystemExit(f"steering vector not found: {vector}")
    strengths = list(dict.fromkeys(args.strengths or suite.strengths))

    options = merged_eval_options(suite, spec)
    _override(options, "seed", args.seed)
    _override(options, "limit", args.limit)
    _override(options, "max_samples", args.max_samples)
    _override(options, "message_limit", args.message_limit)
    _override(options, "time_limit", args.time_limit)
    _override(options, "max_tokens", args.max_tokens)
    options, seed = normalize_eval_options(options)
    needs_auxiliary_model = spec.needs_auxiliary_model
    auxiliary_spec = suite.auxiliary_model if needs_auxiliary_model else None
    if auxiliary_spec is not None and args.auxiliary_base_url is None:
        raise SystemExit(
            f"{spec.name} requires --auxiliary-base-url for {auxiliary_spec.model}"
        )

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    root = (args.output / spec.name / run_id).resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "suite_name": suite.name,
        "suite_config": str(suite.path),
        "suite_config_sha256": suite.sha256,
        "eval_name": spec.name,
        "description": spec.description,
        "task": spec.task,
        "task_args": spec.task_args,
        "judge_args": list(spec.judge_args),
        "model_roles": list(spec.model_roles),
        "tags": spec.tags,
        "metrics": [metric.__dict__ for metric in spec.metrics],
        "model": args.model,
        "vector": str(vector),
        "vector_sha256": sha256(vector),
        "strengths": strengths,
        "steering_target": args.steering_target,
        "layers": args.layers,
        "think": args.think,
        "seed": seed,
        "auxiliary_model": auxiliary_spec.__dict__ if auxiliary_spec else None,
        "auxiliary_base_url": args.auxiliary_base_url if auxiliary_spec else None,
        "eval_options": options,
        "inspect_ai_version": importlib.metadata.version("inspect-ai"),
        "inspect_evals_version": importlib.metadata.version("inspect-evals"),
        "vllm_version": importlib.metadata.version("vllm"),
        "vllm_lens_version": importlib.metadata.version("vllm-lens"),
        "transformers_version": importlib.metadata.version("transformers"),
        "steering_backend": "vllm_lens",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "started_at": started_at.isoformat(),
        "results": [],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    target = get_model(
        f"trustmi/{args.model}",
        memoize=False,
        vector_path=str(vector),
        strength=0.0,
        layers=args.layers,
        steering_target=args.steering_target,
        think=args.think,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    try:
        auxiliary_model = None
        if auxiliary_spec is not None:
            auxiliary_model = get_model(
                auxiliary_spec.model,
                base_url=args.auxiliary_base_url,
                api_key=args.auxiliary_api_key,
                memoize=False,
                config=GenerateConfig(
                    max_tokens=auxiliary_spec.max_tokens,
                    reasoning_effort=auxiliary_spec.reasoning_effort,
                    temperature=0.0,
                    seed=seed,
                ),
            )
        model_roles = {
            role: auxiliary_model for role in spec.model_roles if auxiliary_model
        }

        for strength in strengths:
            condition = condition_name(strength)
            condition_dir = root / condition
            condition_dir.mkdir()
            target.api.strength = strength
            print(
                f"\n=== {spec.name}: {condition} (strength={strength:g}) ===",
                flush=True,
            )
            logs = inspect_eval(
                build_task(spec, auxiliary_model),
                model=target,
                model_roles=model_roles or None,
                log_dir=str(condition_dir),
                metadata={
                    "trustmi_suite": suite.name,
                    "trustmi_eval": spec.name,
                    "trustmi_condition": condition,
                    "trustmi_strength": strength,
                    "trustmi_auxiliary_model": (
                        auxiliary_spec.model if auxiliary_spec else None
                    ),
                    "trustmi_auxiliary_reasoning_effort": (
                        auxiliary_spec.reasoning_effort if auxiliary_spec else None
                    ),
                },
                **options,
                seed=seed,
            )
            if len(logs) != 1:
                raise RuntimeError(f"expected one Inspect log, got {len(logs)}")
            log = logs[0]
            result = {
                "condition": condition,
                "strength": strength,
                "status": log.status,
                "log": log.location,
                "total_samples": log.results.total_samples if log.results else 0,
                "completed_samples": (
                    log.results.completed_samples if log.results else 0
                ),
                "metrics": extract_metrics(log),
            }
            manifest["results"].append(result)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            write_summary(root, manifest)
            if log.status != "success":
                message = log.error.message if log.error else "unknown error"
                raise RuntimeError(f"{condition} failed: {message}")
    finally:
        target.api.backend.shutdown_engine(target.api.model_name)

    finished_at = datetime.now(timezone.utc)
    manifest["finished_at"] = finished_at.isoformat()
    manifest["elapsed_seconds"] = (finished_at - started_at).total_seconds()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_summary(root, manifest)
    print(f"\nEvaluation artifacts: {root}", flush=True)
    return root


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
