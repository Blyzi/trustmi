"""Run a configured evaluation suite locally, one evaluation at a time."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.steering_policy import DEFAULT_STEERING_TARGET, STEERING_TARGETS

from benchmarks.run_eval_parallel import default_devices
from benchmarks.suite import load_suite, validate_task


def eval_command(
    args: argparse.Namespace,
    eval_name: str,
    output: Path,
    run_id: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.run_eval_parallel",
        "--suite-config",
        str(args.suite_config),
        "--eval-name",
        eval_name,
        "--model",
        args.model,
        "--vector",
        str(args.vector),
        "--devices",
        *args.devices,
        "--steering-target",
        args.steering_target,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--output",
        str(output),
        "--run-id",
        run_id,
    ]
    if args.auxiliary_base_url:
        command.extend(["--auxiliary-base-url", args.auxiliary_base_url])
    if args.auxiliary_model_path:
        command.extend(["--auxiliary-model-path", str(args.auxiliary_model_path)])
    if args.auxiliary_devices:
        command.extend(["--auxiliary-devices", *args.auxiliary_devices])
    command.extend(
        [
            "--auxiliary-api-key",
            args.auxiliary_api_key,
            "--auxiliary-port",
            str(args.auxiliary_port),
            "--auxiliary-startup-timeout",
            str(args.auxiliary_startup_timeout),
            "--auxiliary-gpu-memory-utilization",
            str(args.auxiliary_gpu_memory_utilization),
            "--auxiliary-max-model-len",
            str(args.auxiliary_max_model_len),
            "--auxiliary-reasoning-parser",
            args.auxiliary_reasoning_parser,
        ]
    )
    if args.strengths:
        command.extend(["--strengths", *[str(value) for value in args.strengths]])
    if args.layers:
        command.extend(["--layers", args.layers])
    if args.think:
        command.append("--think")
    for name in (
        "seed",
        "limit",
        "max_samples",
        "message_limit",
        "time_limit",
        "max_tokens",
    ):
        value = getattr(args, name)
        if value is not None:
            command.extend([f"--{name.replace('_', '-')}", str(value)])
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-config", type=Path, required=True)
    parser.add_argument("--eval-name", action="append")
    parser.add_argument("--tag", action="append")
    parser.add_argument("--model", required=True)
    parser.add_argument("--vector", type=Path, required=True)
    parser.add_argument("--strengths", type=float, nargs="+")
    parser.add_argument("--devices", nargs="+", default=default_devices())
    parser.add_argument(
        "--steering-target",
        choices=STEERING_TARGETS,
        default=DEFAULT_STEERING_TARGET,
    )
    parser.add_argument("--layers")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--message-limit", type=int)
    parser.add_argument("--time-limit", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="GPUs per strength worker; models too large for one GPU need more",
    )
    parser.add_argument("--auxiliary-base-url")
    parser.add_argument("--auxiliary-api-key", default="inspectai")
    parser.add_argument("--auxiliary-model-path", type=Path)
    parser.add_argument("--auxiliary-devices", nargs="+")
    parser.add_argument("--auxiliary-port", type=int, default=8001)
    parser.add_argument("--auxiliary-startup-timeout", type=int, default=900)
    parser.add_argument("--auxiliary-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--auxiliary-max-model-len", type=int, default=32768)
    parser.add_argument("--auxiliary-reasoning-parser", default="openai_gptoss")
    parser.add_argument("--output", type=Path, default=Path("inspect_logs") / "suites")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the suite and print commands without running evaluations",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    suite = load_suite(args.suite_config)
    selected = suite.select(args.eval_name, args.tag)
    for spec in selected:
        validate_task(spec)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = (args.output / f"{suite.name}-{run_id}").resolve()
    commands = [eval_command(args, spec.name, output, run_id) for spec in selected]
    plan: dict[str, Any] = {
        "suite": suite.name,
        "steering_backend": "vllm_lens",
        "steering_target": args.steering_target,
        "suite_config": str(suite.path),
        "suite_config_sha256": suite.sha256,
        "evals": [spec.name for spec in selected],
        "strengths": args.strengths or list(suite.strengths),
        "devices": args.devices,
        "commands": commands,
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return

    output.mkdir(parents=True, exist_ok=False)
    (output / "suite_manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    failures: list[str] = []
    for spec, command in zip(selected, commands):
        print(f"\n=== Running evaluation {spec.name} ===", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            continue
        failures.append(f"{spec.name}: exit {result.returncode}")
        if not args.continue_on_error:
            break
    if failures:
        raise SystemExit("evaluation failures: " + ", ".join(failures))


if __name__ == "__main__":
    main()
