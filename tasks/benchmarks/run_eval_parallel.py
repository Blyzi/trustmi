"""Run one configured Inspect evaluation concurrently across independent GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from utils.steering_policy import DEFAULT_STEERING_TARGET, STEERING_TARGETS

from utils.local_vllm_server import local_vllm_server
from benchmarks.run_eval import write_summary
from benchmarks.suite import load_suite, validate_task


def default_devices() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [device.strip() for device in visible.split(",") if device.strip()]
    return ["0"]


def partition_strengths(strengths: list[float], worker_count: int) -> list[list[float]]:
    """Split contiguous strengths evenly, putting remainder work first."""
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    worker_count = min(worker_count, len(strengths))
    size, remainder = divmod(len(strengths), worker_count)
    groups: list[list[float]] = []
    offset = 0
    for index in range(worker_count):
        group_size = size + (index < remainder)
        groups.append(strengths[offset : offset + group_size])
        offset += group_size
    return groups


def device_groups(devices: list[str], size: int) -> list[list[str]]:
    """Split the visible devices into one tensor-parallel group per worker."""
    if size < 1:
        raise ValueError("tensor_parallel_size must be positive")
    groups = [
        devices[start : start + size]
        for start in range(0, len(devices) - size + 1, size)
    ]
    if not groups:
        raise ValueError(
            f"{len(devices)} devices cannot hold a tensor-parallel group of {size}"
        )
    return groups


def worker_environment(devices: str) -> dict[str, str]:
    """The environment for one strength worker on ``devices``."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = devices
    env["PYTHONUNBUFFERED"] = "1"
    return env


def set_aside_previous_attempt(root: Path) -> Path | None:
    """Move an earlier attempt's output out of this attempt's way.

    Rerunning under the same run id, for example after a crash or a lost host,
    starts a new attempt. The earlier attempt's files are kept under
    ``previous_attempts/``: its logs explain what happened, and the arms it
    finished are carried into this attempt by ``earlier_results``.
    """
    entries = [entry for entry in root.iterdir() if entry.name != "previous_attempts"]
    if not entries:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    destination = root / "previous_attempts" / stamp
    destination.mkdir(parents=True)
    for entry in entries:
        entry.rename(destination / entry.name)
    return destination


def earlier_results(
    root: Path, eval_name: str, strengths: list[float]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """The whole arms earlier attempts of this run finished, one per strength.

    A worker records each strength as it finishes, so an attempt abandoned
    mid-run still leaves whole arms behind, and they need not be run again.
    Every earlier attempt is searched, newest first, so arms survive repeated
    abandonment. Their logs moved with their attempt, so each result's log is
    pointed at its new place. Also returns one earlier worker manifest, which
    stands in for the run's settings when no arm is left to run.
    """
    finished: dict[float, dict[str, Any]] = {}
    base = None
    attempts = root / "previous_attempts"
    for attempt in sorted(attempts.iterdir(), reverse=True) if attempts.is_dir() else []:
        for path in sorted((attempt / "workers" / eval_name).glob("worker_*/manifest.json")):
            manifest = json.loads(path.read_text())
            base = base or manifest
            for result in manifest.get("results", []):
                strength = float(result["strength"])
                whole = (
                    result["status"] == "success"
                    and result["completed_samples"] == result["total_samples"]
                )
                if whole and strength in strengths and strength not in finished:
                    finished[strength] = {
                        **result,
                        "log": result["log"].replace(
                            str(root / "workers"), str(attempt / "workers"), 1
                        ),
                    }
    return sorted(finished.values(), key=lambda result: result["strength"]), base


def child_command(
    args: argparse.Namespace,
    strengths: list[float],
    worker_root: Path,
    worker_id: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.run_eval",
        "--suite-config",
        str(args.suite_config),
        "--eval-name",
        args.eval_name,
        "--model",
        args.model,
        "--vector",
        str(args.vector),
        "--strengths",
        *[str(strength) for strength in strengths],
        "--steering-target",
        args.steering_target,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--output",
        str(worker_root),
        "--run-id",
        worker_id,
    ]
    if args.auxiliary_base_url:
        command.extend(["--auxiliary-base-url", args.auxiliary_base_url])
    if args.auxiliary_api_key:
        command.extend(["--auxiliary-api-key", args.auxiliary_api_key])
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


def merge_manifests(
    root: Path,
    eval_name: str,
    strengths: list[float],
    devices: list[str],
    worker_groups: list[list[float]],
    started_at: datetime,
    auxiliary_devices: list[str] | None = None,
    carried: list[dict[str, Any]] | None = None,
    carried_base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    worker_manifests = [
        root / "workers" / eval_name / f"worker_{index}" / "manifest.json"
        for index in range(len(worker_groups))
    ]
    manifests = [json.loads(path.read_text()) for path in worker_manifests]
    carried = carried or []
    results = [*carried, *(result for manifest in manifests for result in manifest["results"])]
    results.sort(key=lambda result: result["strength"])

    observed = [float(result["strength"]) for result in results]
    if observed != sorted(strengths):
        raise RuntimeError(
            f"merged strengths do not match request: observed={observed}, "
            f"requested={sorted(strengths)}"
        )

    incomplete = [
        result
        for result in results
        if result["completed_samples"] != result["total_samples"]
    ]
    if incomplete:
        conditions = ", ".join(result["condition"] for result in incomplete)
        raise RuntimeError(f"incomplete evaluation results: {conditions}")
    sample_counts = {int(result["total_samples"]) for result in results}
    if len(sample_counts) != 1:
        raise RuntimeError(
            "inconsistent sample counts across steering strengths: "
            + ", ".join(
                f"{result['condition']}={result['total_samples']}" for result in results
            )
        )

    base = manifests[0] if manifests else carried_base
    if base is None:
        raise RuntimeError("no worker manifest to merge")
    manifest = dict(base)
    finished_at = datetime.now(timezone.utc)
    manifest.update(
        {
            "run_id": root.name,
            "strengths": strengths,
            "results": results,
            "parallel": {
                "devices": devices[: len(worker_groups)],
                "auxiliary_devices": auxiliary_devices or [],
                "worker_groups": worker_groups,
                "worker_manifests": [str(path) for path in worker_manifests],
                "carried_over": [result["condition"] for result in carried],
            },
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": (finished_at - started_at).total_seconds(),
        }
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_summary(root, manifest)
    return manifest


def stop_workers(
    processes: list[tuple[str, list[float], Path, TextIO, subprocess.Popen[str]]],
) -> None:
    """Stop unfinished workers and close every log file."""
    for _, _, _, _, process in processes:
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
    for _, _, _, log_file, process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if not log_file.closed:
            log_file.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-config", type=Path, required=True)
    parser.add_argument("--eval-name", required=True)
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
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--auxiliary-base-url")
    parser.add_argument("--auxiliary-api-key", default="inspectai")
    parser.add_argument("--auxiliary-model-path", type=Path)
    parser.add_argument("--auxiliary-devices", nargs="+")
    parser.add_argument("--auxiliary-port", type=int, default=8001)
    parser.add_argument("--auxiliary-startup-timeout", type=int, default=900)
    parser.add_argument("--auxiliary-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--auxiliary-max-model-len", type=int, default=32768)
    parser.add_argument("--auxiliary-reasoning-parser", default="openai_gptoss")
    parser.add_argument("--output", type=Path, default=Path("inspect_logs"))
    parser.add_argument("--run-id")
    return parser


def run(args: argparse.Namespace) -> Path:
    suite = load_suite(args.suite_config)
    spec = suite.eval(args.eval_name)
    validate_task(spec)
    strengths = list(dict.fromkeys(args.strengths or suite.strengths))
    if not strengths:
        raise SystemExit("at least one steering strength is required")
    if not args.devices:
        raise SystemExit("at least one CUDA device is required")
    if args.auxiliary_base_url and args.auxiliary_model_path:
        raise SystemExit(
            "choose either --auxiliary-base-url or --auxiliary-model-path, not both"
        )
    if spec.needs_auxiliary_model:
        if args.auxiliary_model_path and not args.auxiliary_devices:
            raise SystemExit(
                "--auxiliary-devices is required with --auxiliary-model-path"
            )
        if not args.auxiliary_base_url and not args.auxiliary_model_path:
            raise SystemExit(
                f"{spec.name} requires either --auxiliary-base-url or "
                "--auxiliary-model-path"
            )
        overlap = set(args.devices).intersection(args.auxiliary_devices or [])
        if overlap:
            raise SystemExit(
                "target and auxiliary CUDA devices must be disjoint: "
                + ", ".join(sorted(overlap))
            )
    vector = args.vector.resolve()
    if not vector.is_file():
        raise SystemExit(f"steering vector not found: {vector}")
    args.vector = vector
    args.suite_config = args.suite_config.resolve()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    root = (args.output / args.eval_name / run_id).resolve()
    carried: list[dict[str, Any]] = []
    carried_base = None
    if root.exists():
        set_aside_previous_attempt(root)
        carried, carried_base = earlier_results(root, args.eval_name, strengths)
        if carried:
            print(
                "Keeping arms an earlier attempt finished: "
                + ", ".join(result["condition"] for result in carried),
                flush=True,
            )
    finished = {float(result["strength"]) for result in carried}
    remaining = [strength for strength in strengths if strength not in finished]
    worker_root = root / "workers"
    worker_root.mkdir(parents=True, exist_ok=False)
    worker_devices = [
        ",".join(group)
        for group in device_groups(args.devices, args.tensor_parallel_size)
    ]
    groups = partition_strengths(remaining, len(worker_devices)) if remaining else []
    started_at = datetime.now(timezone.utc)

    server_context = nullcontext(args.auxiliary_base_url)
    if groups and spec.needs_auxiliary_model and args.auxiliary_model_path:
        auxiliary_spec = suite.auxiliary_model
        assert auxiliary_spec is not None
        served_model_name = auxiliary_spec.model.removeprefix("trustmi-aux/")
        server_context = local_vllm_server(
            model_path=args.auxiliary_model_path.resolve(),
            served_model_name=served_model_name,
            devices=args.auxiliary_devices,
            log_path=root / "auxiliary-server.log",
            port=args.auxiliary_port,
            startup_timeout=args.auxiliary_startup_timeout,
            gpu_memory_utilization=args.auxiliary_gpu_memory_utilization,
            max_model_len=args.auxiliary_max_model_len,
            reasoning_parser=args.auxiliary_reasoning_parser,
        )

    processes: list[tuple[str, list[float], Path, TextIO, subprocess.Popen[str]]] = []
    failures: list[str] = []
    with server_context as auxiliary_base_url:
        args.auxiliary_base_url = auxiliary_base_url
        try:
            for index, group in enumerate(groups):
                worker_id = f"worker_{index}"
                device = worker_devices[index]
                log_path = root / f"{worker_id}.log"
                log_file = log_path.open("w")
                env = worker_environment(device)
                command = child_command(args, group, worker_root, worker_id)
                print(
                    f"Launching {worker_id} on CUDA devices {device}: "
                    f"strengths {', '.join(f'{value:g}' for value in group)}",
                    flush=True,
                )
                try:
                    process = subprocess.Popen(
                        command,
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                except BaseException:
                    log_file.close()
                    raise
                processes.append((worker_id, group, log_path, log_file, process))

            pending = list(processes)
            while pending:
                for item in list(pending):
                    worker_id, group, log_path, log_file, process = item
                    return_code = process.poll()
                    if return_code is None:
                        continue
                    log_file.close()
                    pending.remove(item)
                    print(
                        f"Finished {worker_id}: exit={return_code}, "
                        f"strengths={','.join(f'{value:g}' for value in group)}",
                        flush=True,
                    )
                    if return_code != 0:
                        tail = "\n".join(
                            log_path.read_text(errors="replace").splitlines()[-40:]
                        )
                        failures.append(f"{worker_id} exited {return_code}:\n{tail}")
                if failures:
                    break
                if pending:
                    time.sleep(1)
        finally:
            stop_workers(processes)

    if failures:
        raise RuntimeError("\n\n".join(failures))

    manifest = merge_manifests(
        root,
        args.eval_name,
        strengths,
        worker_devices,
        groups,
        started_at,
        args.auxiliary_devices,
        carried=carried,
        carried_base=carried_base,
    )
    print(
        f"Parallel {args.eval_name} sweep completed in "
        f"{manifest['elapsed_seconds']:.1f}s: {root}",
        flush=True,
    )
    return root


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
