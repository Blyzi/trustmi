"""Lifecycle management for a local OpenAI-compatible vLLM server."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO


def server_environment(model_path: Path, devices: list[str]) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    env["PYTHONUNBUFFERED"] = "1"
    encodings = model_path / "tiktoken_encodings"
    if encodings.is_dir():
        env["TIKTOKEN_ENCODINGS_BASE"] = str(encodings)
    return env


def wait_until_ready(
    api_base: str,
    process: subprocess.Popen[bytes],
    timeout: int,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"auxiliary vLLM server exited with status {process.returncode}"
            )
        try:
            with urllib.request.urlopen(f"{api_base}/models", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError(f"auxiliary vLLM server was not ready after {timeout}s")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _server_command(
    *,
    model_path: Path,
    served_model_name: str,
    port: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    reasoning_parser: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--served-model-name",
        served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--gdn-prefill-backend",
        "triton",
        "--default-chat-template-kwargs",
        '{"enable_thinking": false}',
        "--enforce-eager",
        "--reasoning-parser",
        reasoning_parser,
    ]


@contextmanager
def local_vllm_server(
    *,
    model_path: Path,
    served_model_name: str,
    devices: list[str],
    log_path: Path,
    port: int,
    startup_timeout: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    reasoning_parser: str,
) -> Iterator[str]:
    if not model_path.exists():
        raise FileNotFoundError(f"auxiliary model not found: {model_path}")
    if not devices:
        raise ValueError("at least one auxiliary CUDA device is required")

    api_base = f"http://127.0.0.1:{port}/v1"
    command = _server_command(
        model_path=model_path,
        served_model_name=served_model_name,
        port=port,
        tensor_parallel_size=len(devices),
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        reasoning_parser=reasoning_parser,
    )
    env = server_environment(model_path, devices)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file: TextIO = log_path.open("w", encoding="utf-8")
    print(
        f"Starting {served_model_name} on CUDA device(s) {','.join(devices)}; "
        f"server log: {log_path}",
        flush=True,
    )
    process = subprocess.Popen(
        command,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_until_ready(api_base, process, startup_timeout)
        print(f"Auxiliary model ready at {api_base}", flush=True)
        yield api_base
    except BaseException as error:
        log_file.flush()
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
        if tail:
            error.add_note(f"Auxiliary server log tail:\n{tail}")
        raise
    finally:
        stop_server(process)
        log_file.close()


@contextmanager
def local_vllm_server_pool(
    *,
    model_path: Path,
    served_model_name: str,
    devices: list[str],
    log_dir: Path,
    base_port: int,
    startup_timeout: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    reasoning_parser: str,
) -> Iterator[list[str]]:
    """Serve one independent model replica per GPU and return all API bases."""
    if not model_path.exists():
        raise FileNotFoundError(f"auxiliary model not found: {model_path}")
    if not devices:
        raise ValueError("at least one auxiliary CUDA device is required")
    log_dir.mkdir(parents=True, exist_ok=True)
    servers = []
    try:
        for index, device in enumerate(devices):
            port = base_port + index
            api_base = f"http://127.0.0.1:{port}/v1"
            log_path = log_dir / f"judge-server-{index:02d}.log"
            log_file: TextIO = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                _server_command(
                    model_path=model_path,
                    served_model_name=served_model_name,
                    port=port,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=max_model_len,
                    reasoning_parser=reasoning_parser,
                ),
                env=server_environment(model_path, [device]),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            servers.append((api_base, process, log_file, log_path))
        with ThreadPoolExecutor(max_workers=len(servers)) as pool:
            futures = [
                pool.submit(wait_until_ready, api_base, process, startup_timeout)
                for api_base, process, _log_file, _log_path in servers
            ]
            for future in futures:
                future.result()
        print(f"{len(servers)} auxiliary model replicas ready", flush=True)
        yield [api_base for api_base, _process, _log_file, _log_path in servers]
    except BaseException as error:
        tails = []
        for _api_base, _process, log_file, log_path in servers:
            log_file.flush()
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
            if tail:
                tails.append(f"{log_path.name}:\n{tail}")
        if tails:
            error.add_note("Auxiliary server log tails:\n" + "\n".join(tails))
        raise
    finally:
        for _api_base, process, log_file, _log_path in servers:
            stop_server(process)
            log_file.close()
