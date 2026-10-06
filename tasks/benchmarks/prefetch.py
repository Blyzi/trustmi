"""Materialize datasets needed by a configured Inspect evaluation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


HF_DATASETS = {
    "gpqa_diamond": {
        "repo_id": "Idavidrein/gpqa",
        "revision": "633f5ee89ab8ad4522a9f850766b73f62147ffdd",
        "filename": "gpqa_diamond.csv",
        "sha256": "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305",
    },
}

BFCL_WHEEL = {
    "version": "2026.3.23",
    "url": (
        "https://files.pythonhosted.org/packages/ba/41/"
        "ed458527c770c50225b60bae3b0c3444b26804ee455fa2d8f187018d2cb2/"
        "bfcl_eval-2026.3.23-py3-none-any.whl"
    ),
    "sha256": "3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-config", type=Path, required=True)
    parser.add_argument("--eval-name", action="append")
    parser.add_argument("--tag", action="append")
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_hf_dataset(name: str) -> tuple[Path, dict[str, str]]:
    from huggingface_hub import hf_hub_download

    source = HF_DATASETS[name]
    path = Path(
        hf_hub_download(
            repo_id=source["repo_id"],
            filename=source["filename"],
            repo_type="dataset",
            revision=source["revision"],
        )
    )
    actual_sha256 = _sha256(path)
    if actual_sha256 != source["sha256"]:
        raise ValueError(f"unexpected SHA-256 for {name}: {actual_sha256}")
    return path, source


def _prefetch_bfcl(cache_dir: Path) -> dict[str, str]:
    """Materialize BFCL data and backend code from its pinned upstream wheel."""
    destination = cache_dir / "BFCL"
    canary = destination / "BFCL_v4_simple_python.json"
    backend_canary = destination / "func_source_code" / "gorilla_file_system.py"

    if not canary.is_file() or not backend_canary.is_file():
        with tempfile.TemporaryDirectory(prefix="trustmi-bfcl-") as directory:
            wheel = Path(directory) / "bfcl_eval.whl"
            with urllib.request.urlopen(BFCL_WHEEL["url"], timeout=120) as response:
                with wheel.open("wb") as output:
                    shutil.copyfileobj(response, output)
            actual_sha256 = _sha256(wheel)
            if actual_sha256 != BFCL_WHEEL["sha256"]:
                raise ValueError(
                    f"unexpected SHA-256 for BFCL wheel: {actual_sha256}"
                )

            data_prefix = "bfcl_eval/data/"
            backend_prefix = (
                "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/"
            )
            with zipfile.ZipFile(wheel) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    if member.filename.startswith(data_prefix):
                        relative = Path(member.filename.removeprefix(data_prefix))
                        target = destination / relative
                    elif member.filename.startswith(backend_prefix):
                        relative = Path(member.filename.removeprefix(backend_prefix))
                        target = destination / "func_source_code" / relative
                    else:
                        continue
                    if ".." in relative.parts:
                        raise ValueError(f"unsafe path in BFCL wheel: {member.filename}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)

        for path in (destination / "func_source_code").glob("*.py"):
            content = path.read_text(encoding="utf-8")
            path.write_text(
                content.replace(
                    "from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.",
                    "from ",
                ),
                encoding="utf-8",
            )

    from inspect_evals.bfcl.data import _validate_processed_data

    _validate_processed_data(destination)
    return dict(BFCL_WHEEL)


def _prefetch_mirrored_datasets(
    eval_names: set[str], cache_dir: Path
) -> dict[str, dict[str, str]]:
    """Fetch pinned copies of datasets that upstream would download at run time."""
    sources: dict[str, dict[str, str]] = {}
    if "bfcl_core" in eval_names:
        sources["bfcl_core"] = _prefetch_bfcl(cache_dir)
    if "gpqa_diamond" in eval_names:
        source, metadata = _download_hf_dataset("gpqa_diamond")
        destination = cache_dir / "gpqa" / "gpqa_diamond.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        sources["gpqa_diamond"] = metadata

    return sources


def main() -> None:
    args = _parser().parse_args()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["INSPECT_EVALS_CACHE_DIR"] = str(cache_dir)
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "huggingface" / "datasets")
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "huggingface" / "hub")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from inspect_ai.model import get_model

    from benchmarks.suite import build_task, load_suite, validate_task

    suite = load_suite(args.suite_config)
    selected = suite.select(args.eval_name, args.tag)
    sources = _prefetch_mirrored_datasets(
        {spec.name for spec in selected},
        cache_dir,
    )
    mock_model = get_model("mockllm/model")
    prefetched: list[str] = []
    for spec in selected:
        validate_task(spec)
        build_task(spec, mock_model if spec.needs_auxiliary_model else None)
        prefetched.append(spec.name)
        print(f"Prefetched {spec.name}", flush=True)

    manifest = {
        "suite": suite.name,
        "suite_config_sha256": suite.sha256,
        "evals": prefetched,
        "mirrored_sources": sources,
    }
    (cache_dir / "trustmi-prefetch.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Dataset cache: {cache_dir}")


if __name__ == "__main__":
    main()
