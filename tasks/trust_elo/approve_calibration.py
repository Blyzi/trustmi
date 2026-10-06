"""Inspect and explicitly approve the Trust-Elo rubric calibration gate."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from trust_elo.experiment import file_sha256, load_experiment
from trust_elo.run_campaign import (
    GRADING_IDENTITY_FIELDS,
    _validate_recorded_outputs,
    validate_grading,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Write approval.json after printing the calibration diagnostics.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    experiment = load_experiment(args.experiment_config)
    run = experiment.run(experiment.calibration_run_id)
    run_root = args.output_root / "runs" / run.run_id
    generation_path = run_root / "generation" / "generations.jsonl"
    calibration = run_root / "calibration"
    results_path = calibration / "results.json"
    validation = validate_grading(
        calibration,
        run,
        expected_rows=min(experiment.judging.calibration_rows, run.expected_rows),
        generation_sha256=file_sha256(generation_path),
    )
    _validate_recorded_outputs(calibration, validation)
    calibration_manifest_path = calibration / "manifest.json"
    calibration_manifest = json.loads(calibration_manifest_path.read_text())
    grading_identity = {
        key: calibration_manifest.get(key) for key in GRADING_IDENTITY_FIELDS
    }
    results = json.loads(results_path.read_text())
    scores = [
        json.loads(line)
        for line in (calibration / "scores.jsonl").read_text().splitlines()
        if line.strip()
    ]
    at_ceiling = sum(float(row["score"]) == 100 for row in scores)
    diagnostics = {
        "validation": validation,
        "pole_scores": (results.get("trust_score") or {}).get("poles"),
        "calibration_accuracy": results.get("calibration_accuracy"),
        "position_bias_a_share": results.get("position_bias_a_share"),
        "scores_at_100": at_ceiling,
        "scores_total": len(scores),
    }
    print(json.dumps(diagnostics, indent=2))
    if not args.approve:
        print("Calibration not approved; pass --approve after reviewing diagnostics.")
        return
    approval = {
        "campaign": experiment.name,
        "experiment_config_sha256": experiment.sha256,
        "run_id": run.run_id,
        "results_sha256": file_sha256(results_path),
        "calibration_manifest_sha256": file_sha256(calibration_manifest_path),
        "grading_identity": grading_identity,
        "approved": True,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "approved_by": os.environ.get("USER"),
        "diagnostics": diagnostics,
    }
    approvals = args.output_root / "approvals"
    approvals.mkdir(parents=True, exist_ok=True)
    path = approvals / f"{run.run_id}.json"
    if path.exists():
        existing = json.loads(path.read_text())
        identity_fields = (
            "campaign",
            "experiment_config_sha256",
            "run_id",
            "results_sha256",
            "calibration_manifest_sha256",
            "grading_identity",
            "approved",
        )
        if {
            key: existing.get(key)
            for key in identity_fields
        } != {
            key: approval[key]
            for key in identity_fields
        }:
            raise ValueError(f"refusing to replace conflicting approval: {path}")
        print(f"Calibration already approved: {path}")
        return
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=approvals,
        prefix=f".{run.run_id}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(approval, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)
    print(f"Calibration approved: {path}")


if __name__ == "__main__":
    main()
