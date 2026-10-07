from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from trust_elo.experiment import load_experiment
from trust_elo.run_campaign import (
    GENERATION_CODE_FILES,
    GRADING_CODE_FILES,
    _git_commit,
    _manifest_base,
    adopt_generation,
    main,
    run_generation,
    validate_generations,
    validate_grading,
)
from trust_elo.judging import (
    SOLO_FIELDS,
    grade_one,
    parse_verdict,
    play,
    solo_system_prompt,
)
from utils.run_id import vector_run_id


TASKS_ROOT = Path(__file__).parents[1]
EXPERIMENT = TASKS_ROOT / "trust_elo/configs/rubric_20260921_v1.json"


class ExperimentConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = load_experiment(EXPERIMENT)

    def test_campaign_has_expected_runs(self) -> None:
        self.assertEqual(len(self.experiment.runs), 13)
        self.assertEqual(self.experiment.calibration_run_id, "20260913-125025")

    def test_every_shipped_config_loads(self) -> None:
        for path in sorted((TASKS_ROOT / "trust_elo/configs").glob("*.json")):
            with self.subTest(config=path.name):
                load_experiment(path)

    def test_run_ids_match_vector_directories(self) -> None:
        for run in self.experiment.runs:
            self.assertEqual(vector_run_id(run.vector), run.run_id)

    def test_unknown_run_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown run IDs"):
            self.experiment.select(["missing"])

    def test_empty_generation_is_preserved_as_collapse_outcome(self) -> None:
        run = replace(
            self.experiment.run("20260913-234838"), expected_rows=1
        )
        source = {
            "id": "row-1",
            "context": [{"role": "user", "content": "question"}],
            "messages_trust": [{"role": "assistant", "content": "trust"}],
            "messages_distrust": [{"role": "assistant", "content": "distrust"}],
        }
        row = {
            "id": "row-1",
            "context": source["context"],
            "reference_trust": "trust",
            "reference_distrust": "distrust",
            "vector_run_id": run.run_id,
            "steering_target": run.steering_target,
            "vector": str(run.vector),
            "generations": {
                "-2": ["", "reply"],
                "-1": ["reply", "reply"],
                "0": ["reply", "reply"],
                "1": ["reply", "reply"],
                "2": ["reply", "reply"],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generations.jsonl"
            path.write_text(json.dumps(row) + "\n")
            with patch("trust_elo.run_campaign._dataset_rows", return_value=[source]):
                validation = validate_generations(path, run, self.experiment)
        self.assertEqual(validation["rows"], 1)

    def test_manifest_uses_explicit_source_commit(self) -> None:
        run = self.experiment.run("20260913-125025")
        manifest = _manifest_base(
            self.experiment,
            run,
            "generation",
            ["generate"],
            "source-commit",
        )
        self.assertEqual(manifest["git_commit"], "source-commit")

    def test_generation_records_its_commit_and_runtime(self) -> None:
        run = self.experiment.run("20260913-234838")
        versions = {"vllm": "0.27.1", "vllm-lens": "1.2.1"}
        validation = {"rows": run.expected_rows, "draws": 2}

        def generate(command: list[str], *, cwd: Path) -> None:
            Path(command[command.index("--out") + 1]).write_text("{}\n")

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch(
                    "trust_elo.run_campaign.validate_model_checkpoint",
                    return_value="model-config",
                ),
                patch("trust_elo.run_campaign._run", side_effect=generate),
                patch(
                    "trust_elo.run_campaign.validate_generations",
                    return_value=validation,
                ),
            ):
                final = run_generation(
                    self.experiment,
                    run,
                    model_path="/model",
                    expected_model_type="qwen3_5",
                    source_git_commit="source-commit",
                    runtime_versions=versions,
                    tensor_parallel_size=1,
                    output_root=Path(tmp),
                )
            manifest = json.loads((final / "manifest.json").read_text())
            self.assertEqual(manifest["source_git_commit"], "source-commit")
            self.assertEqual(manifest["runtime_versions"], versions)
            self.assertEqual(manifest["validation"], validation)
            self.assertFalse([key for key in manifest if "package" in key])
            self.assertEqual(final.stat().st_mode & 0o777, 0o755)

    def test_generation_adoption_records_source_artifacts(self) -> None:
        run = self.experiment.run("20260913-234838")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source" / "runs" / run.run_id / "generation"
            source.mkdir(parents=True)
            (source / "generations.jsonl").write_text("{}\n")
            (source / "manifest.json").write_text(
                json.dumps({"campaign": "old-campaign"}) + "\n"
            )
            validation = {"rows": run.expected_rows, "draws": 2}
            with patch(
                "trust_elo.run_campaign.validate_generations",
                return_value=validation,
            ):
                final = adopt_generation(
                    self.experiment,
                    run,
                    source_output_root=root / "source",
                    output_root=root / "target",
                    source_git_commit="adoption-commit",
                )
            manifest = json.loads((final / "manifest.json").read_text())
            self.assertEqual(manifest["source_campaign"], "old-campaign")
            self.assertEqual(manifest["validation"], validation)
            self.assertEqual(len(manifest["source_generation_sha256"]), 64)

    def test_grading_validation_rejects_nonzero_failures(self) -> None:
        run = self.experiment.run("20260913-125025")
        conditions = {
            "dataset_distrust_pole": 1,
            "baseline": 1,
            "steer-2": 1,
            "steer-1": 1,
            "steer+1": 1,
            "steer+2": 1,
            "dataset_trust_pole": 1,
        }
        results = {
            "run": {"vector_run_id": run.run_id, "n_rows": 1},
            "elo": conditions,
            "games_in_fit": 1,
            "judge_failures": 1,
            "trust_score": {
                "grading": {"graded": 1, "requested": 2, "failed": 1}
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "results.json").write_text(json.dumps(results))
            (directory / "scores.jsonl").write_text("{}\n")
            (directory / "verdicts.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "failed solo judge calls"):
                validate_grading(
                    directory,
                    run,
                    expected_rows=1,
                    generation_sha256="generation",
                )


class CampaignCommandTests(unittest.TestCase):
    def test_every_hashed_code_file_exists(self) -> None:
        for path in (*GENERATION_CODE_FILES, *GRADING_CODE_FILES):
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file(), f"{path} is hashed but missing")

    def test_uncommitted_tracked_changes_are_refused(self) -> None:
        status = subprocess.CompletedProcess([], 0, stdout=" M tasks/trust_elo/main.py\n")
        with (
            patch("trust_elo.run_campaign.subprocess.run", return_value=status),
            self.assertRaisesRegex(ValueError, "commit tracked changes"),
        ):
            _git_commit()

    def test_calibration_is_fixed_to_the_configured_run(self) -> None:
        argv = [
            "run_campaign",
            "calibrate",
            "--experiment-config",
            str(EXPERIMENT),
            "--run-id",
            "20260913-234838",
            "--output-root",
            "unused",
            "--judge-model-path",
            "judge",
            "--expected-judge-model-type",
            "gpt_oss",
            "--judge-devices",
            "0",
        ]
        with (
            patch("sys.argv", argv),
            patch("trust_elo.run_campaign._git_commit", return_value="commit"),
            patch("trust_elo.run_campaign._runtime_versions", return_value={}),
            self.assertRaisesRegex(ValueError, "calibration is fixed to 20260913-125025"),
        ):
            main()


class RubricTests(unittest.TestCase):
    def test_coherence_uses_conversation_language(self) -> None:
        prompt = solo_system_prompt()
        self.assertIn("wrong language for the conversation", prompt)
        self.assertNotIn("non-English output", prompt)

    def test_score_must_be_in_five_point_steps(self) -> None:
        with self.assertRaisesRegex(ValueError, "SCORE"):
            parse_verdict("NOTE: x\nCOHERENT: YES\nSCORE: 83", SOLO_FIELDS)
        verdict = parse_verdict("NOTE: x\nCOHERENT: YES\nSCORE: 85", SOLO_FIELDS)
        self.assertEqual(verdict["SCORE"], "85")

    @patch("trust_elo.judging.complete")
    def test_solo_prompt_names_focal_uncertainty(self, complete) -> None:
        complete.return_value = {"COHERENT": "YES", "SCORE": "0"}
        grade_one(
            "judge",
            [{"role": "user", "content": "Question"}],
            "the user claims the staging migration completed",
            "Reply",
            100,
            {},
            "test",
        )
        messages = complete.call_args.args[1]
        self.assertIn(
            "FOCAL UNCERTAINTY:\nthe user claims the staging migration completed",
            messages[1]["content"],
        )

    @patch("trust_elo.judging.complete")
    def test_pair_prompt_names_focal_uncertainty(self, complete) -> None:
        complete.return_value = {"TRUST_SIDE": "A"}
        play(
            "judge",
            [{"role": "user", "content": "Question"}],
            "the user claims the staging migration completed",
            "Reply A",
            "Reply B",
            100,
            {},
            "test",
            True,
        )
        messages = complete.call_args.args[1]
        self.assertIn(
            "FOCAL UNCERTAINTY:\nthe user claims the staging migration completed",
            messages[1]["content"],
        )


if __name__ == "__main__":
    unittest.main()
