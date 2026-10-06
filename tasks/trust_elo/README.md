# Trust-Elo campaign execution

`main.py` is the scientific implementation. The campaign layer adds two things
around it:

- `configs/rubric_20260921_v1.json` fixes the selected vectors and protocol;
- `run_campaign.py` runs one phase of one run, validates it, and publishes only
  complete atomic outputs.

## Output layout

Every phase writes below one output root:

```text
<output-root>/
  runs/<run-id>/generation/
  runs/<run-id>/calibration/
  runs/<run-id>/grading/
  approvals/
```

Every completed phase includes `manifest.json` with the experiment/config hash,
the git commit, the runtime package versions, input/output hashes, and
validation counts. Attempts write below `.attempts/` and are promoted with a
same-filesystem rename only after validation. A matching completed phase is a
no-op; a conflicting phase is never overwritten.

A phase refuses to run from a checkout with uncommitted tracked changes: only
its own code files are hashed, so the recorded commit stands for the rest of
the code it runs. Model and judge paths must be local checkpoints, whose
`config.json` is checked against `--expected-model-type` or
`--expected-judge-model-type` and hashed into the manifest.

## Workflow

All commands are run from `tasks/`. Each one runs a single phase of a single run.

```bash
OUTPUT=/path/to/trustmi-elo/rubric-20260921-v1
EXPERIMENT=trust_elo/configs/rubric_20260921_v1.json

# One generation sweep, for each run marked generate_and_grade.
uv run python -m trust_elo.run_campaign generate \
  --experiment-config "$EXPERIMENT" --output-root "$OUTPUT" --run-id <run-id> \
  --model-path /path/to/<model> --expected-model-type <model_type> \
  [--tensor-parallel-size <n>]

# Required 40-row rubric check, fixed to run 20260913-125025.
uv run python -m trust_elo.run_campaign calibrate \
  --experiment-config "$EXPERIMENT" --output-root "$OUTPUT" --run-id 20260913-125025 \
  --judge-model-path /path/to/gpt-oss-120b --expected-judge-model-type gpt_oss \
  --judge-devices 0 1 2 3

# After reviewing the calibration diagnostics it prints.
uv run python -m trust_elo.approve_calibration \
  --experiment-config "$EXPERIMENT" --output-root "$OUTPUT" --approve

# Full grading, for each of the thirteen runs.
uv run python -m trust_elo.run_campaign grade \
  --experiment-config "$EXPERIMENT" --output-root "$OUTPUT" --run-id <run-id> \
  --judge-model-path /path/to/gpt-oss-120b --expected-judge-model-type gpt_oss \
  --judge-devices 0 1 2 3
```

Calibration is deliberately not chained to full grading: its pole separation,
score histogram, calibration accuracy, and position bias must be reviewed first.

Three more phases cover larger or split campaigns:

- `generate-full-node` shards the rows over GPUs `0` to `--sampler-replicas`
  minus one, one sampler each, then merges and validates the shards.
- `grade-full-node` serves one judge replica per `--judge-devices` entry. It
  does not consult the calibration approval, so it requires an explicit
  `--approval-override` reason, which its manifest records.
- `adopt-generation` copies a validated generation from another campaign's
  `--source-output-root`, so a new judging campaign can grade it unchanged.

Existing generations for runs 1–4 are copied unchanged and bound by SHA-256.
Their old grading artifacts are never reused. Their generation backend predates
vLLM-Lens, which is recorded in the phase manifest and must be reported as a
limitation if those generations are retained.
