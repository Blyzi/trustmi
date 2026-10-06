# Tasks

The evaluations of the TrustMI steering vectors. Both run the steered model
in-process on vLLM, with [vLLM-Lens](https://github.com/UKGovernmentBEIS/vllm-lens)
hooks adding the vector over exact token spans, and use `gpt-oss-120b` as the
judge.

| Folder | Contents |
| --- | --- |
| [`trust_elo/`](trust_elo/) | Trust-Elo: turns a steering vector into an Elo rating on the benevolence test split, with a forced-choice judge and a Bradley-Terry fit. |
| [`benchmarks/`](benchmarks/) | Safety and capability benchmarks run through Inspect (AgentHarm, AgentDojo, Agentic Misalignment, τ²-bench, GPQA Diamond, BBEH, BFCL), and the paper's results on them. |
| [`utils/`](utils/) | Code both evaluations share: the vLLM-Lens backend, steering targets and span helpers, and the local judge server. |
| [`tests/`](tests/) | Unit tests for all of the above. |

## Requirements

- Linux with NVIDIA GPUs: one for the steered model and one with 80 GB or more
  for the judge. PyTorch comes from the CUDA 13.2 wheel index.
- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- The rest of this repository. The vectors live in
  `interpretability/data/steering_vectors/`, and the token spans to steer come
  from `interpretability/utils.py`, so a vector is applied exactly where
  training fitted it.

## Setup

```bash
cd tasks
uv sync --extra inspect-suite
```

Plain `uv sync` is enough for Trust-Elo; the `inspect-suite` extra adds the
benchmarks. Run every command from `tasks/`, the import root for `trust_elo`,
`benchmarks` and `utils`.

## Running

- **Trust-Elo:** [`trust_elo/README.md`](trust_elo/README.md) describes the
  campaign (generation, rubric calibration, grading).
- **Benchmarks:** [`benchmarks/README.md`](benchmarks/README.md) describes the
  suite; the published scores and settings are in
  [`benchmarks/reports/`](benchmarks/reports/README.md).

## Tests

```bash
uv run python -m unittest discover -s tests
```

Some tests import vLLM, so run them in the full Linux environment.
