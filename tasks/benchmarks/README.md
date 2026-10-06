# Benchmarks

This package evaluates a TrustMI-steered local model with upstream
`inspect-evals` tasks:

- `model.py` adapts the in-process vLLM-Lens backend used by Trust Elo to Inspect.
- `suite.py` validates JSON suite definitions and loads task factories.
- `run_eval.py` runs one evaluation and multiple steering strengths.
- `run_eval_parallel.py` assigns those strengths to independent local GPUs.
- `run_suite.py` runs selected evaluations locally, one after another.
- `prefetch.py` materializes version-pinned upstream datasets for offline runs.

Every entry point takes the model, vector, devices and output location on the
command line. A run needs Linux and at least two CUDA GPUs: one for the steered
model, and one with 80 GB or more for the `gpt-oss-120b` judge.

The suite is `configs/paper.json`: AgentHarm (harmful and benign),
AgentDojo (with and without prompt injections), Agentic Misalignment (three
scenarios under four conditions), τ²-bench airline, GPQA Diamond, BBEH mini,
and BFCL, each at steering strengths −2 to +2.
[`reports/README.md`](reports/README.md)
records every setting the published runs used. The published scores are in
[`reports/main_results/scores_with_ci.csv`](reports/main_results/scores_with_ci.csv),
with qualitative examples in
[`reports/qualitative_examples`](reports/qualitative_examples/README.md).
[`reports/qwen_agentdojo_user_steering_ablation`](reports/qwen_agentdojo_user_steering_ablation/scores_with_ci.csv)
is an ablation: Qwen3.5-9B and Qwen3.5-27B on AgentDojo with only user messages
steered.

All launch paths accept the same `--steering-target` setting as Trust Elo:

- `latest_user` (default) steers only the latest user message in each request.
- `latest_user_and_tools` steers the latest user message and every subsequent
  tool result in the same agentic turn.
- `all_users` steers every user message when the full context is prefetched.
- `all_users_and_tools` steers every user message and tool result in the prompt.
- `generated_assistant` steers tokens generated after the first assistant token.

The prompt-span targets require a user-trained vector; `generated_assistant`
requires an assistant-trained vector. The published results use
`all_users_and_tools`, which must be passed explicitly.

## Local execution

Resolve the `tasks` environment and validate the suite without loading a model:

```bash
cd tasks
uv sync --extra inspect-suite
VECTOR=../interpretability/data/steering_vectors/20260913-125025_Qwen3.5-9B_benevolence_Lall_user/Qwen3.5-9B_benevolence_Lall_user_trust.pt
uv run python -m benchmarks.run_suite \
  --suite-config benchmarks/configs/paper.json \
  --model Qwen/Qwen3.5-9B \
  --vector "$VECTOR" \
  --dry-run
```

Run one two-sample baseline smoke test. AgentHarm requires the suite's pinned
`gpt-oss-120b` judge, so the parallel runner starts a local judge server on a
separate GPU:

```bash
uv run python -m benchmarks.run_eval_parallel \
  --suite-config benchmarks/configs/paper.json \
  --eval-name agentharm_benign \
  --model Qwen/Qwen3.5-9B \
  --vector "$VECTOR" \
  --strengths 0 \
  --devices 0 \
  --auxiliary-model-path /path/to/gpt-oss-120b \
  --auxiliary-devices 1 \
  --limit 2
```

Run every evaluation with the vector added to user messages and tool results,
as in the published users+tools results, using one model process per GPU:

```bash
uv run python -m benchmarks.run_suite \
  --suite-config benchmarks/configs/paper.json \
  --model Qwen/Qwen3.5-9B \
  --vector "$VECTOR" \
  --steering-target all_users_and_tools \
  --devices 0 1 2 \
  --auxiliary-model-path /path/to/gpt-oss-120b \
  --auxiliary-devices 3
```

A model too large for one GPU takes `--tensor-parallel-size`: each strength
worker then holds that many devices, so `--tensor-parallel-size 2 --devices 0 1 2 3`
runs two strengths at a time.

Each evaluation gets its own output directory. Every run records the suite
hash, task arguments, metric directions, model/vector identity, raw Inspect
logs, tidy metrics, and a Markdown summary. The suite pins every judge,
grader, and simulator role to `gpt-oss-120b` with low reasoning effort.
Auxiliary grading uses temperature zero and the run seed to reduce grader
variance. The auxiliary model runs unsteered behind a local vLLM endpoint and
never shares a target-model process or GPU. Pass `--auxiliary-base-url` instead
of `--auxiliary-model-path` to connect to an already-running compatible server.

The provider supports text messages and structured tools. Multimodal
benchmarks are intentionally absent because token-span steering is currently
defined only for text chat prompts.

To run without internet access, download the pinned datasets on a connected
machine first:

```bash
uv run python -m benchmarks.prefetch \
  --suite-config benchmarks/configs/paper.json \
  --cache-dir /path/to/inspect-evals-cache
```

Then point the offline machine at that directory:

```bash
export INSPECT_EVALS_CACHE_DIR=/path/to/inspect-evals-cache
export HF_HOME="$INSPECT_EVALS_CACHE_DIR/huggingface"
export HF_HUB_OFFLINE=1
```

GPQA comes from its gated Hugging Face dataset: accept its terms and log in to
Hugging Face before prefetching. BFCL comes from its pinned upstream wheel.

The benchmarks run through upstream `inspect-evals`, with the adjustments listed
in [`reports/README.md`](reports/README.md#harness-adjustments-to-the-benchmarks).
