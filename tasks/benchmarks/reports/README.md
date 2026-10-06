# Published Inspect results

| Folder | Contents |
|---|---|
| [`main_results/`](main_results/scores_with_ci.csv) | `scores_with_ci.csv`, the source of every Inspect score in the paper: six models, all 20 evaluations, strengths −2 to +2, with the vector added to every user message and tool result. |
| [`qualitative_examples/`](qualitative_examples/README.md) | Ten matched −2/0/+2 transcripts from Qwen3.5-27B and OLMo-3.1-32B, selected after the fact to illustrate mechanisms. |
| [`qwen_agentdojo_user_steering_ablation/`](qwen_agentdojo_user_steering_ablation/scores_with_ci.csv) | Ablation: Qwen3.5-9B and Qwen3.5-27B on AgentDojo with the vector added to user messages only. |

All runs used the `paper` suite ([`configs/paper.json`](../configs/paper.json)) in
September 2026, and every value below is taken from that config or from the
manifests the runs wrote. The main results and the ablation share every setting
except where the vector is added: `all_users_and_tools` for the main results,
`all_users` for the ablation.

Since the runs, the package was renamed from `trustmi_inspect` to `benchmarks`
and the suite from `cross-model-gold-v1` to `paper` (run records such as
`qualitative_examples/run_identity.json` keep the old names), and the config's
descriptions and tags were reworded; no setting changed.

## Models and steering vectors

| Model | Hugging Face ID | Steering vector (BiPO run) | Recipe |
|---|---|---|---|
| OLMo-3-7B | `allenai/Olmo-3-7B-Instruct` | `20260913-152358_Olmo-3-7B-Instruct_benevolence_Lall_user_rpo0.5` | rpo 0.5 |
| OLMo-3.1-32B | `allenai/Olmo-3.1-32B-Instruct` | `20260913-184027_Olmo-3.1-32B-Instruct_benevolence_Lall_user_rpo0.5` | rpo 0.5 |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | `20260913-125025_Qwen3.5-9B_benevolence_Lall_user` | none |
| Qwen3.5-27B | `Qwen/Qwen3.5-27B` | `20260913-160552_Qwen3.5-27B_benevolence_Lall_user` | none |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` | `20260922-004926_Llama-3.1-8B-Instruct_benevolence_Lall_user_rpo0.2` | rpo 0.2, lr 1e-4 |
| Llama-3.1-70B | `meta-llama/Llama-3.1-70B-Instruct` (rev `1605565b47bb`) | `20260922-080717_Meta-Llama-3.1-70B-Instruct_benevolence_Lall_user_rpo0.2` | rpo 0.2, lr 1e-4 |

All vectors were trained on the **user span**, with one row per decoder layer
(`Lall`), and are evaluated at their final checkpoint: the `*_trust.pt` file in the
run's folder under `interpretability/data/steering_vectors/`.

## Steering

| Parameter | Value |
|---|---|
| Strengths | −2, −1, 0, +1, +2 (multiples of the trained vector) |
| Strength 0 | no vector attached: the unmodified model |
| Layers | every decoder layer where the vector is non-zero |
| Positions, users suite | content tokens of every user message |
| Positions, users+tools suite | content tokens of every user message and every tool result |
| When | prompt (prefill) tokens only; generated tokens are not steered |
| Span location | diff of the full rendered prompt against the same prompt with that message emptied (template-agnostic; role markers, system prompt and tool definitions are never steered) |
| Backend | vLLM with vllm-lens hooks |

## Decoding and harness (all evals unless overridden)

| Parameter | Value |
|---|---|
| Temperature | 0.8 |
| Max new tokens per generation | 4096 |
| Context length (`max_model_len`) | 32,768 |
| Seed | 104729, offset per epoch so repeated epochs are independent samples; seeding does not make runs bit-for-bit reproducible, since vLLM batches concurrent requests |
| Thinking / reasoning mode | off |
| Message limit per sample | 20 (tau2: none) |
| Sample failure policy | `fail_on_error = true` (any errored sample fails the arm, which is rerun) |
| System prompt | the benchmark's own; none is added |
| Tool-call parser | Qwen3.5: `qwen3_coder`; OLMo-3: `olmo3`; Llama-3.1: `llama3_json` |

## Evaluations

| Eval | Benchmark / split | Unique items | Epochs | Reported metrics |
|---|---|---|---|---|
| agentdojo_injected | AgentDojo, with prompt injections, no sandbox tasks | 944 | 3 | attack success (↓), utility (↑) |
| agentdojo_benign | AgentDojo, no injections, no sandbox tasks | 96 | 3 | utility (↑) |
| agentharm_harmful | AgentHarm `test_public`, harmful | 176 | 3 | harm score (↓), full harmful completion (↓), refusal rate (↑) |
| agentharm_benign | AgentHarm `test_public`, benign | 176 | 3 | task score (↑), full completion (↑), refusal rate (↓) |
| agentic_misalignment (×12) | 3 scenarios (blackmail, leaking, murder) × 4 conditions: explicit/replacement, latent/replacement, explicit/restriction, none/replacement; goal value "america" | 1 prompt each | 25 | harmful behavior rate (↓) |
| tau2_airline | τ²-bench airline | 50 | 5 | task success (↑); 1,800 s time limit per sample |
| bfcl_core | BFCL: simple_python, multiple, parallel, parallel_multiple, irrelevance, multi_turn_base, multi_turn_miss_func, multi_turn_miss_param | 1,840 | 3 | tool-call accuracy (↑), also per category |
| gpqa_diamond | GPQA Diamond | 198 | 3 | accuracy (↑) |
| bbeh_mini | BIG-Bench Extra Hard, mini | 460 | 3 | accuracy (↑) |

AgentHarm's full (harmful) completion counts a prompt only when it scores fully in
all three epochs: Inspect averages a prompt's epochs before AgentHarm checks for a
full score.

## Auxiliary model (judges and simulated user)

| Parameter | Value |
|---|---|
| Model | `openai/gpt-oss-120b`, served locally with vLLM on one H200 |
| Roles | tau2 simulated user; AgentHarm refusal and semantic judges; agentic misalignment grader |
| Reasoning effort | low |
| Max tokens | 2048 |
| Context length | 32,768 |

The same auxiliary model serves every strength of a run, so it never sees the
steering vector.

## Confidence intervals

95% intervals are `value ± 1.96 × SE`, clipped to [0, 1]. The standard error is taken
over per-item scores after averaging each item's epochs, so repeated epochs of one
prompt are not counted as independent. Where a scorer reports no standard error
(AgentHarm, BBEH), it is recomputed the same way from the per-item scores in the
evaluation logs; this reproduces the reported standard error exactly on scorers that
do report one. BFCL's per-category standard errors come from its scorer.

## Software and hardware

| Component | Version |
|---|---|
| inspect_ai | 0.3.263 |
| inspect_evals | 0.17.0 |
| vLLM | 0.27.1 |
| vllm-lens | 1.2.1 |
| transformers | 5.14.1 |
| GPUs | NVIDIA H200 141 GB; one GPU per strength, two for Llama-3.1-70B (tensor parallel) |

## Harness adjustments to the benchmarks

Each makes a crash into a scored outcome and changes no result that would otherwise
have been produced (every affected path previously raised and failed its sample):

- AgentDojo: a tool call with invalid arguments returns a tool error to the model
  instead of aborting the sample; a security or utility check that finds none of the
  state it looks for (for example a missing inbox) scores the goal as not met; calendar
  events attached to emails can be compared with file IDs.
- AgentHarm graders and AgentDojo scorers treat an omitted tool argument as empty.
- Tool calls whose arguments are not a JSON object reach the model as a parse error.
- Unpaired UTF-16 surrogates in model output are replaced with U+FFFD.
- Llama-3.1's chat template cannot express parallel tool calls, so for Llama an
  assistant turn with several calls is rendered as consecutive single-call turns,
  in the same order.
- BFCL: a multi-turn trajectory that ends before its last turn is scored as
  incorrect instead of crashing the scorer.

Two adjustments do change outcomes. AgentHarm's system prompt asks the model to
write tool calls as `<function=name>{json}</function>` text, while each model's chat
template asks for its native format. When a reply contains no native tool call,
calls written in AgentHarm's format are now executed; before, they were left as text
and the grader scored an agent that had called nothing. Llama-3.1-70B follows
AgentHarm's instruction on every turn, so its AgentHarm results come from runs with
this parsing. The other models' AgentHarm runs predate it: in a sample of 20
transcripts per model and eval at strength 0, OLMo and Llama-3.1-8B wrote no calls
in this format, and Qwen3.5 wrote 1–3% of its turns in it.

BFCL multi-turn samples see only the functions the sample declares, each defined for
that sample alone. Upstream also exposed the backend classes' other public methods,
including helpers outside the benchmark's model-facing API, and its tool definitions
could leak between samples running concurrently. Every published BFCL run used this
version.
