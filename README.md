# TrustMI: Causally controlling how assistants trust their users

[![arXiv](https://img.shields.io/badge/arXiv-2610.06064-b31b1b.svg)](https://arxiv.org/abs/2610.06064)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-TrustMI-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/TrustMI)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-yellow.svg)](LICENSE)
[![Data license: CC BY 4.0](https://img.shields.io/badge/data%20license-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)

Code and data for the paper
[*TrustMI: Causally controlling how assistants trust their users*](https://arxiv.org/abs/2610.06064)
(Lasnier, Froger, Lasbordes and Seddah, 2026).

<p align="center">
  <img src="assets/hero_figure.png" width="85%" alt="An agent asked for its user's March spending reads a transaction list that hides an instruction to send money to another account. Steered to distrust (α = −2) it ignores the instruction and makes no transfer; unsteered (α = 0) it sends 50€; steered to trust (α = +2) it sends 100€.">
</p>

*Steering trust modulates susceptibility to indirect prompt injections. Under
trust steering (h<sub>ℓ</sub> ← h<sub>ℓ</sub> + α·v<sub>ℓ</sub>), Qwen3.5-27B
evaluates a tool output containing an injection to send money to an
unauthorized account. Lowering trust (α = −2) lets the agent identify the
threat, refuse the transfer and complete the user's original request; raising
it (α = +2) leads to compliance with the malicious payload.*

LLM assistants constantly decide whether to trust users and third parties whose
competence, intentions and integrity they cannot verify, and trusting the wrong
party can lead an agent to comply with a harmful request or act on a malicious
instruction met during tool use. Following Mayer, Davis & Schoorman (1995), we
define trust as an assistant's willingness to accept vulnerability to another
party's actions, and show that it can be causally controlled through model
activations: steering matrices learned from contrastive conversations shift
trust decisions monotonically in both directions across six instruction-tuned
models from three families, and the effect carries over to safety-relevant
agent settings.

## How it works

1. **Data** ([`data/`](data/)): 2,000 contrastive conversations spanning
   ability, benevolence and integrity. Each pair of replies completes the same
   request but differs in whether the assistant trusts the user. The corpus is
   also translated into Chinese, French and Spanish.
2. **Steering** ([`interpretability/`](interpretability/)): one steering vector
   per decoder layer, learned with BiPO while the model's own weights stay
   frozen.
3. **Evaluation** ([`tasks/`](tasks/)): Trust-Elo measures how much trust each
   steering strength buys; the benchmarks measure what that does to harmful
   requests (AgentHarm), prompt injections (AgentDojo) and insider threats
   (Agentic Misalignment), with benign-task and reasoning benchmarks as
   controls.

## Repository layout

| Path | Contents |
| --- | --- |
| [`data/`](data/) | Generates the corpus the vectors are trained on: conversations whose final assistant reply either trusts the user or does not, and their translations. |
| [`interpretability/`](interpretability/) | Trains the steering vectors with BiPO ([`steering-vector-train.py`](interpretability/steering-vector-train.py)); the trained vectors are in [`interpretability/data/steering_vectors/`](interpretability/data/steering_vectors/). |
| [`tasks/trust_elo/`](tasks/trust_elo/) | Trust-Elo: turns a steering vector into an Elo rating on the benevolence test split, with a forced-choice judge and a Bradley-Terry fit. The paper's Trust-Elo results are in [`tasks/trust_elo/reports/trust-elo-results.csv`](tasks/trust_elo/reports/trust-elo-results.csv). |
| [`tasks/benchmarks/`](tasks/benchmarks/) | Safety and capability benchmarks run through Inspect, and the paper's results on them. |
| [`tasks/utils/`](tasks/utils/) | Code both evaluations share: the vLLM-Lens backend, steering targets and span helpers, and the local judge server. |

## Setup

Each of `data/`, `interpretability/` and `tasks/` is its own
[uv](https://docs.astral.sh/uv/) project on Python 3.12:

```bash
(cd data && uv sync)                          # corpus generation
(cd interpretability && uv sync)              # training: Linux with NVIDIA GPUs
(cd tasks && uv sync --extra inspect-suite)   # evaluation: Linux with NVIDIA GPUs
```

## Reproducing the paper

The paper's corpus, steering vectors and results are all in the repository, so
any step can start from the previous step's published output.

1. Generate a corpus with any LiteLLM model; it is written to
   `data/data/benevolence/<model-name>/`. The paper's, made with Claude Opus 5,
   is `data/data/benevolence/opus-5/`.

   ```bash
   cd data && uv run benevolence.py <litellm-model>
   ```

2. Train a steering vector on it; vectors are written to
   `interpretability/data/steering_vectors/`.

   ```bash
   cd interpretability && uv run steering-vector-train.py \
       -m Qwen/Qwen3.5-9B -d benevolence --data_dir ../data/data/benevolence/opus-5
   ```

3. Evaluate it with Trust-Elo and the benchmarks, as described in
   [`tasks/README.md`](tasks/README.md). The Trust-Elo results for every
   vector in `interpretability/data/steering_vectors/` except
   Meta-Llama-3-8B-Instruct, which was not evaluated, are in
   [`tasks/trust_elo/reports/trust-elo-results.csv`](tasks/trust_elo/reports/trust-elo-results.csv):
   one row per vector, judging run and steering strength (the unsteered
   baseline and α = ±1, ±2), with the Elo rating, trust score, their 95%
   confidence intervals and coherence counts. `judged_by` names the judging
   run that produced each row; four vectors were judged twice, and rows from
   different judging runs should not be compared directly. The settings behind
   every published benchmark score, including each model's vector, are in
   [`tasks/benchmarks/reports/README.md`](tasks/benchmarks/reports/README.md).

Pass `-h` to any script for its full list of options.

## Citation

```bibtex
@misc{lasnier2026trustmicausallycontrollingassistants,
      title={TrustMI: Causally controlling how assistants trust their users},
      author={Théo Lasnier and Romain Froger and Maxence Lasbordes and Djamé Seddah},
      year={2026},
      eprint={2610.06064},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.06064},
}
```

## License

The code is under the [MIT License](LICENSE). The data, steering vectors and
results are under [CC BY 4.0](LICENSE-DATA). Content reproduced from
third-party benchmarks keeps its own license, and using a steering vector also
requires the model it was trained on, under that model's license.
