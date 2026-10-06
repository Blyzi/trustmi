"""Turn a trust steering vector into an Elo rating, on the benevolence test split.

The question this answers is "how much trust does strength +1 buy, on a scale
with a known top and bottom", rather than "did the judge prefer the steered
reply". It rebuilds an earlier pair of generate and judge scripts around a
forced-choice judge and a Bradley-Terry fit.

Three differences from that pair, all of them deliberate:

  * **Forced choice.** The judge must name A or B. No NEITHER, no ties. Ties are
    what a Bradley-Terry fit has no room for, and on this rubric they were also
    the judge's escape hatch: two replies that both hand over the same artefact
    differ only in whether they lean on the user's unverified word, which is a
    real difference the judge can always resolve if it has to.
  * **Contrastive rows only.** The control subset is skipped: its two poles are
    the same reply, so it carries no trust contrast to rate.
  * **Elo.** Every comparison is one game between two *conditions*, and the
    conditions are fitted jointly instead of each being reported as its own win
    rate against the baseline. That buys the thing win rates cannot give: a
    common scale. With `--anchors` the dataset's own trust and distrust poles
    play the baseline too, so they land on that same scale and a steered
    condition can be read off as a fraction of the distance between them —
    "+1 gets you 40% of the way from the distrust pole to the trust pole" says
    something a 62% win rate does not.

What is compared. For every held-out row the model writes the final assistant
reply once per `--strengths` value, from the same prompt, so the only thing that
differs between a row's generations is the injection. Decoding is greedy unless
`--temperature` is set, and `--num_generations k` draws k continuations per
(row, strength) instead of one — the baseline included, so both arms of every
comparison carry decoding noise rather than a single greedy baseline standing
in for the model. Draw k plays the baseline's draw k, so k multiplies the games
rather than squaring them. Strength 0
attaches no hook at all, so the baseline is the unmodified model, and it is the
opponent in every game: steered-vs-baseline for each non-zero strength, plus,
under `--anchors`, trust-pole-vs-baseline, distrust-pole-vs-baseline and
trust-pole-vs-distrust-pole. That last pair is also the calibration read — its
direction is known, so the share the judge gets right is a ceiling on what any
other number here can show.

Two judging passes, in this order.

**One.** Every distinct reply is graded on its own, in its own call, with no
other reply in the context: a NOTE of at most fifty words, then COHERENT, then a
trust SCORE from 0 to 100. `<stem>-scores.jsonl` carries one line per reply and
the scores are aggregated per strength.

**Two.** The replies that came back coherent are then compared in pairs against
that row's own strength-0 baseline — a NOTE, then which of the two extends more
trust, and nothing else. Every pair is judged twice with the slots swapped,
which is the position-bias control and what `position_bias` reads its estimate
off. A reply that failed pass one plays no games at all, so the pairwise judge
is only ever handed two usable replies and has one thing to decide. It keeps
the grade pass one gave it, though, and by default that grade counts in the
per-condition score like any other; `--score_coherent_only` averages the
coherent replies alone, the set the Elo was fitted on.

COHERENT drops a reply that is mechanically broken — a repetition loop,
disfluent output, a fragment that stops mid-sentence — and also one that does
not answer the question put to it: changes the subject, dodges the request, or
stays general where a specific answer was wanted. **Read the coherent rate per
strength before reading anything else.** Withholding trust often looks like not
answering head-on, so this check can remove the low end of the trust axis from
the games, and a strength rated on a fraction of its replies is described by
whichever fraction stayed answerable. That is why the score, by default, counts
every reply: it is the one number here the gate does not reach.

The scale is `data/benevolence.py`'s trust axis. Its two ends are that
generator's DISTRUSTFUL and TRUSTFUL endings and its definition of trust is
quoted whole, direction included — but only the definition. The criteria that
generator uses to accept or reject its own rows exist to keep a synthetic pair
clean and say nothing about where a real steered reply sits on the axis, so none
of them are here. Grading the dataset's own poles through this same path is the
calibration read: a run whose poles come back at 30 and 60 has said its judge
resolves under a third of what the data varies, and nothing else in the table is
worth much.

What the score adds over the Elo is level. Elo is differences only, the anchor
putting the baseline at 1500 by fiat, so it cannot say whether a strength failed
to move because the vector did nothing or because the baseline already sat at
the top of the scale. Two numbers keep the two passes honest against each other:
`cross_check` asks how often the grades agree with the forced choice, which is
two instruments with no shared context rather than one call agreeing with
itself, and `--score_repeats N --score_temperature T` measures the judge's
scatter on one unchanged reply, below which a strength gap is not a measurement.

The judge layer — both prompts, the parse, and one call of each kind — is in
`judging.py`; this file is what is done with the answers. It is not named
`utils`, which `tasks/` already owns on this sys.path.

How the vector is added. Generation runs on stock vLLM with the `vllm-lens`
plugin, through `utils/vllm_lens_wrapper.py`, which adds the vector with one
persistent `torch` forward hook per decoder layer.

The hook receives this request's own slice of the flat `[total_tokens, hidden]`
batch and carries its own absolute offset by counting the tokens the request has
been scheduled for, which is what makes a chunked prefill harmless and retires
the in-flight cap that `--batch_size` used to be clamped to. See
`make_span_hook` for the arithmetic.

The price is reproducibility. Interleaving is scheduler-dependent, so two runs
of one command differ, on top of greedy decoding not being bitwise stable across
positions within a vLLM batch. Neither is new — the sweep was already built to
tolerate it, which is what the shuffle below is for — but do not expect a rerun
to reproduce a JSONL byte for byte. The HuggingFace path both engines replaced
is still the reference the span is checked against, and still lives in
`interpretability/utils.py`.

Where the vector goes. `--steering-target latest_user` adds it at the latest
user turn's content tokens during prefill; `latest_user_and_tools` also adds it
to each later tool-result span; `all_users` adds it at every user turn; and
`all_users_and_tools` adds it at every user and tool-result span.
`generated_assistant` adds it to tokens the model writes. Prompt-span targets
steer how the model reads the scenario and reach the answer through the KV cache.
They require a user-trained vector, while `generated_assistant` requires an
assistant-trained vector; the script warns when they disagree.

Two phases, because the halves want different machines. Generation needs a GPU
and the model, so it is a batch job on a network-less compute node; the judge is
worth running from a login node where it can be a strong external model rather
than the one that wrote the text. Running with `--judge_model` and no `--from`
does both in one go, which is right when the judge is served locally.

Outputs. All four artefacts land in `data/` beside this file and share one
stem, which carries the vector's **training-run id** — the timestamp of the
directory `steering-vector-train.py` wrote the checkpoint into, plus its step
when the vector is a checkpoint, which is also the name of that run's
TensorBoard directory. A curve, the checkpoints it describes and the results of
evaluating them are then all found by one string. `generate` also records the id
inside the generations JSONL, so a `--from` pass names its outputs after the run
even when the caller chose the filename (the sbatch launcher names generations
after the SLURM job, which is unique but says nothing about which vector was
evaluated); the id is appended to that name rather than replacing it, so
uniqueness and provenance both survive. The four are the generations JSONL, `<stem>-verdicts.jsonl` with the
per-game verdicts, `<stem>-scores.jsonl` with one line per reply carrying its
score and its text, and `<stem>.json` with the ratings and the summary numbers.
`logs/` is left to the job's own stdout and the vLLM server log. `--out`,
`--verdicts`, `--scores_out` and `--results` each override one of the four.

Usage:
  # phase 1, on a compute node, from tasks/
  uv run trust_elo/main.py \\
      -v ../interpretability/data/steering_vectors/<run>/<vec>.pt \\
      -m Qwen/Qwen3.5-9B \\
      --data_dir ../data/data/benevolence/gemma-4-31B-it \\
      --steering-target latest_user --strengths -1 0 1
  # phase 2, from a login node, no GPU and no model load
  uv run trust_elo/main.py --from data/trust-elo-<vec>.jsonl \\
      --judge_model anthropic/claude-sonnet-4-5
  # both at once against a local judge server
  uv run trust_elo/main.py -v <vec>.pt --data_dir <dir> \\
      --judge_model hosted_vllm/Qwen/Qwen3.5-9B --api_base http://127.0.0.1:8001/v1
"""

from __future__ import annotations

import json
import logging
import math
import random
import statistics
import sys
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

# tasks/ is the import root; utils lives there.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stdlib only, so importing it costs nothing on the judge-only path.
from utils.run_id import vector_run_id  # noqa: E402

# The judge layer: prompts, parse, and one call of each kind. Deliberately not
# `utils`, which `tasks/` already owns on this sys.path -- see judging.py.
from judging import (  # noqa: E402
    SCORE_MAX,
    SCORE_MIN,
    grade_one,
    play,
)

# litellm's own logger inherits the root level and emits noisy per-request
# internal messages; pin it above INFO.
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Where generated artefacts go when no path is given -- all three of them: the
# generations JSONL, the per-game verdicts JSONL and the summary JSON. Kept
# apart from logs/, which holds the job's own stdout and the vLLM server log;
# this is the run's data. Relative to the cwd, which for this task is
# tasks/trust_elo/.
DEFAULT_OUT_DIR = Path("data")

# The engine's context window. Spelled out here rather than imported from
# `utils.vllm_lens_wrapper`, whose module-level `from vllm import SamplingParams`
# would pull vLLM in at argument-parsing time and break the judge-only `--from`
# path on a login node. `generate` checks the two agree once the import is paid
# for anyway.
DEFAULT_MAX_MODEL_LEN = 32768

BASELINE = "baseline"
TRUST_POLE = "dataset_trust_pole"
DISTRUST_POLE = "dataset_distrust_pole"

# Elo's own constant: 400 points is a 10:1 odds ratio, so a Bradley-Terry
# log-strength converts at 400/ln(10).
ELO_SCALE = 400.0 / math.log(10.0)
ELO_ANCHOR = 1500.0


# --------------------------------------------------------------------------- #
# Phase 1: steered generation
# --------------------------------------------------------------------------- #


def rows_run_id(rows: list[dict]) -> str | None:
    """The training-run id a generations JSONL records, if it carries one.

    `generate` stamps every row, so a file judged later under `--from` can still
    be named after the run that produced it whatever the file itself is called.
    Files written before that field existed carry nothing and fall back to
    their own stem rather than being given an id from somewhere else.
    """
    for row in rows:
        found = row.get("vector_run_id")
        if found:
            return str(found)
    return None


def out_stem(generations_path: Path, run_id: str | None) -> str:
    """The stem this run's four artefacts share.

    The vector's training-run id is in it whenever one is knowable, which is the
    whole point of naming outputs after the run rather than after the vector
    file: the TensorBoard curve under `runs/`, the checkpoints in that run's
    directory and these results then all answer to one string.

    A caller that named the generations file itself keeps that name and gains
    the run id beside it, so both properties hold at once -- uniqueness from the
    caller (the sbatch launcher uses the SLURM job id, and two evaluations of
    one checkpoint must not overwrite each other) and provenance from here. A
    stem that already contains the id is left alone, so the default path and a
    launcher that already builds the id in are unchanged.
    """
    stem = generations_path.stem
    if run_id and run_id not in stem:
        return f"{stem}-{run_id}"
    return stem


def load_contrastive_test(data_dir: Path, split: str) -> list[dict]:
    """Read the contrastive subset's held-out split.

    The control subset is not read at all. Its two poles are the same reply, so
    it holds no trust contrast to rate and would only add games between a
    condition and itself.
    """
    from datasets import load_from_disk

    path = data_dir / "contrastive"
    if not path.is_dir():
        raise SystemExit(
            f"no contrastive/ subset under {data_dir}. --data_dir must point at "
            "the directory benevolence.py wrote (the one holding contrastive/ "
            "and neutral/)."
        )
    ds = load_from_disk(str(path))
    if split not in ds:
        raise SystemExit(
            f"{path} has no '{split}' split (it has {list(ds)}). A dataset "
            "generated with --test_ratio 0 holds nothing out."
        )
    return list(ds[split])


def generate(
    vector_path: Path,
    model_name: str,
    data_dir: Path,
    split: str,
    layer_filter: list[int] | None,
    strengths: list[float],
    steering_target: str,
    max_new_tokens: int,
    think: bool,
    num_samples: int | None,
    sample_offset: int,
    batch_size: int | None,
    out_path: Path,
    num_generations: int = 1,
    temperature: float = 0.0,
    seed: int = 0,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
) -> list[dict]:
    """Write one generation per (row, strength) to `out_path` and return them.

    Imported lazily: torch, vLLM and the model live behind this call, and the
    judge-only path (--from) has no use for any of them and may be running on a
    login node.

    Generation goes through `utils/vllm_lens_wrapper.py`, which owns the engine,
    its thread and its event loop; this function only ever hands it
    rendered prompt layouts and strengths and gets text back.
    """
    from utils.steering import (
        build_steered_prompt,
        load_steering_vector,
        trained_span_tag,
    )
    from utils.steering_policy import (
        resolve_steering_ranges,
        training_span_for_target,
        validate_steering_target,
    )

    from utils.vllm_lens_wrapper import (
        DEFAULT_MAX_MODEL_LEN as WRAPPER_DEFAULT_MAX_MODEL_LEN,
        VLLMLens,
        default_tensor_parallel_size,
        num_decoder_layers,
    )

    # Two spellings of one number, kept honest here rather than left to drift:
    # the copy above is what --help prints and what a caller of this function
    # gets by default, the wrapper's (which it re-exports from
    # `utils/vllm_common.py`) is what every other in-process engine here runs on.
    assert DEFAULT_MAX_MODEL_LEN == WRAPPER_DEFAULT_MAX_MODEL_LEN, (
        f"default max_model_len disagrees: {DEFAULT_MAX_MODEL_LEN} here vs "
        f"{WRAPPER_DEFAULT_MAX_MODEL_LEN} in utils/vllm_common.py"
    )

    rows = load_contrastive_test(data_dir, split)
    if sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")
    rows = rows[sample_offset:]
    if num_samples is not None:
        rows = rows[:num_samples]
    if not rows:
        raise SystemExit(f"no rows in the {split} split under {data_dir}")

    if tensor_parallel_size is None:
        tensor_parallel_size = default_tensor_parallel_size()

    # The provider owns the engine, its thread and its event loop; this script
    # only ever hands it (ids, span, strength) triples. `default_model` preloads
    # eagerly so the engine is up before anything below needs the tokenizer, and
    # `max_loaded_models=1` makes a typo in --model a clear error rather than a
    # second 18 GB of weights.
    provider = VLLMLens(
        default_model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_loaded_models=1,
    )
    tokenizer = provider.get_tokenizer(model_name)
    # Layer count from the checkpoint config rather than from a model object:
    # stock vLLM keeps its decoder layers in a worker process, so there is
    # nothing here to count. The row-selection policy stays in
    # `interpretability/utils.py` where the other consumers read it.
    vector, layer_rows = load_steering_vector(
        vector_path, layer_filter=layer_filter,
        n_layers=num_decoder_layers(model_name),
    )

    print(f"backend   vllm-lens (async), tensor-parallel "
          f"{tensor_parallel_size}")
    print(f"vector    {vector_path.name}")
    print(f"layers    {layer_rows}")
    steering_target = validate_steering_target(steering_target)
    print(f"target    {steering_target}")
    print(f"rows      {len(rows)} (contrastive/{split})")
    print(f"strengths {strengths}")
    print(f"draws     {num_generations} per (row, strength) at temperature "
          f"{temperature}" if num_generations > 1 or temperature > 0
          else "draws     1 per (row, strength), greedy")
    print("batch     uncapped; vLLM schedules what fits" if batch_size is None
          else f"batch     up to {batch_size} requests in flight")
    if num_generations > 1 and temperature <= 0:
        raise SystemExit(
            f"--num_generations {num_generations} with temperature 0 would write "
            "the same string that many times; pass --temperature > 0"
        )

    # The filename records the span the vector was trained on. Applying it over a
    # different one produces plausible text rather than an error, so say so
    # loudly rather than letting a mismatched sweep be read as a result.
    trained_on = trained_span_tag(vector_path)
    required_training_span = training_span_for_target(steering_target)
    if trained_on is None:
        print("warning:  filename carries no span tag; cannot check that "
              f"--steering-target {steering_target} matches vector training")
    elif trained_on != required_training_span:
        print(f"warning:  this vector was trained on the {trained_on} turn but is "
              f"being applied to {steering_target}, which requires a "
              f"{required_training_span}-trained vector")

    encoded = []
    for row in rows:
        context = list(row["context"])
        try:
            layout = build_steered_prompt(tokenizer, context, think)
            resolve_steering_ranges(
                steering_target,
                len(layout.token_ids),
                layout.user_spans,
                layout.tool_spans,
            )
        except ValueError as exc:
            # A row whose span cannot be located is not one this vector can be
            # applied to; drop it rather than steer the wrong tokens.
            print(f"warning: row {row.get('id')} skipped: {exc}")
            continue
        encoded.append((row, layout))
    if not encoded:
        raise SystemExit("no row produced a usable injection span")

    # Every draw is its own job, including the baseline's: sampling n steered
    # continuations against one greedy baseline would leave all the decoding
    # noise on one arm, and a row whose single baseline draw happened to land
    # trusting would tilt every game in that row.
    #
    # Shuffled rather than sorted by prompt length. The batch is flat and
    # unpadded, so length sorting buys nothing — and a sorted order is actively
    # harmful, because generation is not bitwise stable across positions within
    # a vLLM batch (identical greedy requests diverge late, but they do
    # diverge). Under the sorted order a row's
    # strengths sit at consecutive offsets, so each condition keeps landing in
    # the same position class and a per-position numerical quirk stops being
    # noise and starts being a small offset attached to a condition. Shuffling
    # decorrelates condition from offset, which is all that is needed: the
    # floor stays, but it stays noise.
    jobs = [
        (i, strength, draw)
        for i in sorted(
            range(len(encoded)), key=lambda j: len(encoded[j][1].token_ids)
        )
        for strength in strengths
        for draw in range(num_generations)
    ]
    random.Random(seed).shuffle(jobs)

    # Shuffled, not sorted by length. Generation is not bitwise stable across
    # positions within a vLLM batch, so under a sorted order a row's strengths
    # sit at consecutive offsets, each
    # condition keeps landing in the same position class, and a per-position
    # quirk stops being noise and becomes a small offset attached to a
    # condition. Shuffling decorrelates condition from offset.

    # No prefill-width cap here: a prefill vLLM splits across steps still lands
    # the span on the right tokens.
    #
    # `make_span_hook` derives its absolute position by counting the tokens the
    # request has been scheduled for, so a prefill split across steps adds up to
    # the same offsets and a chunk boundary is invisible to the span. The only
    # thing that breaks that counter is a preemption-and-recompute, and
    # `verify_seen` audits every request against its own token count and raises
    # rather than letting a shifted span read as a result. So `--batch_size` is
    # now purely throughput and nothing here needs to clamp it.
    longest = max(len(layout.token_ids) for _, layout in encoded)
    print(f"prefill   {longest} tokens in the longest prompt (chunking is safe "
          "here; the span is counted, not narrowed)")

    generations: dict[int, dict[str, list[str]]] = {
        i: {f"{s:g}": [] for s in strengths} for i in range(len(encoded))
    }
    with tqdm(total=len(jobs), desc="Generating") as bar:
        texts = provider.generate_steered(
            model_name,
            [(encoded[i][1], strength) for i, strength, _ in jobs],
            vector,
            layer_rows,
            steering_target=steering_target,
            max_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            max_in_flight=batch_size,
            on_done=bar.update,
        )
    for (i, strength, _), text in zip(jobs, texts):
        generations[i][f"{strength:g}"].append(text.strip())

    # An engine holds its KV cache for the life of the process, and the judge
    # half of this script is often a second model served on the same node.
    # Nothing below needs the model, so let it go before the judge starts
    # competing with it. `shutdown_engine` also cancels what vLLM parked on the
    # engine's event loop, which otherwise prints an EngineDeadError traceback
    # after a perfectly good run.
    import gc

    provider.shutdown_engine(model_name)
    del provider
    gc.collect()

    out_rows = []
    for i, (row, layout) in enumerate(encoded):
        ranges = resolve_steering_ranges(
            steering_target,
            len(layout.token_ids),
            layout.user_spans,
            layout.tool_spans,
        )
        prompt_ranges = [(start, end) for start, end in ranges if end is not None]
        out_rows.append(
            {
                **{
                    k: row.get(k)
                    for k in ("id", "kind", "family", "cue", "scenario",
                              "length_style", "num_user_turns")
                },
                "subset": "contrastive",
                "context": list(row["context"]),
                # The dataset's own poles, carried through so they can play the
                # baseline and put the ratings on a scale with a known top and
                # bottom.
                "reference_trust": row["messages_trust"][-1]["content"],
                "reference_distrust": row["messages_distrust"][-1]["content"],
                "steering_target": steering_target,
                "span_tokens": sum(end - start for start, end in prompt_ranges),
                "prompt_tokens": len(layout.token_ids),
                # Which vector wrote these, and the id of the training run it
                # came out of. Carried in the file rather than only in its name
                # so that a --from pass can still name its outputs after the run
                # even when the caller chose the filename -- the sbatch launcher
                # names generations after the SLURM job, which is unique but
                # says nothing about which vector was evaluated.
                "vector": str(vector_path),
                "vector_run_id": vector_run_id(vector_path),
                "generations": generations[i],
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(out_rows)} rows x {len(strengths)} strengths x "
          f"{num_generations} draw(s) to {out_path}")
    return out_rows


# --------------------------------------------------------------------------- #
# Phase 2: judging
# --------------------------------------------------------------------------- #
# The prompts, the parse and one call of each kind live in judging.py; what is
# left here is what is done with the answers.

def draws_at(generations: dict, strength: str) -> list[str]:
    """The continuations a row holds for one strength, always as a list.

    `--num_generations` made this field a list. Files written before that carry
    a bare string, and they are still valid input to `--from`, so one draw and
    the old shape are the same thing here rather than two code paths.
    """
    text = generations.get(strength)
    if text is None:
        return []
    return [text] if isinstance(text, str) else list(text)


def build_games(rows: list[dict], anchors: bool) -> list[dict]:
    """One job per game to play.

    Every steered condition meets that row's own strength-0 baseline, which is
    what makes the row its own control: the row-to-row variation an absolute
    rating would have to average out cancels inside the pair.

    With several draws per condition, draw k meets the baseline's draw k rather
    than every baseline draw: n independent replicates of the matchup, not n^2
    games most of which share a text. The baseline is drawn as many times as the
    steered arm, so both sides carry decoding noise and no row's single lucky
    continuation can tilt every game in that row. A condition with fewer draws
    than the baseline wraps around, which is what lets a one-draw steered arm
    still play an n-draw baseline.

    With `anchors`, the dataset's own poles join in — each against the baseline,
    and against each other. The pole-vs-pole game has a known answer, so it
    measures the judge; the pole-vs-baseline games are what connect the poles to
    the same fitted scale as the steered conditions, and without them the poles
    would form a component of their own that no Elo could compare against. The
    poles are fixed dataset text with nothing to resample, so they play every
    baseline draw but meet each other once per row.

    `draw` indexes the game; `first_draw` and `second_draw` index each *side*,
    which is what identifies the reply that side showed. They come apart wherever
    a game pairs unlike arms: the poles are one fixed text however many baseline
    draws they play, and a steered arm with more draws than the baseline wraps
    around it. Scoring replies needs the per-side index, or one text would be
    counted as several responses and several texts as one.
    """
    games: list[dict] = []
    for i, row in enumerate(rows):
        generations = row.get("generations") or {}
        baselines = draws_at(generations, "0")
        if not baselines:
            continue
        rid = row.get("id")
        for strength in generations:
            if float(strength) == 0.0:
                continue
            for k, text in enumerate(draws_at(generations, strength)):
                games.append(
                    {
                        "row": i,
                        "kind": "sweep",
                        "strength": strength,
                        "draw": k,
                        "first_draw": k,
                        "second_draw": k % len(baselines),
                        "first": f"steer{float(strength):+g}",
                        "second": BASELINE,
                        "first_text": text,
                        "second_text": baselines[k % len(baselines)],
                        "what": f"row{rid}-s{strength}-d{k}",
                    }
                )
        if anchors:
            for k, base in enumerate(baselines):
                games.append(
                    {
                        "row": i, "kind": "anchor", "strength": None, "draw": k,
                        "first_draw": 0, "second_draw": k,
                        "first": TRUST_POLE, "second": BASELINE,
                        "first_text": row["reference_trust"],
                        "second_text": base,
                        "what": f"row{rid}-trustref-vs-base-d{k}",
                    }
                )
                games.append(
                    {
                        "row": i, "kind": "anchor", "strength": None, "draw": k,
                        "first_draw": 0, "second_draw": k,
                        "first": DISTRUST_POLE, "second": BASELINE,
                        "first_text": row["reference_distrust"],
                        "second_text": base,
                        "what": f"row{rid}-distrustref-vs-base-d{k}",
                    }
                )
            games.append(
                {
                    "row": i, "kind": "calibration", "strength": None, "draw": 0,
                    "first_draw": 0, "second_draw": 0,
                    # Known direction: the trust pole must win, so the share it
                    # does is a ceiling on the judge's resolution.
                    "first": TRUST_POLE, "second": DISTRUST_POLE,
                    "first_text": row["reference_trust"],
                    "second_text": row["reference_distrust"],
                    "what": f"row{rid}-ref",
                }
            )
    return games


def expand_orders(games: list[dict]) -> list[dict]:
    """Two judging jobs per game, the replies swapped between them.

    Not optional. Doubling is what makes the slot allocation exactly balanced
    instead of balanced in expectation: every condition meets every opponent
    once from each side, so a per-game coin flip's leftover imbalance — worth
    tens of Elo at the slot preference this judge has — cannot reach the ratings
    at all. It is also the only thing that makes the preference *measurable*:
    `position_bias` reads it off how the doubled matchups split, and the true
    gap between the conditions cancels out of that ratio, so the estimate needs
    no assumption about the effect being measured. A single judgement per game
    gives neither, which is why the choice was removed rather than defaulted —
    every judgement in an undoubled run carries a slot bias nothing can size.

    Both jobs are ordinary games and both enter the fit. A matchup the judge
    splits contributes one win and one loss, which is BT being told the two are
    even for that row; dropping split matchups instead would discard exactly the
    closest comparisons and push every rating away from the baseline.
    """
    jobs = []
    for i, game in enumerate(games):
        for order, first_is_a in ((0, True), (1, False)):
            jobs.append(
                dict(
                    game,
                    game_index=i,
                    order=order,
                    first_is_a=first_is_a,
                    what=f"{game['what']}-o{order}",
                )
            )
    return jobs


def position_bias(jobs: list[dict], results: list) -> dict:
    """Estimate the judge's slot preference off matchups played both ways.

    A doubled matchup lands in one of three states. The judge names the same
    reply both times (it followed the text), or it names the reply shown as A
    both times, or the reply shown as B both times — and naming the same *slot*
    twice is exactly naming a different condition each time, i.e. a split.

    The two kinds of split are what carry the bias. Writing p1 = sigma(delta + g)
    for the order where the first player sits in slot A and p2 = sigma(delta - g)
    for the other, the split counts are p1(1 - p2) and (1 - p1)p2, whose ratio is

        p1(1 - p2) / ((1 - p1)p2) = exp(delta + g) * exp(-(delta - g)) = exp(2g)

    The true gap cancels. So half the log ratio of the two split counts is the
    slot advantage in logits, exactly, with no rating model fitted and no
    assumption about how far apart the conditions really are. Under no
    preference the two are equally common whatever the gap, and g comes out 0.
    """
    per_game: dict[int, list[tuple[dict, str]]] = defaultdict(list)
    for job, res in zip(jobs, results):
        if res is None:
            continue
        verdict, _ = res
        per_game[job["game_index"]].append((job, verdict))

    agree = 0
    locked = {"A": 0, "B": 0}
    for pair in per_game.values():
        # Every game is doubled, so a matchup here with one judgement lost its
        # other one to a judge failure and has no split to read.
        if len(pair) != 2:
            continue
        (_, v0), (_, v1) = pair
        if v0["TRUST_SIDE"] == v1["TRUST_SIDE"]:
            locked[v0["TRUST_SIDE"]] += 1
        else:
            agree += 1

    doubled = agree + locked["A"] + locked["B"]
    out = {
        "matchups_doubled": doubled,
        "agreed": agree,
        "split_a_slot": locked["A"],
        "split_b_slot": locked["B"],
        "slot_advantage_logit": None,
        "slot_advantage_elo": None,
        "slot_advantage_elo_ci95": None,
    }
    if locked["A"] and locked["B"]:
        # Haldane correction keeps the estimate finite when one cell is small.
        a, b = locked["A"] + 0.5, locked["B"] + 0.5
        g = 0.5 * math.log(a / b)
        se = 0.5 * math.sqrt(1.0 / a + 1.0 / b)
        out["slot_advantage_logit"] = g
        out["slot_advantage_elo"] = g * ELO_SCALE
        out["slot_advantage_elo_ci95"] = [
            (g - 1.96 * se) * ELO_SCALE,
            (g + 1.96 * se) * ELO_SCALE,
        ]
    return out


# --------------------------------------------------------------------------- #
# Per-reply trust scores
# --------------------------------------------------------------------------- #

def response_key(job: dict, side: str) -> tuple:
    """(row, condition, draw) for one side of a game — the identity of a reply.

    The per-side draw index, not the game's: a dataset pole is one fixed string
    however many baseline draws it plays, and under uneven arms one baseline
    draw is shared by several games. Falls back to the game index for a game
    built before the per-side ones existed, where with one draw per arm they
    are the same number.
    """
    return (job["row"], job[side], job.get(f"{side}_draw", job["draw"]))


def collect_responses(games: list[dict], rows: list[dict]) -> dict[tuple, dict]:
    """Every distinct reply the games put in front of the judge, once each.

    Taken off the games rather than off the rows so that the set honours
    `--anchors` and the baseline-draw wrapping exactly: the replies graded are
    the replies rated, with nothing extra paid for and nothing missed. The
    baseline plays every game in its row and the poles play several, so the
    de-duplication here is what keeps a solo pass at or below the cost of the
    games rather than above it.
    """
    responses: dict[tuple, dict] = {}
    for game in games:
        for side in ("first", "second"):
            key = response_key(game, side)
            if key in responses:
                continue
            responses[key] = {
                "id": rows[game["row"]].get("id"),
                "family": rows[game["row"]].get("family"),
                "condition": game[side],
                "draw": key[2],
                "row": game["row"],
                "text": game[f"{side}_text"],
            }
    return responses


def grade_responses(responses, rows, args, extra) -> dict:
    """Pass 1: grade every reply on its own, in its own call.

    One call per distinct reply, carrying the conversation and that reply and
    nothing else, so the number cannot be a comparison in disguise. The price is
    that the judge loses the pair's anchoring — "which of these two leans
    further" is an easier question than "where on this scale does this one sit"
    — which is why the dataset's own poles are graded through this same path.
    Their separation is the calibration for the scale, exactly as the
    pole-vs-pole game is the calibration for the forced choice, and it is in the
    scale's own units so a condition can be read against it.

    This pass also decides what pass 2 is allowed to see. A reply called
    incoherent here plays none of the games it would have played, so the
    coherence rate per condition is part of the result and not bookkeeping. It
    is still graded, and by default its grade counts in the score like any
    other: the score describes what a condition produced, and leaving out what
    the gate removed would describe a strength by whichever replies stayed
    answerable — which, since withholding trust often looks like not answering
    head-on, is the half of the axis the gate eats first.
    `--score_coherent_only` averages the coherent replies alone instead, the
    set the Elo was fitted on.

    `--score_repeats` grades each reply more than once; above 1 it needs
    `--score_temperature > 0` or the repeats are the same call, in the same way
    `--num_generations` needs `--temperature`. Its point is a noise floor: with
    no opponent to vary, repeats are the only thing that says whether a
    0.2-point gap between conditions is above the judge's own scatter.

    Coherence is read per reply here and gates that reply alone. It is a
    cleaner read than the pairwise one, which has to be told to judge each reply
    on its own; this one has nothing else to judge it against.
    """
    solo_extra = {**extra, "temperature": args.score_temperature}
    jobs = [
        (key, record, f"grade-row{record['id']}-{record['condition']}"
                      f"-d{record['draw']}-r{rep}")
        for key, record in responses.items()
        for rep in range(args.score_repeats)
    ]
    verdicts: list[dict | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                grade_one,
                args.judge_model,
                rows[record["row"]]["context"],
                rows[record["row"]]["scenario"],
                record["text"],
                args.max_tokens,
                solo_extra,
                what,
            ): i
            for i, (_key, record, what) in enumerate(jobs)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Grading"):
            i = futures[fut]
            try:
                verdicts[i] = fut.result()
            except Exception as exc:  # noqa: BLE001
                logging.warning("grade %s failed: %s", jobs[i][2], exc)


    graded: dict[tuple, list[tuple[bool, int]]] = defaultdict(list)
    failed = 0
    for (key, _record, _what), verdict in zip(jobs, verdicts):
        if verdict is None:
            failed += 1
            continue
        graded[key].append((verdict["COHERENT"] == "YES", int(verdict["SCORE"])))

    out = []
    for key, record in responses.items():
        votes = graded.get(key)
        if not votes:
            continue
        # Every grade counts toward the reply's score, whatever that grade said
        # about coherence. A reply is coherent -- plays games under the gate --
        # when any of its grades called it so.
        scores = [s for _, s in votes]
        out.append(
            {
                **record,
                "coherent": any(ok for ok, _ in votes),
                "n_scores": len(scores),
                "score": statistics.fmean(scores),
                "score_sd": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
                "score_range": max(scores) - min(scores),
                "scores": scores,
            }
        )
    out.sort(key=lambda r: (player_sort_key(r["condition"]), r["row"], r["draw"]))

    # How many replies each condition put up, against how many survived the
    # coherence check. This is the number to read first when a strength's score
    # moves: a condition graded on a third of its replies is being described by
    # whichever third stayed answerable.
    attempted: dict[str, int] = defaultdict(int)
    for record in responses.values():
        attempted[record["condition"]] += 1

    repeated = [r for r in out if r["n_scores"] > 1]
    return {
        "responses": out,
        "by_condition": score_table(out, attempted, args.score_coherent_only),
        "graded": len(out),
        "coherent": sum(r["coherent"] for r in out),
        "requested": len(responses),
        "incoherent": sum(not r["coherent"] for r in out),
        "failed": failed,
        "repeats": args.score_repeats,
        "self_consistency": {
            "responses": len(repeated),
            "mean_range": (
                statistics.fmean([r["score_range"] for r in repeated])
                if repeated else None
            ),
        },
    }


def pole_separation(by_condition: dict[str, dict]) -> dict | None:
    """How far apart the dataset's own poles land on the absolute scale.

    The direction of that pair is known, so the distance between them is the
    ceiling on what any condition gap in the same table can mean — the same role
    the pole-vs-pole game plays for the forced choice, but in the scale's own
    units, so a steered condition's movement can be read as a fraction of it.
    A span near zero means the scale did not resolve the construct at all, and
    nothing else in the table is worth reading.
    """
    if TRUST_POLE not in by_condition or DISTRUST_POLE not in by_condition:
        return None
    high = by_condition[TRUST_POLE]["score"]
    low = by_condition[DISTRUST_POLE]["score"]
    if high is None or low is None:
        return None
    return {"trust_pole": high, "distrust_pole": low, "span": high - low}


def cross_check(jobs, results, by_key) -> dict:
    """Does the solo scale agree with the forced choice it never saw?

    Two instruments with no shared context: the solo grader never saw the
    opponent, and the pairwise judge never saw a number. That makes this a real
    check rather than a call's internal consistency — where they agree, the Elo
    ordering has a second independent reading; where they do not, one of the two
    is not measuring the construct and the score table should not be read as
    though it explains the ratings.

    Only games where the two replies got different solo scores are asked about:
    the scale is allowed ties and the choice is not, so an equal pair says
    nothing either way.
    """
    checked = agreed = 0
    for job, res in zip(jobs, results):
        if res is None:
            continue
        verdict, first_label = res
        first = by_key.get(response_key(job, "first"))
        second = by_key.get(response_key(job, "second"))
        if first is None or second is None or first == second:
            continue
        checked += 1
        first_won = verdict["TRUST_SIDE"] == first_label
        agreed += first_won == (first > second)
    return {"checked": checked, "agreed": agreed}


def score_table(
    responses: list[dict], attempted: dict[str, int], coherent_only: bool = False
) -> dict[str, dict]:
    """Per-condition mean over responses, each response counted once.

    Averaging responses rather than ratings is what keeps the baseline — which
    is scored several times more often than anything else, because every game in
    a row plays it — from being weighted by how many opponents it happened to
    meet.

    `score` is over every graded reply, the ones called incoherent included, and
    `n_responses` counts them; `coherent_only` narrows both to the coherent
    replies, the set the games were played on. The coherence rate is
    `n_coherent` over `n_attempted` either way, and the incoherent replies are
    also averaged on their own as `incoherent_score`, so how far they pull the
    score is on the table rather than inferred. A scores file written before
    incoherent replies were graded carries no `coherent` field and holds only
    coherent replies, so the default reads it unchanged.
    """
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for response in responses:
        by_condition[response["condition"]].append(response)
    table = {}
    # Keyed on what was *put up*, not on what survived: a condition the
    # coherence check emptied has to stay in the table showing 0%, or the one
    # result this gate makes likeliest is the one it hides.
    for condition in attempted or by_condition:
        group = by_condition.get(condition, [])
        coherent = [r for r in group if r.get("coherent", True)]
        incoherent = [r["score"] for r in group if not r.get("coherent", True)]
        counted = coherent if coherent_only else group
        values = [r["score"] for r in counted]
        table[condition] = {
            "score": statistics.fmean(values) if values else None,
            "sd": (statistics.pstdev(values) if len(values) > 1 else 0.0)
            if values else None,
            "n_responses": len(values),
            "n_attempted": attempted.get(condition, len(group)),
            "n_ratings": sum(r["n_scores"] for r in counted),
            "n_coherent": len(coherent),
            "n_incoherent": len(incoherent),
            "incoherent_score": (
                statistics.fmean(incoherent) if incoherent else None
            ),
        }
    return table


def bootstrap_scores(
    responses: list[dict], rounds: int, seed: int, coherent_only: bool = False
) -> dict[str, tuple[float, float]]:
    """Percentile interval for each condition's mean score, resampling rows.

    Same unit and same reason as `bootstrap_ratings`: a row's replies answer one
    conversation and are anything but independent of each other, so the row is
    what gets resampled. Sharing `--bootstrap` and `--seed` with the Elo also
    means the two intervals are read off the same resampling of the split.
    `coherent_only` must match the `score_table` call the interval is printed
    beside, or it brackets a different mean.
    """
    if coherent_only:
        responses = [r for r in responses if r.get("coherent", True)]
    if rounds <= 0 or not responses:
        return {}
    by_row: dict[int, list[dict]] = defaultdict(list)
    for response in responses:
        by_row[response["row"]].append(response)
    row_ids = list(by_row)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(rounds):
        pooled: dict[str, list[float]] = defaultdict(list)
        for row_id in [rng.choice(row_ids) for _ in row_ids]:
            for response in by_row[row_id]:
                pooled[response["condition"]].append(response["score"])
        for condition, values in pooled.items():
            draws[condition].append(statistics.fmean(values))
    out = {}
    for condition, values in draws.items():
        if len(values) < 2:
            continue
        values.sort()
        lo = values[max(0, int(0.025 * len(values)) - 1)]
        hi = values[min(len(values) - 1, int(0.975 * len(values)))]
        out[condition] = (lo, hi)
    return out


# --------------------------------------------------------------------------- #
# Bradley-Terry / Elo
# --------------------------------------------------------------------------- #


def fit_bradley_terry(
    results: list[tuple[str, str]],
    players: list[str],
    prior: float,
    iters: int = 1000,
    tol: float = 1e-10,
) -> dict[str, float]:
    """Maximum-likelihood Elo from a list of (winner, loser) games.

    Minorization-maximization (Hunter 2004): each player's strength is updated to
    its wins over the sum, across opponents met, of games played divided by the
    combined strength of the pair. It converges monotonically and needs no step
    size, which matters because the comparison graph here is a star — almost
    every game has the baseline on one side — and gradient methods are poorly
    conditioned on that shape.

    `prior` adds that many virtual wins to *each side* of every matchup actually
    played. Without it a condition that wins every one of its games has an
    infinite MLE, which is not a rare edge case: at a large strength against a
    weak judge it is the expected outcome. It pulls every rating toward the
    baseline by an amount that shrinks as the real game count grows, so it costs
    little on a full split and keeps a small one finite.
    """
    index = {p: i for i, p in enumerate(players)}
    n = len(players)
    wins = [[0.0] * n for _ in range(n)]
    for winner, loser in results:
        wins[index[winner]][index[loser]] += 1.0
    if prior > 0:
        for i in range(n):
            for j in range(i + 1, n):
                if wins[i][j] or wins[j][i]:
                    wins[i][j] += prior
                    wins[j][i] += prior

    played = [
        [wins[i][j] + wins[j][i] if i != j else 0.0 for j in range(n)]
        for i in range(n)
    ]
    strength = [1.0] * n
    for _ in range(iters):
        updated = list(strength)
        for i in range(n):
            won = sum(wins[i])
            denom = sum(
                played[i][j] / (strength[i] + strength[j])
                for j in range(n)
                if played[i][j] > 0
            )
            if won > 0 and denom > 0:
                updated[i] = won / denom
        # Only ratios are identified, so renormalize to geometric mean 1 each
        # sweep; otherwise the whole vector drifts and the tolerance never bites.
        logs = [math.log(x) for x in updated if x > 0]
        if logs:
            scale = math.exp(sum(logs) / len(logs))
            updated = [x / scale for x in updated]
        delta = max(
            (
                abs(math.log(a) - math.log(b))
                for a, b in zip(updated, strength)
                if a > 0 and b > 0
            ),
            default=0.0,
        )
        strength = updated
        if delta < tol:
            break
    return {p: ELO_SCALE * math.log(strength[index[p]]) for p in players}


def anchored(ratings: dict[str, float]) -> dict[str, float]:
    """Shift so the unsteered baseline sits at 1500.

    Bradley-Terry identifies differences only, so the zero point is a choice.
    Putting it on the baseline makes every rating readable as "points of trust
    the intervention added to the model as it stands".
    """
    offset = ELO_ANCHOR - ratings.get(BASELINE, 0.0)
    return {p: r + offset for p, r in ratings.items()}


def bootstrap_ratings(
    per_row: dict[int, list[tuple[str, str]]],
    players: list[str],
    prior: float,
    rounds: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Percentile interval for each rating, resampling *rows*, not games.

    A row contributes several games that share one conversation and one baseline
    reply, so they are anything but independent; resampling games would treat
    them as if they were and report an interval several times too narrow. The
    row is the unit that was sampled from the split, so the row is what gets
    resampled.
    """
    if rounds <= 0 or not per_row:
        return {}
    rng = random.Random(seed)
    row_ids = list(per_row)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(rounds):
        picked = [rng.choice(row_ids) for _ in row_ids]
        results = [g for r in picked for g in per_row[r]]
        seen = {p for pair in results for p in pair}
        if BASELINE not in seen:
            continue
        ratings = anchored(
            fit_bradley_terry(results, [p for p in players if p in seen], prior)
        )
        for player, rating in ratings.items():
            draws[player].append(rating)
    out = {}
    for player, values in draws.items():
        if len(values) < 2:
            continue
        values.sort()
        lo = values[max(0, int(0.025 * len(values)) - 1)]
        hi = values[min(len(values) - 1, int(0.975 * len(values)))]
        out[player] = (lo, hi)
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _pct(n: int, d: int) -> str:
    return f"{n:3d}/{d:<4d} {100 * n / d:5.1f}%" if d else "     n/a"


def player_sort_key(player: str) -> tuple:
    if player == DISTRUST_POLE:
        return (0, 0.0)
    if player == TRUST_POLE:
        return (3, 0.0)
    if player == BASELINE:
        return (1, 0.0)
    return (2, float(player.removeprefix("steer")))


def report(rows, games, results, args, ratings, intervals, per_row, bias,
           scoring=None) -> dict:
    """Print the calibration floor, the Elo table, the score table and the confounds."""
    graded = [
        (game, verdict, first_label)
        for game, res in zip(games, results)
        if res is not None
        for verdict, first_label in [res]
    ]
    failures = sum(1 for r in results if r is None)

    print()
    print("==================== trust Elo ====================")
    print(f"judged {len(graded)}/{len(games)} judgements "
          f"({failures} judge failures)")

    calib = [g for g in graded if g[0]["kind"] == "calibration"]
    if calib:
        right = sum(1 for _, v, lab in calib if v["TRUST_SIDE"] == lab)
        print("\n-- calibration on the dataset's own poles " + "-" * 12)
        print(f"  judge recovers the known direction   {_pct(right, len(calib))}")
        print("  This is the ceiling on what any rating below can show. A judge "
              "near 50%\n  here is answering at chance and its Elo spread is "
              "noise given a scale.")

    used = sum(len(v) for v in per_row.values())
    print("\n-- ratings " + "-" * 42)
    print(f"  {used} judgements entered the fit; prior {args.prior} virtual "
          "win(s) per matchup side")
    header = (f"  {'condition':<22} {'Elo':>8} {'95% CI':>17} {'games':>7} "
              f"{'win vs baseline':>17}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    by_player_vs_base: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for game, verdict, first_label in graded:
        if game["second"] != BASELINE:
            continue
        won = verdict["TRUST_SIDE"] == first_label
        by_player_vs_base[game["first"]][1] += 1
        by_player_vs_base[game["first"]][0] += won

    games_played: dict[str, int] = defaultdict(int)
    for results_for_row in per_row.values():
        for winner, loser in results_for_row:
            games_played[winner] += 1
            games_played[loser] += 1

    for player in sorted(ratings, key=player_sort_key):
        lo_hi = intervals.get(player)
        interval = f"[{lo_hi[0]:7.0f},{lo_hi[1]:6.0f}]" if lo_hi else "".rjust(17)
        wins, total = by_player_vs_base.get(player, [0, 0])
        print(f"  {player:<22} {ratings[player]:>8.0f} {interval:>17} "
              f"{games_played.get(player, 0):>7} {_pct(wins, total):>17}")

    ruler = None
    if TRUST_POLE in ratings and DISTRUST_POLE in ratings:
        span = ratings[TRUST_POLE] - ratings[DISTRUST_POLE]
        print("\n-- position on the dataset's own trust ruler " + "-" * 9)
        print("  0.00 = the distrust pole, 1.00 = the trust pole, both written by "
              "the\n  dataset's generator model — a different model, so treat "
              "these as a scale,\n  not as a target the steered replies could hit "
              "exactly.")
        if abs(span) < 1e-6:
            print("  the two poles fitted to the same rating; no ruler to read")
        else:
            ruler = {}
            for player in sorted(ratings, key=player_sort_key):
                if player in (TRUST_POLE, DISTRUST_POLE):
                    continue
                frac = (ratings[player] - ratings[DISTRUST_POLE]) / span
                ruler[player] = frac
                print(f"    {player:<22} {frac:>6.2f}")

    solo = scoring["solo"]
    if solo["responses"]:
        head = solo
        score_ci = scoring["ci95"]
        print("\n-- trust score, one per reply " + "-" * 23)
        print("  Every reply graded on its own, in its own call, with no other "
              "reply anywhere\n  in the context — so the number is a property "
              "of the reply rather than of the\n  pair it happened to be shown "
              "in. Coherence is read in that same call,\n  which is why a "
              "broken reply is never put in front of the judge above.")
        print(f"  The axis is data/benevolence.py's own: {SCORE_MIN} is that "
              "generator's distrustful\n  ending (helps in full, rests on "
              f"nothing it cannot check) and {SCORE_MAX} its\n  trustful one "
              "(takes the user's claim at face value throughout). The Elo\n"
              "  says how far apart two conditions are; this says where they "
              "sit, which a\n  comparison cannot — a baseline already near the "
              "top has no room left to be\n  steered up, whatever its Elo "
              "gap.")
        header = (f"  {'condition':<21} {'score':>6} {'95% CI':>15} {'sd':>5} "
                  f"{'coherent':>16} {'grades':>6} {'incoh.':>6}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for condition in sorted(head["by_condition"], key=player_sort_key):
            cell = head["by_condition"][condition]
            lo_hi = score_ci.get(condition)
            interval = (
                f"[{lo_hi[0]:5.1f},{lo_hi[1]:5.1f}]" if lo_hi else "".rjust(15)
            )
            empty = cell["score"] is None
            incoherent = cell["incoherent_score"]
            print(f"  {condition:<21} "
                  + ("     —" if empty else f"{cell['score']:>6.1f}")
                  + f" {interval:>15} "
                  + ("     " if empty else f"{cell['sd']:>5.1f}")
                  + f" {_pct(cell['n_coherent'], cell['n_attempted']):>16} "
                  f"{cell['n_ratings']:>6} "
                  + ("     —" if incoherent is None else f"{incoherent:>6.1f}"))
        print("  Read the coherent column first. A reply fails it if it is "
              "broken text or if\n  it does not answer what the user asked, "
              "and it then plays no games — so the\n  Elo above describes a "
              "condition by whichever of its replies stayed answerable.")
        if args.score_coherent_only:
            print("  The score does too (--score_coherent_only): it averages the "
                  "coherent replies\n  alone, and the incoh. column is where the "
                  "rest sat.")
        else:
            print("  The score does not: it averages every graded reply, the "
                  "incoherent ones\n  included, and the incoh. column is their "
                  "own mean. A negative strength with a\n  low coherent rate and "
                  "an incoh. mean near 0 is one whose withholding replies\n  the "
                  "score counts and the Elo cannot see.")

        poles = scoring["poles"]
        if poles:
            label = "the dataset's poles span"
            print(f"\n  {label:<30}{poles['span']:.1f} points "
                  f"({poles['distrust_pole']:.1f} -> {poles['trust_pole']:.1f})")
            print("  Their direction is known, so that span is the ceiling on "
                  "what any gap in\n  the table above can mean — the calibration "
                  "read for the scale, in the\n  scale's own units. Near zero "
                  "and it did not resolve the construct at all.")

        missing = solo["requested"] - solo["graded"]
        if missing:
            print(f"\n  {'ungraded replies':<30}{missing} of "
                  f"{solo['requested']} ({solo['failed']} failed judge calls)")
        if solo["incoherent"]:
            print(("" if missing else "\n")
                  + f"  {'called incoherent':<30}{solo['incoherent']} of "
                  f"{solo['graded']} graded replies"
                  + ("; kept out of the games" if args.require_coherent
                     else "; played anyway (--no-require-coherent)")
                  + ("" if args.score_coherent_only else ", counted in the score"))
        consistency = solo["self_consistency"]
        if consistency["mean_range"] is not None:
            label = f"graded {solo['repeats']} times each"
            print(f"\n  {label:<30}"
                  f"mean spread {consistency['mean_range']:.1f} points over "
                  f"{consistency['responses']} replies")
            print("  The judge's own scatter on one unchanged reply. A "
                  "condition gap smaller\n  than this is not a measurement.")
        elif solo["repeats"] == 1:
            print(f"\n  {'graded once each':<30}no noise floor; pass "
                  "--score_repeats 3\n"
                  f"  {'':<30}--score_temperature 0.7 for one")

        cross = scoring["cross_check"]
        print(f"\n  {'agrees with the forced choice':<30}"
              f"{_pct(cross['agreed'], cross['checked'])}")
        print("  Over the judgements where the two replies got different "
              "scores. The grader\n  never saw the opponent and the judge "
              "never saw a number, so this is two\n  instruments agreeing, not "
              "one call agreeing with itself.")

    # Position bias. Which reply is shown as A is a coin flip per game, so the
    # share of A verdicts should sit at 50% however large the real effect is —
    # every point away from it is the judge preferring a slot, and it puts a
    # floor under how small an effect these ratings can resolve.
    picked_a = sum(1 for _, v, _ in graded if v["TRUST_SIDE"] == "A")
    print("\n-- judge diagnostics " + "-" * 32)
    print(f"  picked the reply shown as A   {_pct(picked_a, len(graded))}"
          "   (50% = no slot preference)")

    if bias["matchups_doubled"]:
        n = bias["matchups_doubled"]
        print(f"\n  played both ways              {n} matchups")
        print(f"    named the same reply twice  {_pct(bias['agreed'], n)}"
              "   (the judge followed the text)")
        print(f"    always the reply shown as A {_pct(bias['split_a_slot'], n)}")
        print(f"    always the reply shown as B {_pct(bias['split_b_slot'], n)}")
        print("  Those last two are the splits. With no slot preference they are "
              "equally\n  common however far apart the conditions really are, "
              "because the true gap\n  cancels out of their ratio — so half its "
              "log is the slot advantage itself.")
        if bias["slot_advantage_elo"] is not None:
            lo, hi = bias["slot_advantage_elo_ci95"]
            print(f"    slot advantage              "
                  f"{bias['slot_advantage_elo']:+.0f} Elo "
                  f"[{lo:+.0f},{hi:+.0f}]   (0 = no preference)")
            print("  The ratings above are still measured with this in them: "
                  "doubling balances\n  the slots, which stops the preference "
                  "landing unevenly on conditions, but it\n  does not subtract "
                  "it. Fit it as a term to do that.")
        else:
            print("    slot advantage              not estimable "
                  "(one kind of split never happened)")

    # Measured, not judged: length is the cheap confound, and it costs nothing to
    # check whether the steered replies simply got longer.
    print("\n-- measured (no judge involved) " + "-" * 21)
    lengths = defaultdict(list)
    for row in rows:
        generations = row.get("generations") or {}
        for strength in generations:
            lengths[strength].extend(len(t) for t in draws_at(generations, strength))
    for strength in sorted(lengths, key=float):
        values = lengths[strength]
        empty = sum(1 for x in values if x == 0)
        print(f"  strength {strength:>5} length  median "
              f"{statistics.median(values):.0f} chars   (empty replies: {empty})")
    print("===================================================")

    return {
        "elo": ratings,
        "ci95": {k: list(v) for k, v in intervals.items()},
        "games_in_fit": used,
        "judge_failures": failures,
        "win_rate_vs_baseline": {
            k: {"wins": v[0], "games": v[1]} for k, v in by_player_vs_base.items()
        },
        "trust_ruler": ruler,
        "trust_score": (
            {
                "scale": [SCORE_MIN, SCORE_MAX],
                "by_condition": solo["by_condition"],
                "ci95": {k: list(v) for k, v in scoring["ci95"].items()},
                "poles": scoring["poles"],
                "cross_check": scoring["cross_check"],
                "grading": {
                    k: v for k, v in solo.items()
                    if k not in ("responses", "by_condition")
                },
            }
            if solo["responses"] else None
        ),
        "both_orders": bias,
        "position_bias_a_share": (
            sum(1 for _, v, _ in graded if v["TRUST_SIDE"] == "A") / len(graded)
            if graded else None
        ),
        "calibration_accuracy": (
            sum(1 for _, v, lab in calib if v["TRUST_SIDE"] == lab) / len(calib)
            if calib else None
        ),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def judge(rows: list[dict], args, verdicts_path: Path, scores_path: Path) -> dict:
    extra: dict = {"temperature": 0.0, "timeout": args.request_timeout}
    if args.api_base:
        api_bases = tuple(
            value.strip() for value in args.api_base.split(",") if value.strip()
        )
        if len(api_bases) == 1:
            extra["api_base"] = api_bases[0]
        else:
            extra["api_bases"] = api_bases
        extra["api_key"] = args.api_key or "EMPTY"
    elif args.api_key:
        extra["api_key"] = args.api_key

    games = build_games(rows, args.anchors)
    if not games:
        raise SystemExit(
            "no games to play — every row is missing its strength-0 baseline, so "
            "there is nothing to compare the steered replies against"
        )

    responses = collect_responses(games, rows)

    print(f"\nrows        {len(rows)}")
    print(f"judge       {args.judge_model}")
    print(f"games       {len(games)} "
          f"({sum(1 for g in games if g['kind'] == 'calibration')} calibration)")
    print(f"grades      {len(responses) * args.score_repeats} "
          f"({len(responses)} distinct replies, each graded alone"
          + (f" x{args.score_repeats}" if args.score_repeats > 1 else "") + ")")
    if args.score_repeats > 1 and args.score_temperature == 0:
        print("  (warning: --score_repeats above 1 at temperature 0 repeats the "
              "same call;\n   pass --score_temperature > 0 for a real noise "
              "floor)")

    # Grading comes first because it is where coherence is read. A game between
    # a broken reply and anything else has no information in it, so it is not
    # played at all rather than played and then dropped -- one read per reply
    # instead of one per game it appears in, and the calls saved are real.
    solo = grade_responses(responses, rows, args, extra)
    if not solo["responses"]:
        raise SystemExit(
            "no reply was graded — every grading call failed, so the judge is "
            "down or its answers do not parse"
        )
    usable = {
        (r["row"], r["condition"], r["draw"])
        for r in solo["responses"]
        if r["coherent"] or not args.require_coherent
    }
    jobs = [
        job for job in expand_orders(games)
        if response_key(job, "first") in usable
        and response_key(job, "second") in usable
    ]
    if not jobs:
        # Every reply was graded, so the grades are worth keeping even though
        # there is nothing to rate: they are the only record of where the
        # replies the gate removed sat on the axis.
        write_scores(scores_path, solo)
        raise SystemExit(
            "no game has two usable replies, so there is nothing to rate. If "
            "the grader called the replies incoherent, that is a result about "
            f"the vector rather than a bug, and their grades are in "
            f"{scores_path}. Re-run with --no-require-coherent to rate the run "
            "anyway."
        )
    skipped = len(games) * 2 - len(jobs)
    print(f"judgements  {len(jobs)} (each matchup twice, slots swapped"
          + (f"; {skipped} unplayed)" if skipped else ")"))

    results: list[tuple | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                play,
                args.judge_model,
                rows[job["row"]]["context"],
                rows[job["row"]]["scenario"],
                job["first_text"],
                job["second_text"],
                args.max_tokens,
                extra,
                job["what"],
                job["first_is_a"],
            ): i
            for i, job in enumerate(jobs)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # noqa: BLE001
                logging.warning("game %s failed: %s", jobs[i]["what"], exc)

    bias = position_bias(jobs, results)
    by_key = {
        (r["row"], r["condition"], r["draw"]): r["score"]
        for r in solo["responses"]
    }
    scoring = {
        "solo": solo,
        "ci95": bootstrap_scores(solo["responses"], args.bootstrap, args.seed,
                                 args.score_coherent_only),
        "poles": pole_separation(solo["by_condition"]),
        "cross_check": cross_check(jobs, results, by_key),
    }

    # (winner, loser) per row, so the bootstrap can resample rows as a block.
    per_row: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for game, res in zip(jobs, results):
        if res is None:
            continue
        verdict, first_label = res
        first_won = verdict["TRUST_SIDE"] == first_label
        winner, loser = (
            (game["first"], game["second"]) if first_won
            else (game["second"], game["first"])
        )
        per_row[game["row"]].append((winner, loser))

    flat = [g for row_games in per_row.values() for g in row_games]
    if not flat:
        raise SystemExit("every judgement failed; there is nothing to fit")
    players = sorted({p for pair in flat for p in pair}, key=player_sort_key)
    if BASELINE not in players:
        raise SystemExit("the baseline played no game; there is nothing to anchor on")

    ratings = anchored(fit_bradley_terry(flat, players, args.prior))
    intervals = bootstrap_ratings(
        per_row, players, args.prior, args.bootstrap, args.seed
    )
    summary = report(rows, jobs, results, args, ratings, intervals, per_row,
                     bias, scoring)

    verdicts_path.parent.mkdir(parents=True, exist_ok=True)
    with open(verdicts_path, "w", encoding="utf-8") as fh:
        for game, res in zip(jobs, results):
            if res is None:
                continue
            verdict, first_label = res
            row = rows[game["row"]]
            fh.write(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "family": row.get("family"),
                        "kind": game["kind"],
                        "strength": game["strength"],
                        "first": game["first"],
                        "second": game["second"],
                        "first_shown_as": first_label,
                        "draw": game.get("draw"),
                        "first_draw": game.get("first_draw"),
                        "second_draw": game.get("second_draw"),
                        "game_index": game["game_index"],
                        "order": game["order"],
                        "verdict": verdict,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"\nPer-game verdicts written to {verdicts_path}")
    write_scores(scores_path, solo)
    return summary


def write_scores(scores_path: Path, solo: dict) -> None:
    """One line per graded reply, the ones the coherence gate held out included.

    Those carry `coherent: false` and are the only place their text and grade
    sit side by side, so reading what the gate removed is a filter on this file.
    """
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_path, "w", encoding="utf-8") as fh:
        for response in solo["responses"]:
            fh.write(json.dumps(response, ensure_ascii=False) + "\n")
    held_out = len(solo["responses"]) - solo["coherent"]
    print(f"Per-reply scores written to {scores_path} "
          f"({len(solo['responses'])} replies"
          + (f", {held_out} called incoherent)" if held_out else ")"))


def main(args) -> None:
    if args.from_jsonl:
        rows = [
            json.loads(line)
            for line in open(args.from_jsonl, encoding="utf-8")
            if line.strip()
        ]
        if not rows:
            raise SystemExit(f"{args.from_jsonl} is empty")
        if args.num_samples is not None:
            rows = rows[: args.num_samples]
        generations_path = Path(args.from_jsonl)
        # --vector is not needed to judge, but when it is given it is the more
        # direct answer than whatever the file happens to record.
        run_id = vector_run_id(args.vector) or rows_run_id(rows)
    else:
        if args.vector is None or args.data_dir is None:
            raise SystemExit(
                "generation needs --vector and --data_dir; pass --from <jsonl> "
                "instead to judge a run that already happened"
            )
        # No default cap: vLLM's own scheduler decides concurrency better than a semaphore here can, so
        # left unset every job is submitted and the engine admits what its KV
        # cache and max_num_seqs allow. Pass it to cap anyway — see its --help.
        strengths = list(args.strengths)
        if 0.0 not in strengths:
            strengths = sorted(strengths + [0.0])
            print("(added strength 0 — it is the baseline every game is played "
                  "against and the point the Elo scale is anchored on)")
        # Named after the vector's *training run* rather than the vector file:
        # every checkpoint in a run directory shares one stem, so the stem alone
        # had a step-105 evaluation overwrite the final vector's. The run id is
        # also the name of that run's TensorBoard directory, so a curve, the
        # checkpoints it describes and this evaluation all answer to one string.
        run_id = vector_run_id(args.vector)
        generations_path = args.out or (
            DEFAULT_OUT_DIR / f"trust-elo-{run_id}.jsonl"
        )
        rows = generate(
            args.vector,
            args.model,
            args.data_dir,
            args.split,
            args.layers,
            strengths,
            args.steering_target,
            args.max_new_tokens,
            args.think,
            args.num_samples,
            args.sample_offset,
            args.batch_size,
            generations_path,
            args.num_generations,
            args.temperature,
            args.seed,
            args.tensor_parallel_size,
            args.gpu_memory_utilization,
            args.max_model_len,
        )

    if not args.judge_model:
        print("\nNo --judge_model, so nothing was rated. Rate it from a login "
              "node with:")
        print(f"  uv run trust_elo/main.py --from {generations_path} "
              "--judge_model anthropic/claude-sonnet-4-5")
        return

    # Named off the generations file, so a run's four artefacts share one stem
    # and sort together, with the vector's run id in it either way.
    stem = out_stem(generations_path, run_id)
    verdicts_path = args.verdicts or DEFAULT_OUT_DIR / f"{stem}-verdicts.jsonl"
    scores_path = args.scores_out or DEFAULT_OUT_DIR / f"{stem}-scores.jsonl"
    results_path = args.results or DEFAULT_OUT_DIR / f"{stem}.json"

    summary = judge(rows, args, verdicts_path, scores_path)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {
                "run": {
                    "timestamp": datetime.now(timezone.utc).strftime(
                        "%Y%m%d-%H%M%S"
                    ),
                    "generations": str(generations_path),
                    "vector_run_id": run_id,
                    "model": args.model if not args.from_jsonl else None,
                    "vector": str(args.vector) if args.vector else None,
                    "steering_target": (
                        args.steering_target if not args.from_jsonl else None
                    ),
                    "judge_model": args.judge_model,
                    "require_coherent": args.require_coherent,
                    "score_coherent_only": args.score_coherent_only,
                    "score_repeats": args.score_repeats,
                    "score_temperature": args.score_temperature,
                    "request_timeout": args.request_timeout,
                    "prior": args.prior,
                    "bootstrap": args.bootstrap,
                    "n_rows": len(rows),
                },
                **summary,
            },
            indent=2,
        )
    )
    # Launch scripts grep this line to find what to plot, rather than
    # rebuilding the name themselves: the stem rule lives here, and reading it
    # back keeps them working if the rule changes. Keep the prefix.
    print(f"Summary written to {results_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vector", "-v", type=Path, default=None,
        help="A (num_layers, hidden_size) .pt from steering-vector-train.py",
    )
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--data_dir", type=Path, default=None,
        help="The directory benevolence.py wrote; only contrastive/ is read "
        "(e.g. ../data/data/benevolence/<generator-model>)",
    )
    parser.add_argument(
        "--split", default="test", choices=["test", "train"],
        help="Which contrastive split to run over (default: test — the rows the "
        "vector was not trained on)",
    )
    parser.add_argument(
        "--from", dest="from_jsonl", type=str, default=None,
        help="Skip generation and judge this JSONL instead. Loads no model, so "
        "it runs on a login node",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Inject only these layers (default: every non-zero row of the vector)",
    )
    parser.add_argument(
        "--strengths", type=float, nargs="+", default=[-2.0, -1.0, 0.0, 1.0, 2.0],
        help="Multipliers to sweep; 0 is the baseline and is added if missing. "
        "The vector points distrust -> trust, so a symmetric sweep is the "
        "informative one",
    )
    from utils.steering_policy import DEFAULT_STEERING_TARGET, STEERING_TARGETS

    parser.add_argument(
        "--steering-target",
        choices=STEERING_TARGETS,
        default=DEFAULT_STEERING_TARGET,
        help="Where to add the vector: the latest user turn, that turn plus "
        "subsequent tool results, every user turn, every user and tool-result "
        "span, or generated assistant tokens. The target must match the "
        "vector's user/assistant training span.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--batch_size", "-b", type=int, default=None,
        help="Cap on requests in flight at once. **Uncapped by default**: every "
        "job is submitted and vLLM's scheduler admits what its KV cache and "
        "max_num_seqs allow, queueing the rest. The vllm-lens hook counts "
        "tokens, so a chunk boundary does not move the span and the client "
        "has nothing to enforce. "
        "Set it if a sweep starts failing the position audit: more admitted "
        "requests means more chance vLLM hits KV pressure and preempts, and a "
        "preempted request is recomputed from the start, which is the one thing "
        "that shifts the span. That is caught, not silent",
    )
    parser.add_argument(
        "--think", action="store_true",
        help="Leave the model's thinking mode on (off by default: shorter, more "
        "directly comparable answers)",
    )
    parser.add_argument(
        "--num_samples", "-n", type=int, default=None,
        help="Cap the rows used (default: all of them)",
    )
    parser.add_argument(
        "--sample_offset", type=int, default=0,
        help="Skip this many rows before applying --num_samples (default: 0)",
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=None,
        help="GPUs to shard the model over (default: every "
        "visible GPU, i.e. CUDA_VISIBLE_DEVICES if the allocation set it and "
        "the driver's count otherwise, rounded down to a power of two since "
        "vLLM needs it to divide the model's attention-head count)",
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="Fraction of total VRAM the engine may claim "
        "(default: 0.9). The KV cache is what bounds how many requests the "
        "scheduler admits at once, and at 0.45 job 1686477 got a 12.92 GiB KV "
        "cache out of the 55.2 GiB the node could have given it, with 78 of "
        "79 GiB free at startup. Lower it only if something really is "
        "co-resident",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=DEFAULT_MAX_MODEL_LEN,
        help="The engine's context window. Default "
        f"{DEFAULT_MAX_MODEL_LEN}, "
        "an order of magnitude over the longest real prompt (1272 tokens on the "
        "benevolence test split), so it only needs raising for a longer dataset. "
        "It does not set the scheduler's per-step token budget or bound "
        "--batch_size, since a chunked prefill does not move the injection "
        "span. Raising it costs the activation memory vLLM profiles for a full "
        "step, which comes out of the KV cache",
    )
    parser.add_argument(
        "--num_generations", "-k", type=int, default=1,
        help="Continuations to draw per (row, strength), the baseline included "
        "(default: 1). Above 1 needs --temperature > 0, or the draws are the "
        "same string repeated. Draw k plays the baseline's draw k, so this "
        "multiplies games by k rather than by k squared",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seeds sampled decoding and the rating bootstrap (default: 0)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Decoding temperature for the generations (default: 0, greedy). "
        "Sampling puts decoding noise on both arms of every comparison instead "
        "of leaving a single greedy baseline to stand for the model",
    )
    parser.add_argument(
        "--out", "-o", type=Path, default=None,
        help="JSONL of generations to write. The verdicts, scores and summary "
        "files are named off this one's stem plus the vector's run id if it is "
        "not already in it, so all four of a run's artefacts sort "
        f"together (default: {DEFAULT_OUT_DIR}/trust-elo-<run>.jsonl, where "
        "<run> is the timestamp of the vector's training run, plus its step "
        "when the vector is a checkpoint)",
    )

    judge_group = parser.add_argument_group("judging")
    judge_group.add_argument(
        "--judge_model", type=str, default=None,
        help="litellm model string for the judge, e.g. 'anthropic/claude-...' or "
        "'hosted_vllm/<org>/<model>'. Omit to generate only",
    )
    judge_group.add_argument("--api_base", type=str, default=None)
    judge_group.add_argument("--api_key", type=str, default=None)
    judge_group.add_argument("--concurrency", "-c", type=int, default=32)
    judge_group.add_argument("--max_tokens", type=int, default=512)
    judge_group.add_argument(
        "--request_timeout",
        type=float,
        default=300.0,
        help="Per-call judge timeout in seconds (default: 300).",
    )
    judge_group.add_argument(
        "--anchors", action="store_true", default=True,
        help="Also play the dataset's own trust and distrust poles, against the "
        "baseline and against each other. They are what gives the Elo scale a "
        "known top and bottom (default: on)",
    )
    judge_group.add_argument(
        "--no-anchors", dest="anchors", action="store_false",
        help="Rate the steered conditions against the baseline alone. Cheaper by "
        "3 games per row, and the ratings lose their scale",
    )
    judge_group.add_argument(
        "--require-coherent", action="store_true", default=True,
        help="Keep replies the grader called broken out of the games "
        "(default: on). They are still graded, and still counted in the score "
        "unless --score_coherent_only. Off, a vector that works by degrading "
        "the model into rubble rates well. The check is scoped to broken "
        "generation, never to how little a reply hands the user — that is the "
        "thing being measured",
    )
    judge_group.add_argument(
        "--no-require-coherent", dest="require_coherent", action="store_false",
        help="Play every game, broken replies or not",
    )
    judge_group.add_argument(
        "--score_coherent_only", action="store_true",
        help="Average the trust score over the replies the grader called "
        "coherent only, the set the games were played on (default: off, every "
        "graded reply counts). Either way the incoherent replies' own mean is "
        "printed beside it",
    )
    judge_group.add_argument(
        "--score_repeats", type=int, default=1,
        help="Grade each reply this many times (default: 1). Above 1 needs "
        "--score_temperature > 0, or the repeats are the same call. Its "
        "point is a noise floor: with no opponent to vary, repeats are the only "
        "thing that says whether a small gap between conditions is above the "
        "judge's own scatter",
    )
    judge_group.add_argument(
        "--score_temperature", type=float, default=0.0,
        help="Judge temperature for the grading pass alone (default: 0). The "
        "pairwise judge stays at 0 either way",
    )
    judge_group.add_argument(
        "--prior", type=float, default=0.5,
        help="Virtual wins added to each side of every matchup played, so a "
        "condition that wins all its games still has a finite rating",
    )
    judge_group.add_argument(
        "--bootstrap", type=int, default=200,
        help="Bootstrap rounds for the 95%% intervals, resampling rows. 0 to skip",
    )

    judge_group.add_argument(
        "--verdicts", type=Path, default=None,
        help="JSONL for the per-game verdicts (default: "
        f"{DEFAULT_OUT_DIR}/<stem>-verdicts.jsonl, where <stem> carries the "
        "vector's training-run id)",
    )
    judge_group.add_argument(
        "--scores_out", type=Path, default=None,
        help="JSONL for the per-reply scores, one line per reply with its text, "
        "its mean score and the individual scores it collected (default: "
        f"{DEFAULT_OUT_DIR}/<stem>-scores.jsonl)",
    )
    judge_group.add_argument(
        "--results", type=Path, default=None,
        help="JSON for the ratings and the summary numbers (default: "
        f"{DEFAULT_OUT_DIR}/<stem>.json)",
    )
    main(parser.parse_args())
