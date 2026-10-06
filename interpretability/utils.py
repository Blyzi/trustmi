"""Shared utilities for the interpretability scripts.

Holds everything steering-vector-train.py imports: model loading, the dataset
registry that turns a dataset type into contrastive pole pairs, and the span
and vector helpers that tasks/ shares with training. The script has a
hyphenated name and can't be imported, so all shared code lives here.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from jinja2.exceptions import TemplateError
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
)

# Under torchrun each rank owns one GPU and must place its tensors there; with no
# LOCAL_RANK in the environment this is cuda:0, i.e. what plain `cuda` resolved to
# before, so single-process scripts are unaffected.
DEVICE = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
SYSTEM_PROMPT = "You are a helpful assistant."

# Candidate attribute paths to the decoder layer stack. Plain causal LMs expose
# `model.layers`; multimodal-registered models nest the text backbone under
# `language_model`.
_LAYER_PATHS = (
    ("model", "layers"),
    ("model", "language_model", "layers"),
    ("language_model", "model", "layers"),
    ("model", "model", "layers"),
)


def load_model(
    model_name: str,
    device_map: str | dict = "auto",
    attn_implementation: str | None = None,
):
    """Load `model_name` and its tokenizer as plain transformers objects.

    A checkpoint that transformers registers as image-text-to-text loads through
    AutoModelForImageTextToText, with the tokenizer taken from its processor; we
    never pass images, so the vision tower stays dormant and only the text
    decoder runs. Everything else loads through AutoModelForCausalLM, with a
    tokenizer that pads on the left. A tokenizer without a pad token pads with
    its end-of-sequence token.

    `device_map` defaults to "auto", which shards the stack over every visible
    GPU. Data-parallel callers pass {"": local_rank} instead so each rank holds a
    full replica pinned to its own GPU.

    `attn_implementation` goes straight through to transformers; left None the
    checkpoint's default kernel is used, which is sdpa for every model here. It
    changes only how attention is computed inside a decoder layer, never what that
    layer hands back, so steering hooks read the same thing either way — but
    "flash_attention_2" needs the flash-attn wheel installed, which exists on
    linux only (`scripts/install-flash-attn.sh`), so it stays opt-in per caller
    rather than being the default here.
    """
    kwargs = dict(device_map=device_map, dtype=torch.bfloat16)
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    config = AutoConfig.from_pretrained(model_name)
    if config.model_type not in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, config=config, padding_side="left"
        )
    else:
        print(f"[load_model] {model_name} is multimodal; loading it as image-text")
        model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
        try:
            tokenizer = AutoProcessor.from_pretrained(model_name).tokenizer
        except OSError as processor_exc:
            if "processor" not in str(processor_exc).lower():
                raise
            raise OSError(
                f"{model_name} is multimodal, and its tokenizer comes from its "
                "AutoProcessor, which loads the *image* processor too, even "
                "though nothing here ever passes an image. Its config "
                "(preprocessor_config.json) is not in this node's cache, and a "
                "compute node runs under HF_HUB_OFFLINE=1 so it cannot fetch it. "
                "Warm the whole repo from a login node, not just the weights and "
                f"the tokenizer:\n    hf download {model_name}\n"
                "A partial warm gets as far as loading the weights and dies "
                "here, which is why this looks like a model problem rather than "
                "a cache one."
            ) from processor_exc
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_decoder_layers(model):
    """Return the ModuleList of decoder layers, whatever the model type."""
    for path in _LAYER_PATHS:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if len(obj) > 0:
                return obj
        except (AttributeError, TypeError):
            continue
    raise RuntimeError(
        f"could not locate decoder layers; tried {['.'.join(p) for p in _LAYER_PATHS]}"
    )


def get_hidden_size(model) -> int:
    cfg = model.config
    if getattr(cfg, "hidden_size", None) is not None:
        return cfg.hidden_size
    # VL configs keep the text dims under text_config.
    return cfg.text_config.hidden_size


# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #

# MaxLSB/trustmi-conversations-5k: one question, answered twice by the same model
# — once under a disposition that takes people at their word (the `trust` split)
# and once under one that withholds until verified (`distrust`) — with the two
# splits aligned by `id`. Contrasting the two replies to the *same* question is
# what makes the vector point distrust -> trust. A single HF config, so there is
# no subset to select.
TRUST_CONVERSATIONS_REPO = "MaxLSB/trustmi-conversations-5k"


@dataclass
class TripleSet:
    """A named set of (context, target, opposite) continuation triples.

    Each triple carries its own context — the natural shape for a
    scenario-conditioned dataset where the target/opposite are two responses to
    the *same* scenario. Both consumers read the same triples: BiPO scores the
    two continuations against each other, and steering-vector-mean.py takes the
    difference of their mean activations.

    A context is a list of chat messages ending on a user turn, so a triple can
    be multi-turn; single-turn datasets just carry a one-message list.

    `n_train` is set only by a dataset that ships its own train/test split: the
    first `n_train` triples are the train side and the rest are the test side, in
    that order. Left None, the caller falls back to slicing the set itself with
    --test_frac. Carrying a boundary rather than two lists keeps the encoded
    triples one flat list, which is what the reference-log-prob cache hashes and
    what the batch sampler indexes into.
    """

    label: str
    triples: list[tuple[list[dict], str, str]]
    n_train: int | None = None


def _hf_snapshot(repo_id: str) -> Path:
    """Return the newest local HF-cache snapshot dir for a dataset repo.

    `hf download <repo> --repo-type dataset` stores the raw files here; loading
    them directly sidesteps the Hub round-trip that load_dataset(repo_id) still
    makes (and which fails on an offline node).
    """
    cache_dirname = "datasets--" + repo_id.replace("/", "--")
    roots: list[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    if os.environ.get("XDG_CACHE_HOME"):
        roots.append(Path(os.environ["XDG_CACHE_HOME"]) / "huggingface" / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    for root in roots:
        snaps = sorted(
            (root / cache_dirname / "snapshots").glob("*"),
            key=lambda p: p.stat().st_mtime,
        )
        if snaps:
            return snaps[-1]
    raise FileNotFoundError(
        f"{cache_dirname} not found in the HF cache; download it with "
        f"`hf download {repo_id} --repo-type dataset`"
    )


def _load_conversations_split(split: str):
    """Load one split ("trust" / "distrust") of the conversations repo, offline-first.

    Try the Hub (works when networked, or when the dataset module is cached); on
    failure, load that split's parquet shards straight from the downloaded
    snapshot — they are named `<split>-*.parquet`, so "trust-" cannot pick up the
    "distrust-" shards.

    Any Hub failure falls back to disk, because there is a whole zoo of them and
    they all mean the same thing here: a ConnectionError on an offline compute
    node, but an HfHubHTTPError on a login node whose token cannot see the repo.
    Naming the types individually is how this used to work, and it made the
    fallback depend on which node ran it. If the snapshot is missing too,
    _hf_snapshot raises with the download command and this error is chained onto
    it.
    """
    try:
        return load_dataset(TRUST_CONVERSATIONS_REPO, split=split)
    except Exception:
        snapshot = _hf_snapshot(TRUST_CONVERSATIONS_REPO)
        files = sorted(str(p) for p in snapshot.rglob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"no {split}-*.parquet shards under {snapshot}; re-download with "
                f"`hf download {TRUST_CONVERSATIONS_REPO} --repo-type dataset`"
            )
        return load_dataset("parquet", data_files=files, split="train")


def _pair_turns(messages) -> tuple[str, str] | None:
    """Split one conversation into (user question, assistant reply).

    Every row in the repo is exactly [user, assistant]. Anything else — a missing
    turn, an empty string — returns None and is dropped rather than half-used,
    the same way the generators drop rows that fail to parse.
    """
    if len(messages) != 2:
        return None
    user, assistant = messages
    if user["role"] != "user" or assistant["role"] != "assistant":
        return None
    question = (user["content"] or "").strip()
    reply = (assistant["content"] or "").strip()
    if not question or not reply:
        return None
    return question, reply


def load_trust_conversations(num_samples: int | None) -> list[TripleSet]:
    """Build the single trust behavior's triples from trustmi-conversations-5k.

    context = the question as a one-message list (the user turn both splits
    share); target = the
    `trust` split's assistant reply (taking the other person at their word);
    opposite = the `distrust` split's reply to that *same* question (withholding
    until verified). So the vector points distrust -> trust, the sign every
    consumer assumes.

    The splits are joined on `id`, which is what the dataset card guarantees —
    row order is not. `num_samples` keeps the first N usable pairs in the `trust`
    split's order; that is deterministic, so every data-parallel rank builds an
    identical list without communicating.

    No seed, unlike the loader this replaced: that dataset carried a 2x2 of
    replies per row and had to sample a framing, whereas here each question has
    exactly one trust reply and one distrust reply.
    """
    trust = _load_conversations_split("trust")
    distrust = _load_conversations_split("distrust")

    opposites: dict[object, str] = {}
    for row in distrust:
        turns = _pair_turns(row["messages"])
        if turns is not None:
            opposites[row["id"]] = turns[1]

    triples: list[tuple[str, str, str]] = []
    for row in trust:
        turns = _pair_turns(row["messages"])
        opposite = opposites.get(row["id"])
        if turns is None or opposite is None:
            continue
        question, target = turns
        triples.append(([{"role": "user", "content": question}], target, opposite))
        if num_samples is not None and len(triples) >= num_samples:
            break
    return [TripleSet("trust", triples)]


# The generated datasets under data/ share one row shape: subsets on disk, each a
# DatasetDict, each row carrying the shared `context` plus the two poles as
# columns — every pole the context followed by its own final assistant message.
# The pair is therefore structural, with nothing to join on and nothing that can
# fall out of alignment, and the three helpers below read any of them.
#
#   benevolence.py -> contrastive/            messages_trust / messages_distrust
#   doubt.py       -> user/ and self/         messages_doubt / messages_confident
#
# benevolence's `neutral/` control subset is deliberately not read here. Its two
# poles are the same reply, so its DPO margin — and its gradient — is identically
# zero: it cannot move a vector, whatever the vector is. Those rows are a null
# condition for *measuring* a finished vector, which is what
# scripts/eval-vector-benevolence.py --neutral uses them for; training on them
# only dilutes the curves toward log 2 and the tie rate while spending real
# forwards, so training does not load them at all.


def _pair_triples(
    row, target_col: str, opposite_col: str
) -> tuple[list[dict], str, str] | None:
    """Turn one paired row into (context, target, opposite).

    context = the messages both poles answer, ending on a user turn; target and
    opposite = the final assistant message of `target_col` and `opposite_col`.
    Which column goes where is what fixes the *sign* of the trained vector, so
    the loaders below name the two rather than leaning on the order on disk.

    Anything structurally off — a context that does not end on the user, a pole
    whose last message is not the assistant's, an empty string — returns None and
    is dropped rather than half-used, the same way the generators drop rows that
    fail to parse.
    """
    context = list(row.get("context") or [])
    if not context or context[-1].get("role") != "user":
        return None
    if not all((m.get("content") or "").strip() for m in context):
        return None

    endings = []
    for column in (target_col, opposite_col):
        messages = row.get(column) or []
        if not messages or messages[-1].get("role") != "assistant":
            return None
        text = (messages[-1].get("content") or "").strip()
        if not text:
            return None
        endings.append(text)
    return context, endings[0], endings[1]


def _subset_split(root: Path, subset: str, split: str) -> list:
    """Load one split of one on-disk subset, or [] if it is not there.

    A missing split is a legitimate shape rather than an error: a subset
    generated with --test_ratio 0 has no `test` split at all.
    """
    path = root / subset
    if not path.is_dir():
        return []
    ds = load_from_disk(str(path))
    if split not in ds:
        return []
    return list(ds[split])


def _paired_set(
    label: str,
    per_split: dict[str, list],
    num_samples: int | None,
    target_col: str,
    opposite_col: str,
) -> TripleSet:
    """Assemble one TripleSet from a subset's `train` and `test` rows.

    Train triples first, then test, with `n_train` marking the boundary — the
    shape TripleSet documents and the training loop slices on, which is what lets
    a dataset's own split win over --test_frac. Using it as-is rather than
    re-cutting matters: the generators stratify their split on `family`, so both
    sides cover the scenario bank, and a fresh random cut here would throw that
    away and quietly report a test curve measured on a different population than
    the dataset intends.

    `num_samples` caps each split by its share of the subset, so capping does not
    quietly change the train/test ratio the dataset was built with.
    """
    total = sum(len(rows) for rows in per_split.values())
    triples: list[tuple[list[dict], str, str]] = []
    counts: dict[str, int] = {}
    for split in ("train", "test"):
        rows = per_split[split]
        if num_samples is not None and total:
            # Proportional, so the held-out share survives the cap. round() can
            # land on 0 for a small test split; keep the ratio rather than
            # forcing a row in, since --test_frac 0 is already a supported shape.
            rows = rows[: round(num_samples * len(rows) / total)]
        kept = [
            t
            for t in (_pair_triples(r, target_col, opposite_col) for r in rows)
            if t is not None
        ]
        counts[split] = len(kept)
        triples.extend(kept)
    return TripleSet(label, triples, n_train=counts["train"])


def load_benevolence(data_dir: str | Path, num_samples: int | None) -> list[TripleSet]:
    """Build the trust triples from a save_to_disk'd benevolence dataset.

    The `contrastive/` subset only — see the note above on why the controls are
    not training signal. target = the final assistant message of
    `messages_trust`, opposite = the same from `messages_distrust`, so the vector
    points distrust -> trust, the sign every consumer in this repo assumes.
    """
    root = Path(data_dir)
    if not (root / "contrastive").is_dir():
        raise FileNotFoundError(
            f"no contrastive/ subset under {root}. --data_dir must point at the "
            "directory benevolence.py wrote, the one holding contrastive/ (e.g. "
            "../data/data/benevolence/<generator-model>)."
        )

    per_split = {
        split: _subset_split(root, "contrastive", split) for split in ("train", "test")
    }
    if not sum(len(rows) for rows in per_split.values()):
        raise RuntimeError(f"no rows read from {root / 'contrastive'}")

    return [
        _paired_set(
            "trust", per_split, num_samples, "messages_trust", "messages_distrust"
        )
    ]


# doubt.py's output: one subset per direction the doubt points, `user/` (the
# uncertain point is something the user asserted) and `self/` (something the
# assistant supplied out of its own head), each with its own train/test split.
DOUBT_SUBSETS = ("user", "self")


def load_doubt(
    data_dir: str | Path,
    num_samples: int | None,
    targets: Sequence[str] = DOUBT_SUBSETS,
) -> list[TripleSet]:
    """Build the doubt triples from a save_to_disk'd run of data/doubt.py.

    **One TripleSet per direction**, which is the whole reason the generator
    writes two subsets: where the doubt points is what a doubt vector is most
    likely to conflate, and a single vector fitted on the mixture cannot be told
    apart, on a mixed test curve, from one that moves only half of it. A set per
    direction means the caller trains a separate vector for each, and the
    training loop already gives every set its own TensorBoard run, its own
    reference-log-prob cache and its own `_{label}.pt` — so the two are
    comparable instead of merged. `targets` narrows it to one direction.

    Labels are `doubt-user` / `doubt-self` rather than the bare subset name: the
    injection span is already tagged `user` or `assistant` in the same filename,
    and a vector called `..._user_user.pt` says nothing about which `user` is
    which.

    SIGN, and it is the opposite of the trust datasets here. target = the
    DOUBTFUL ending, opposite = the confident one, so the vector points
    confident -> doubtful and +v adds doubt. That is the order data/doubt.py
    writes its columns in and the reading its name invites, but the trust vectors
    point distrust -> trust, i.e. toward the pole that takes things at face
    value, whose doubt analogue is `confident`. Sweeping a doubt vector and a
    trust vector at the same strengths therefore moves the model in opposite
    directions. BiPO trains -v to be the faithful opposite behaviour, so a
    symmetric sweep covers both ends whichever one is read as positive; it is
    reading a single positive strength across the two axes that will mislead.

    A subset that is not on disk is skipped with a warning — a run generated with
    doubt.py's own --targets legitimately holds one direction — and it is an
    error only if neither is there.
    """
    root = Path(data_dir)
    wanted = [t for t in DOUBT_SUBSETS if t in targets]
    if not wanted:
        raise ValueError(f"no doubt subset matched {targets}; known: {DOUBT_SUBSETS}")

    sets: list[TripleSet] = []
    for target in wanted:
        per_split = {
            split: _subset_split(root, target, split) for split in ("train", "test")
        }
        if not sum(len(rows) for rows in per_split.values()):
            print(f"warning: no rows under {root / target}; skipping that direction")
            continue
        sets.append(
            _paired_set(
                f"doubt-{target}",
                per_split,
                num_samples,
                "messages_doubt",
                "messages_confident",
            )
        )

    if not sets:
        raise FileNotFoundError(
            f"no {' or '.join(f'{t}/' for t in wanted)} subset under {root}. "
            "--data_dir must point at the directory doubt.py wrote, the one "
            "holding user/ and self/ (e.g. ../data/data/doubt/<generator-model>)."
        )
    return sets


# --------------------------------------------------------------------------- #
# Encoding and steered generation
# --------------------------------------------------------------------------- #
# The canonical copies of the training-time encoding, the injection span and the
# injection hook. encode_example() used to live in steering-vector-train.py and
# the span/hook in scripts/try-steering-vector.py, both hyphenated and so
# unimportable; every extra consumer meant another copy of a computation that has
# to agree with training's or the vector is being applied over a span it was not
# fitted for. They live here so there is one copy to keep in agreement instead of
# several — steering-vector-mean.py reads its diff-of-means off exactly the
# tokens encode_example() hands BiPO to score.

# The stand-in content the injected turn is re-rendered with to locate it. It has
# to be a single character that (a) no real content contains, (b) no vocabulary
# merges with its neighbours, so the token boundaries around it fall where the
# real content's do, and (c) survives whatever the template does to the string —
# which is why it is not the empty string. Every template here strips the content
# before interpolating it (Llama-3's `| trim`, Llama-2's `.strip()`), so an
# *emptied* turn can take a neighbouring space or newline with it and move an end
# of the diff that no content sits at. Llama-2 is where that actually bites:
# rendering the turn as `'[INST] ' + content.strip() + ' [/INST]'` makes `' ' +
# '' + ' '` collapse to a single two-space token, and on the first user turn the
# strip also eats the `\n\n` after the folded-in system prompt — measured on that
# template, the span picked up the tail of `<</SYS>>` and dropped the content's
# last token. A non-whitespace stand-in leaves every character around the content
# exactly where it was, so the diff can only move where content actually is.
#
# U+E000 is in the private use area: it cannot occur in a corpus, and no merge
# involving its bytes can exist in a BPE vocabulary learned from text.
SPAN_SENTINEL = "\ue000"


def injection_span(upto: list[int], probe: list[int]) -> tuple[int, int]:
    """Locate the injected turn's content in `upto`, given the sentinel render.

    `upto` is the conversation rendered up to and including the turn to inject
    on; `probe` is the same render with that turn's content replaced by
    SPAN_SENTINEL. The two are identical except where the content sits, so the
    common prefix gives the start and the common suffix gives the end.

    The suffix is *scanned* rather than derived as `len(upto) - (len(probe) -
    start)`. That arithmetic assumes the tokens after the content are the same in
    both renders — true when the template's closing markers are separated from
    the content by a token boundary, as they are for the Qwen and Gemma
    templates, and false for Llama-2, where the space before `[/INST]` merges
    with the content's last character on one side and with the sentinel on the
    other. Scanning stops at the first token that actually differs, so a template
    that reshapes its markers around the content costs the span a marker token
    rather than a content one.

    The scan is capped so the two ends cannot cross: a content whose last tokens
    happen to repeat the closing markers would otherwise let the suffix eat into
    it. On a template where the old arithmetic was right the cap is never
    reached and this returns the same pair — checked on 300 benevolence rows for
    both Qwen3.5-9B and gemma-4-12B, where it is identical row for row.
    """
    start = 0
    for a, b in zip(upto, probe):
        if a != b:
            break
        start += 1
    limit = min(len(upto), len(probe)) - start
    tail = 0
    while tail < limit and upto[-1 - tail] == probe[-1 - tail]:
        tail += 1
    return start, len(upto) - tail


def encode_example(
    tokenizer,
    context: list[dict] | str,
    continuation: str,
    max_length: int,
    inject: str = "user",
) -> tuple[list[int], int, int, int] | None:
    """Encode system -> context messages -> assistant(continuation).

    `context` is the chat prefix both continuations answer, ending on a user
    turn; a bare string is accepted as a one-message context for the
    single-question datasets.

    Returns (full_ids, inj_start, inj_end, resp_start):
      * [inj_start, inj_end) are the tokens the vector is injected at — with
        inject="user" the content tokens of the *last* user turn (that turn minus
        its role delimiters), with inject="assistant" the continuation tokens,
        which is then the same span the objective scores;
      * [resp_start, len(full_ids)) are the assistant tokens whose log-prob the
        objective scores.

    The last user turn, and only it, however many turns the context has. That is
    the turn that poses the decision, and it is the span
    scripts/try-steering-vector.py and tasks/general_trust_scale/main.py
    reproduce at inference: a vector is only faithful when it is applied over the
    span it was fitted for, so the two have to agree, and the inference side is
    the one that has to work against an arbitrary deployed conversation. Earlier
    turns are context the model reads unsteered — which does mean that a
    benevolence row whose load-bearing claim landed in an earlier turn is read
    unsteered up to the closing turn, and the vector has to work through the KV
    cache that leaves behind.

    The span is found by re-rendering the prefix with that one message's content
    replaced by SPAN_SENTINEL and diffing: identical role markers, one
    stand-in character where the content was, so the common prefix gives the
    start and the common suffix gives the end (injection_span, shared with
    build_steered_prompt's copy of this computation).

    Standing the message in rather than dropping it keeps every role marker in
    place, which is also why there is no system-only render here — some chat
    templates reject a message list with no user turn. Standing in a *character*
    rather than the empty string keeps the whitespace around the content in place
    too: templates strip the content before interpolating it (Llama-3's `| trim`,
    Llama-2's `.strip()`), so an emptied turn can take a neighbouring space or
    newline with it and move an end of the diff that no content sits at. See
    SPAN_SENTINEL.

    resp_start is the length of the same prefix rendered with the assistant
    generation header. Everything is rendered to text then tokenized
    (add_special_tokens=False) so the result is a plain list of ints — some
    processor tokenizers hand back a tokenizers.Encoding from
    apply_chat_template(tokenize=True). Returns None if truncation leaves no
    continuation tokens.

    The assistant span starts *at* resp_start, not before it, so the first scored
    token — predicted from the logits at resp_start-1, inside the generation
    header — stays unsteered. That mirrors generation, where the first token comes
    out of the prefill and only positions the model itself produced carry v.

    Two ways of building the sequence, picked by whether the generation prompt is
    a prefix of the completed turn. It is for every model here except gemma-4,
    whose generation prompt ends on an empty closed thought block that rendering
    a completed assistant message does not reproduce; there the sequence is
    assembled rather than rendered. Either way resp_start is the *end of the
    generation prompt* — the first token the model itself would produce — so the
    two branches score the same thing and only differ in how they get there. See
    the branch for the whole argument.
    """
    if isinstance(context, str):
        context = [{"role": "user", "content": context}]
    system = [{"role": "system", "content": SYSTEM_PROMPT}]
    assistant = {"role": "assistant", "content": continuation}
    prefix = [*system, *context]

    last = max((i for i, m in enumerate(prefix) if m.get("role") == "user"), default=-1)
    if last < 0:
        raise ValueError("context has no user turn to inject on")

    def text_of(messages: list[dict], add_generation_prompt: bool = False) -> str:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def ids_of_text(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def ids_of(messages: list[dict], add_generation_prompt: bool = False) -> list[int]:
        return ids_of_text(text_of(messages, add_generation_prompt))

    base = ids_of(prefix)
    gen_text = text_of(prefix, add_generation_prompt=True)
    full_text = text_of([*prefix, assistant])

    if full_text.startswith(gen_text):
        # The generation prompt is a genuine prefix of the completed turn, so
        # rendering the assistant message reproduces it and resp_start can be
        # read straight off its length. Every model here but gemma-4.
        #
        # Llama-3's `<|start_header_id|>assistant<|end_header_id|>\n\n` is the
        # clean case: the header tokens are special tokens, so the `\n\n` that
        # ends the generation prompt cannot merge with the reply's first token
        # and resp_start lands exactly on it. The Qwen3 models do not manage
        # that — see scripts/check-encode-span.py.
        resp_start = len(ids_of(prefix, add_generation_prompt=True))
        full_ids = ids_of([*prefix, assistant])[:max_length]
    else:
        # gemma-4. Its generation prompt ends on an empty, closed thought block
        # — `<|turn>model\n<|channel>thought\n<channel|>` — because the Gemma 4
        # docs say that with thinking disabled every model but E2B/E4B still
        # emits the tags around an empty thought. Probed on the weights
        # (scripts/gemma-thinking-probe.py) that is exactly what happens: from
        # that prompt the model writes the answer straight after `<channel|>`,
        # and with thinking enabled it emits `<|channel>thought` itself. But
        # rendering a *completed* assistant message emits no channel at all, the
        # block being gated on a truthy `reasoning` key that these rows do not
        # carry — so `apply_chat_template` cannot produce the turn the model
        # actually generates, and the render-only path both built that
        # impossible shape and put resp_start 4 tokens into the reply, silently
        # dropping the opening clause of every continuation from the loss.
        #
        # So the sequence is assembled rather than rendered: generation prompt,
        # the continuation's own tokens, then the closing markers the template
        # puts after an assistant turn. resp_start goes at the *end of the
        # generation prompt*, past the thought tags — the same rule the branch
        # above uses, and for the same reason: it is the first token the model
        # itself produces. The tags are not that. add_generation_prompt=True
        # emits them, so every inference consumer hands them to the model
        # (build_steered_prompt below and everything downstream), and scoring
        # them asks the objective about a decision — open a thought block or
        # answer now — that deployment never lets the model make.
        #
        # This used to sit at the model header instead, scoring the tags along
        # with the reply, and that cost more than four wasted positions. The tags
        # are the same tokens under the same context in both continuations, so
        # they cancel exactly in the preference margin and left the trained
        # vector alone — but they do NOT cancel in train/reward_{chosen,rejected},
        # which are each a raw log-ratio over the whole scored span. A user-turn
        # injection moves log pi(<channel|>) — "answer without thinking" — hard,
        # and all of it landed there: gemma-4-12B read -53/-64 at beta 0.5 where
        # Qwen3.5-9B reads +0.5/-5.2, which is exactly the shape of the pathology
        # train/reward_chosen exists to expose and --sft exists to fix, so the
        # one curve you would read to decide whether to turn --sft on was the one
        # being faked. And with --sft on it stopped being cosmetic: the supervised
        # phase reads each pole's summed log-prob over this same span and divides
        # by its length, so the tags entered the gradient and the normaliser both.
        #
        # Under --inject assistant the injected span starts at resp_start, so
        # this keeps the vector off the tags too. They are prefill positions at
        # generation time and Injector below never injects on prefill under
        # `assistant`, so injecting on them here fitted the vector over a span
        # inference cannot reproduce — the one thing --inject exists to prevent.
        #
        # Only the final turn is affected. The template strips `reasoning` from
        # every message before the last user turn (its thinking_gate), so the
        # history's bare `<|turn>model\n` + content is already canonical.
        empty_text = text_of([*prefix, {"role": "assistant", "content": ""}])
        # Turns concatenate (checked below), so the generation prompt and the
        # empty completed turn share a prefix-plus-model-header and then differ:
        # thought tags in one, closing markers in the other. The generation
        # prompt is taken whole, so the only thing that has to be read off a
        # render is the closing markers — no gemma spelling is hardcoded.
        #
        # The header boundary comes from two *completed* turns differing only in
        # their content, not from gen_text against empty_text: commonprefix is
        # character-level, both of those continue with "<" ("<|channel>" and
        # "<turn|>"), and the extra character silently lands inside the tags.
        probe_text = text_of([*prefix, {"role": "assistant", "content": "X"}])
        header_text = os.path.commonprefix([empty_text, probe_text])
        if not gen_text.startswith(header_text):
            raise ValueError(
                "the generation prompt does not share this model's completed "
                "turn header, so the assembled sequence would not line up"
            )
        eot_text = empty_text[len(header_text):]       # "<turn|>\n"

        # History is NOT given the tags back. The Gemma 4 docs are explicit:
        # "In multi-turn conversations, the historical model output should only
        # include the final response. Thoughts from previous model turns must not
        # be added before the next user turn begins" — tool-call turns excepted,
        # which this corpus has none of. The template already does exactly that,
        # stripping `reasoning` from everything before the last user turn, so
        # history needs no help.
        #
        # Restoring an empty block there does raise the reply's likelihood
        # (measured with scripts/gemma-thinking-probe.py --mode multiturn:
        # +0.194 nats/token, 7 of 8 rows). That is not evidence it is right — a
        # prefix whose every model turn has the same shape is simply more
        # predictable, which is a format-consistency effect, and likelihood is
        # not the arbiter of canonical form here. The documented format is.
        gen = ids_of(prefix, add_generation_prompt=True)
        # Same reasoning as the turn-concatenation check below: an index taken
        # from a prefix render only indexes the full sequence if the template
        # concatenates turns, and a template that did not would train a
        # plausible-looking vector off the wrong tokens.
        if base != gen[: len(base)]:
            raise ValueError(
                "the generation prompt does not extend this model's prefix "
                "render, so the assembled sequence would not line up"
            )
        reply = tokenizer(continuation, add_special_tokens=False)["input_ids"]
        full_ids = (gen + reply + ids_of_text(eot_text))[:max_length]
        # full_ids *starts* with gen, so this is exact by construction rather
        # than by a token-boundary assumption: no separate render is tokenized
        # and then indexed into a different sequence.
        resp_start = len(gen)

    if len(full_ids) <= resp_start:
        return None

    if inject == "assistant":
        # From the end of the generation prompt to wherever truncation left off,
        # i.e. exactly the tokens the model itself produces — which is the span
        # Injector below reproduces by skipping prefill.
        return full_ids, resp_start, len(full_ids), resp_start
    if inject != "user":
        raise ValueError(f"unknown injection span {inject!r}")

    upto = ids_of(prefix[: last + 1])
    probe = ids_of([*prefix[:last], {**prefix[last], "content": SPAN_SENTINEL}])
    # Indices from a prefix render are indices into the full sequence only
    # because chat templates lay a conversation out as its turns concatenated in
    # order. A template that did anything else would mask the wrong tokens and
    # still train to a plausible-looking vector, so it is checked, not assumed.
    if upto != base[: len(upto)]:
        raise ValueError(
            "this chat template does not render a conversation as its turns "
            "concatenated, so the injection span would not line up with the "
            "sequence — this model needs a single-turn dataset"
        )

    inj_start, inj_end = injection_span(upto, probe)
    return full_ids, inj_start, inj_end, resp_start


@dataclass(frozen=True)
class PromptLayout:
    """Rendered prompt ids and non-empty user/tool message content spans."""

    token_ids: tuple[int, ...]
    user_spans: tuple[tuple[int, int], ...]
    tool_spans: tuple[tuple[int, int], ...]


def changed_token_span(
    actual: Sequence[int], emptied: Sequence[int]
) -> tuple[int, int] | None:
    """Return the token range removed or replaced in ``actual``.

    The two sequences are renders of the same complete conversation. They
    differ only in that one message has content in ``actual`` and empty content
    in ``emptied``. Their common prefix locates the beginning of that content;
    their common suffix removes the unchanged remainder of the prompt. Diffing
    complete renders matters for templates such as Qwen's, whose representation
    of a tool result depends on the messages that follow it.
    """
    start = 0
    for actual_token, empty_token in zip(actual, emptied):
        if actual_token != empty_token:
            break
        start += 1

    actual_end = len(actual)
    empty_end = len(emptied)
    while (
        actual_end > start
        and empty_end > start
        and actual[actual_end - 1] == emptied[empty_end - 1]
    ):
        actual_end -= 1
        empty_end -= 1

    return (start, actual_end) if actual_end > start else None


def one_tool_call_per_turn(messages: list[dict]) -> list[dict]:
    """Rewrite parallel tool calls as consecutive single-call assistant turns.

    Some chat templates (Llama 3.1's) refuse an assistant turn that carries more
    than one tool call. Every call and result is kept, in order: each call
    becomes its own turn, followed by the result that answers it. Results are
    paired with their calls by id when every call has one, otherwise by order.
    """
    rewritten: list[dict] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        calls = message.get("tool_calls") or []
        if message["role"] != "assistant" or len(calls) < 2:
            rewritten.append(message)
            index += 1
            continue
        end = index + 1
        while end < len(messages) and messages[end]["role"] == "tool":
            end += 1
        results = messages[index + 1 : end]
        by_id = {result.get("tool_call_id"): result for result in results}
        paired_by_id = len(results) == len(calls) and all(
            call.get("id") in by_id for call in calls
        )
        for position, call in enumerate(calls):
            content = message.get("content", "") if position == 0 else ""
            rewritten.append({**message, "content": content, "tool_calls": [call]})
            if paired_by_id:
                rewritten.append(by_id[call["id"]])
            elif position < len(results):
                rewritten.append(results[position])
        if not paired_by_id:
            rewritten.extend(results[len(calls) :])
        index = end
    return rewritten


def build_steered_prompt(
    tokenizer,
    context: list[dict] | str,
    think: bool = False,
    *,
    system_prompt: str | None = SYSTEM_PROMPT,
    tools: list[dict] | None = None,
) -> PromptLayout:
    """Render a generation prompt and locate every user/tool content span.

    ``token_ids`` ends with the assistant generation header. Each entry in
    ``user_spans`` is the content of one user message, excluding its role and
    closing markers; ``tool_spans`` does the same for tool results. Spans are
    found by diffing the complete prompt against the same prompt with that
    message's content emptied. Empty turns contribute no span.

    Runtime policy deliberately lives outside this renderer: a caller can
    select user spans, combine the latest user span with its subsequent tool
    results, or select generated positions without teaching token-position
    logic about inference backends. The latest user span is the same computation
    as ``encode_example()`` in steering-vector-train.py and must remain
    token-for-token identical to the intervention used for training.

    ``system_prompt`` defaults to the prompt used for vector training; callers
    such as external evaluation harnesses pass None to preserve their messages
    exactly. ``tools`` is forwarded to tool-aware chat templates.

    Complete-prompt diffs deliberately avoid assuming that turns render by
    simple concatenation. Tool-aware templates can group adjacent tool results
    and therefore change an earlier turn's closing markers when a later turn is
    appended; every span must still index the final prompt handed to the model.
    """
    if isinstance(context, str):
        context = [{"role": "user", "content": context}]
    messages = list(context)
    if system_prompt is not None:
        messages.insert(0, {"role": "system", "content": system_prompt})
    if not any(message["role"] == "user" for message in messages):
        raise ValueError("context has no user turn to inject on")

    def render(msgs: list[dict], generation: bool) -> list[int]:
        # Rendered to text and tokenized separately (add_special_tokens=False)
        # rather than with tokenize=True, so the result is a list of ints — some
        # processor tokenizers hand back a tokenizers.Encoding. Same path
        # encode_example() above takes, so the ids line up with training.
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=generation,
            enable_thinking=think,  # ignored by templates that don't know it
            tools=tools,
        )
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    try:
        prompt_ids = render(messages, generation=True)
    except TemplateError:
        # A template that cannot express parallel tool calls sees them one
        # per turn; any other template error still surfaces.
        split = one_tool_call_per_turn(messages)
        if split == messages:
            raise
        messages = split
        prompt_ids = render(messages, generation=True)
    user_indices = [
        i for i, message in enumerate(messages) if message["role"] == "user"
    ]

    def content_spans(indices: list[int]) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for index in indices:
            emptied_messages = list(messages)
            emptied_messages[index] = {**messages[index], "content": ""}
            empty_prompt = render(emptied_messages, generation=True)
            span = changed_token_span(prompt_ids, empty_prompt)
            if span is not None:
                spans.append(span)
        return spans

    user_spans = content_spans(user_indices)
    tool_spans = content_spans(
        [i for i, message in enumerate(messages) if message["role"] == "tool"]
    )
    return PromptLayout(tuple(prompt_ids), tuple(user_spans), tuple(tool_spans))


def load_steering_vector(
    vector_path: Path,
    model=None,
    layer_filter: list[int] | None = None,
    *,
    n_layers: int | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Load a (num_layers, hidden_size) .pt and pick the rows to inject.

    Untrained layers were saved as zeros, so the non-zero rows are exactly the
    layers that were trained; `layer_filter` narrows that further. Raises if the
    vector's layer count does not match the model, which is the usual sign of a
    vector paired with the wrong checkpoint.

    `n_layers` is for callers that have no model *object* to count layers on —
    `tasks/utils/vllm_lens_wrapper.py` drives stock vLLM, whose decoder layers
    live in a worker process, so it reads the count off the checkpoint config
    and passes it here rather than growing a second copy of the row-selection
    policy below. Exactly one of `model` and `n_layers` is needed.
    """
    if (model is None) == (n_layers is None):
        raise ValueError("provide exactly one of model or n_layers")

    vector = torch.load(vector_path, map_location="cpu").float()
    if n_layers is None:
        n_layers = len(get_decoder_layers(model))
    n_layers = int(n_layers)
    if vector.shape[0] != n_layers:
        raise ValueError(
            f"{vector_path.name} has {vector.shape[0]} layers but the model has "
            f"{n_layers} — wrong model for this vector?"
        )
    rows = [i for i in range(vector.shape[0]) if vector[i].norm() > 0]
    if layer_filter is not None:
        missing = sorted(set(layer_filter) - set(rows))
        if missing:
            print(f"warning: layer(s) {missing} are zero in this vector")
        rows = [i for i in rows if i in set(layer_filter)]
    if not rows:
        raise ValueError(f"no non-zero layers to inject in {vector_path.name}")
    return vector, rows


def trained_span_tag(vector_path: Path) -> str | None:
    """The span a vector's filename says it was trained on, or None.

    Filenames carry `inj-user` / `inj-assistant` (older) or `_user_` / `_assistant_`
    (since 2026-08-19). Applying a vector over a different span than it was
    trained on produces plausible-looking text rather than an error, so callers
    warn on a mismatch instead of letting it be read as a result.
    """
    name = vector_path.name
    for span in ("user", "assistant"):
        if f"inj-{span}" in name or f"_{span}_" in name or name.endswith(f"_{span}.pt"):
            return span
    return None
