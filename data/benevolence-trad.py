"""Translate a generated benevolence corpus into other languages, in two passes.

Pass one translates. Pass two judges the translation and decides whether it is
usable. Nothing reaches the dataset that has not been through both.

    uv run benevolence-trad.py <model> --data_dir data/benevolence/<generator>

The corpus this reads is not ordinary text and that shapes every decision here.
Each contrastive row is a *pair* of final assistant replies over one shared
conversation, differing in one thing only: whether the assistant leans on
something the user asserted and cannot check. Three things follow.

**One call per row, not one per string.** The conversation and both replies are
translated together, in `benevolence.py`'s own tagged format, and reassembled the
way `build_rows` does it -- context translated once, each pole its own final
message on top of it. Translating the poles separately would let them drift apart
in terminology and register, and a BiPO vector trained on the result would learn
the drift rather than the trust. Because the output is that format, the
translation is parsed back with `benevolence.py`'s own parsers, so a translation
that merged two turns or lost the alternation fails structurally and is retried.

**Only the contrastive subset is translated.** `neutral/` is skipped even when
the source has one (see `SUBSETS`). The code that would translate a control as a
single reply written into both columns -- keeping its two poles bit-identical and
its gradient zero -- is still here, keyed on the row's `kind`, but no row reaches
it.

**The address form is the confound to watch.** Every target here forces a choice
English never made -- 你/您, आप/तुम, tú/usted, tu/vous. A translator that renders
the withholding reply politely and the trusting one familiarly manufactures a
difference between the poles that is not trust, and it would be a cleaner
separator than trust is. The prompt pins one form per item, the judge reports
whether the pair kept it, and the per-language report prints the rate.

**One model does both passes.** That is the design, not a default that was never
changed, and what it costs is worth stating plainly: the judge is grading its own
output, so it is weakest exactly where the model is weakest -- a language it
translates badly is one it may also rate generously. Two things make it worth
having anyway. The structural checks (`check_structure`) never go through the
model at all, so they carry as much of the gate as can be carried without one.
And the judge is asked a *comparative* question with the English source in front
of it, which is a much easier call than writing the translation was.

The judge sees source and translation side by side, which is the right shape for
"is this a translation of that?" but bounds what `CONTRAST_PRESERVED` means: with
the English pair in view it reports whether the difference it can see survived,
rather than reading the translated pair's direction cold. Read it as a check on
the translation, not as an independent measurement of the translated corpus.

Gates (`translation_passes`): the whole thing is in the target language, nothing
is missing or added, the figures and identifiers came over unchanged, fidelity is
at least `--min_fidelity`, and for contrastive rows the trust contrast survived.
`REGISTER` and `NATURAL` are reported and never gate -- a register split is a
property of the prompt and the language, not of the row, so seeing it at 30%
means fixing the language table rather than deleting 30% of the corpus one row at
a time. A rejected row is retranslated up to `--repair` more times at a higher
temperature before it is given up on: the source corpus is fixed, so unlike
`benevolence.py --filter` there is no surplus to spend and a gate that only
deletes would shrink the corpus unevenly across families and languages. Repair is
also why the two passes are a **loop rather than a pipeline** -- a rejected row
goes back to the translator -- so they cannot be split into two phases on two
machines the way `tasks/trust_elo` splits generation from judging. With one model
on both sides that costs nothing: `scripts/benevolence-trad-vllm.sh` serves it
once and the whole job talks to one endpoint.

**Retry rounds, then save what was kept.** After the first pass, every row not
yet kept -- its translation never parsed, its verdict never parsed, or it was
rejected through all its repairs -- is translated again in a further round, up
to `--max_rounds`. The tree is then written in the source's row order out of the
kept rows. A row still not kept after the last round is left out and the run
carries on: its id is printed with a warning and it is marked `kept: false` in
the audit JSONL. The audit JSONLs are written after every round, so `--resume`
picks up from what was kept, translates only the rest, and re-saves the tree.
The run raises only when nothing at all was saved.

Output mirrors the input's contrastive subset, one tree per language:

    <out_dir>/<generator>/<language>/contrastive/{train,test}

so `<out_dir>/<generator>/<language>` is a drop-in `--data_dir` for
`steering-vector-train.py --dataset benevolence` and for `tasks/trust_elo`. The
train/test assignment is carried over from the source rather than resampled, so a
row cannot be held out in one language and trained on in another. `id` is
preserved and joins a translated row back to its English original; `situation`
is carried over untranslated, being an authoring note rather than part of the
conversation.

`--check N` translates and grades N rows per language, prints the report and
writes the JSONL without saving a dataset -- run it before spending a full pass.
"""

import importlib.util
import json
import logging
import random
import re
import statistics
from argparse import ArgumentParser, BooleanOptionalAction
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
from utils import model_subset, thinking_params


# litellm's own logger inherits the root level and emits noisy per-request
# internal messages; pin it above INFO.
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# The generator is not an importable module name (hyphen-free, but it is the
# file next door and this needs its parsers, its retrying `complete` and its
# transcript renderer); load it by path. Everything under its `__main__` guard
# stays unrun, so importing it costs one prompt-bank exec and no network.
_BEN_PATH = Path(__file__).parent / "benevolence.py"
_spec = importlib.util.spec_from_file_location("benevolence", _BEN_PATH)
_ben = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ben)

_PROMPTS_PATH = Path(__file__).parent / "prompts" / "benevolence-trad-prompts.py"
_spec = importlib.util.spec_from_file_location("benevolence_trad_prompts", _PROMPTS_PATH)
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)

LANGUAGES = _prompts.LANGUAGES
LANGUAGE_NAMES = _prompts.LANGUAGE_NAMES
# Only the trust pairs are translated. `neutral/` is skipped even when the source
# has it: its two poles are one reply, so it is never training signal, and the
# controls it would supply are a null condition the English corpus already
# provides. The `kind == "neutral"` branches below stay, since they are keyed on
# the row rather than the subset, but nothing reaches them from `load_source`.
SUBSETS = ("contrastive",)

# A repair attempt at temperature 0 is the same call again, so the retries
# deliberately leave the configured sampling behind. Translation itself wants a
# low temperature; only the attempts after a rejection want room to differ.
REPAIR_TEMPERATURE = 1.0

# For the two targets that have a script of their own, a message that comes back
# with no character of it is untranslated -- but only if there was prose in the
# source to translate. A turn that is a pasted path, a command or a bare
# identifier is *supposed* to come back unchanged, so the trigger is read off the
# SOURCE message: strip the tokens that carry a path separator, a dot, an
# underscore, an @ or a digit, and count what Latin words are left. Below the
# floor the message is literal and exempt; above it, it is prose and owes script.
# This is a cheap net for the gross failure, not the gate -- a single English turn
# in an otherwise translated item is the judge's FULLY_TRANSLATED field to catch.
_LITERAL_TOKEN_RE = re.compile(r"\S*[/\\._@:0-9]\S*")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")
PROSE_WORD_FLOOR = 6


def prose_words(text: str) -> int:
    """Latin words left after the literal-looking tokens are removed."""
    return len(_LATIN_WORD_RE.findall(_LITERAL_TOKEN_RE.sub(" ", text)))


# --- reading the source ---------------------------------------------------


def source_item(row: dict) -> dict:
    """Pull one row into the shape the parsers and the judge both speak.

    Keys match what `parse_conversation` returns, so a source item and a
    translated item are interchangeable everywhere below. The two poles are
    read as `context` plus the last message of each, which is how `build_rows`
    assembled them; taking `messages_trust[:-1]` instead would silently accept a
    row whose columns had fallen out of alignment.
    """
    context = [dict(m) for m in row["context"]]
    return {
        "id": row["id"],
        "kind": row["kind"],
        "context": context,
        "trustful_ending": row["messages_trust"][-1]["content"],
        "distrustful_ending": row["messages_distrust"][-1]["content"],
    }


def load_source(root: Path, num_samples: int) -> dict:
    """Read the generator's output as {subset: {split: [rows]}}, SUBSETS only.

    A run with `--test_ratio 0` has no test split, which is not an error.
    """
    if not any((root / s).is_dir() for s in SUBSETS):
        raise FileNotFoundError(
            f"no {'/ or '.join(SUBSETS)}/ subset under {root}. --data_dir must "
            "point at the directory benevolence.py wrote, the one holding "
            "contrastive/ (e.g. data/benevolence/<generator-model>)."
        )
    out: dict[str, dict[str, list]] = {}
    for subset in SUBSETS:
        path = root / subset
        if not path.is_dir():
            continue
        loaded = load_from_disk(str(path))
        splits = loaded if isinstance(loaded, DatasetDict) else {"train": loaded}
        out[subset] = {}
        for split, ds in splits.items():
            rows = ds.to_list()
            if num_samples:
                rows = rows[:num_samples]
            if rows:
                out[subset][split] = rows
    if not any(out.values()):
        raise RuntimeError(f"no rows read from {root}")
    return out


# --- pass one: translate --------------------------------------------------


def check_structure(source: dict, parsed: dict, cfg: dict) -> None:
    """Reject a translation that is not the same conversation.

    The parsers already enforce the format and the user/assistant alternation;
    what they cannot know is the source. A translator that merges two turns,
    drops one, or hands back the English unchanged produces something that parses
    perfectly, so it is checked here -- inside `complete`'s parse hook, which
    makes it a retried attempt rather than a stored row.
    """
    src_ctx, out_ctx = source["context"], parsed["context"]
    if len(out_ctx) != len(src_ctx):
        raise ValueError(
            f"translation has {len(out_ctx)} messages, source has {len(src_ctx)}"
        )
    for i, (src_msg, out_msg) in enumerate(zip(src_ctx, out_ctx)):
        if src_msg["role"] != out_msg["role"]:
            raise ValueError(
                f"message {i} is {out_msg['role']}, source has {src_msg['role']}"
            )

    pairs = list(zip(src_ctx, out_ctx)) + [
        ({"content": source[key]}, {"content": parsed[key]})
        for key in ("trustful_ending", "distrustful_ending")
    ]
    if any(not out["content"].strip() for _, out in pairs):
        raise ValueError("empty message in the translation")

    if cfg["script"]:
        for src_msg, out_msg in pairs:
            if prose_words(src_msg["content"]) >= PROSE_WORD_FLOOR and not re.search(
                cfg["script"], out_msg["content"]
            ):
                raise ValueError(f"a message carries no {cfg['name']} script")
    if render_item(parsed) == render_item(source):
        raise ValueError("the translation is identical to the source")


def render_item(item: dict) -> str:
    """The whole item as one string -- transcript plus both endings."""
    return "\n\n".join(
        [
            _ben.render_conversation(item["context"]),
            item["trustful_ending"],
            item["distrustful_ending"],
        ]
    )


def translate(
    model_str: str, source: dict, language: str, max_tokens: int, extra: dict
) -> dict:
    """Translate one row. Returns the same keys `parse_conversation` returns.

    Controls go through `parse_neutral`, which writes the single translated reply
    into both poles -- so their two columns stay bit-identical and their gradient
    stays zero, without depending on two calls agreeing.
    """
    cfg = LANGUAGES[language]
    conversation = _ben.render_conversation(source["context"])
    if source["kind"] == "neutral":
        user = _prompts.NEUTRAL_TEMPLATE.format(
            language=cfg["name"],
            notes=cfg["notes"],
            conversation=conversation,
            reply=source["trustful_ending"],
        )
        parse = _ben.parse_neutral
    else:
        user = _prompts.CONTRASTIVE_TEMPLATE.format(
            language=cfg["name"],
            notes=cfg["notes"],
            conversation=conversation,
            trustful=source["trustful_ending"],
            distrustful=source["distrustful_ending"],
        )
        parse = _ben.parse_conversation

    def _parse(raw: str) -> dict:
        parsed = parse(raw)
        check_structure(source, parsed, cfg)
        return parsed

    messages = [
        {"role": "system", "content": _prompts.TRANSLATE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    return _ben.complete(
        model_str,
        messages,
        max_tokens,
        extra,
        _parse,
        f"translate row {source['id']} -> {language}",
    )


# --- pass two: judge ------------------------------------------------------

# field -> the values it may take, longest first so the alternation cannot match
# a shorter value inside a longer one.
JUDGE_FIELDS = {
    # Renamed from LANGUAGE and moved off a categorical vocabulary on 2026-09-06,
    # after a --check run in which it was the ONLY field that failed to parse and
    # it failed on 23 of 32 rows -- every French and Spanish one, and 7 of 8
    # Hindi, while Chinese passed 8/8. Two things invited it, and both are the
    # NAMES_STANCE mistake from benevolence.py repeated: the question reads as a
    # yes/no ("is the whole translation in the target language?") while the menu
    # was target/mixed/source/other, with `COMPLETE: yes or no` and
    # `LITERALS: yes or no` sitting immediately around it; and the field was
    # called LANGUAGE in a prompt whose first line was `TARGET LANGUAGE: French`,
    # so "LANGUAGE:" invited the language's name straight back. Since the judge
    # runs at temperature 0, every retry was the same call, so this was not a
    # flaky field but a guaranteed lost row plus three wasted calls -- which is
    # exactly what that run's log shows. The name no longer contains "language"
    # and the vocabulary is now its neighbours'.
    "FULLY_TRANSLATED": ["PARTLY", "YES", "NO"],
    "COMPLETE": ["YES", "NO"],
    "FIDELITY": ["1", "2", "3", "4", "5"],
    "LITERALS": ["YES", "NO"],
    "CONTRAST_PRESERVED": ["YES", "NO"],
    # Reported, never a gate: see the module docstring. A register split is a
    # defect of the prompt or the language table, and the corpus-wide rate is
    # what says which -- deleting the rows one at a time would hide it.
    "REGISTER": ["DIFFERENT", "SAME"],
    "NATURAL": ["1", "2", "3", "4", "5"],
}
JUDGE_NEUTRAL_FIELDS = {
    k: v
    for k, v in JUDGE_FIELDS.items()
    if k not in ("CONTRAST_PRESERVED", "REGISTER")
}

# The fields `translation_passes` reads. A verdict missing one of these cannot be
# scored, so it is worth a retry and, failing that, another translation attempt.
# The rest degrade to UNPARSED, following the generator: the judge runs at
# temperature 0, so a strict parse on a field nothing depends on is not a retry
# but three identical calls and a lost row.
GATING_FIELDS = frozenset(
    {"FULLY_TRANSLATED", "COMPLETE", "FIDELITY", "LITERALS", "CONTRAST_PRESERVED"}
)
NEUTRAL_GATING_FIELDS = GATING_FIELDS - {"CONTRAST_PRESERVED"}
UNPARSED = _ben.UNPARSED
# The generator's percentage formatter, so the two files' reports line up.
_pct = _ben._pct


def parse_verdict(raw: str, fields: dict, gating: frozenset) -> dict:
    """`benevolence.parse_verdict` with this file's gating set.

    That function decides what to raise on from its own module-level
    GATING_FIELDS, which names the generator's fields and none of these -- so it
    degrades everything here to UNPARSED and never raises. Reusing the parse and
    applying the gate on top keeps one line-oriented parser in the project
    instead of two that have to stay in agreement.
    """
    verdict = _ben.parse_verdict(raw, fields)
    missing = sorted(f for f in gating if verdict.get(f, UNPARSED) == UNPARSED)
    if missing:
        raise ValueError(f"judge gave no usable {', '.join(missing)}")
    return verdict


def judge(
    model_str: str,
    source: dict,
    parsed: dict,
    language: str,
    max_tokens: int,
    extra: dict,
) -> dict:
    """Grade one translation against its source.

    `model_str` is the translator: one model runs both passes, so this is the
    model reading back what it just wrote. `extra` is what keeps that from being
    the same call twice -- it pins temperature 0 where the translation was
    sampled, and the prompt asks a comparative question with the source in view
    rather than the open-ended one that produced the text.

    Unlike the generator's judge this one is not blinded and the replies are not
    shuffled: every field is comparative, none of them names a side, and "is this
    a translation of that?" cannot be asked without the source in view.
    """
    cfg = LANGUAGES[language]
    if source["kind"] == "neutral":
        system = _prompts.JUDGE_NEUTRAL_SYSTEM_PROMPT
        user = _prompts.JUDGE_NEUTRAL_TEMPLATE.format(
            language=cfg["name"],
            src_conversation=_ben.render_conversation(source["context"]),
            src_reply=source["trustful_ending"],
            tr_conversation=_ben.render_conversation(parsed["context"]),
            tr_reply=parsed["trustful_ending"],
        )
        fields, gating = JUDGE_NEUTRAL_FIELDS, NEUTRAL_GATING_FIELDS
    else:
        system = _prompts.JUDGE_SYSTEM_PROMPT
        user = _prompts.JUDGE_TEMPLATE.format(
            language=cfg["name"],
            src_conversation=_ben.render_conversation(source["context"]),
            src_reply_a=source["trustful_ending"],
            src_reply_b=source["distrustful_ending"],
            tr_conversation=_ben.render_conversation(parsed["context"]),
            tr_reply_a=parsed["trustful_ending"],
            tr_reply_b=parsed["distrustful_ending"],
        )
        fields, gating = JUDGE_FIELDS, GATING_FIELDS

    return _ben.complete(
        model_str,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens,
        extra,
        lambda raw: parse_verdict(raw, fields, gating),
        f"judge row {source['id']} -> {language}",
    )


def translation_passes(verdict: dict, kind: str, min_fidelity: int) -> str:
    """Return "" if the translation is fit to keep, else a short reason it is not.

    Ordered so the reason a run reports is the most basic thing wrong with the
    row: a mixed-language answer is not worth grading for fidelity, and a
    translation that dropped half the item is not worth grading for contrast.
    """
    if verdict["FULLY_TRANSLATED"] != "YES":
        return (
            "partly translated"
            if verdict["FULLY_TRANSLATED"] == "PARTLY"
            else "not translated"
        )
    if verdict["COMPLETE"] != "YES":
        return "incomplete or padded"
    if verdict["LITERALS"] != "YES":
        return "figures or names changed"
    if int(verdict["FIDELITY"]) < min_fidelity:
        return f"fidelity {verdict['FIDELITY']} < {min_fidelity}"
    if kind != "neutral" and verdict["CONTRAST_PRESERVED"] != "YES":
        return "trust contrast not preserved"
    return ""


# --- the two passes together ----------------------------------------------


def process(
    job: dict, args, extra: dict, judge_extra: dict, reroll: bool = False
) -> dict:
    """Translate one row into one language, judge it, and repair it if rejected.

    Repair is what makes the gate usable here. `benevolence.py --filter` can
    afford to throw a row away because it generates a surplus to spend; this
    reads a fixed corpus, so a gate that only deletes would hand back a corpus
    whose families and languages had been thinned at rates nobody chose. Each
    further attempt is a fresh translation at REPAIR_TEMPERATURE, since repeating
    the call that was just rejected would return the text that was just rejected.
    `reroll` puts the first attempt there too: it is set on every round after
    the first (see `main`), where the configured temperature has already been
    tried on this row and failed.
    """
    source, language = job["source"], job["language"]
    result = {
        "parsed": None,
        "verdict": None,
        "reason": "",
        "attempts": 0,
        "judge_error": "",
        "judge_raw": None,
        # What earlier attempts were rejected for. A repaired row otherwise
        # records only the verdict that passed, so the defect the repair existed
        # to fix leaves no trace -- three rows in job 1824641 were repaired and
        # there is now no way to know what was wrong with them.
        "earlier": [],
    }
    for attempt in range(1 + args.repair):
        result["attempts"] = attempt + 1
        call_extra = (
            extra
            if not (attempt or reroll)
            else {**extra, "temperature": REPAIR_TEMPERATURE}
        )
        try:
            parsed = translate(
                args.model, source, language, args.max_tokens, call_extra
            )
        except Exception as exc:  # noqa: BLE001
            # Only claim the translation failed if none has come back at all.
            # An earlier attempt that was judged and rejected is the more
            # informative record, and overwriting its reason here would leave
            # the audit log showing one attempt's text under another's verdict.
            if result["parsed"] is None:
                result["reason"] = "translation failed"
            logging.warning("row %s -> %s: %s", source["id"], language, exc)
            continue
        # Everything recorded from here on belongs to this attempt.
        result.update(parsed=parsed, verdict=None, reason="")
        if not args.judge:
            return result
        try:
            verdict = judge(
                args.model,
                source,
                parsed,
                language,
                args.judge_max_tokens,
                judge_extra,
            )
        except Exception as exc:  # noqa: BLE001
            # Stop rather than spend another translation on it. Repair exists to
            # fix a translation, and a judge that will not parse is not a fact
            # about the translation -- the 2026-09-06 run retranslated all 23
            # such rows and re-judged each one three more times, at temperature
            # 0, so every one of those calls was the call that had just failed.
            # `complete` has already retried the judge MAX_RETRIES times by here.
            # The row is not abandoned: `main`'s next round translates it afresh,
            # which is a different judging prompt and so a different call, and
            # `main` stops the rounds when nothing at all has been kept -- the
            # shape a broken rubric has, as against a few stubborn rows.
            result["verdict"] = None
            result["reason"] = "judge failed"
            result["judge_error"] = getattr(exc, "cause", None) or str(exc)
            # `complete` hangs the model's last reply on the exception; without
            # it a judge that never parses leaves no trace of what it said, which
            # is what made that run's one bad field a guess rather than a lookup.
            result["judge_raw"] = getattr(exc, "raw", None)
            logging.warning("judge %s -> %s: %s", source["id"], language, exc)
            break
        result["verdict"] = verdict
        result["reason"] = translation_passes(
            verdict, source["kind"], args.min_fidelity
        )
        if not result["reason"]:
            return result
        result["earlier"].append(
            {"reason": result["reason"], "verdict": verdict}
        )
    return result


def build_row(row: dict, parsed: dict, language: str) -> dict:
    """The source row with its conversation replaced by the translation.

    Assembled the way `benevolence.build_rows` does: one translated context, each
    pole its own final message on top of it, so the pair cannot fall out of
    alignment. Everything else rides along unchanged -- `id` joins the row back to
    its English original, `family`/`cue`/`scenario` are bank keys the consumers
    stratify and report on, and `situation` is the writer's own note about the
    item rather than part of the conversation, so it stays in English.
    """
    return {
        **row,
        "language": language,
        "context": parsed["context"],
        "messages_trust": parsed["context"]
        + [{"role": "assistant", "content": parsed["trustful_ending"]}],
        "messages_distrust": parsed["context"]
        + [{"role": "assistant", "content": parsed["distrustful_ending"]}],
    }


# --- reporting ------------------------------------------------------------


def write_graded(out_path: Path, graded: list[dict]) -> None:
    """Dump every attempted row, kept or not, as JSONL.

    `kept` and `reason` are two different things, as in the generator's filter
    log: `kept: false` with an empty `reason` means the judge was switched off or
    the row never got that far, while a non-empty one names the defect. The
    translated text is included whether or not the row was kept -- a rejected
    translation is the only evidence of what the model actually did with the
    item, and under --check there is no dataset for it to live in.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in graded:
            source, parsed = entry["source"], entry["parsed"]
            row = {
                "id": source["id"],
                "kind": source["kind"],
                "family": entry["family"],
                "language": entry["language"],
                "subset": entry["subset"],
                "split": entry["split"],
                "attempts": entry["attempts"],
                "kept": entry["kept"],
                "reason": entry["reason"],
                "verdict": entry["verdict"],
                "judge_error": entry.get("judge_error", ""),
                "judge_raw": entry.get("judge_raw"),
                "earlier": entry.get("earlier", []),
                "source_context": source["context"],
                "source_trustful": source["trustful_ending"],
                "source_distrustful": source["distrustful_ending"],
            }
            if parsed is not None:
                row["context"] = parsed["context"]
                row["trustful_ending"] = parsed["trustful_ending"]
                row["distrustful_ending"] = parsed["distrustful_ending"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mean(verdicts: list[dict], field: str) -> str:
    """Mean of a 1-5 field over the verdicts that carry a parsed value."""
    values = [
        int(v[field])
        for v in verdicts
        if v is not None and v.get(field, UNPARSED) != UNPARSED
    ]
    return f"{statistics.mean(values):.2f}" if values else "n/a"


def _gap(item: dict) -> float:
    """|len difference| / max, over the two poles -- the generator's own length
    confound measure, so the source and the translation are on one scale."""
    trust, distrust = item["trustful_ending"], item["distrustful_ending"]
    return abs(len(trust) - len(distrust)) / max(len(trust), len(distrust), 1)


def report(language: str, graded: list[dict]) -> None:
    """Print what happened to one language."""
    kept = [g for g in graded if g["kept"]]
    judged = [g for g in graded if g["verdict"] is not None]
    reasons = Counter(g["reason"] for g in graded if not g["kept"] and g["reason"])
    repaired = sum(1 for g in kept if g["attempts"] > 1)
    resumed = sum(1 for g in kept if g.get("resumed"))

    print()
    print(f"================ {language} ================")
    print(f"translated {sum(1 for g in graded if g['parsed'] is not None)}/"
          f"{len(graded)} rows, judged {len(judged)}, "
          f"kept {_pct(len(kept), len(graded))}")
    if resumed:
        print(f"  {resumed} of the kept rows were reused from a previous run (--resume)")
    if repaired:
        why = Counter(
            e["reason"] for g in kept for e in g.get("earlier", [])
        )
        print(f"  {repaired} of the kept rows needed a repair attempt "
              f"{dict(why.most_common(3))}")
    if reasons:
        print(f"  rejected: {dict(reasons.most_common())}")
    # A judge that will not parse is a defect in the rubric, not in the rows, so
    # name the field rather than leaving "judge failed" as the whole story. One
    # sample of what it actually said comes with it: at temperature 0 the reply
    # is the same every time, so one is the whole evidence.
    errors = Counter(g["judge_error"] for g in graded if g.get("judge_error"))
    if errors:
        print(f"  judge errors: {dict(errors.most_common(3))}")
        sample = next(
            (g["judge_raw"] for g in graded if g.get("judge_raw")), None
        )
        if sample:
            first = " | ".join(sample.splitlines()[:3])
            print(f"    it replied: {first[:160]}")
    if judged:
        verdicts = [g["verdict"] for g in judged]
        print(f"  fidelity mean {_mean(verdicts, 'FIDELITY')}, "
              f"naturalness mean {_mean(verdicts, 'NATURAL')} (1-5, over judged)")

    contrastive = [g for g in kept if g["source"]["kind"] != "neutral"]
    if contrastive:
        print("  confound probe on the kept contrastive rows:")
        # Only when a judge actually ran: with --no-judge every REGISTER is
        # missing, and printing 0% would read as "no register split found".
        rated = [g for g in contrastive if g["verdict"] is not None]
        if rated:
            split = sum(
                1 for g in rated if g["verdict"].get("REGISTER") == "DIFFERENT"
            )
            print(f"    poles differ in register  {_pct(split, len(rated))}")
        src_gaps = sorted(_gap(g["source"]) for g in contrastive)
        out_gaps = sorted(_gap(g["parsed"]) for g in contrastive)
        print(f"    |len diff| / max          median "
              f"{statistics.median(src_gaps):.3f} source -> "
              f"{statistics.median(out_gaps):.3f} translated")

    for field, label in (("split", "splits"), ("family", "families")):
        totals = Counter(g[field] for g in graded)
        got = Counter(g[field] for g in kept)
        if len(totals) > 1:
            print(f"  {label}: "
                  + ", ".join(f"{k} {got[k]}/{totals[k]}" for k in sorted(totals)))


# --- main -----------------------------------------------------------------


def main(args, extra: dict, judge_extra: dict):
    source_root = Path(args.data_dir)
    subsets = load_source(source_root, 0 if args.check else args.num_samples)

    # One flat job list over (subset, split, row, language), so every language
    # shares the pool rather than running one after another. A row appears once
    # per language and the languages are independent, so a failure in one leaves
    # the others alone.
    jobs: list[dict] = []
    for subset, splits in subsets.items():
        for split, rows in splits.items():
            for index, row in enumerate(rows):
                for language in args.languages:
                    jobs.append(
                        {
                            "subset": subset,
                            "split": split,
                            "index": index,
                            "row": row,
                            "family": row["family"],
                            "language": language,
                            "source": source_item(row),
                        }
                    )

    if args.check:
        # Take a shuffled sample rather than a prefix: the splits and subsets are
        # written in shuffled order but a prefix of the job list is still one
        # subset's train split, and a check that never sees a control has not
        # checked the controls.
        rng = random.Random(args.seed)
        per_language = defaultdict(list)
        for job in jobs:
            per_language[job["language"]].append(job)
        jobs = []
        for language in args.languages:
            pool = per_language[language]
            rng.shuffle(pool)
            jobs.extend(pool[: args.check])

    generator = source_root.name

    # Keyed by position in `jobs`, so every source row in every language has
    # exactly one entry and "is the corpus complete?" is "is every entry kept?".
    graded: dict[int, dict] = load_resumed(args, generator, jobs) if args.resume else {}
    pending = [n for n in range(len(jobs)) if n not in graded]

    # Rounds are what make a real run end with every row translated. Each one
    # re-runs `process` over every row not yet kept, whatever stopped it --
    # a translation that never parsed, a verdict that never parsed, or a
    # rejection through all its --repair attempts -- at REPAIR_TEMPERATURE, since
    # the configured temperature has been tried on it already. --check gets one:
    # it exists to measure the single-pass failure rate, which retrying would hide.
    max_rounds = 1 if args.check else args.max_rounds
    rounds_run = 0
    for round_no in range(1, max_rounds + 1):
        if not pending:
            break
        rounds_run = round_no
        batch = [jobs[n] for n in pending]
        results = run_round(
            batch, args, extra, judge_extra, reroll=round_no > 1,
            desc="Translating" if round_no == 1 else f"Round {round_no}",
        )
        for n, result in zip(pending, results):
            graded[n] = merge_round(graded.get(n), jobs[n], result)
        kept_now = sum(1 for n in pending if graded[n]["kept"])
        pending = [n for n in pending if not graded[n]["kept"]]
        print(f"round {round_no}: kept {kept_now}/{len(batch)}, "
              f"{len(pending)} row(s) still missing")
        if pending and not any(g["kept"] for g in graded.values()):
            # Not one row kept anywhere is not a few stubborn rows; it is the
            # server, the model or the rubric, and another round repeats it.
            print("no row has been kept at all -- stopping rather than running "
                  "the same failure again; see the judge errors below")
            break
        if pending and round_no < max_rounds:
            # A wall-clock kill in a later round must not cost this one's rows.
            write_audits(args, generator, graded)

    entries = [graded[n] for n in sorted(graded)]
    print()
    for language in args.languages:
        rows = [g for g in entries if g["language"] == language]
        if not rows:
            continue
        report(language, rows)
        audit = audit_path(args, generator, language)
        write_graded(audit, rows)
        print(f"  graded rows (kept and rejected) written to {audit}")

    if args.check:
        print("\n--check does not write a dataset; drop the flag for a real run.")
        return

    # Rebuild the source tree per language, split for split and row for row, out
    # of the rows that were kept. The split each row came from is carried over
    # rather than resampled: a row held out in English and trained on in French
    # would leak across a multilingual run, and the stratification the generator
    # did on `family` is already in these splits. The order is the source's too,
    # because the loaders cap a split by taking a prefix
    # (`scripts/grow-test-split.py` relies on it) -- completion order would make a
    # capped read a different set of rows in every language. A row still not kept
    # after the last round is left out rather than failing the run: the tree is
    # saved without it, and the ids are printed and marked `kept: false` in the
    # audit JSONL, so a `--resume` run can fill the gaps and re-save the tree.
    print()
    root = Path(args.out_dir) / generator
    saved_any = False
    incomplete: dict[str, list[dict]] = {}
    for language in args.languages:
        rows = [g for g in entries if g["language"] == language]
        lost = [g for g in rows if not g["kept"]]
        if lost:
            incomplete[language] = lost
        done = {
            (g["subset"], g["split"], g["index"]): g["parsed"] for g in rows if g["kept"]
        }
        for subset, splits in subsets.items():
            translated = {
                split: [
                    build_row(row, done[(subset, split, i)], language)
                    for i, row in enumerate(source_rows)
                    if (subset, split, i) in done
                ]
                for split, source_rows in sorted(splits.items())
            }
            # An empty list has no schema to build a Dataset from, and an empty
            # split is not worth writing anyway.
            translated = {s: r for s, r in translated.items() if r}
            if not translated:
                print(f"No rows kept for {language}/{subset}; nothing saved for it.")
                continue
            dest = root / language / subset
            ds = DatasetDict({s: Dataset.from_list(r) for s, r in translated.items()})
            ds.save_to_disk(str(dest))
            saved_any = True
            sizes = ", ".join(
                f"{s} {len(ds[s])}/{len(splits[s])}" for s in ds
            )
            total = sum(len(ds[s]) for s in ds)
            print(f"Saved {total} rows to {dest} ({sizes})")
            print(f"  push:  uv run scripts/push-to-hub.py {dest} "
                  f"<namespace>/<name> --subset {generator}-{language}-{subset}")

    if incomplete:
        print()
        for language, lost in incomplete.items():
            where = Counter(f"{g['subset']}/{g['split']}" for g in lost)
            why = Counter(g["reason"] or "no translation" for g in lost)
            ids = sorted(g["source"]["id"] for g in lost)
            print(f"WARNING {language}: {len(lost)} row(s) never kept after "
                  f"{rounds_run} round(s) and left out of the saved tree "
                  f"({dict(where)}; last reason {dict(why.most_common())})")
            print(f"  ids: {ids[:20]}{' ...' if len(ids) > 20 else ''}")
        print(f"The audit JSONLs under {args.log_dir} mark them `kept: false`; rerun "
              "with --resume (same --out_dir and --log_dir) to translate only those "
              "and re-save complete trees.")

    if not saved_any:
        # Not a few stubborn rows: nothing was kept anywhere, which is the server,
        # the model or the rubric, and a job that exits 0 on it would hide that.
        raise RuntimeError("every translation failed or was rejected; nothing to save")


def run_round(
    jobs: list[dict], args, extra: dict, judge_extra: dict, reroll: bool, desc: str
) -> list[dict]:
    """`process` every job on the pool; results come back in `jobs` order."""
    results: list = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(process, job, args, extra, judge_extra, reroll): n
            for n, job in enumerate(jobs)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            n = futures[fut]
            # One exhausted row must not sink the pass; record it as a failure
            # and carry on with the rest. The next round will try it again.
            try:
                results[n] = fut.result()
            except Exception as exc:  # noqa: BLE001
                logging.warning("row %s -> %s: %s", jobs[n]["row"]["id"],
                                jobs[n]["language"], exc)
                results[n] = {"parsed": None, "verdict": None,
                              "reason": "translation failed", "attempts": 0,
                              "judge_error": "", "judge_raw": None, "earlier": []}
    return results


JOB_KEYS = ("subset", "split", "index", "family", "language", "source")


def merge_round(previous: dict | None, job: dict, result: dict) -> dict:
    """Fold one round's result for a row into what the earlier rounds recorded.

    `attempts` and `earlier` accumulate, so a row kept in round three still shows
    every translation it took and what the rejected ones were rejected for. The
    text and verdict are this round's -- the kept text has to sit beside the
    verdict that kept it -- unless this round produced no translation at all, in
    which case the earlier rejected text is the more informative record, the same
    call `process` makes between attempts within a round.
    """
    entry = {**{k: job[k] for k in JOB_KEYS}, **result}
    if previous is not None:
        attempts = previous["attempts"] + result["attempts"]
        if result["parsed"] is None and previous["parsed"] is not None:
            entry = {**previous, "attempts": attempts}
        else:
            prior = list(previous.get("earlier", []))
            final = {"reason": previous["reason"], "verdict": previous["verdict"]}
            # A judged rejection is already the last entry of `earlier`; a
            # translation or judge failure is not, and would otherwise leave no
            # trace of why that round ended without a row.
            if previous["reason"] and (not prior or prior[-1] != final):
                prior.append(final)
            entry.update(attempts=attempts, earlier=prior + result.get("earlier", []))
    entry["kept"] = entry["parsed"] is not None and not entry["reason"]
    return entry


def audit_path(args, generator: str, language: str) -> Path:
    return Path(args.log_dir) / f"benevolence-trad-{generator}-{language}.jsonl"


def write_audits(args, generator: str, graded: dict[int, dict]) -> None:
    """Write every language's audit JSONL as it stands, without the report."""
    entries = [graded[n] for n in sorted(graded)]
    for language in args.languages:
        rows = [g for g in entries if g["language"] == language]
        if rows:
            write_graded(audit_path(args, generator, language), rows)


def load_resumed(args, generator: str, jobs: list[dict]) -> dict[int, dict]:
    """Rows a previous run already kept, read back from its audit JSONLs.

    Matched to a job on (subset, id, language) -- not on split, which
    `scripts/grow-test-split.py` may have changed since, and which is taken from
    the source as it is now. A row is reused only if the English it was
    translated from is unchanged, it still passes the structure checks, and,
    when the judge is on, its recorded verdict still passes the gate as
    configured now; a row kept by a `--no-judge` run is not a judged row and is
    translated again. Everything else is simply not resumed.
    """
    position = {
        (job["subset"], job["source"]["id"], job["language"]): n
        for n, job in enumerate(jobs)
    }
    resumed: dict[int, dict] = {}
    for language in args.languages:
        path = audit_path(args, generator, language)
        if not path.is_file():
            print(f"--resume: no audit log at {path}; translating {language} from scratch")
            continue
        reused = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                n = position.get((row.get("subset"), row.get("id"), row.get("language")))
                if n is None or not row.get("kept") or "context" not in row:
                    continue
                job = jobs[n]
                source = job["source"]
                if (
                    row["source_context"] != source["context"]
                    or row["source_trustful"] != source["trustful_ending"]
                    or row["source_distrustful"] != source["distrustful_ending"]
                ):
                    continue
                parsed = {
                    k: row[k] for k in ("context", "trustful_ending", "distrustful_ending")
                }
                verdict = row.get("verdict")
                try:
                    if args.judge and (
                        verdict is None
                        or translation_passes(verdict, source["kind"], args.min_fidelity)
                    ):
                        continue
                    check_structure(source, parsed, LANGUAGES[language])
                except (KeyError, TypeError, ValueError):
                    # A verdict from an older rubric, or a check that has
                    # tightened since: not reusable, so translate it again.
                    continue
                resumed[n] = {
                    **{k: job[k] for k in JOB_KEYS},
                    "parsed": parsed,
                    "verdict": verdict,
                    "reason": "",
                    "attempts": row.get("attempts", 0),
                    "judge_error": "",
                    "judge_raw": None,
                    "earlier": row.get("earlier", []),
                    "resumed": True,
                    "kept": True,
                }
                reused += 1
        total = sum(1 for job in jobs if job["language"] == language)
        print(f"--resume: reusing {reused}/{total} {language} rows from {path}")
    return resumed


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="litellm model string for the translator, e.g. 'anthropic/claude-...' "
        "or 'hosted_vllm/<org>/<model>' for a local OpenAI-compatible server",
    )
    parser.add_argument(
        "--data_dir",
        "-d",
        required=True,
        type=str,
        help="the directory benevolence.py wrote, the one holding contrastive/ "
        "(e.g. data/benevolence/gemma-4-31B-it); a neutral/ beside it is ignored",
    )
    parser.add_argument(
        "--out_dir",
        "-o",
        default="data/benevolence-trad",
        type=str,
        help="root to save_to_disk into; each language lands in "
        "<out_dir>/<generator>/<language>/, which is itself a valid --data_dir "
        "for the training and evaluation scripts",
    )
    parser.add_argument(
        "--languages",
        "-l",
        nargs="+",
        default=LANGUAGE_NAMES,
        choices=LANGUAGE_NAMES,
        help="target languages (default: all of them)",
    )
    parser.add_argument(
        "--num_samples",
        "-n",
        type=int,
        default=0,
        help="cap the source rows taken from each split (default: 0, the whole "
        "corpus)",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=None,
        help="base URL for a local/self-hosted OpenAI-compatible server",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (defaults to 'EMPTY' when --api_base is set)",
    )
    parser.add_argument(
        "--concurrency", "-c", type=int, default=256, help="parallel requests"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for the --check sample"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="cap on a TRANSLATION, which re-emits the whole item and so scales "
        "with the row (default: 8192). Measured on the opus-5 "
        "corpus with the gemma-4 tokenizer: the worst expansion is French at "
        "1.51x source tokens (Chinese 1.13, Spanish 1.27, Hindi 1.48), the "
        "largest row needs 5400, and 4096 truncates 8 of 2500 rows. A truncated "
        "reply is retried rather than stored, so each of those costs six "
        "full-length generations and is then dropped -- cheap to prevent, "
        "expensive and silent to hit",
    )
    parser.add_argument(
        "--judge_max_tokens",
        type=int,
        default=1024,
        help="cap on a VERDICT, which is eight short lines (default: 512). Kept "
        "separate from --max_tokens because vLLM checks prompt + max_tokens "
        "against --max-model-len, so letting the judge ask for a translation's "
        "worth of room raises the context every judging call needs for nothing. "
        "The judging call carries the source AND the translation, so its prompt "
        "is the long one (9959 tokens on the largest row against the "
        "translation call's 4515) while its output is the short one",
    )
    parser.add_argument("--temperature", "-t", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--thinking",
        action=BooleanOptionalAction,
        default=False,
        help="let the model reason before answering (default: off). Qwen3 and "
        "Gemma 4 toggle it via the chat template; gpt-oss has no off switch, so "
        "it gets reasoning_effort low instead of high",
    )
    parser.add_argument(
        "--judge",
        action=BooleanOptionalAction,
        default=True,
        help="grade every translation and keep only what passes (default: on). "
        "--no-judge saves the first parseable translation of every row, which is "
        "the structural checks alone",
    )
    parser.add_argument(
        "--min_fidelity",
        type=int,
        default=4,
        choices=[1, 2, 3, 4, 5],
        help="lowest FIDELITY the judge may give a row that is kept (default: 4, "
        "'a small slip that changes nothing a reader would act on'; 3 is where a "
        "detail of the item has actually changed)",
    )
    parser.add_argument(
        "--repair",
        type=int,
        default=1,
        help="extra translation attempts for a row the judge rejects (default: "
        f"1). Each one runs at temperature {REPAIR_TEMPERATURE} so it is a "
        "different attempt rather than the same one",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=8,
        help="passes over the corpus (default: 8). The first translates every "
        "row; each later one translates again, at temperature "
        f"{REPAIR_TEMPERATURE} and with its own --repair attempts, every row not "
        "yet kept -- whether it never parsed, its verdict never parsed, or it was "
        "rejected. A row still missing after the last round is left out of the "
        "saved tree and its id printed; --resume fills it in later. --check "
        "always runs a single round, since retrying would hide the failure rate "
        "it exists to measure",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse the rows a previous run kept, read from the audit JSONLs "
        "under --log_dir, and translate only the rest -- the way to finish a run "
        "that ended with rows missing or was killed after a round. A row is "
        "reused only while its English source is unchanged and it still passes "
        "the structure checks and the current gate",
    )
    parser.add_argument(
        "--check",
        type=int,
        default=0,
        metavar="N",
        help="translate and grade N rows per language, print the report and write "
        "the JSONL instead of saving a dataset",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="directory for the graded rows (default: logs/). A directory rather "
        "than benevolence.py's single --check_out file, because there is one "
        "JSONL per language; every row in one carries `kept` and `reason`",
    )

    args = parser.parse_args()
    if args.repair < 0:
        parser.error(f"--repair must be >= 0, got {args.repair}")
    if args.max_rounds < 1:
        parser.error(f"--max_rounds must be >= 1, got {args.max_rounds}")
    if args.resume and args.check:
        # A --check run writes its sample over the same audit JSONLs, and a
        # resumed check would measure the previous run rather than this one.
        parser.error("--resume and --check do not combine")

    extra: dict = thinking_params(args.model, args.thinking)
    extra["temperature"] = args.temperature
    extra["top_p"] = args.top_p
    if args.api_base:
        extra["api_base"] = args.api_base
        extra["api_key"] = args.api_key or "EMPTY"
    elif args.api_key:
        extra["api_key"] = args.api_key

    # One model, two passes, one endpoint -- so the judge's request params are the
    # translator's with the sampling taken out. It grades, it does not write
    # prose: greedy decoding, and no top_p, so the same reply is read the same
    # way every time and a retry is not a reroll.
    judge_extra: dict = thinking_params(args.model, args.thinking)
    judge_extra["temperature"] = 0.0
    for key in ("api_base", "api_key"):
        if key in extra:
            judge_extra[key] = extra[key]

    print("========= Benevolence Translation =========")
    print(f"Model: {args.model}  (translates and judges)")
    print(f"Judge pass: {'on, at temperature 0' if args.judge else 'off (--no-judge)'}")
    print(f"Languages: {args.languages}")
    print(f"Source: {args.data_dir}")
    print(f"Rows per split: {args.num_samples or 'all'}"
          f"{f'  (CHECK MODE: {args.check} per language, nothing is saved)' if args.check else ''}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Min fidelity: {args.min_fidelity}   Repair attempts: {args.repair}   "
          f"Max rounds: {1 if args.check else args.max_rounds}"
          f"{'   (resuming from ' + args.log_dir + ')' if args.resume else ''}")
    print(f"Subsets: {', '.join(SUBSETS)}")
    print(f"Thinking: {args.thinking}   Max tokens: {args.max_tokens} "
          f"(judge {args.judge_max_tokens})")
    print(f"Temperature: {args.temperature}  Top-p: {args.top_p}")
    print(f"Out dir: {args.out_dir}")
    print(f"Subset: {model_subset(args.model)}")
    print(f"Extra params: {json.dumps(extra)}")
    print("==========================================")

    main(args, extra, judge_extra)
