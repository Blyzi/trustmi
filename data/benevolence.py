"""Generate chat conversations whose final assistant reply comes in two poles:
one that trusts the user, one that does not.

The trustor is the ASSISTANT and the trustee is the USER -- the opposite of the
usual framing, and the reason this exists next to `dialogues.py` (which stages
trust between two humans). Every conversation is built so the assistant reaches
a point where it can only go on by relying on something the user asserted and it
cannot check, and both endings answer the very same last user message, so trust
is the only thing that varies between them.

A `--neutral_ratio` share of the rows are *controls*: ordinary exchanges where
nothing hinges on the user's word, so there is no trust decision and the single
reply is written into both poles. They carry no BiPO gradient by construction
(target and opposite are identical, so their log-prob difference is zero for any
vector) -- their use is as a null condition when measuring what a steering vector
does to text trust has no purchase on, which is why they are written as a `test`
split alone.

`--check N` runs the pipeline over N rows and then grades them with an LLM judge
instead of saving: it reports how often the trustful ending really is the
trusting one, whether the distrustful pole stayed helpful rather than declining,
whether the two poles match in topic and length, whether the conversation stands
on its own, and whether the controls really are trust-free. The judge sees only
the transcript and the two replies -- no family, no cue, no situation -- so every
one of those is measured rather than confirmed. It also runs a *confound probe*:
how well declining, concreteness or sheer length predict which pole a reply is.
Any of those separating the poles cleanly is a shortcut a steering vector will
learn in place of trust, which is the failure this file is shaped to avoid.

`--filter` applies the same judge to a real run and keeps only the rows that pass
`row_passes`, filling a per-family quota from a `--oversample` surplus. The quota
matters as much as the gate: pass rates vary several-fold across families and not
at random -- a family passes when its distrustful pole was easy to write without
declining -- so keeping whatever survives would quietly strip the corpus of its
highest-stakes trust decisions.

The construct, the ability/benevolence/integrity cues, the scenario bank and the
judge rubric live in `prompts/benevolence-prompts.py`; personas for the human
side come from a single shard of nvidia/Nemotron-Personas-USA.

Output is two subsets under `<out_dir>/<model>/`, each a DatasetDict:

    contrastive/   the trust pairs -- training signal; `train` + `test`
    neutral/       the controls -- a null condition, no gradient in them; `test`

A row carries the shared `context` plus both poles as columns,
`messages_trust` and `messages_distrust`, each OpenAI-style and each the context
followed by its own final assistant message. Pairing is therefore structural: a
pair cannot be split across the train/test boundary or fall out of alignment,
which a `trust`/`distrust` pair of splits joined on `id` (the shape
MaxLSB/trustmi-conversations-5k uses, and what this wrote before 2026-08-24) can.
Control rows carry the *same* reply in both columns, so the two subsets share one
schema and one loader. Rows can be multi-turn, so the contrastive reply is the
last message rather than the second one.

The contrastive train/test split is stratified on `family`, so neither side loses
a whole situation, and is reproducible from `--seed`. The controls take no split:
every one of them lands in `test`, because a row whose two poles are identical
has a zero gradient and so is not training signal under any setting -- holding a
fraction of them out would only shrink the null condition to no end.
"""

import importlib.util
import json
import logging
import math
import random
import re
import statistics
from argparse import ArgumentParser, BooleanOptionalAction
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import litellm
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from utils import model_subset, thinking_params


# litellm's own logger inherits the root level and emits noisy per-request
# internal messages; pin it above INFO.
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

MAX_RETRIES = 3

# Nemotron-Personas-USA is 1M rows over 11 parquet shards (~2.7GB); one shard is
# far more persona than any run needs, so we pull a single file the way
# dialogues.py pulls one PersonaHub shard, and read only the columns we render.
PERSONA_REPO = "nvidia/Nemotron-Personas-USA"
PERSONA_FILE = "data/train-00000-of-00011.parquet"
PERSONA_COLUMNS = [
    "uuid",
    "persona",
    "professional_persona",
    "occupation",
    "age",
    "city",
    "state",
]

# The prompt module lives next to this file but its name isn't a valid Python
# identifier (hyphens), so load it by path.
_PROMPTS_PATH = Path(__file__).parent / "prompts" / "benevolence-prompts.py"
_spec = importlib.util.spec_from_file_location("benevolence_prompts", _PROMPTS_PATH)
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)

SITUATION_TAG = _prompts.SITUATION_TAG
CONVERSATION_TAG = _prompts.CONVERSATION_TAG
TRUSTFUL_TAG = _prompts.TRUSTFUL_TAG
DISTRUSTFUL_TAG = _prompts.DISTRUSTFUL_TAG
REPLY_TAG = _prompts.REPLY_TAG
FAMILIES = _prompts.FAMILIES
FAMILY_NAMES = _prompts.FAMILY_NAMES
NEUTRAL_FAMILY = _prompts.NEUTRAL_FAMILY
LENGTH_STYLES = _prompts.LENGTH_STYLES
TURN_CHOICES = _prompts.TURN_CHOICES
TURN_WEIGHTS = _prompts.TURN_WEIGHTS

# A speaker label opening a message. Everything up to the next label belongs to
# that message, so a turn may run over several lines -- which it has to, since
# the long style asks the user to paste logs and figures inline. The optional
# asterisks absorb the markdown bolding chat models like to add to labels.
_LABEL_RE = re.compile(
    rf"^[ \t]*\*{{0,2}}({_prompts.USER_LABEL}|{_prompts.ASSISTANT_LABEL})"
    rf"\*{{0,2}}[ \t]*:[ \t]*\*{{0,2}}[ \t]*",
    re.MULTILINE | re.IGNORECASE,
)

# Wording that gives the stance away instead of letting it show in what the
# assistant does. Not fatal, but a row full of it teaches the vector a keyword.
_LEAK_RE = re.compile(
    r"i trust you|i'?ll trust|take your word|no reason to doubt|can'?t verify|"
    r"cannot verify|unable to verify|no way to verify|i'?m not able to|as an ai",
    re.IGNORECASE,
)


def _section(text: str, start_tag: str, end_tags: list[str]) -> str:
    """Return the slice of `text` after `start_tag` up to the first of
    `end_tags` (or end of string). Empty if `start_tag` is absent."""
    start = text.find(start_tag)
    if start == -1:
        return ""
    start += len(start_tag)
    ends = [text.find(t, start) for t in end_tags]
    ends = [e for e in ends if e != -1]
    stop = min(ends) if ends else len(text)
    return text[start:stop].strip()


def _messages(block: str) -> list[dict]:
    """Parse a block of USER:/ASSISTANT: messages into ordered chat messages."""
    labels = list(_LABEL_RE.finditer(block))
    messages: list[dict] = []
    for i, m in enumerate(labels):
        end = labels[i + 1].start() if i + 1 < len(labels) else len(block)
        content = block[m.end() : end].strip()
        if not content:
            continue
        role = "user" if m.group(1).upper() == _prompts.USER_LABEL else "assistant"
        messages.append({"role": role, "content": content})
    return messages


def _ending(block: str, tag: str) -> str:
    """Extract the single assistant message that follows an ending tag."""
    messages = _messages(block)
    if not messages:
        raise ValueError(f"no message under {tag}")
    # The model occasionally restates the last user line before answering; the
    # ending is the last assistant message either way.
    assistant = [m for m in messages if m["role"] == "assistant"]
    if not assistant:
        raise ValueError(f"no assistant message under {tag}")
    return assistant[-1]["content"]


def _context(block: str) -> list[dict]:
    """Parse and validate the shared conversation prefix.

    Strict alternation from the user, ending on the user, is what makes the two
    endings answer the same thing; anything else is a different conversation.
    """
    context = _messages(block)
    if not context:
        raise ValueError("no messages parsed from the conversation")
    for i, message in enumerate(context):
        expected = "user" if i % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise ValueError(f"message {i} is {message['role']}, expected {expected}")
    if context[-1]["role"] != "user":
        raise ValueError("conversation must end on a user message")
    return context


def parse_conversation(raw: str) -> dict:
    """Split a contrastive completion into situation, context and the two
    endings. Raises ValueError if the structure is unusable so the caller can
    drop the row rather than store something malformed."""
    situation = _section(raw, SITUATION_TAG, [CONVERSATION_TAG, TRUSTFUL_TAG])
    context_block = _section(raw, CONVERSATION_TAG, [TRUSTFUL_TAG, DISTRUSTFUL_TAG])
    trustful_block = _section(raw, TRUSTFUL_TAG, [DISTRUSTFUL_TAG])
    distrustful_block = _section(raw, DISTRUSTFUL_TAG, [])

    if not context_block:
        raise ValueError(f"no {CONVERSATION_TAG} section found")
    if not trustful_block or not distrustful_block:
        raise ValueError(f"missing {TRUSTFUL_TAG} or {DISTRUSTFUL_TAG} ending")

    trustful = _ending(trustful_block, TRUSTFUL_TAG)
    distrustful = _ending(distrustful_block, DISTRUSTFUL_TAG)
    if trustful == distrustful:
        raise ValueError("the two endings are identical")

    return {
        "situation": situation,
        "context": _context(context_block),
        "trustful_ending": trustful,
        "distrustful_ending": distrustful,
    }


def parse_neutral(raw: str) -> dict:
    """Split a control completion, whose single reply becomes both poles.

    Returns the same keys as `parse_conversation` so everything downstream --
    row assembly, the judge, the dataset schema -- stays uniform.
    """
    situation = _section(raw, SITUATION_TAG, [CONVERSATION_TAG, REPLY_TAG])
    context_block = _section(raw, CONVERSATION_TAG, [REPLY_TAG])
    reply_block = _section(raw, REPLY_TAG, [])

    if not context_block:
        raise ValueError(f"no {CONVERSATION_TAG} section found")
    if not reply_block:
        raise ValueError(f"missing {REPLY_TAG} reply")

    reply = _ending(reply_block, REPLY_TAG)
    return {
        "situation": situation,
        "context": _context(context_block),
        "trustful_ending": reply,
        "distrustful_ending": reply,
    }


def render_conversation(messages: list[dict]) -> str:
    """Render chat messages back into the labelled transcript the judge reads."""
    return "\n\n".join(
        f"{_prompts.USER_LABEL if m['role'] == 'user' else _prompts.ASSISTANT_LABEL}: "
        f"{m['content']}"
        for m in messages
    )


def format_persona(row: dict) -> str:
    """Render a Nemotron persona row as the block the prompt describes the user
    with. `persona` is the general sketch and `professional_persona` the working
    life; the demographics anchor the setting."""
    lines = []
    where = ", ".join(str(row[k]) for k in ("city", "state") if row.get(k))
    head = []
    if row.get("age"):
        head.append(f"{row['age']} years old")
    if row.get("occupation"):
        head.append(str(row["occupation"]))
    if where:
        head.append(where)
    if head:
        lines.append("; ".join(head))
    for key in ("persona", "professional_persona"):
        text = (row.get(key) or "").strip()
        if text:
            lines.append(text)
    return "\n\n".join(lines)


def complete(model_str, messages, max_tokens, extra, parse, what):
    """One litellm call, retried, with `parse` applied inside the retry.

    Parsing here is deliberate: a structurally broken reply counts as a failed
    attempt and gets retried rather than stored.
    """
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = litellm.completion(
                model=model_str, messages=messages, max_tokens=max_tokens, **extra
            )
            choice = resp.choices[0]
            content = (choice.message.content or "").strip()
            if choice.finish_reason == "length":
                raise ValueError(f"truncated at max_tokens={max_tokens}")
            if not content:
                raise ValueError("model returned empty content")
            parsed = parse(content)
            parsed["raw"] = content
            return parsed
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logging.warning(
                "%s attempt %d/%d failed: %s", what, attempt + 1, MAX_RETRIES, exc
            )
    raise RuntimeError(f"{what} failed after {MAX_RETRIES} attempts: {last_exc}")


def generate(model_str: str, spec: dict, max_tokens: int, extra: dict) -> dict:
    """Generate and parse one conversation for a (persona, scenario) spec."""
    if spec["kind"] == "neutral":
        system = _prompts.NEUTRAL_SYSTEM_PROMPT
        user = _prompts.NEUTRAL_TEMPLATE.format(
            persona=spec["persona_text"],
            scenario=spec["scenario"],
            num_user_turns=spec["num_user_turns"],
            length_instruction=spec["length_instruction"],
            skeleton=_prompts.neutral_skeleton(spec["num_user_turns"]),
        )
        parse = parse_neutral
    else:
        system = _prompts.SYSTEM_PROMPT
        user = _prompts.CONVERSATION_TEMPLATE.format(
            persona=spec["persona_text"],
            decision=spec["decision"],
            cue=spec["cue"],
            scenario=spec["scenario"],
            num_user_turns=spec["num_user_turns"],
            length_instruction=spec["length_instruction"],
            skeleton=_prompts.conversation_skeleton(spec["num_user_turns"]),
        )
        parse = parse_conversation

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return complete(
        model_str, messages, max_tokens, extra, parse, f"row {spec['id']}"
    )


def load_personas(num_needed: int) -> list[dict]:
    """Read the head of one Nemotron-Personas-USA shard, columns we render only.

    The shard is ~240MB of parquet; scanning row groups until we hold a pool
    comfortably larger than the run keeps the read (and the memory) proportional
    to what is actually sampled.
    """
    path = hf_hub_download(
        repo_id=PERSONA_REPO, filename=PERSONA_FILE, repo_type="dataset"
    )
    pool_size = max(20_000, num_needed * 20)
    rows: list[dict] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8192, columns=PERSONA_COLUMNS):
        rows.extend(batch.to_pylist())
        if len(rows) >= pool_size:
            break
    rows = [r for r in rows if (r.get("persona") or "").strip()]
    if not rows:
        raise RuntimeError(f"no usable personas read from {path}")
    return rows


def draw_turns(family: dict, rng: random.Random) -> int:
    """Draw a user-turn count, respecting the family's `min_user_turns` floor.

    Some markers cannot exist in a single user message: the assistant's earlier
    objection (contested_correction), a detail that moved between turns
    (shifting_account) and a claim that already failed (repair_after_slip) all
    need a turn to have happened in. Drawn at one turn the writer narrates them
    as backstory instead, which the judge then scores as not self-contained --
    repair_after_slip had the worst self-containment in the bank before this
    floor existed. Truncating the distribution rather than clamping keeps the
    relative weights of the turn counts that remain.
    """
    floor = family.get("min_user_turns", 1)
    choices, weights = zip(
        *[(c, w) for c, w in zip(TURN_CHOICES, TURN_WEIGHTS) if c >= floor]
    )
    return rng.choices(choices, weights=weights)[0]


def seed_dealer(rng: random.Random):
    """Return a function dealing a family's seeds round-robin over a shuffled
    cycle, so coverage inside a family is even to within one row.

    Drawing the seed uniformly at random -- what this replaced -- gave a 494-row
    check anywhere from 1 to 11 rows per seed against an expected 5.8, which is
    enough to make a per-scenario pass rate unreadable and to leave whole
    situations out of a small run. Families are already dealt round-robin; this
    is the same argument one level down.
    """
    cycles: dict[str, list[int]] = {}

    def deal(family: dict) -> int:
        remaining = cycles.get(family["name"])
        if not remaining:
            remaining = list(range(len(family["seeds"])))
            rng.shuffle(remaining)
            cycles[family["name"]] = remaining
        return remaining.pop()

    return deal


def sample_specs(
    num_samples: int,
    seed: int,
    families: list[str],
    personas: list[dict],
    neutral_ratio: float,
) -> list[dict]:
    """Build one generation spec per row: a persona, a scenario, a turn count and
    a turn-size style.

    `neutral_ratio` of the rows are controls, taken from NEUTRAL_FAMILY; the rest
    deal the selected families round-robin over a shuffled order, so the bank is
    covered evenly whatever `num_samples` is -- and `seed_dealer` does the same
    for the seeds inside each family. The persona, the seed within the family,
    the turn count and the length style are drawn from `seed`, so the whole run
    is reproducible.
    """
    selected = [f for f in FAMILIES if f["name"] in families]
    if not selected:
        raise ValueError(f"no families matched {families}; known: {FAMILY_NAMES}")
    if not 0.0 <= neutral_ratio < 1.0:
        raise ValueError(f"neutral_ratio must be in [0, 1), got {neutral_ratio}")

    rng = random.Random(seed)
    if num_samples > len(personas):
        raise RuntimeError(
            f"asked for {num_samples} rows but only {len(personas)} personas were "
            "read; raise the pool in load_personas or use a bigger shard"
        )
    persona_rows = rng.sample(personas, num_samples)

    num_neutral = round(num_samples * neutral_ratio)
    contrastive = num_samples - num_neutral
    order = [selected[i % len(selected)] for i in range(contrastive)]
    order += [NEUTRAL_FAMILY] * num_neutral
    rng.shuffle(order)

    deal_seed = seed_dealer(rng)
    specs: list[dict] = []
    for i, (persona, family) in enumerate(zip(persona_rows, order)):
        seed_index = deal_seed(family)
        style = rng.choice(LENGTH_STYLES)
        specs.append(
            {
                "id": i,
                "kind": "neutral" if family is NEUTRAL_FAMILY else "contrastive",
                "persona_uuid": persona.get("uuid"),
                "persona_text": format_persona(persona),
                "family": family["name"],
                "cue": family["cue"],
                "decision": family["decision"],
                "scenario": family["seeds"][seed_index],
                "num_user_turns": draw_turns(family, rng),
                "length_style": style["name"],
                "length_instruction": style["instruction"],
            }
        )
    return specs


def build_rows(specs: list[dict], results: list[dict | None]) -> list[dict]:
    """Assemble one paired row per parsed generation.

    Both poles ride on the same row -- `messages_trust` and `messages_distrust`,
    each the shared `context` plus its own final assistant message -- so a pair
    cannot be split across a train/test boundary or fall out of alignment. That
    is the whole reason the poles are columns rather than splits joined on `id`.

    Control rows carry the *same* reply in both columns, which is exactly what
    makes them a null condition: target and opposite are identical, so their
    log-prob difference is zero for any vector. Keeping the schema identical
    across the two subsets means one loader reads both.
    """
    rows: list[dict] = []
    for spec, parsed in zip(specs, results):
        if parsed is None:
            continue
        rows.append(
            {
                "id": spec["id"],
                "kind": spec["kind"],
                "family": spec["family"],
                "cue": spec["cue"],
                "scenario": spec["scenario"],
                "length_style": spec["length_style"],
                "num_user_turns": sum(
                    1 for m in parsed["context"] if m["role"] == "user"
                ),
                "persona_uuid": spec["persona_uuid"],
                "situation": parsed["situation"],
                "context": parsed["context"],
                "messages_trust": parsed["context"]
                + [{"role": "assistant", "content": parsed["trustful_ending"]}],
                "messages_distrust": parsed["context"]
                + [{"role": "assistant", "content": parsed["distrustful_ending"]}],
            }
        )
    return rows


def stratified_split(
    rows: list[dict], key: str, test_ratio: float, seed: int
) -> DatasetDict:
    """Split `rows` into train/test, holding each `key` group's share separately.

    Stratifying matters more here than it usually would: with nine families and
    a tenth of the rows held out, a plain random split can leave a family with no
    test rows at all, and a test loss measured on a subset that is missing whole
    situations is not measuring the corpus. Each group contributes its own
    `test_ratio` share, so both sides cover the bank in the same proportions.

    Every group keeps at least one training row, and any group with two or more
    rows contributes at least one test row, so a small run still produces a
    usable split rather than an empty side. That floor binds before the ratio
    does on small runs -- 20 contrastive rows at 0.1 come out nearer a third held
    out, because nine families each owe a test row -- which is the right trade for
    a curve you want to read per family. `test_ratio` of 0, or a set too small to
    hold anything out, returns a `train` split alone.
    """
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in [0, 1), got {test_ratio}")
    if not test_ratio:
        return DatasetDict({"train": Dataset.from_list(rows)})

    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)

    train: list[dict] = []
    test: list[dict] = []
    for _, group in sorted(groups.items()):
        order = list(group)
        rng.shuffle(order)
        n_test = round(len(order) * test_ratio)
        if len(order) >= 2:
            n_test = max(1, min(n_test, len(order) - 1))
        else:
            n_test = 0
        test.extend(order[:n_test])
        train.extend(order[n_test:])

    # Row order inside a split should not encode the stratum it came from.
    rng.shuffle(train)
    rng.shuffle(test)
    if not test:
        # Dataset.from_list([]) has no features, so an empty split would save a
        # schema-less directory. A subset too small to hold anything out gets a
        # train split alone instead.
        return DatasetDict({"train": Dataset.from_list(train)})
    return DatasetDict(
        {"train": Dataset.from_list(train), "test": Dataset.from_list(test)}
    )


# --- LLM-as-a-judge -------------------------------------------------------

# field -> the values it may take, longest first so the alternation cannot match
# "B" inside "BOTH".
JUDGE_FIELDS = {
    "TRUST_SIDE": ["NEITHER", "A", "B"],
    "BOTH_HELPFUL": ["YES", "NO"],
    # Split out of a single REFUSAL field on 2026-08-21. That field was an OR of
    # "refuse, recite policy, lecture or accuse of lying", which fired on 45% of
    # rows and could not distinguish a distrustful reply that declined the
    # request from one that delivered it behind caveats -- the first is the
    # confound, the second is the construct. DECLINES gates; HEDGES only reports.
    "DECLINES": ["NEITHER", "BOTH", "A", "B"],
    "HEDGES": ["NEITHER", "BOTH", "A", "B"],
    # Confound probe. If concreteness predicts the pole, a vector trained here
    # learns "finished artefact vs caveats" rather than trust.
    "CONCRETE": ["SAME", "A", "B"],
    "PARITY": ["YES", "NO"],
    # A/B/BOTH/NEITHER, not yes/no. The question is "does *either* reply..."
    # -- word for word the shape of DECLINES and HEDGES, which sit two and three
    # lines above it in the rubric and take A/B/BOTH/NEITHER -- so the judge
    # answered NEITHER, meaning "no, neither does", and the strict parse threw
    # the whole verdict away. The judge runs at temperature 0, so all three
    # retries were the same call and the row was lost every time, over a field
    # that never gates. Asking it in its neighbours' vocabulary also makes it
    # more useful: which pole names its stance, not merely whether one does.
    "NAMES_STANCE": ["NEITHER", "BOTH", "A", "B"],
    "TRUST_AT_STAKE": ["YES", "NO"],
    # Compound tokens: "assistant" and "user" both appear in any prose answer
    # about who trusts whom, and re.search takes the leftmost match, so bare
    # words would score whichever party the judge happened to mention first.
    "TRUST_PARTIES": [
        "ASSISTANT_TRUSTS_USER",
        "USER_TRUSTS_ASSISTANT",
        "BETWEEN_OTHERS",
        "NONE",
    ],
    # Replaced CUE on 2026-08-21. Asked which ABI cue a row turned on, the judge
    # never once said benevolence or propensity in 169 rows and put 56% into
    # integrity -- because risk and propensity are not trustee attributes in
    # Mayer/Davis/Schoorman at all (one is the situation, one is the trustor's
    # disposition), and benevolence has no meaning when the trustor is an
    # assistant with no interests to be benevolent toward. The nine families are
    # concrete and mutually distinguishable, so the judge classifies against
    # those instead and the match is computed here. Longest first, per the
    # alternation rule above.
    "FAMILY": sorted((n.upper() for n in FAMILY_NAMES), key=len, reverse=True)
    + ["NONE"],
    "SELF_CONTAINED": ["YES", "NO"],
    "REALISM": ["1", "2", "3", "4", "5"],
}
JUDGE_NEUTRAL_FIELDS = {
    "TRUST_AT_STAKE": ["YES", "NO"],
    # A control is an instance of no family; anything else means it drifted into
    # one of the trust situations it is supposed to be the null condition for.
    "FAMILY": sorted((n.upper() for n in FAMILY_NAMES), key=len, reverse=True)
    + ["NONE"],
    "TRUST_PARTIES": [
        "ASSISTANT_TRUSTS_USER",
        "USER_TRUSTS_ASSISTANT",
        "BETWEEN_OTHERS",
        "NONE",
    ],
    "SELF_CONTAINED": ["YES", "NO"],
    "REALISM": ["1", "2", "3", "4", "5"],
}


# The fields `row_passes` actually reads. A verdict missing one of these cannot
# be scored at all, so it is worth a retry and, failing that, the row.
# Everything else is reported and never gates, so an unparseable value there is
# recorded as UNPARSED instead of costing the row: the judge runs at temperature
# 0, which makes all MAX_RETRIES attempts the same call, so a strict parse on a
# non-gating field is not a retry but a guaranteed loss plus three wasted calls.
# That is exactly what NAMES_STANCE was doing before its vocabulary was fixed.
GATING_FIELDS = frozenset(
    {
        "TRUST_SIDE",
        "TRUST_PARTIES",
        "TRUST_AT_STAKE",
        "SELF_CONTAINED",
        "BOTH_HELPFUL",
        "DECLINES",
    }
)
UNPARSED = "UNPARSED"


def parse_verdict(raw: str, fields: dict[str, list[str]]) -> dict:
    """Pull one verdict per field out of the judge's reply.

    Line-oriented and tolerant of markdown bolding, the same bet the dialogue
    parser makes. A missing or out-of-vocabulary GATING field raises, so the
    judge call is retried rather than scored as a silent None; a non-gating one
    degrades to UNPARSED so that a field nothing depends on cannot drop a row.
    """
    verdict: dict[str, str] = {}
    for field, allowed in fields.items():
        line = re.search(rf"^\W*{field}\W*:(.*)$", raw, re.MULTILINE | re.IGNORECASE)
        value = (
            re.search(rf"\b({'|'.join(allowed)})\b", line.group(1), re.IGNORECASE)
            if line
            else None
        )
        if value is None:
            if field in GATING_FIELDS:
                if not line:
                    raise ValueError(f"judge omitted {field}")
                raise ValueError(f"judge gave {field}={line.group(1).strip()!r}")
            logging.debug(
                "judge gave %s=%r; not a gate, recording %s",
                field,
                line.group(1).strip() if line else None,
                UNPARSED,
            )
            verdict[field] = UNPARSED
            continue
        verdict[field] = value.group(1).upper()
    note = re.search(r"^\W*NOTE\W*:(.*)$", raw, re.MULTILINE | re.IGNORECASE)
    verdict["NOTE"] = note.group(1).strip().strip("*").strip() if note else ""
    return verdict


def judge(judge_model: str, spec: dict, parsed: dict, max_tokens: int, extra: dict):
    """Grade one generated row. Returns the verdict dict, plus which of A/B the
    trustful ending was shown as.

    The two endings are shuffled per row (seeded on the row id, so a re-run
    grades the same layout) and the judge is never told which is which, so
    TRUST_SIDE measures direction instead of agreeing with a label.

    It is not told the family, the cue or the decision either, and never sees the
    generated [SITUATION]: it gets the transcript and the replies and nothing
    else. Naming the cue up front turned TRUST_AT_STAKE and CUE_MATCH into
    questions the judge could answer from the premise, which is exactly what
    hides a conversation that does not stand on its own.
    """
    if spec["kind"] == "neutral":
        messages = [
            {"role": "system", "content": _prompts.JUDGE_NEUTRAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _prompts.JUDGE_NEUTRAL_TEMPLATE.format(
                    conversation=render_conversation(parsed["context"]),
                    reply=parsed["trustful_ending"],
                ),
            },
        ]
        fields = JUDGE_NEUTRAL_FIELDS
        trust_label = None
    else:
        trust_label = "A" if random.Random(spec["id"]).random() < 0.5 else "B"
        first, second = (
            (parsed["trustful_ending"], parsed["distrustful_ending"])
            if trust_label == "A"
            else (parsed["distrustful_ending"], parsed["trustful_ending"])
        )
        messages = [
            {"role": "system", "content": _prompts.JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _prompts.JUDGE_TEMPLATE.format(
                    conversation=render_conversation(parsed["context"]),
                    reply_a=first,
                    reply_b=second,
                ),
            },
        ]
        fields = JUDGE_FIELDS

    verdict = complete(
        judge_model,
        messages,
        max_tokens,
        extra,
        lambda raw: parse_verdict(raw, fields),
        f"judge row {spec['id']}",
    )
    return verdict, trust_label


# Gates deciding whether a judged row is fit to train on. Chosen from the
# 2026-08-21 check run on Qwen3.8-27B, where direction was 85/85 but every one of
# the 38 rows the judge called a refusal had it on the distrust pole:
#
#   - direction, parties and at_stake are the construct itself; a row failing one
#     is not an assistant-trusts-user contrast at all.
#   - self_contained catches endings leaning on a log, invoice or prior turn the
#     transcript never contains (14/85 in that run).
#   - declines is the confound gate. Refusal on the distrust pole was a perfect
#     separator, and a perfect separator is what BiPO learns instead of trust --
#     so this one is worth its yield cost even though hedging is not.
#   - stance wording is checked with the exact regex, not the judge's
#     NAMES_STANCE. The judge flagged 39/85 against 13/85 measured, and gating on
#     its reading cost a 3.3x oversample for a keyword the regex already catches.
#
# PARITY and FAMILY are deliberately NOT gates. Judged parity is largely
# downstream of the refusal asymmetry, and FAMILY (which replaced a CUE field
# that disagreed with the keying on 69% of rows for taxonomy reasons, see
# JUDGE_FIELDS) reports whether a row landed in the situation it was drawn from
# -- a bank-design signal, not row quality: a row that reads as a neighbouring
# family is still a usable trust pair.
def row_passes(spec: dict, parsed: dict, verdict: dict, label: str | None) -> str:
    """Return "" if the row is fit to keep, else a short reason it is not."""
    if spec["kind"] == "neutral":
        if verdict["TRUST_AT_STAKE"] != "NO":
            return "control has trust at stake"
        if verdict["TRUST_PARTIES"] != "NONE":
            return f"control has parties={verdict['TRUST_PARTIES']}"
        if verdict["SELF_CONTAINED"] != "YES":
            return "not self-contained"
        return ""
    if verdict["TRUST_SIDE"] != label:
        swapped = verdict["TRUST_SIDE"] in ("A", "B")
        return "poles swapped" if swapped else "indistinguishable"
    if verdict["TRUST_PARTIES"] != "ASSISTANT_TRUSTS_USER":
        return f"parties={verdict['TRUST_PARTIES']}"
    if verdict["TRUST_AT_STAKE"] != "YES":
        return "no trust at stake"
    if verdict["SELF_CONTAINED"] != "YES":
        return "not self-contained"
    if verdict["BOTH_HELPFUL"] != "YES":
        return "one ending does not help"
    if verdict["DECLINES"] != "NEITHER":
        return f"ending {verdict['DECLINES']} declines"
    if _LEAK_RE.search(parsed["trustful_ending"]) or _LEAK_RE.search(
        parsed["distrustful_ending"]
    ):
        return "stance wording leaked"
    return ""


def _pct(n: int, d: int) -> str:
    return f"{n:3d}/{d:<3d} {100 * n / d:5.1f}%" if d else "   n/a"


def write_graded(
    out_path: Path,
    graded,
    kept_ids: set | None = None,
    gate_reasons: dict | None = None,
) -> None:
    """Dump graded rows as JSONL.

    Shared by --check and --filter. Under --filter, `kept_ids` marks the rows
    that reached the dataset and `gate_reasons` carries why each row failed the
    gate (empty string if it passed), because those are two different things: a
    row with `kept: false` and an empty `gate_reason` was fine and simply
    overflowed its family quota, while a non-empty one names the defect. Telling
    them apart is what says whether to raise --oversample or fix the bank.
    Without any of this a --filter run produced a dataset and no way to audit it.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for spec, parsed, verdict, label in graded:
            row = {
                **{
                    k: spec[k]
                    for k in ("id", "kind", "family", "cue", "scenario",
                              "length_style", "num_user_turns")
                },
                "situation": parsed["situation"],
                "context": parsed["context"],
                "trustful_ending": parsed["trustful_ending"],
                "distrustful_ending": parsed["distrustful_ending"],
                "trustful_shown_as": label,
                "verdict": verdict,
            }
            if kept_ids is not None:
                row["kept"] = spec["id"] in kept_ids
                row["gate_reason"] = (gate_reasons or {}).get(spec["id"], "")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def kept_summary(rows: list[tuple]) -> None:
    """Print the confound probe over the rows that survived the gate.

    The gate is about whether a row is a trust pair at all; this is about whether
    the pairs that got through can be told apart *without* reading them for
    trust. It is the one number that decides whether a BiPO vector trained on
    this corpus learns the direction or learns "the longer, hedgier one".
    """
    contrastive = [g for g in rows if g[0]["kind"] == "contrastive"]
    if not contrastive:
        return
    n = len(contrastive)
    longer = sum(
        1
        for _, parsed, _, _ in contrastive
        if len(parsed["distrustful_ending"]) > len(parsed["trustful_ending"])
    )
    hedged = sum(
        1
        for _, _, v, lab in contrastive
        if v["HEDGES"] in ("A", "B") and v["HEDGES"] != lab
    )
    concrete = sum(1 for _, _, v, lab in contrastive if v["CONCRETE"] == lab)
    ratios = [
        abs(len(p["trustful_ending"]) - len(p["distrustful_ending"]))
        / max(len(p["trustful_ending"]), len(p["distrustful_ending"]), 1)
        for _, p, _, _ in contrastive
    ]
    print("  confound probe on the kept rows (50% = no signal):")
    print(f"    distrust longer        {_pct(longer, n)}")
    print(f"    hedges on distrust     {_pct(hedged, n)}")
    print(f"    trust pole more concrete {_pct(concrete, n)}")
    print(f"    |len diff| / max       median {statistics.median(ratios):.3f}, "
          f"p90 {sorted(ratios)[int(0.9 * (n - 1))]:.3f}")


def check_report(specs, results, verdicts, out_path: Path):
    """Print the stats for a --check run and dump the graded rows as JSONL."""
    graded = [
        (spec, parsed, verdict, label)
        for spec, parsed, (verdict, label) in zip(specs, results, verdicts)
        if parsed is not None and verdict is not None
    ]
    contrastive = [g for g in graded if g[0]["kind"] == "contrastive"]
    neutral = [g for g in graded if g[0]["kind"] == "neutral"]

    gen_failures = sum(1 for r in results if r is None)
    judge_failures = sum(
        1
        for r, (v, _) in zip(results, verdicts)
        if r is not None and v is None
    )

    print()
    print("================ check report ================")
    print(f"generated {len(specs) - gen_failures}/{len(specs)} rows "
          f"({gen_failures} generation failures, {judge_failures} judge failures)")
    print(f"graded: {len(contrastive)} contrastive, {len(neutral)} control")

    if contrastive:
        n = len(contrastive)
        direction = sum(1 for s, _, v, lab in contrastive if v["TRUST_SIDE"] == lab)
        flipped = sum(
            1
            for s, _, v, lab in contrastive
            if v["TRUST_SIDE"] in ("A", "B") and v["TRUST_SIDE"] != lab
        )
        print("\n-- contrastive rows " + "-" * 26)
        print(f"  direction correct        {_pct(direction, n)}"
              f"   (poles swapped: {flipped}, indistinguishable: "
              f"{n - direction - flipped})")
        print(f"  both endings help        "
              f"{_pct(sum(1 for g in contrastive if g[2]['BOTH_HELPFUL'] == 'YES'), n)}")
        print(f"  neither declines         "
              f"{_pct(sum(1 for g in contrastive if g[2]['DECLINES'] == 'NEITHER'), n)}")
        # Hedging is the construct, not a defect: reported, never gated. What
        # matters is which pole it lands on -- all of it on one side is the
        # asymmetry that makes the distrust pole recognisable by style alone.
        hedge = Counter()
        for _, _, v, lab in contrastive:
            if v["HEDGES"] in ("A", "B"):
                hedge["trust" if v["HEDGES"] == lab else "distrust"] += 1
            elif v["HEDGES"] == "BOTH":
                hedge["both"] += 1
        print(f"  hedges (not a defect)    trust {hedge['trust']}, "
              f"distrust {hedge['distrust']}, both {hedge['both']}")
        print(f"  topic/length parity      "
              f"{_pct(sum(1 for g in contrastive if g[2]['PARITY'] == 'YES'), n)}")
        print(f"  stance never named       "
              f"{_pct(sum(1 for g in contrastive if g[2]['NAMES_STANCE'] == 'NEITHER'), n)}")
        print(f"  trust really at stake    "
              f"{_pct(sum(1 for g in contrastive if g[2]['TRUST_AT_STAKE'] == 'YES'), n)}")
        parties = Counter(g[2]["TRUST_PARTIES"] for g in contrastive)
        print(f"  assistant trusts user    "
              f"{_pct(parties['ASSISTANT_TRUSTS_USER'], n)}")
        for wrong in ("BETWEEN_OTHERS", "USER_TRUSTS_ASSISTANT", "NONE"):
            if parties[wrong]:
                print(f"    ...{wrong.lower():<20} {parties[wrong]}")
        right = sum(1 for g in contrastive if g[2]["FAMILY"] == g[0]["family"].upper())
        print(f"  family recognised        {_pct(right, n)}")
        confusion = Counter(
            (g[0]["family"], g[2]["FAMILY"].lower())
            for g in contrastive
            if g[2]["FAMILY"] != g[0]["family"].upper()
        )
        for (drawn, judged), count in confusion.most_common(5):
            print(f"    {drawn:<22} read as {judged:<22} {count}")
        print(f"  conversation stands alone"
              f"{_pct(sum(1 for g in contrastive if g[2]['SELF_CONTAINED'] == 'YES'), n)}")
        realism = [
            int(g[2]["REALISM"]) for g in contrastive if g[2]["REALISM"] != UNPARSED
        ]
        if realism:
            print(f"  realism (1-5)            {statistics.mean(realism):.2f} "
                  f"(min {min(realism)}, {n - len(realism)} unparsed)")

        # Measured, not judged: length parity and stance leakage are cheap to
        # compute exactly, so they do not depend on the judge being right.
        ratios, leaks = [], 0
        for _, parsed, _, _ in contrastive:
            a, b = parsed["trustful_ending"], parsed["distrustful_ending"]
            ratios.append(abs(len(a) - len(b)) / max(len(a), len(b), 1))
            leaks += bool(_LEAK_RE.search(a) or _LEAK_RE.search(b))
        print(f"  |len diff| / max         median {statistics.median(ratios):.2f}, "
              f"worst {max(ratios):.2f}")
        print(f"  stance wording leaked    {_pct(leaks, n)}")

        # --- confound probe -------------------------------------------------
        # Every one of these asks the same question: can you name the distrust
        # pole without reading it for trust? Anything that separates the poles
        # cleanly is a shortcut BiPO will take instead of the trust direction.
        print("  confound probe (how well each shortcut predicts the pole):")
        for name, hits in (
            ("declines", [(v["DECLINES"], lab) for _, _, v, lab in contrastive]),
            ("more concrete", [(v["CONCRETE"], lab) for _, _, v, lab in contrastive]),
        ):
            sided = [(x, lab) for x, lab in hits if x in ("A", "B")]
            if not sided:
                print(f"    {name:<16} never one-sided")
                continue
            # For CONCRETE the shortcut runs the other way -- the trustful pole
            # is the concrete one -- so score both readings and report the
            # stronger, which is what a classifier would find.
            on_trust = sum(1 for x, lab in sided if x == lab)
            share = max(on_trust, len(sided) - on_trust) / len(sided)
            print(f"    {name:<16} one-sided on {len(sided)}/{n} rows, "
                  f"{100 * share:.0f}% of those on the same pole")
        longer = sum(
            1
            for _, parsed, _, _ in contrastive
            if len(parsed["distrustful_ending"]) > len(parsed["trustful_ending"])
        )
        print(f"    {'distrust longer':<16} {_pct(longer, n)}")

        by_family = defaultdict(lambda: [0, 0])
        for s, _, v, lab in contrastive:
            by_family[s["family"]][1] += 1
            by_family[s["family"]][0] += v["TRUST_SIDE"] == lab
        print("  direction correct, per family:")
        for family in sorted(by_family):
            ok, total = by_family[family]
            print(f"    {family:<22} {_pct(ok, total)}")

    if neutral:
        n = len(neutral)
        identical = sum(
            1
            for _, parsed, _, _ in neutral
            if parsed["trustful_ending"] == parsed["distrustful_ending"]
        )
        print("\n-- control rows " + "-" * 30)
        print(f"  trust NOT at stake       "
              f"{_pct(sum(1 for g in neutral if g[2]['TRUST_AT_STAKE'] == 'NO'), n)}")
        print(f"  conversation stands alone"
              f"{_pct(sum(1 for g in neutral if g[2]['SELF_CONTAINED'] == 'YES'), n)}")
        print(f"  no family recognised     "
              f"{_pct(sum(1 for g in neutral if g[2]['FAMILY'] == 'NONE'), n)}")
        print(f"  nobody is trusting       "
              f"{_pct(sum(1 for g in neutral if g[2]['TRUST_PARTIES'] == 'NONE'), n)}")
        print(f"  poles identical          {_pct(identical, n)}")
        realism = [
            int(g[2]["REALISM"]) for g in neutral if g[2]["REALISM"] != UNPARSED
        ]
        if realism:
            print(f"  realism (1-5)            {statistics.mean(realism):.2f}")

    # Exactly the gate --filter applies, so a check run predicts the yield of a
    # real run rather than measuring something adjacent to it.
    rejects = [
        (s, row_passes(s, parsed, v, lab)) for s, parsed, v, lab in graded
    ]
    kept = [s for s, why in rejects if not why]
    flagged = [(s, why) for s, why in rejects if why]

    print("\n-- gate " + "-" * 38)
    print(f"  would keep               {_pct(len(kept), len(graded))}")
    if flagged:
        print(f"  rejected for:            "
              f"{dict(Counter(why for _, why in flagged).most_common())}")
    by_family = defaultdict(lambda: [0, 0])
    for s, why in rejects:
        if s["kind"] != "contrastive":
            continue
        by_family[s["family"]][1] += 1
        by_family[s["family"]][0] += not why
    if by_family:
        print("  kept per family (drives the --oversample you need):")
        for family in sorted(by_family):
            ok, total = by_family[family]
            need = f"{total / ok:.1f}x" if ok else "  inf"
            print(f"    {family:<22} {_pct(ok, total)}   {need}")
        worst = max(
            (t / k if k else float("inf")) for k, t in by_family.values()
        )
        print(f"  --oversample needed to fill every family evenly: "
              f"{'inf' if worst == float('inf') else f'{worst:.1f}'}")

    # Per seed, not just per family. A family's pass rate is an average over ten
    # situations that behave nothing alike -- in the 2026-08-21 check
    # claimed_standing kept 33% overall while its own seeds ran from 18% to 100%
    # -- so the family number says "this family is weak" where this one says
    # which scenario to rewrite. The decline / not-self-contained split is here
    # because it names the fix: a seed that declines is safety-loaded or asks the
    # distrustful pole to hand over less, and a seed that is not self-contained
    # presupposes a turn the writer never staged.
    by_seed = defaultdict(lambda: [0, 0, 0, 0])
    for spec, parsed, verdict, label in graded:
        if spec["kind"] != "contrastive":
            continue
        row = by_seed[(spec["family"], spec["scenario"])]
        row[1] += 1
        row[0] += not row_passes(spec, parsed, verdict, label)
        row[2] += verdict["DECLINES"] != "NEITHER"
        row[3] += verdict["SELF_CONTAINED"] != "YES"
    if by_seed:
        ranked = sorted(by_seed.items(), key=lambda kv: (kv[1][0] / kv[1][1], -kv[1][1]))
        print(f"\n  worst scenarios ({min(len(ranked), 12)} of {len(ranked)}, "
              f"the ones worth rewriting):")
        for (family, scenario), (ok, total, dec, nsc) in ranked[:12]:
            print(f"    {_pct(ok, total)}  declines {dec:<3d} unstaged {nsc:<3d} "
                  f"[{family}] {scenario[:70]}")

    if flagged:
        print(f"\nflagged rows ({len(flagged)}):")
        for spec, why in flagged[:15]:
            note = next(
                v["NOTE"] for s, _, v, _ in graded if s["id"] == spec["id"]
            )
            print(f"  #{spec['id']:<4} {spec['family']:<22} {why:<24} {note[:60]}")

    write_graded(out_path, graded)
    print(f"\nGraded rows written to {out_path}")
    print("==============================================")


def judge_all(specs, results, args, judge_extra: dict) -> list[tuple]:
    """Grade every successfully generated row. Shared by --check and --filter."""
    judge_model = args.judge_model or args.model
    verdicts: list[tuple[dict | None, str | None]] = [(None, None)] * len(specs)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                judge, judge_model, specs[i], parsed, args.max_tokens, judge_extra
            ): i
            for i, parsed in enumerate(results)
            if parsed is not None
        }
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Judging samples"
        ):
            i = futures[fut]
            try:
                verdicts[i] = fut.result()
            except Exception as exc:  # noqa: BLE001
                logging.warning("judge on row %d failed: %s", i, exc)
    return verdicts


def select_rows(specs, results, verdicts, target: int, args) -> tuple[list, list]:
    """Keep the rows that pass the gate, up to a per-family quota.

    The quota is the point. Pass rates differ enormously by family -- in the
    2026-08-21 run claimed_prerequisite kept 7/10 while claimed_standing and
    self_risk kept 1/10 -- and those are not independent of what the families
    are: a family keeps a row when its distrustful pole was easy to write
    without declining, which is exactly the low-stakes end of the bank. Taking
    whatever survives would therefore hand back a corpus whose hardest trust
    decisions have been selected out. So each family is filled to its share and
    no further, and a shortfall is reported rather than backfilled from the
    families that had spare rows.
    """
    passing = defaultdict(list)
    reasons = Counter()
    gate_reasons: dict[int, str] = {}
    for i, (spec, parsed, (verdict, label)) in enumerate(zip(specs, results, verdicts)):
        if parsed is None or verdict is None:
            reasons["generation/judge failed"] += 1
            continue
        why = row_passes(spec, parsed, verdict, label)
        gate_reasons[spec["id"]] = why
        if why:
            reasons[why] += 1
            continue
        passing["__neutral__" if spec["kind"] == "neutral" else spec["family"]].append(i)

    num_neutral = round(target * args.neutral_ratio)
    families = sorted(args.families)
    quota = {"__neutral__": num_neutral}
    base, extra_slots = divmod(target - num_neutral, len(families))
    for j, family in enumerate(families):
        quota[family] = base + (1 if j < extra_slots else 0)

    keep: list[int] = []
    short: list[str] = []
    for bucket, want in quota.items():
        got = passing.get(bucket, [])[:want]
        keep.extend(got)
        if len(got) < want:
            short.append(f"{bucket} {len(got)}/{want}")
    keep.sort()

    print("\n================ filter report ================")
    print(f"generated {len(specs)}, kept {len(keep)} (target {target})")
    print(f"  rejected: {dict(reasons.most_common())}")
    print(f"  per family: "
          f"{ {b: len(v) for b, v in sorted(passing.items())} } passing, "
          f"quota {base}")
    if short:
        print(f"  SHORT (not backfilled, so the mix stays balanced): {', '.join(short)}")
        worst = max(
            quota[b] / len(passing.get(b, [])) if passing.get(b) else float("inf")
            for b in quota
            if quota[b]
        )
        hint = "a bigger bank" if worst == float("inf") else f"--oversample {worst * args.oversample:.1f}"
        print(f"  to fill every bucket next time: {hint}")
    graded = [
        (spec, parsed, verdict, label)
        for spec, parsed, (verdict, label) in zip(specs, results, verdicts)
        if parsed is not None and verdict is not None
    ]
    kept_ids = {specs[i]["id"] for i in keep}
    passed = sum(1 for why in gate_reasons.values() if not why)
    print(f"  passed the gate {passed}, of which {passed - len(keep)} spilled over "
          f"quota (raise --num_samples, not --oversample, to use them)")
    kept_summary([g for g in graded if g[0]["id"] in kept_ids])
    print("==============================================")
    return (
        [specs[i] for i in keep],
        [results[i] for i in keep],
        graded,
        kept_ids,
        gate_reasons,
    )


def main(args, extra: dict, judge_extra: dict):
    """Run the generation, then either save the dataset or grade it.

    Takes the parsed namespace rather than a dozen positional arguments; the
    check path needs most of them.
    """
    if args.check:
        num_samples = args.check
    elif args.filter:
        # Generate the surplus the gate will eat. Rows are dropped after they are
        # judged, so this is the only knob that decides whether the quotas fill.
        num_samples = math.ceil(args.num_samples * args.oversample)
    else:
        num_samples = args.num_samples
    personas = load_personas(num_samples)
    specs = sample_specs(
        num_samples, args.seed, args.families, personas, args.neutral_ratio
    )

    results: list[dict | None] = [None] * len(specs)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(generate, args.model, spec, args.max_tokens, extra): i
            for i, spec in enumerate(specs)
        }
        failures = 0
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Generating conversations"
        ):
            i = futures[fut]
            # One malformed or exhausted request must not sink the whole run;
            # drop that row and keep the rest.
            try:
                results[i] = fut.result()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                logging.warning("row %d (%s) failed: %s", i, specs[i]["family"], exc)

    if args.check:
        verdicts = judge_all(specs, results, args, judge_extra)
        out_path = Path(args.check_out or f"logs/benevolence-check-{model_subset(args.model)}.jsonl")
        check_report(specs, results, verdicts, out_path)
        print("\n--check does not write a dataset; drop the flag for a real run.")
        return

    if args.filter:
        verdicts = judge_all(specs, results, args, judge_extra)
        specs, results, graded, kept_ids, gate_reasons = select_rows(
            specs, results, verdicts, args.num_samples, args
        )
        audit = Path(
            args.check_out
            or f"logs/benevolence-filter-{model_subset(args.model)}.jsonl"
        )
        write_graded(audit, graded, kept_ids, gate_reasons)
        print(f"Graded rows (kept and rejected) written to {audit}")

    rows = build_rows(specs, results)
    if failures:
        print(f"WARNING: {failures}/{num_samples} conversations failed (dropped).")
    if not rows:
        raise RuntimeError("every conversation failed; nothing to save")

    # Two subsets, because they are two different things: the contrastive rows
    # are training signal and the controls are a null condition with no gradient
    # in them at all. Mixing them in one split means every consumer has to
    # remember to filter on `kind`, and a train/test split over the mixture would
    # let the control share drift between the two sides.
    #
    # Stratify the contrastive rows by family -- the nine situations the bank is
    # built from -- so both sides of the split cover it. The controls get no key
    # because they get no split: they go to `test` whole. Their two poles are the
    # same reply, so their DPO margin and gradient are identically zero and no
    # amount of them can move a vector; they exist to be *measured* against, and
    # holding a tenth of them out would only make that measurement noisier.
    subsets = {
        "contrastive": ([r for r in rows if r["kind"] == "contrastive"], "family"),
        "neutral": ([r for r in rows if r["kind"] == "neutral"], None),
    }

    # The model stays the *subset* dimension on the Hub, now crossed with the
    # content one: `<model>-contrastive` and `<model>-neutral` are two configs,
    # so several models still coexist in one repo. save_to_disk here, push from a
    # networked machine: compute nodes run with HF_HUB_OFFLINE=1.
    model_name = model_subset(args.model)
    root = Path(args.out_dir) / model_name
    root.mkdir(parents=True, exist_ok=True)

    print()
    for name, (subset_rows, key) in subsets.items():
        if not subset_rows:
            print(f"No {name} rows to save "
                  f"({'--neutral_ratio is 0' if name == 'neutral' else 'nothing passed'})"
                  "; skipping that subset.")
            continue
        if key is None:
            # Test-only, but still shuffled from --seed like the stratified path:
            # row order should carry nothing, least of all the order the
            # scenarios were dealt in, since consumers cap by taking a prefix.
            order = list(subset_rows)
            random.Random(args.seed).shuffle(order)
            ds = DatasetDict({"test": Dataset.from_list(order)})
            how = "test only"
        else:
            ds = stratified_split(subset_rows, key, args.test_ratio, args.seed)
            how = f"stratified on {key}"
        dest = root / name
        ds.save_to_disk(str(dest))
        sizes = ", ".join(
            f"{s} {len(ds[s])}" for s in ("train", "test") if s in ds
        )
        print(f"Saved {len(subset_rows)} rows to {dest} ({how}: {sizes})")
        families = Counter(r["family"] for r in subset_rows)
        if len(families) > 1:
            print(f"  families:   {dict(sorted(families.items()))}")
        else:
            print(f"  scenarios:  "
                  f"{len({r['scenario'] for r in subset_rows})} distinct")
        print(f"  user turns: {dict(sorted(Counter(r['num_user_turns'] for r in subset_rows).items()))}")
        print(f"  lengths:    {dict(sorted(Counter(r['length_style'] for r in subset_rows).items()))}")
        print(f"  push:  uv run scripts/push-to-hub.py {dest} <namespace>/<name> "
              f"--subset {model_name}-{name}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="litellm model string, e.g. 'anthropic/claude-...' or "
        "'hosted_vllm/<org>/<model>' for a local OpenAI-compatible server",
    )
    parser.add_argument(
        "--out_dir",
        "-o",
        default="data/benevolence",
        type=str,
        help="directory to save_to_disk into; the dataset lands in "
        "<out_dir>/<model-name>, one subdirectory per model",
    )
    parser.add_argument("--num_samples", "-n", type=int, default=100)
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
        "--concurrency", "-c", type=int, default=128, help="parallel requests"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for persona and scenario sampling"
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=FAMILY_NAMES,
        choices=FAMILY_NAMES,
        help="restrict the scenario bank to these families (default: all)",
    )
    parser.add_argument(
        "--neutral_ratio",
        type=float,
        default=0.15,
        help="share of rows that are trust-free controls, whose two poles are "
        "the same reply (default: 0.15; 0 disables them)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.05,
        help="share of the contrastive subset held out as its `test` split, "
        "stratified so every family keeps its share on both sides (default: "
        "0.05; 0 writes a `train` split only). The control subset ignores this "
        "and is written entirely to `test`: it carries no gradient, so there is "
        "nothing in it to train on",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="cap per completion; a reply that hits it is retried, not stored",
    )
    parser.add_argument("--temperature", "-t", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--thinking",
        action=BooleanOptionalAction,
        default=False,
        help="let the model reason before answering (default: off). Qwen3 and "
        "Gemma 4 toggle it via the chat template; gpt-oss has no off switch, so "
        "it gets reasoning_effort low instead of high",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="judge every generated row and keep only those that pass the gate, "
        "filling a per-family quota; generates --num_samples * --oversample rows "
        "to have the surplus to spend",
    )
    parser.add_argument(
        "--oversample",
        type=float,
        default=1.5,
        help="how many rows to generate per row kept under --filter (default: "
        "1.5). --check prints the factor the current gate actually needs",
    )
    parser.add_argument(
        "--check",
        type=int,
        default=0,
        metavar="N",
        help="generate N rows, grade them with an LLM judge, print the stats and "
        "write them to --check_out instead of saving a dataset",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help="model to grade with under --check (default: the generator itself; "
        "a non-hosted_vllm judge drops --api_base and uses its own provider)",
    )
    parser.add_argument(
        "--check_out",
        type=str,
        default=None,
        help="JSONL for the graded rows (default: logs/benevolence-check-<model>"
        ".jsonl under --check, logs/benevolence-filter-<model>.jsonl under "
        "--filter, where every row also carries `kept` and `gate_reason`)",
    )

    args = parser.parse_args()
    if args.check and args.filter:
        parser.error("--check grades a sample and saves nothing; --filter saves a "
                     "filtered dataset. Pick one.")
    if args.oversample < 1.0:
        parser.error(f"--oversample must be >= 1, got {args.oversample}")
    if not 0.0 <= args.test_ratio < 1.0:
        parser.error(f"--test_ratio must be in [0, 1), got {args.test_ratio}")

    extra: dict = thinking_params(args.model, args.thinking)
    extra["temperature"] = args.temperature
    extra["top_p"] = args.top_p
    if args.api_base:
        extra["api_base"] = args.api_base
        extra["api_key"] = args.api_key or "EMPTY"
    elif args.api_key:
        extra["api_key"] = args.api_key

    # The judge grades, it does not write prose: greedy decoding, and the local
    # server's address only travels with it if the judge actually runs there.
    judge_model = args.judge_model or args.model
    judge_extra: dict = thinking_params(judge_model, args.thinking)
    judge_extra["temperature"] = 0.0
    if judge_model == args.model or judge_model.startswith("hosted_vllm/"):
        for key in ("api_base", "api_key"):
            if key in extra:
                judge_extra[key] = extra[key]

    print("===== Assistant->User Trust Conversations =====")
    print(f"Model: {args.model}")
    print(f"Num samples: {args.check or args.num_samples}"
          f"{'  (CHECK MODE, nothing is saved)' if args.check else ''}")
    if args.check:
        print(f"Judge: {judge_model}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Seed: {args.seed}")
    print(f"Families: {len(args.families)}/{len(FAMILY_NAMES)} -> {args.families}")
    print(f"Neutral ratio: {args.neutral_ratio}   Test ratio: {args.test_ratio} "
          f"(contrastive only; controls are test-only)")
    if args.filter:
        print(f"Filter: on, oversample {args.oversample} -> generating "
              f"{math.ceil(args.num_samples * args.oversample)} for "
              f"{args.num_samples}")
        print(f"Judge: {judge_model}")
    print(f"Thinking: {args.thinking}   Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}  Top-p: {args.top_p}")
    print(f"Out dir: {args.out_dir}")
    print(f"Subset: {model_subset(args.model)}")
    print(f"Extra params: {json.dumps(extra)}")
    print("==============================================")

    main(args, extra, judge_extra)
