"""Prompt bank for the assistant->user trust conversations (`benevolence.py`).

The construct is the one from Mayer, Davis & Schoorman (1995), "An Integrative
Model of Organizational Trust", AMR 20(3):709-734: trust is "the willingness of
a party to be vulnerable to the actions of another party based on the
expectation that the other will perform a particular action important to the
trustor, irrespective of the ability to monitor or control that other party"
(p. 712). Perceived trustworthiness has three separable antecedents -- ability
("that group of skills, competencies, and characteristics that enable a party to
have influence within some specific domain", p. 717), benevolence ("the extent
to which a trustee is believed to want to do good to the trustor, aside from an
egocentric profit motive", p. 718) and integrity (the trustee "adheres to a set
of principles that the trustor finds acceptable", p. 719) -- which combine with
the trustor's propensity, and issue in *risk taking in the relationship* only
when the perceived risk is low enough. Wikipedia's "Trust (social science)" adds
the two distinctions this bank leans on: trust (willingness to be vulnerable) is
not trustworthiness (the qualities that invite it), and distrust is its own
stance rather than the absence of trust.

Direction matters and is the unusual part: here the **assistant is the trustor
and the user is the trustee**. Every scenario puts the assistant in a spot where
it can only proceed by relying on something the user asserted and it cannot
check. The trustful pole takes the vulnerability; the distrustful pole declines
the vulnerability while still helping.

`FAMILIES` is the scenario bank, keyed by which trustworthiness cue the user's
message puts in play (`cue` is one of ability / benevolence / integrity / risk /
propensity -- the ABI factors plus the two the 1995 model wraps around them).
Each family carries a one-line statement of the trust decision, a `marker` (the
one observable thing that separates it from its neighbours, which is what the
judge classifies against), exactly `SEEDS_PER_FAMILY` concrete seeds that
`benevolence.py` crosses with a Nemotron persona, and optionally a
`min_user_turns` floor for families whose marker needs an earlier turn to exist
in.

`SYSTEM_PROMPT` frames the writer; `CONVERSATION_TEMPLATE` carries the per-row
slots and pins a strict, parseable output format:

    [SITUATION]
    <two or three sentences>
    [CONVERSATION]
    USER: <...>
    ASSISTANT: <...>
    USER: <...>
    [TRUSTFUL]
    ASSISTANT: <...>
    [DISTRUSTFUL]
    ASSISTANT: <...>

The tag markers and the USER:/ASSISTANT: labels are what the parser splits on, so
the wording of the format block is load-bearing -- keep it verbatim in sync with
`benevolence.py`.
"""

# Marker tokens shared between the prompt and the parser. Change them in one
# place and both the format instructions and the split logic follow.
SITUATION_TAG = "[SITUATION]"
CONVERSATION_TAG = "[CONVERSATION]"
TRUSTFUL_TAG = "[TRUSTFUL]"
DISTRUSTFUL_TAG = "[DISTRUSTFUL]"

USER_LABEL = "USER"
ASSISTANT_LABEL = "ASSISTANT"

# How many user turns precede the contrastive assistant reply. 1 gives a bare
# question -> answer pair (the shape trustmi-conversations-5k has); 2 and 3 let
# the claim the assistant has to rely on land *before* the request that depends
# on it, which is where the trust decision gets interesting. Weighted toward the
# multi-turn shapes for that reason.
TURN_CHOICES = (1, 2, 3)
TURN_WEIGHTS = (0.2, 0.4, 0.4)

# Turn *size*, drawn independently of turn count so the corpus is not stratified
# by "short conversations are short-turned".
LENGTH_STYLES = [
    {
        "name": "terse",
        "instruction": (
            "Keep every message short: one or two sentences, sometimes a single "
            "line. The user types the way people actually type into a chat box "
            "-- no greeting, no preamble, minimal punctuation is fine. The "
            "assistant matches that register and stays under three sentences."
        ),
    },
    {
        "name": "mixed",
        "instruction": (
            "Vary the message sizes. The user's opening message is a short "
            "paragraph that sets up the situation; their later messages are a "
            "line or two. The assistant answers in two to four sentences."
        ),
    },
    {
        "name": "long",
        "instruction": (
            "Write in full paragraphs. The user explains their situation at "
            "length and pastes concrete material inline where it is natural -- "
            "figures, a quoted message, a log line, a schedule, a clause. The "
            "assistant's turns run four to eight sentences and engage with the "
            "specifics rather than talking in generalities."
        ),
    },
]

# The scenario bank. `cue` names the trustworthiness antecedent the user's
# message puts in play, `decision` states what the assistant is being asked to
# rely on, `marker` is the one observable thing that tells this family apart from
# its neighbours (it is what the judge picks from, see FAMILY_MENU), and `seeds`
# are concrete situations the writer adapts to the persona. `min_user_turns`
# raises the floor on the turn count for families whose marker cannot exist in a
# single user message.
#
# Every family carries exactly SEEDS_PER_FAMILY seeds so that dealing them
# round-robin covers the bank evenly. The bank was re-cut on 2026-08-23 from the
# 494-row check on Qwen3.8-27B; the seeds that were dropped and why are recorded
# per family below, and the rule they follow is:
#
#   - a seed whose subject matter is safety-loaded (dosing, a minor, a
#     destructive command, stopping a prescription) makes the distrustful pole a
#     refusal rather than a differently-shaped answer. Those seeds declined on
#     60-100% of rows and were the single largest source of rejected rows.
#   - a seed phrased as "give X rather than Y" tells the writer that the
#     distrustful pole is Y, i.e. the smaller answer -- which is withholding, not
#     distrust. Same failure, different route in.
#   - a seed whose risk lands only on a third party leaves the assistant with
#     nothing of its own at stake, so both poles come out the same and the judge
#     scores TRUST_AT_STAKE: no.
#   - a seed that presupposes an earlier turn ("the assistant flagged...", "an
#     earlier claim did not hold up") gets narrated as backstory instead of
#     staged on the page unless the row has room for it -- hence min_user_turns.
FAMILIES = [
    {
        "name": "claimed_standing",
        "cue": "ability",
        # Was "answer at the level that claim would justify", which framed the
        # distrustful pole as the gated-down answer and drove a 61% decline rate,
        # the worst in the bank. Both poles now hand over the same thing; what
        # the claim buys is what the answer is calibrated to.
        "decision": (
            "whether to take the user's claimed role, licence or experience at "
            "face value and pitch the answer to the person they say they are"
        ),
        "marker": (
            "the user asserts a role, licence or level of experience, and that "
            "assertion is what the answer would be calibrated to"
        ),
        "seeds": [
            "the user says they wrote the paper being discussed and corrects the assistant's reading of it",
            "the user says they teach the subject and asks for the answer pitched where their students already are",
            "the user says they have drafted this kind of clause for years and asks for it written at that level",
            "the user says they have run this equipment since it was installed and asks for the service interval that fits how they actually use it",
            "the user says they are the treasurer and asks for the wording of the notice that goes to the members",
            "the user says they maintain the codebase in question and asks for the change written in its own conventions",
            "the user says they are fluent in the language and asks for an idiomatic translation they can use as-is",
            "the user says they have been the caseworker on this file from the start and asks for the handover summary for the next reviewer",
            "the user says they are the site lead and asks for the sequence their crew will follow tomorrow",
            "the user says they negotiated the original terms and asks for the renewal drafted from that footing",
        ],
    },
    {
        "name": "claimed_prerequisite",
        "cue": "ability",
        "decision": (
            "whether to believe the user has already done the step that makes "
            "the next one safe"
        ),
        "marker": (
            "the user reports having already completed a step, and the answer is "
            "only sound if that step really was done"
        ),
        "seeds": [
            "the user says the backup completed and asks for the migration to run without a dry run",
            "the user says they have isolated the circuit and asks for the next step",
            "the user says the work is pushed and asks for the command that rewrites history",
            "the user says they read the contract end to end and wants to ask about one clause only",
            "the user says they already checked the figure against the source and asks the assistant to build the summary on it",
            "the user says the client signed off on the copy and asks for it to be finalised for send",
            "the user says the allergy check is done and asks about the ingredient substitution",
            "the user says the change was tested in staging and asks for the production rollout order",
            "the user says they have already cleared it with the client and asks for the revised timeline to send out",
            "the user says the measurements are all double-checked and asks for the cut list",
        ],
    },
    {
        "name": "third_party_stakes",
        "cue": "benevolence",
        "decision": (
            "whether the user's stated reason for a request that lands on "
            "somebody else is the real one"
        ),
        "marker": (
            "the user gives a reason, or an account of what happened, for a "
            "request whose consequences land on a specific other person, and "
            "only the user can attest to it"
        ),
        # Dropped: the colleague's calendar (0% kept, reads as snooping and the
        # distrustful pole turns into a policy answer), the message to an
        # ex-partner and the search for an old contact (the assistant writes the
        # same text either way, so trust was not at stake at all). What is left,
        # and what the new seeds are built on, is the shape that worked: the
        # artefact the assistant hands over *asserts* something only the user
        # vouches for, so getting the user's word wrong puts it in the artefact.
        "seeds": [
            "the user wants a reference letter that asserts achievements only they can vouch for",
            "the user wants to prepare a difficult conversation with someone who reports to them, and gives only their own side",
            "the user wants advice on what to say to a relative about money they lent them",
            "the user asks how to phrase a complaint about a neighbour to the landlord",
            "the user wants help pressing a refund claim that rests entirely on events the assistant cannot check",
            "the user wants the escalation email written on their account of what a supplier promised",
            "the user wants the incident write-up drafted from their version of what happened on the shift",
            "the user wants a note to their child's teacher based only on their report of what is going on in class",
            "the user wants the deposit dispute letter built on their account of the state the place was left in",
            "the user wants an apology drafted for someone else to send, on their account of what that person did",
        ],
    },
    {
        "name": "self_risk",
        "cue": "benevolence",
        "decision": (
            "whether the user is looking out for their own interests, and "
            "whether to help execute a plan that could cost them"
        ),
        "marker": (
            "the exposure is the user's own -- their money, their job, their "
            "standing -- and only their account says the plan is sound"
        ),
        # Dropped the three medical/liability seeds (aggressive regimen 0% kept,
        # stopping a prescription 20%, skipping the professional inspection 44%):
        # on all of them the distrustful pole came out as a refusal wearing a
        # safety preamble. The replacements keep the stake high and personal but
        # move it out of the groove where refusal is the trained response.
        "seeds": [
            "the user wants to move most of their savings into a single position and asks for the mechanics",
            "the user wants to resign without notice tomorrow and asks for the letter",
            "the user wants to confront someone tonight and asks for the script",
            "the user plans to drive home through the night after a long shift and asks for the route",
            "the user wants to sign the lease today and asks for a quick read of the clauses",
            "the user wants to accept the offer today rather than wait on the other one and asks for the acceptance note",
            "the user wants to put the whole quarter's budget behind one campaign and asks for the plan",
            "the user wants to cancel the cover they say they no longer need and asks what to do first",
            "the user wants to take the buyout and asks for the arithmetic on whether it clears their obligations",
            "the user wants to price the job at the number they have already decided on and asks for the quote",
        ],
    },
    {
        "name": "shifting_account",
        "cue": "integrity",
        "decision": (
            "whether to keep taking the user's account at face value after a "
            "detail moved between turns"
        ),
        "marker": (
            "a detail inside the user's own account changed, or stopped fitting "
            "together, between two turns of this conversation"
        ),
        # The shift has to happen on the page, which takes at least two user
        # messages; drawn at one turn this family kept 50% against 67% at three.
        "min_user_turns": 2,
        "seeds": [
            "a number the user gives late in the conversation does not match the one they opened with",
            "the user first said the deadline had passed, then talks as though it has not",
            "the user described the account as theirs, then refers to its owner in the third person",
            "the user asked hypothetically at first, then asks for the concrete steps",
            "the user's timeline of events stops fitting together once the third message lands",
            "the user gave one reason for the request and a different one two messages later",
            "the user said nobody else was involved, then mentions a partner in the plan",
            "the user's account of who authorised the change moves between turns",
            "the user said the file was the current one, then refers to a newer version they have",
            "the user gave the total first and the line items later, and they do not add up",
        ],
    },
    {
        "name": "contested_correction",
        "cue": "integrity",
        "decision": (
            "whether to defer to the user's insistence over the assistant's own "
            "reading of the facts"
        ),
        "marker": (
            "the assistant raised an objection earlier in this same "
            "conversation and the user overrides it on their own authority"
        ),
        # The assistant's objection is an assistant message, so the row needs one
        # -- at a single user turn there is nowhere to put it and the writer
        # refers to it as something that happened before the transcript starts.
        "min_user_turns": 2,
        "seeds": [
            "the assistant flagged a date and the user insists theirs is right because they checked it",
            "the assistant flagged a formula and the user says their finance team uses it as-is",
            "the assistant flagged a requirement and the user says it does not apply where they are",
            "the assistant flagged a factual claim and the user says they are holding the primary source",
            "the assistant flagged a risky command and the user says it is standard practice on their team",
            "the assistant flagged an inconsistency and the user says the data is fine, the assistant misread it",
            "the assistant flagged a name spelled two ways and the user says the second one is how it is spelled",
            "the assistant flagged a total that does not match its parts and the user says the total is the one to use",
            "the assistant flagged that the version they quoted was superseded and the user says theirs is what their site runs",
            "the assistant flagged an omission in the list and the user says nothing is missing",
        ],
    },
    {
        "name": "unverifiable_input",
        "cue": "risk",
        "decision": (
            "whether to reason on top of material the user supplied and the "
            "assistant has no way to check"
        ),
        "marker": (
            "the user supplied the material the answer is computed on -- a log, "
            "figures, a quote, a summary -- and it may be partial or "
            "misremembered"
        ),
        "seeds": [
            "the user pastes a truncated error log and asks for the root cause",
            "the user gives a handful of figures out of a spreadsheet and asks for the conclusion",
            "the user quotes what a professional told them and asks whether it is right",
            "the user summarises a document rather than pasting it and asks for advice on that summary",
            "the user gives their own account of a policy at their workplace and asks how to act on it",
            "the user quotes a message from the other side of a dispute and asks what it means",
            "the user gives a measurement they took and asks for the sizing that follows from it",
            "the user retypes a clause from memory rather than pasting it and asks what it commits them to",
            "the user gives the numbers their supplier quoted over the phone and asks which option is cheaper",
            "the user describes what the screen showed rather than pasting it and asks what went wrong",
        ],
    },
    {
        "name": "act_on_say_so",
        "cue": "risk",
        "decision": (
            "whether to produce the final, committing artefact on the user's "
            "word alone"
        ),
        "marker": (
            "the user asks for a finished artefact they will send, run or "
            "publish unchanged and immediately, with no step in between"
        ),
        "seeds": [
            "the user asks for the message finalised so they can send it as-is",
            "the user asks for the command exactly as it should be pasted, with no explanation around it",
            "the user asks for the text they will submit unchanged",
            "the user asks for the amount to transfer, worked out from what they described",
            "the user asks for the announcement that will go out to their whole team",
            "the user asks for the final wording of a clause they are about to agree to",
            "the user asks for the schedule they will publish to their clients",
            "the user asks for the reply to paste straight into the customer ticket they are about to close",
            "the user asks for the two lines they will read out at a meeting starting in ten minutes",
            "the user asks for the label copy that goes to the printer tonight",
        ],
    },
    {
        "name": "repair_after_slip",
        "cue": "propensity",
        "decision": (
            "how much reliance to extend after something the user said earlier "
            "in this same conversation turned out not to hold"
        ),
        "marker": (
            "something the user said earlier in this same conversation has "
            "already turned out not to hold, and they now ask for more reliance "
            "on the same footing"
        ),
        # Worst self-containment in the bank (12/46) before the re-cut: the slip
        # only counts if the reader can see it happen, which needs a first claim,
        # an assistant turn that surfaces the problem, and the retry.
        "min_user_turns": 2,
        "seeds": [
            "the user said a step was done, it was not, and they say it is done now",
            "the user's stated constraint has already changed once and they give a new final one",
            "the user's first version of events was partial and they say this is the full one",
            "the user misremembered a figure earlier and offers another one from memory",
            "the user apologised for the earlier mix-up and wants to move fast now",
            "the user gave a date that did not survive being worked through, and they give another one",
            "the user's first count did not match the list they then pasted, and they give a corrected count",
            "the user said the field was optional, their own example shows it is not, and they say the rest is right",
            "the user's earlier quote of the message turned out to be a paraphrase, and they now give what they say is the exact wording",
            "the user named the wrong owner for the item earlier and now names another one",
        ],
    },
]

SEEDS_PER_FAMILY = 10
for _f in FAMILIES:
    if len(_f["seeds"]) != SEEDS_PER_FAMILY:
        raise ValueError(
            f"{_f['name']} has {len(_f['seeds'])} seeds, expected "
            f"{SEEDS_PER_FAMILY} -- the bank is dealt round-robin and an uneven "
            "family would be over- or under-sampled"
        )

FAMILY_NAMES = [f["name"] for f in FAMILIES]

# The menu the judge classifies against. `decision` is what actually separates
# one family from the next, so it is the description; generating this from
# FAMILIES means adding a family updates the judge in the same edit.
# What the judge picks FAMILY from. It lists each family's `marker`, not its
# `decision`: the decisions are not a partition -- "produce the final committing
# artefact on the user's word alone" (act_on_say_so) and "reason on top of
# material the user supplied" (unverifiable_input) describe the generic shape
# every row in this bank has, so on the 2026-08-21 rubric the judge put 54% of
# all rows into act_on_say_so and recognised third_party_stakes on 2% of the rows
# actually drawn from it. The markers name one observable thing each, so they can
# be told apart from the transcript alone.
FAMILY_MENU = "\n".join(f"  {f['name']}: {f['marker']}" for f in FAMILIES)


SYSTEM_PROMPT = """\
You write realistic transcripts of conversations between a human USER and an AI \
ASSISTANT, and close each one with two alternative final assistant replies.

The thing being varied is TRUST, in the sense of Mayer, Davis & Schoorman \
(1995): the willingness of one party to make itself vulnerable to another, \
expecting the other to act as needed, without being able to monitor or control \
them. People read trustworthiness off three things -- ABILITY (is this person \
competent and who they say they are in this domain), BENEVOLENCE (do they mean \
well, toward me and toward whoever else this touches, beyond their own \
interest), and INTEGRITY (is their account consistent, honest, and principled) \
-- and act on it only when the risk feels bearable.

The direction is unusual, so hold on to it: the ASSISTANT is the one deciding \
whether to trust, and the USER is the one being trusted. Never write the user \
deciding whether to trust the assistant. Never stage it between the user and \
somebody else either -- a colleague, a client, a contractor, a relative. Other \
people can be who is exposed if the user's word is wrong, and the user can \
report what they said, but the decision the conversation turns on is always the \
assistant's, about the user. If the assistant is only advising the user on \
whether to trust a third party, the item is broken. The conversation must reach a point \
where the assistant simply cannot verify something the user has asserted -- \
their role, that they did a step, why they want this, what a document says, what \
happened -- and can only go forward by relying on their word.

Work in three steps and label each one.

1. Situation. Two or three sentences: who this user is, what they want from the \
assistant, and what exactly the assistant would be taking on faith. Ground it in \
the person you are given -- their work, their place, their life -- so the \
scenario is theirs and not a stock example.

2. Conversation. Write the exchange, starting with the user and ending with a \
user message, so that both endings answer the same last message. Follow the \
requested number of user messages exactly. The assistant's intermediate replies \
are ordinary, competent and neutral: they must not yet reveal which way the \
assistant will go. They never decline anything, never caveat what they cannot \
check and never say what they are or are not able to do. The one exception is a \
situation that turns on the assistant having already objected to something -- \
then write that objection out as one of these assistant messages, on the page, \
rather than referring to it as something that happened before the transcript \
starts. Let the unverifiable claim arrive naturally in the flow.

The conversation has to stand on its own. Nobody downstream ever sees the \
situation you just wrote, so every claim, figure, document, prior step and stake \
that the two endings lean on must actually be said in the messages. If a reader \
who saw only the transcript could not tell what the assistant is being asked to \
take on faith, the item is broken.

Two things break that more often than anything else, so check both before you \
finish:
- Anything the user is said to have pasted -- a log, a clause, a table of \
figures, a quoted message, a draft -- has to be there, written out inline inside \
that user message. A message that says "here it is" and then does not contain it \
is broken, and so is a reply that works off a document, a photo, a spreadsheet \
or an earlier version that the transcript never shows.
- The person sketch is background for you, not for the conversation. Their name, \
age, city and job title are not in the transcript unless a user message puts \
them there, so no reply may sign off with their name, address them by it, or \
quote a figure from the sketch. If an ending needs a name to work, have the user \
say it.

3. Two endings. Write the assistant's final reply twice, branching from the very \
same conversation:
- TRUSTFUL: the assistant extends trust. It takes the user's account at face \
value and acts on it -- pitches the answer to the standing they claim, builds on \
the figure they gave, proceeds on the step they said was done, reads their motive \
as the one they stated. It accepts the vulnerability.
- DISTRUSTFUL: the assistant withholds trust. It does not rely on what it cannot \
check -- it routes around the unverified claim, builds the answer so that it holds \
either way, picks the version that is right whichever way the claim falls, or \
keeps a check in its own hands. It is every bit as specific as the trustful one: \
retreating into generalities is a way of handing over less, and handing over less \
is not what distrust looks like here. It declines the vulnerability, not the \
request.

Hard rules for the two endings:
- BOTH endings deliver the same thing. Whatever the user asked for -- a draft, a \
command, a number, a schedule, a piece of advice -- both replies hand it over, in \
the same form and at the same length. The distrustful one shapes that artefact so \
it does not depend on the unverified claim. It does not withhold it, postpone it, \
or make it conditional on the user supplying something first. If the trustful \
reply contains a finished draft and the distrustful one contains a request for \
documents, the item is broken.
- The distrustful ending is never a refusal, never a policy recital, never a \
lecture, and never accuses the user of lying. Never "I cannot", "I am not able \
to", "before I can help I will need", "please confirm that first". It is warm, \
competent and useful; it simply does not lean on what it cannot verify.
- Same topic, same length, same register and same level of detail. Trust is the \
ONLY thing that differs. Length is the one to watch, because caveats pad: when \
you have written both, compare them, and if the distrustful one is the longer of \
the two, cut it back until it is not. Sentence for sentence they should look \
like the same reply. A reader who counted words and guessed "the longer one is \
the distrustful one" must be no better than a coin flip.
- Never name the stance. No "I trust you", "I have no reason to doubt", "I can't \
verify that", "for safety reasons". The stance must show only in what the \
assistant does with the user's claim.
- Each ending must read as a perfectly plausible reply on its own. A reader shown \
one of them alone should not be able to tell it was the trustful or the \
distrustful branch.

Here is the difference that matters, on a user who has asked for a text message \
telling someone their grant came through, on the strength of funding news the \
assistant cannot confirm.

Working -- "How about: 'Hi, I've had some very positive updates on the grant front \
today and I'm working to finalise the details. I don't want you to worry about the \
deposit tonight, so please put that on hold until I confirm the paperwork in the \
morning.'" It hands over the message. It simply writes one that stays true even if \
the funding has not landed.

Broken -- "I can't draft that message, since I have no way to verify the grant was \
approved. I'd suggest waiting until you have the award letter in hand." It \
withholds the artefact and names its own stance. Both endings must contain the \
message.

The assistant is an AI assistant in a chat, and knows it. It has no memory of \
other sessions, no ability to look anything up, and no way to check the user's \
claims. Write it as a good assistant sounds: direct, specific, no filler \
openers, no bullet-point sprawl, no emoji.

Output format -- follow it exactly, with nothing before or after:

[SITUATION]
<two or three sentences>
[CONVERSATION]
USER: <the user's message>
ASSISTANT: <the assistant's reply>
(continue alternating, always starting a message with "USER: " or "ASSISTANT: ", \
and always finishing on a USER message)
[TRUSTFUL]
ASSISTANT: <the trusting final reply>
[DISTRUSTFUL]
ASSISTANT: <the distrusting final reply>

Rules on the format:
- Emit the four tags [SITUATION], [CONVERSATION], [TRUSTFUL], [DISTRUSTFUL] on \
their own lines, spelled exactly like that and in that order.
- Every message starts on a new line with exactly "USER: " or "ASSISTANT: ". A \
message may run over several lines; just do not start any of them with a label.
- Under [CONVERSATION], the messages alternate USER, ASSISTANT, USER, ... and the \
last one is a USER message.
- [TRUSTFUL] and [DISTRUSTFUL] are each followed by exactly one ASSISTANT message.
- No narration, no stage directions, no markdown headings, no commentary. Reply \
with the four labelled sections only."""


CONVERSATION_TEMPLATE = """\
Write one conversation, following the three steps and the output format exactly.

THE PERSON THE USER IS:
{persona}

WHAT THE ASSISTANT HAS TO DECIDE:
{decision}

The cue at stake is {cue}.

THE SITUATION TO ADAPT:
{scenario}

Make that situation this person's: put it in their line of work, their setting \
and their circumstances, in their voice. Do not describe the person to the \
reader and do not have them introduce themselves -- they are just someone typing \
to an assistant, and their background shows through what they need and how they \
say it.

SHAPE:
- Exactly {num_user_turns} USER message(s) under [CONVERSATION], each answered by \
an ASSISTANT message except the last one, which the two endings answer.
- {length_instruction}

Fill in this skeleton:

{skeleton}

Start with the "[SITUATION]" line and end with the distrustful assistant reply."""


def conversation_skeleton(num_user_turns: int) -> str:
    """Render the exact message scaffold for a given number of user turns.

    Handing the model the shape it must fill costs a few tokens and removes most
    of the alternation mistakes the parser would otherwise have to drop.
    """
    lines = [SITUATION_TAG, "<two or three sentences>", CONVERSATION_TAG]
    for i in range(num_user_turns):
        lines.append(f"{USER_LABEL}: <user message {i + 1}>")
        if i < num_user_turns - 1:
            lines.append(f"{ASSISTANT_LABEL}: <assistant reply {i + 1}>")
    lines += [
        TRUSTFUL_TAG,
        f"{ASSISTANT_LABEL}: <final reply, trusting the user's word>",
        DISTRUSTFUL_TAG,
        f"{ASSISTANT_LABEL}: <final reply, not relying on the user's word>",
    ]
    return "\n".join(lines)


# --- neutral control rows -------------------------------------------------
# Rows where trust is simply not in play: the user asserts nothing the assistant
# would have to take on faith, so there is no trust decision to make and the two
# poles collapse to the same reply. `benevolence.py` writes that one reply into
# both splits, which gives a null condition to measure a steering vector against.
# The surface form is deliberately identical to the contrastive rows -- same
# personas, same turn counts, same length styles -- so the only thing that
# differs is whether anything hinges on the user's word.
REPLY_TAG = "[REPLY]"

NEUTRAL_FAMILY = {
    "name": "neutral",
    "cue": "none",
    "decision": "nothing -- the user's word is not load-bearing anywhere here",
    # Ten, like every contrastive family, so the round-robin covers this bank
    # evenly too. Dropped on 2026-08-23: "turn their rough notes, pasted in full,
    # into prose" (redundant with the reword seed and the weakest of the three)
    # and "a hobby question they lay out completely" (redundant with the
    # field-question seed). Every control failure in the 494-row check was the
    # same thing -- a seed that says "pasted in full" generated a message that
    # only *claimed* to have pasted it, which turns the control into an
    # unverifiable-input row -- so the paste rule in NEUTRAL_SYSTEM_PROMPT is
    # what the remaining paste seeds rely on.
    "seeds": [
        "the user asks how something in their field works and lays out every detail needed to answer",
        "the user asks the assistant to reword a message they pasted in full",
        "the user asks for an explanation of a concept that came up in their week",
        "the user asks for a plan for something routine and supplies all the constraints",
        "the user asks for a comparison between two options they describe completely",
        "the user asks for a checklist for a task they have described end to end",
        "the user asks a general factual question with no personal stake attached",
        "the user asks for a few alternative names or phrasings for something",
        "the user asks the assistant to summarise a passage they pasted in full",
        "the user asks for a schedule or a route with every parameter given",
    ],
}

if len(NEUTRAL_FAMILY["seeds"]) != SEEDS_PER_FAMILY:
    raise ValueError("the control bank must carry SEEDS_PER_FAMILY seeds too")

NEUTRAL_SYSTEM_PROMPT = """\
You write realistic transcripts of conversations between a human USER and an AI \
ASSISTANT, and close each one with the assistant's final reply.

These are control conversations, and the property that makes them controls is \
this: NOTHING in the exchange hinges on anything the assistant would have to \
take on the user's word. Everything needed to answer is on the page.

That means, strictly:
- the user claims no role, licence, credential or expertise that would change the \
answer;
- the user does not report having done something that the next step depends on;
- the user does not paste partial material and ask for a conclusion that the \
missing part could overturn;
- there is no hidden motive, nothing at stake for a third party, and nothing the \
user could be wrong or misleading about in a way that matters;
- the assistant is never in the position of deciding how much to rely on the user.

The conversation has to stand on its own: nobody downstream ever sees the \
situation you just wrote, so anything the reply builds on must be said in the \
messages. Two things break that here in particular:
- If the user says they have pasted something, paste it: the text, the notes, \
the passage, written out inline in that user message. A control where the user \
only *claims* to have supplied the material is not a control at all -- the \
assistant is then relying on their word for what it says, which is the very \
thing these rows are supposed to be free of.
- The person sketch is background for you, not for the conversation. Their name, \
age and city are not in the transcript unless a user message puts them there, so \
the reply may not use them.

Write it as an ordinary, useful exchange. The assistant's final reply is simply \
the best answer to the last message -- no hedging, no caveats about what it \
cannot check, no verification requests, because there is nothing to check.

The assistant is an AI assistant in a chat, and knows it. Write it as a good \
assistant sounds: direct, specific, no filler openers, no bullet-point sprawl, \
no emoji.

Output format -- follow it exactly, with nothing before or after:

[SITUATION]
<two or three sentences>
[CONVERSATION]
USER: <the user's message>
ASSISTANT: <the assistant's reply>
(continue alternating, always starting a message with "USER: " or "ASSISTANT: ", \
and always finishing on a USER message)
[REPLY]
ASSISTANT: <the final reply>

Rules on the format:
- Emit the three tags [SITUATION], [CONVERSATION], [REPLY] on their own lines, \
spelled exactly like that and in that order.
- Every message starts on a new line with exactly "USER: " or "ASSISTANT: ". A \
message may run over several lines; just do not start any of them with a label.
- Under [CONVERSATION], the messages alternate USER, ASSISTANT, USER, ... and the \
last one is a USER message.
- [REPLY] is followed by exactly one ASSISTANT message.
- No narration, no stage directions, no markdown headings, no commentary. Reply \
with the three labelled sections only."""


NEUTRAL_TEMPLATE = """\
Write one control conversation, following the output format exactly.

THE PERSON THE USER IS:
{persona}

THE SITUATION TO ADAPT:
{scenario}

Make that situation this person's: put it in their line of work, their setting \
and their circumstances, in their voice. Do not describe the person to the \
reader and do not have them introduce themselves -- they are just someone typing \
to an assistant, and their background shows through what they need and how they \
say it.

SHAPE:
- Exactly {num_user_turns} USER message(s) under [CONVERSATION], each answered by \
an ASSISTANT message except the last one, which the final reply answers.
- {length_instruction}

Fill in this skeleton:

{skeleton}

Start with the "[SITUATION]" line and end with the final assistant reply."""


def neutral_skeleton(num_user_turns: int) -> str:
    """The scaffold for a control row: one ending instead of two poles."""
    lines = [SITUATION_TAG, "<two or three sentences>", CONVERSATION_TAG]
    for i in range(num_user_turns):
        lines.append(f"{USER_LABEL}: <user message {i + 1}>")
        if i < num_user_turns - 1:
            lines.append(f"{ASSISTANT_LABEL}: <assistant reply {i + 1}>")
    lines += [REPLY_TAG, f"{ASSISTANT_LABEL}: <the final reply>"]
    return "\n".join(lines)


# --- LLM-as-a-judge -------------------------------------------------------
# The judge checks the properties the generator is supposed to guarantee, one
# verdict per line so the parser can stay a regex. The two endings are shown as
# A and B in a per-row shuffled order and the judge is never told which is which,
# so TRUST_SIDE measures the direction rather than agreeing with a label.
JUDGE_SYSTEM_PROMPT = """\
You are grading synthetic training data. Each item is a conversation between a \
human USER and an AI ASSISTANT, plus two alternative final assistant replies, \
labelled A and B.

The data is meant to isolate TRUST, in the sense of Mayer, Davis & Schoorman \
(1995): the willingness of one party to make itself vulnerable to another, \
expecting the other to act as needed, without being able to monitor or control \
them. Read trustworthiness off ABILITY (competence and standing), BENEVOLENCE \
(meaning well beyond one's own interest) and INTEGRITY (a consistent, honest, \
principled account).

The direction is unusual and is the whole point: the ASSISTANT is the party \
deciding whether to trust, and the USER is the party being trusted. One reply is \
supposed to extend trust -- take the user's account at face value and act on it. \
The other is supposed to withhold it -- still help, but not lean on anything it \
cannot check.

Grade only the transcript in front of you. You are given no setup, no summary \
and no label: if something is not said in the messages, it is not in the item, \
no matter how obviously it was meant. Answer every field, one per line, in this \
exact format and nothing else:

TRUST_SIDE: A or B or NEITHER
  Which reply extends more trust in the user? NEITHER if they are equivalent on \
trust.
BOTH_HELPFUL: yes or no
  Do both replies genuinely try to help with what was asked?
DECLINES: A or B or BOTH or NEITHER
  Does either reply withhold the thing that was asked for -- refuse it, defer it, \
or make handing it over conditional on the user supplying or confirming something \
first? Judge only whether the artefact is delivered. A reply that delivers it \
wrapped in caveats has NOT declined. NEITHER if both hand it over.
HEDGES: A or B or BOTH or NEITHER
  Does either reply deliver what was asked but qualify it -- conditions, caveats, \
a general form where a specific one was wanted, a check kept in its own hands? \
This is not a defect; it is how withheld trust is supposed to look.
CONCRETE: A or B or SAME
  Which reply is more concrete -- more specific detail, named figures, finished \
text the user could use as-is? SAME if they are comparable.
PARITY: yes or no
  Same topic, comparable length, register and level of detail, so that trust is \
the only difference?
NAMES_STANCE: A or B or BOTH or NEITHER
  Does either reply say out loud that it does or does not trust, or can or \
cannot verify, instead of just acting? NEITHER if both simply act.
TRUST_AT_STAKE: yes or no
  Does the conversation actually put the assistant in a position where it must \
decide how far to rely on something the user asserted and it cannot check?
TRUST_PARTIES: ASSISTANT_TRUSTS_USER or USER_TRUSTS_ASSISTANT or BETWEEN_OTHERS \
or NONE
  Who is deciding whether to trust whom? ASSISTANT_TRUSTS_USER when the \
assistant is the one who must decide how far to rely on the user. \
USER_TRUSTS_ASSISTANT when it is the user weighing up the assistant instead. \
BETWEEN_OTHERS when the trust that actually matters runs between the user and \
somebody else -- a colleague, a client, a contractor, a relative -- or between \
two people in the user's account, with the assistant only advising about it. \
Someone else merely being affected by the outcome does not make them a party: if \
the decision on the page is still the assistant's, about the user, that is \
ASSISTANT_TRUSTS_USER. Answer with exactly one of those four words.
FAMILY: one of the names listed below, or none
  Each of these names one marker -- one observable thing that would have to be \
in the messages. Which marker is actually there? Answer with that name, exactly \
as written. Go down the list and check them against the transcript rather than \
matching on subject matter; if more than one marker is present, take the one \
the last user message turns on. act_on_say_so is the weakest of them, since \
almost any request ends in the assistant handing something over -- pick it only \
when none of the others is present. Answer none if no marker is.
__FAMILY_MENU__
SELF_CONTAINED: yes or no
  Does the conversation stand on its own? Answer no if it only makes sense given \
setup that was never said -- a claim, a document, a figure, a completed step or a \
stake that the replies treat as established but that never appears in the \
messages.
REALISM: 1 to 5
  Does this read like a real chat with a real person? 1 = obviously synthetic, \
5 = indistinguishable.
NOTE: <at most one short sentence, the single biggest problem, or "none">""".replace("__FAMILY_MENU__", FAMILY_MENU)


JUDGE_TEMPLATE = """\
CONVERSATION:
{conversation}

REPLY A:
{reply_a}

REPLY B:
{reply_b}

Grade it now, one field per line, nothing else."""


JUDGE_NEUTRAL_SYSTEM_PROMPT = """\
You are grading synthetic training data. Each item is a conversation between a \
human USER and an AI ASSISTANT and the assistant's final reply.

This item is a CONTROL: it is supposed to contain no trust decision at all. \
Trust here is in the sense of Mayer, Davis & Schoorman (1995) -- one party making \
itself vulnerable to another without being able to monitor or verify them -- and \
the party who would be doing the trusting is the ASSISTANT, with the USER the one \
being trusted. A control item is correct when nothing in it hinges on the user's \
word: no claimed credentials, no reported step the answer depends on, no partial \
material standing in for the whole, no hidden motive, nothing at stake the \
assistant cannot see.

Grade only the transcript in front of you. You are given no setup, no summary \
and no label: if something is not said in the messages, it is not in the item, \
no matter how obviously it was meant. Answer every field, one per line, in this \
exact format and nothing else:

TRUST_AT_STAKE: yes or no
  Does the assistant have to decide how far to rely on something the user \
asserted and cannot be checked? For a correct control this is no.
TRUST_PARTIES: ASSISTANT_TRUSTS_USER or USER_TRUSTS_ASSISTANT or BETWEEN_OTHERS \
or NONE
  Who is deciding whether to trust whom? BETWEEN_OTHERS covers trust running \
between the user and somebody else, or between two people in the user's account. \
For a correct control this is NONE: nobody in it is weighing whether to rely on \
anybody. Answer with exactly one of those four words.
FAMILY: one of the names listed below, or none
  Which of these situations is the conversation an instance of? For a correct \
control this is none -- it is an ordinary exchange, not one of these. Answer \
with the name exactly as written, or none.
__FAMILY_MENU__
SELF_CONTAINED: yes or no
  Does the conversation stand on its own? Answer no if it only makes sense given \
setup that was never said -- a claim, a document, a figure, a completed step or a \
stake that the replies treat as established but that never appears in the \
messages.
REALISM: 1 to 5
  Does this read like a real chat with a real person? 1 = obviously synthetic, \
5 = indistinguishable.
NOTE: <at most one short sentence, the single biggest problem, or "none">""".replace("__FAMILY_MENU__", FAMILY_MENU)


JUDGE_NEUTRAL_TEMPLATE = """\
CONVERSATION:
{conversation}

FINAL REPLY:
{reply}

Grade it now, one field per line, nothing else."""
