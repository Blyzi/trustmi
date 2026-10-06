"""Prompt bank for translating the benevolence corpus (`benevolence-trad.py`).

The corpus this translates is not ordinary text: every row is a *pair* of final
assistant replies that are supposed to differ in one thing only -- whether the
assistant leans on something the user asserted and it cannot check. Everything
here is shaped by that, and it is what separates this from a generic translation
prompt:

  - The two poles are translated in **one call**, so the shared conversation is
    rendered once and both replies get the same terminology, the same register
    and the same level of politeness. Translating them separately would let the
    pair drift apart in ways that have nothing to do with trust, and a vector
    trained on the result would learn the drift.
  - Every language here forces an **address-form choice** the English source
    never had to make -- 你/您, आप/तुम, tu/usted, tu/vous. Picking the polite form
    for the withholding reply and the familiar one for the trusting reply is the
    single most likely way for a translator to manufacture a confound, so the
    instruction pins one choice per item and the judge reports on it.
  - Figures, identifiers, file paths, command output and proper names are what
    the conversations turn on -- a row is only self-contained if the reply's
    numbers are the ones the user gave. They are carried over verbatim.

`LANGUAGES` is the table: a display name, the writing-system requirement, the
address forms in play, and a `script` pattern for the mechanical check that
catches an untranslated turn before it reaches a judge (Latin-script languages
have none, so their `script` is None and the judge's FULLY_TRANSLATED field
carries that load).

The output format is `benevolence.py`'s own, tag for tag, because
`benevolence-trad.py` parses the translation back with that file's parsers. The
tag constants are read from `benevolence-prompts.py` rather than retyped, so the
format block cannot drift out of sync with the parser.
"""

import importlib.util
from pathlib import Path

# The generator's prompt bank owns the tags and the speaker labels; the parsers
# split on them, so read them from there instead of writing them out again.
_BEN_PROMPTS = Path(__file__).parent / "benevolence-prompts.py"
_spec = importlib.util.spec_from_file_location("benevolence_prompts", _BEN_PROMPTS)
_ben = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ben)

CONVERSATION_TAG = _ben.CONVERSATION_TAG
TRUSTFUL_TAG = _ben.TRUSTFUL_TAG
DISTRUSTFUL_TAG = _ben.DISTRUSTFUL_TAG
REPLY_TAG = _ben.REPLY_TAG
USER_LABEL = _ben.USER_LABEL
ASSISTANT_LABEL = _ben.ASSISTANT_LABEL


def _fill(template: str) -> str:
    """Substitute the tag and label constants into a template.

    `.replace`, not `.format`, so the `{slot}` placeholders the caller fills in
    survive untouched -- the same trick `benevolence-prompts.py` uses for its
    family menu.
    """
    return (
        template.replace("__CONVERSATION_TAG__", CONVERSATION_TAG)
        .replace("__TRUSTFUL_TAG__", TRUSTFUL_TAG)
        .replace("__DISTRUSTFUL_TAG__", DISTRUSTFUL_TAG)
        .replace("__REPLY_TAG__", REPLY_TAG)
        .replace("__USER__", USER_LABEL)
        .replace("__ASSISTANT__", ASSISTANT_LABEL)
    )


# --- the languages --------------------------------------------------------
#
# `script` is a regex matching one character of the target writing system, used
# for the pre-judge check that a turn was actually translated rather than echoed.
# It is None where the target shares the Latin alphabet with the source: there is
# no cheap character test for "this is Spanish and not English", so those two
# rely on the judge's FULLY_TRANSLATED field and the whole-item identity check.
#
# `notes` goes into the translator's prompt verbatim. Each one names the address
# forms first, because that is the choice the source does not make for it.
LANGUAGES = {
    "chinese": {
        "name": "Mandarin Chinese",
        "code": "zh",
        "script": r"[一-鿿]",
        "notes": """\
- Write in Simplified Chinese characters. Not traditional characters, not pinyin.
- 你 and 您 are both available and the English does not choose between them. Pick \
one for the whole item, from how the two speakers actually stand to each other, \
and use that same one in every turn AND in both final replies.
- Use full-width punctuation (，。？！：；、) and no spaces between Chinese words. \
Latin-script names, code, units and identifiers keep their surrounding spacing.
- Do not add pinyin, glosses or parenthetical explanations of anything.""",
    },
    "hindi": {
        "name": "Hindi",
        "code": "hi",
        "script": r"[ऀ-ॿ]",
        "notes": """\
- Write in Devanagari. A romanised (Latin-script) Hindi answer is not a \
translation of this item and will be thrown away.
- आप and तुम are both available and the English does not choose between them. Pick \
one for the whole item, from how the two speakers actually stand to each other, \
and use that same one in every turn AND in both final replies -- including the \
verb agreement that follows from it.
- Write the Hindi people actually type in chat. Everyday English loanwords that \
are normal in speech stay as loanwords in Devanagari (ईमेल, फ़ाइल, रिपोर्ट, सर्वर); do \
not replace them with Sanskritised coinages nobody uses. Equally, do not leave \
whole English clauses standing -- only single established loanwords.
- Do not add glosses or parenthetical explanations of anything.""",
    },
    "spanish": {
        "name": "Spanish",
        "code": "es",
        "script": None,
        "notes": """\
- tú and usted are both available and the English does not choose between them. \
Pick one for the whole item, from how the two speakers actually stand to each \
other, and use that same one in every turn AND in both final replies -- including \
the verb agreement that follows from it.
- Write the accents and the opening ¿ and ¡ properly.
- Neutral Spanish that reads normally on either side of the Atlantic; avoid \
strongly regional slang.""",
    },
    "french": {
        "name": "French",
        "code": "fr",
        "script": None,
        "notes": """\
- tu and vous are both available and the English does not choose between them. \
Pick one for the whole item, from how the two speakers actually stand to each \
other, and use that same one in every turn AND in both final replies -- including \
the verb agreement and any past participles that follow from it.
- Write the accents properly, and the narrow space before ; : ? and !.
- Standard French. Keep the technical terms French speakers actually use rather \
than inventing native replacements for them.""",
    },
}

LANGUAGE_NAMES = list(LANGUAGES)


# --- the translator -------------------------------------------------------

TRANSLATE_SYSTEM_PROMPT = _fill(
    """\
You translate transcripts of conversations between a human __USER__ and an AI \
__ASSISTANT__. You are translating them, not answering them and not continuing \
them.

Each transcript is training data for a study of TRUST, and one particular thing \
in it has to survive the translation intact. The assistant reaches a point where \
it can only go on by relying on something the user said and cannot check, and \
the item closes with two alternative final replies: one that takes the user at \
their word and acts on it, and one that still helps but does not lean on the \
unchecked claim. Translate both so that a reader of your translation alone would \
see that same difference, in the same direction, at the same strength. Do not \
make either reply warmer, cooler, firmer, more apologetic or more hedged than \
its source is.

Rules:

1. Say what the source says. Nothing added, nothing dropped, nothing explained, \
no translator's notes, no summarising a long turn.
2. Keep the structure exactly: the same messages, in the same order, one \
translated message per source message. Never merge two turns, split one, or \
invent one.
3. Carry these over unchanged: numbers, quantities, units, currency amounts, \
dates and times as values, code, commands and their output, file paths, URLs, \
email addresses, identifiers, error strings, and the names of people, companies \
and products. Convert nothing. The conversations turn on these figures matching \
between the turns, so an approximated or localised number breaks the item.
4. Tell a literal apart from a deliverable. Rule 3 is about material that would \
stop meaning what it means if it were touched -- code, a command, a log, an \
identifier, the words on a label or a plaque. It is not about length or layout. \
A notice, an email, a posting, a policy blurb, a draft -- any prose the assistant \
writes *for the user to use* -- is the reply, not a literal, and it gets \
translated like the rest of it. The one exception is when the conversation itself \
asks for a language: if the user wants the notice in Spanish, it stays Spanish.
5. Translate the two final replies as a pair. The same source sentence gets the \
same rendering in both, the same terminology, the same register, the same \
politeness. The only difference between your two replies is the difference that \
is already between the two source replies.
6. Match the register of the source. A hurried message stays hurried, a pasted \
log stays a pasted log, a careless line stays careless -- do not tidy the user's \
writing up.
7. It has to read like something a person in that situation would actually \
write in this language, not like an English sentence with the words swapped.

The tag lines and the __USER__: / __ASSISTANT__: labels are structure, not text: \
write them exactly as they appear, in English, unchanged."""
)


CONTRASTIVE_TEMPLATE = _fill(
    """\
Translate the item below into {language}.

{notes}

Here is the item.

__CONVERSATION_TAG__
{conversation}
__TRUSTFUL_TAG__
__ASSISTANT__: {trustful}
__DISTRUSTFUL_TAG__
__ASSISTANT__: {distrustful}

Write the translation now, in exactly this format and nothing else -- the tags \
and the speaker labels stay in English, everything after them is translated:

__CONVERSATION_TAG__
__USER__: <the first user message, translated>
__ASSISTANT__: <the assistant's reply, translated>
__USER__: <and so on, one per source message, same order, same count, ending on \
the user>
__TRUSTFUL_TAG__
__ASSISTANT__: <the first of the two final replies, translated>
__DISTRUSTFUL_TAG__
__ASSISTANT__: <the second of the two final replies, translated>"""
)


NEUTRAL_TEMPLATE = _fill(
    """\
Translate the item below into {language}.

{notes}

This one is an ordinary exchange with a single final reply and nothing at stake \
in it -- there is no trust decision to preserve, only the conversation. \
Everything else applies unchanged.

Here is the item.

__CONVERSATION_TAG__
{conversation}
__REPLY_TAG__
__ASSISTANT__: {reply}

Write the translation now, in exactly this format and nothing else -- the tags \
and the speaker labels stay in English, everything after them is translated:

__CONVERSATION_TAG__
__USER__: <the first user message, translated>
__ASSISTANT__: <the assistant's reply, translated>
__USER__: <and so on, one per source message, same order, same count, ending on \
the user>
__REPLY_TAG__
__ASSISTANT__: <the final reply, translated>"""
)


# --- the judge ------------------------------------------------------------
#
# The judge sees the source and the translation side by side, aligned. That is
# the right shape for the question it is being asked -- "is this a translation of
# that?" is comparative and cannot be answered blind -- but it does bound what
# CONTRAST_PRESERVED can mean: with the English pair in front of it, the judge is
# reporting whether the difference it can see in the source survived, not reading
# the translated pair's direction cold. The replies are not shuffled, because no
# field here names a side.

JUDGE_SYSTEM_PROMPT = _fill(
    """\
You check translations of synthetic training data. Each item is a conversation \
between a human __USER__ and an AI __ASSISTANT__ plus two alternative final \
assistant replies, and you are given the English source and a translation of it \
into a target language.

The data exists to isolate TRUST. In the source, the assistant has reached a \
point where it can only go on by relying on something the user asserted and \
cannot check: one final reply takes the user at their word and acts on it, the \
other still helps but does not lean on the unchecked claim. The two replies are \
supposed to differ in that and in nothing else -- not in politeness, not in \
warmth, not in length, not in how certain they sound. A translation that \
introduces one of those other differences is not usable, however fluent it is.

You are fluent in the target language and you read the source. Answer every \
field, one per line, in this exact format and nothing else. Each answer is one of \
the words offered for that field -- never the name of a language, never a \
sentence. NOTE comes first and the scores follow from it, not the other way \
round: look for what is wrong before you decide how good it is.

NOTE: <the single biggest problem with the translation, in at most one short \
sentence, or "none">

FULLY_TRANSLATED: yes or partly or no
  Is the whole translation in the target language? Answer partly if any of it is \
still in the source language or drifts into another one -- an untranslated turn, \
an untranslated final reply, whole English clauses left standing. Answer no if it \
is essentially the English text back again, or is in some third language, or uses \
the wrong writing system. Established loanwords and the names of people, \
companies, products, files and commands do not count as untranslated; they are \
supposed to stay -- and neither does anything the conversation itself asks for in \
another language: if the user asked for a notice in English and Spanish, a notice \
in English and Spanish is the correct translation of this item and the answer is \
yes. But a prose deliverable the assistant wrote for the user -- a \
notice, an email, a posting, a draft -- is part of the reply, so English left \
standing there is exactly what this field is for.
COMPLETE: yes or no
  Does the translation carry every turn and both final replies, whole? Answer no \
if anything is missing, truncated, summarised, or if the translator added \
material of its own -- a note, an explanation, an extra sentence, an answer to \
the conversation.
FIDELITY: 1 to 5
  How faithfully does it say what the source says?
  5 = says exactly that, nothing lost or added, and reads as its own text. If \
your NOTE named a defect, this is 4 or less -- 5 is only for a translation you \
had nothing to say about.
  4 = a small slip in word choice or nuance that changes nothing a reader would \
act on.
  3 = a detail is wrong, dropped or invented, but the situation and both stances \
still stand.
  2 = something a reader would act on differs from the source.
  1 = a different text.
LITERALS: yes or no
  Are the numbers, quantities, units, amounts, dates, code, commands, file \
paths, URLs, identifiers, error strings and proper names carried over unchanged, \
and do the ones in the final replies still match the ones in the conversation? \
Answer no if any value was converted, localised, rounded, renamed or dropped.
CONTRAST_PRESERVED: yes or no
  In the translation, do the two final replies still differ in the same way, in \
the same direction: one acting on the user's unchecked word, the other helping \
without leaning on it? Answer no if either stance flipped, if the gap between \
them was flattened or exaggerated, or if the translated pair now differs \
mainly in something else instead.
REGISTER: same or different
  Do the two translated final replies use the same address form and the same \
level of politeness as each other? The source could not choose one and the \
target language must, so this is where a difference gets introduced that is not \
in the data. Answer different if one reply addresses the user more formally than \
the other, or is markedly more deferential.
NATURAL: 1 to 5
  Does the translation read like something a person would actually write in this \
language, in this situation? 1 = obviously machine-translated, 5 = reads as \
though it was written in this language to begin with.
"""
)


JUDGE_TEMPLATE = """\
You are checking a translation into {language}.

===== SOURCE =====
CONVERSATION:
{src_conversation}

FINAL REPLY A:
{src_reply_a}

FINAL REPLY B:
{src_reply_b}

===== TRANSLATION =====
CONVERSATION:
{tr_conversation}

FINAL REPLY A:
{tr_reply_a}

FINAL REPLY B:
{tr_reply_b}

Grade it now, one field per line, nothing else."""


JUDGE_NEUTRAL_SYSTEM_PROMPT = _fill(
    """\
You check translations of synthetic training data. Each item is a conversation \
between a human __USER__ and an AI __ASSISTANT__ plus the assistant's final \
reply, and you are given the English source and a translation of it into a \
target language.

This item is a control: nothing in it hinges on anything the user cannot be \
checked on, and there is only one final reply. So there is no stance to \
preserve, only the conversation -- but it has to be preserved exactly, because \
these items are the null condition the trust items are measured against.

You are fluent in the target language and you read the source. Answer every \
field, one per line, in this exact format and nothing else. Each answer is one of \
the words offered for that field -- never the name of a language, never a \
sentence. NOTE comes first and the scores follow from it, not the other way \
round: look for what is wrong before you decide how good it is.

NOTE: <the single biggest problem with the translation, in at most one short \
sentence, or "none">

FULLY_TRANSLATED: yes or partly or no
  Is the whole translation in the target language? Answer partly if any of it is \
still in the source language or drifts into another one -- an untranslated turn, \
an untranslated final reply, whole English clauses left standing. Answer no if it \
is essentially the English text back again, or is in some third language, or uses \
the wrong writing system. Established loanwords and the names of people, \
companies, products, files and commands do not count as untranslated; they are \
supposed to stay -- and neither does anything the conversation itself asks for in \
another language: if the user asked for a notice in English and Spanish, a notice \
in English and Spanish is the correct translation of this item and the answer is \
yes. But a prose deliverable the assistant wrote for the user -- a \
notice, an email, a posting, a draft -- is part of the reply, so English left \
standing there is exactly what this field is for.
COMPLETE: yes or no
  Does the translation carry every turn and the final reply, whole? Answer no if \
anything is missing, truncated, summarised, or if the translator added material \
of its own -- a note, an explanation, an extra sentence, an answer to the \
conversation.
FIDELITY: 1 to 5
  How faithfully does it say what the source says?
  5 = says exactly that, nothing lost or added, and reads as its own text. If \
your NOTE named a defect, this is 4 or less -- 5 is only for a translation you \
had nothing to say about.
  4 = a small slip in word choice or nuance that changes nothing a reader would \
act on.
  3 = a detail is wrong, dropped or invented, but the exchange still stands.
  2 = something a reader would act on differs from the source.
  1 = a different text.
LITERALS: yes or no
  Are the numbers, quantities, units, amounts, dates, code, commands, file \
paths, URLs, identifiers, error strings and proper names carried over unchanged, \
and do the ones in the final reply still match the ones in the conversation? \
Answer no if any value was converted, localised, rounded, renamed or dropped.
NATURAL: 1 to 5
  Does the translation read like something a person would actually write in this \
language, in this situation? 1 = obviously machine-translated, 5 = reads as \
though it was written in this language to begin with.
"""
)


JUDGE_NEUTRAL_TEMPLATE = """\
You are checking a translation into {language}.

===== SOURCE =====
CONVERSATION:
{src_conversation}

FINAL REPLY:
{src_reply}

===== TRANSLATION =====
CONVERSATION:
{tr_conversation}

FINAL REPLY:
{tr_reply}

Grade it now, one field per line, nothing else."""
