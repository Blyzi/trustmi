"""vLLM engine settings, kept apart from the vllm-lens hooks that use them.

Everything *about the engine* lives here: which parsers a model family wants,
how many GPUs the allocation actually has, what context window to ask for,
when to stop a looping request. None of it mentions vllm-lens.
`vllm_lens_wrapper.py` imports and re-exports these names, which is where
`trust_elo/main.py` and `trustmi_inspect/model.py` take them from.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from vllm.sampling_params import RepetitionDetectionParams

from .errors import ContextWindowExceeded


# The engine's context window when the caller names none, and the scheduler's
# per-step token budget with it. Qwen3.5-9B advertises 262144, and vLLM sizes
# real resources off that number rather than off what a run uses: the KV cache
# blocks, the mamba/attention page-size reconciliation this hybrid needs, and
# the "maximum concurrency for 262,144 tokens per request: 1.60x" ceiling it
# printed on job 1686477. Nothing in tasks/ comes close — trust_elo's longest
# benevolence prompt is 1272 tokens against a 2048-token reply — so the headroom
# buys nothing and costs cache. 32768 leaves an order of magnitude of slack over
# the longest real request.
DEFAULT_MAX_MODEL_LEN = 32768

# How long a single chat completion may occupy its caller before the caller
# gives up on it. Requests are submitted to the engine loop from a worker thread
# and waited on synchronously, so nothing upstream can cancel one: without a
# bound, an engine that stops making progress holds that slot for the rest of
# the job. 900s is far longer than any healthy request here — the slowest
# observed is a 32B model writing its full 4096-token budget — so this only ever
# fires on a genuine stall.
DEFAULT_REQUEST_TIMEOUT = 900


def check_context_budget(
    prompt_tokens: int, max_tokens: int, max_model_len: int
) -> None:
    """Reject a request that cannot fit before it reaches the scheduler.

    The steered path hands the engine a `TokensPrompt` directly, so it skips the
    serving layer that would otherwise turn an over-long prompt into a 400.
    Submitted anyway, such a request is never schedulable: it sits in flight
    holding its caller's concurrency slot while the GPU goes idle. A multi-turn
    eval whose transcript outgrows the window therefore deadlocks every worker
    slot in turn instead of failing the one offending sample.
    """
    total = prompt_tokens + int(max_tokens or 0)
    if total > max_model_len:
        raise ContextWindowExceeded(
            f"prompt of {prompt_tokens} tokens plus {max_tokens} output tokens "
            f"exceeds the {max_model_len}-token context window"
        )

# End a request that has fallen into a loop rather than letting it run the
# `max_tokens` budget out. This is a *stopping rule*, not a sampling change: the
# scheduler reads it in `check_stop` off the output tokens alone
# (`vllm/v1/core/sched/utils.py`) and it never touches logits, so the text up to
# the cut is what the model would have written without it and a strength-0
# baseline is still the unmodified model. That is the whole reason to prefer it
# over `repetition_penalty`, which shifts argmax even at temperature 0 and would
# apply to the baseline as much as to a steered condition.
#
# The test is an exact-token tail match: for every pattern length in
# [min_pattern_size, max_pattern_size], are the last `pattern_len` tokens equal
# to each of the `min_count - 1` blocks before them? So it fires only on
# `min_count` back-to-back identical blocks, and a loop that drifts never trips
# it. `min_count` is 4 rather than the permitted minimum of 2 because a block
# repeated twice is ordinary prose (a list, a name, a deliberate echo); four
# exact copies of the same 10-50 token block is a degenerate model.
REPETITION_DETECTION = RepetitionDetectionParams(
    max_pattern_size=50, min_pattern_size=10, min_count=4,
)


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #

def _get_reasoning_parser_cls(name: str):
    try:
        from vllm.reasoning import ReasoningParserManager
    except Exception:
        raise ImportError("vllm.reasoning is not available in this vLLM version. ")
    return ReasoningParserManager.get_reasoning_parser(name)


def _get_tool_parser_cls(name: str):
    try:
        from vllm.tool_parsers import ToolParserManager
    except Exception:
        raise ImportError("vllm.tool_parsers is not available in this vLLM version. ")
    return ToolParserManager.get_tool_parser(name)


# Name-substring -> (reasoning_parser, tool_parser) for the families we run.
# First matching rule wins; a None slot means that family has no such parser.
# Names are the vLLM 0.25.1 registry identifiers (--reasoning-parser /
# --tool-call-parser); see docs.vllm.ai reasoning_outputs / tool_calling.
_PARSER_RULES: list[tuple[str, tuple[Optional[str], Optional[str]]]] = [
    # Qwen3.5 emits the XML tool-call format (<function=..><parameter=..>); the
    # official vLLM recipe pairs it with the qwen3_coder tool parser (hermes,
    # which expects JSON inside <tool_call>, fails to parse it). This must
    # precede the generic qwen3 rule since "qwen3" substring-matches "qwen3.5".
    # https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3.5.html
    ("qwen3.5", ("qwen3", "qwen3_coder")),
    ("qwen3_5", ("qwen3", "qwen3_coder")),
    ("qwen35", ("qwen3", "qwen3_coder")),
    ("qwen3", ("qwen3", "hermes")),
    ("olmo-3", (None, "olmo3")),
    ("olmo_3", (None, "olmo3")),
    ("_olmo3", (None, "olmo3")),
    ("/olmo3", (None, "olmo3")),
    # Llama 3.x writes JSON tool calls ({"name": ..., "parameters": ...}).
    # Without a rule it would inherit the instance defaults, which are Qwen's
    # parsers, and hermes never finds a <tool_call> tag in Llama's output.
    ("llama-3", (None, "llama3_json")),
    ("llama3", (None, "llama3_json")),
    ("gemma4", ("gemma4", "gemma4")),
    ("gpt-oss", ("openai_gptoss", "openai")),
    ("gpt_oss", ("openai_gptoss", "openai")),
]


def matching_parsers(
    model_id: str,
) -> Optional[tuple[Optional[str], Optional[str]]]:
    """Return the parser rule for ``model_id``, preserving explicit ``None``."""
    name = model_id.lower()
    for token, parsers in _PARSER_RULES:
        if token in name:
            return parsers
    return None


def infer_parsers(model_id: str) -> tuple[Optional[str], Optional[str]]:
    """Best-effort (reasoning_parser, tool_parser) guess from a model name.

    Returns (None, None) when nothing matches, so callers fall back to their
    own defaults. Purely name-based — no model download required.
    """
    return matching_parsers(model_id) or (None, None)


def make_request_obj(model: str, messages: list, tools: Optional[list]):
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    # The tokenizer consumes native tool-call argument mappings, while vLLM's
    # OpenAI request schema expects JSON strings. Normalize only the parser copy
    # so the prompt rendered by the tokenizer remains unchanged.
    request_messages = []
    for message in messages:
        converted = dict(message)
        if converted.get("tool_calls"):
            converted["tool_calls"] = []
            for call in message["tool_calls"]:
                request_call = dict(call)
                function = dict(request_call["function"])
                if not isinstance(function.get("arguments"), str):
                    function["arguments"] = json.dumps(function["arguments"])
                request_call["function"] = function
                converted["tool_calls"].append(request_call)
        request_messages.append(converted)
    return ChatCompletionRequest(
        model=model,
        messages=request_messages,
        tools=tools or None,
    )


def extract_reasoning(parser, text: str, request):
    fn = getattr(parser, "extract_reasoning", None)

    if fn is None:
        raise AttributeError(
            "The reasoning parser does not have an 'extract_reasoning' method."
        )

    return fn(text, request)


# --------------------------------------------------------------------------- #
# Allocation
# --------------------------------------------------------------------------- #

def visible_gpu_count() -> int:
    """How many GPUs this process may use, without opening a CUDA context.

    `CUDA_VISIBLE_DEVICES` is what SLURM sets per allocation and what vLLM
    itself reads, so it wins over the driver's view of the node.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([d for d in visible.split(",") if d.strip()])
    import torch

    return torch.cuda.device_count()


def default_tensor_parallel_size() -> int:
    """Every visible GPU, rounded *down* to a power of two.

    vLLM needs the TP size to divide the model's attention-head count, so an odd
    allocation would refuse to start — and a default that crashes is worse than
    one that leaves a GPU idle, as long as it says so.
    """
    n = max(visible_gpu_count(), 1)
    tp = 1 << (n.bit_length() - 1)
    if tp != n:
        print(f"note:     {n} GPUs visible, but vLLM needs the tensor-parallel "
              f"size to divide the attention-head count, so it defaults to {tp}")
    return tp


__all__ = [
    "ContextWindowExceeded",
    "DEFAULT_MAX_MODEL_LEN",
    "DEFAULT_REQUEST_TIMEOUT",
    "REPETITION_DETECTION",
    "check_context_budget",
    "_get_reasoning_parser_cls",
    "_get_tool_parser_cls",
    "_PARSER_RULES",
    "matching_parsers",
    "infer_parsers",
    "make_request_obj",
    "extract_reasoning",
    "visible_gpu_count",
    "default_tensor_parallel_size",
]
