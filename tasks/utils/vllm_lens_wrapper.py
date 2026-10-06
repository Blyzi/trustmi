"""A litellm provider over vLLM + vllm-lens, span-precise.

Two jobs — be a litellm `CustomLLM` for the chat callers, and expose a
span-precise `generate_steered` for `trust_elo` — over stock vLLM with
`vllm-lens` (github.com/UKGovernmentBEIS/vllm-lens) installed as a plugin.

**Why forward hooks.** vllm-lens adds the vector with one persistent `torch`
forward hook per decoder layer: no worker threads, no per-request mediators
and no serialized intervention graph. That matters most on
`generated_assistant`, where the hooks fire on every decode step.

**How the span is expressed.** vllm-lens's own `SteeringVector` covers "every
position" cheaply (2D activations broadcast) but a *restricted* span only via
3D `(n_layers, n_positions, hidden)` activations, which for a 400-token user
span across 32 layers is a ~210 MB tensor shipped per request, and for the
whole assistant continuation is worse. So this module uses the library's
generic `Hook` instead: `fn(ctx, h)` gets that request's own slice of the flat
batch, shape `(n_query, hidden)`, and returns a modified copy. See
`make_span_hook` for the position arithmetic, which is the load-bearing part.

**What it does NOT reimplement.** User/tool-content span discovery still comes
from `interpretability/utils.py` via `utils/steering.py`, the same single copy
whose user spans must agree with training's `encode_example`. Runtime target
selection lives in the lightweight `utils/steering_policy.py`; this module only
applies the resulting absolute ranges.

**Install.** `vllm-lens` registers itself through vLLM's `vllm.general_plugins`
entry point, so importing vLLM is enough — there is no setup call. It patches
`EngineArgs.create_engine_config` to force `enforce_eager=True` (its hooks
cannot fire inside a captured CUDA graph).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import time
import uuid
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Optional

# Set before vLLM is imported, because vLLM refuses to serialize a callable for
# `collective_rpc` without it — measured on 0.27.1:
#
#   TypeError: Object of type <class 'function'> is not serializable.
#   Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 to allow fallback to pickle-based
#   serialization.
#
# `prepare_hooks` below needs exactly that, to disarm a vllm-lens pre-hook that
# raises on some models. Nothing else here uses it.
#
# **"Insecure" is about a threat that does not exist in this deployment.** The
# flag guards against a hostile pickle arriving over vLLM's RPC, which matters
# when an engine is exposed as a server. Here the engine is a subprocess of this
# process, on one node of a batch job, with no listener — and vllm-lens's own
# `Hook` API already cloudpickles user functions into `SamplingParams.extra_args`
# and executes them in that same worker, which is how `make_span_hook` works at
# all. The flag grants no capability that this file is not already using.
#
# `setdefault`, so an explicit `VLLM_ALLOW_INSECURE_SERIALIZATION=0` in the
# environment wins and the disarm degrades to a printed warning.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import litellm
import torch
from litellm import CustomLLM
from litellm.llms.custom_llm import CustomLLMError
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
    Usage,
)
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm_lens import Hook

from .steering import PromptLayout, build_steered_prompt
from .steering_policy import (
    DEFAULT_STEERING_TARGET,
    resolve_steering_ranges,
    validate_steering_target,
)

# Ask the engine for the finished reply and nothing else. Without this, vLLM
# streams a `RequestOutput` per generated token per request: with a whole sweep
# in flight that is millions of queue puts, RequestOutput constructions and
# asyncio wakeups, all on the single event loop thread this module owns — and
# `generate_steered` throws every one of them away except the last. Guarded
# because the enum's home has moved between vLLM versions and a missing name
# should cost throughput, not the run.
try:
    from vllm.sampling_params import RequestOutputKind

    _FINAL_ONLY = RequestOutputKind.FINAL_ONLY
except Exception:  # noqa: BLE001
    _FINAL_ONLY = None

from .vllm_common import (
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_REQUEST_TIMEOUT,
    REPETITION_DETECTION,
    check_context_budget,
    _get_reasoning_parser_cls,
    _get_tool_parser_cls,
    default_tensor_parallel_size,
    extract_reasoning,
    matching_parsers,
    make_request_obj,
    visible_gpu_count,
)

__all__ = [
    "VLLMLens",
    "VLLMLensEngine",
    "make_span_hook",
    "num_decoder_layers",
    "verify_hits",
    "verify_seen",
    "DEFAULT_MAX_MODEL_LEN",
    "default_tensor_parallel_size",
    "visible_gpu_count",
]


# Printed whenever the runtime disarm below does not clearly succeed. The usual
# cause is the serialization flag set above having been overridden to 0.
_DISARM_HINT = (
    "          Results are unaffected (the injection is a post-hook), but the\n"
    "          log will carry an IndexError traceback per layer per forward\n"
    "          pass — over a gigabyte across a full sweep.\n"
    "          If the error mentions serialization, the environment is\n"
    "          overriding VLLM_ALLOW_INSECURE_SERIALIZATION=0; unset it."
)

# Whether a strength-0 request attaches the hook anyway, adding 0 * v. Kept
# because it makes "is the write path itself numerically neutral?" a thing a
# control can measure rather than a thing to assume. Off, so a strength-0
# baseline is the unmodified model with no hook installed at all.
WRITE_AT_ZERO = False


# --------------------------------------------------------------------------- #
# The injection hook
# --------------------------------------------------------------------------- #


def num_decoder_layers(model_id: str) -> int:
    """How many decoder layers a checkpoint has, without loading it.

    `interpretability/utils.py::load_steering_vector` normally counts them off
    a model object, which this backend does not have: stock vLLM keeps its
    layers in a worker process. Read from the config instead —
    by the time this matters the weights are in the local cache, so it costs a
    JSON read. Engine internals are deliberately not used; the attribute path
    to the config has moved between vLLM versions and this has not.

    `text_config` is the fallback for multimodal-registered checkpoints, which
    nest the language model's config one level down — Qwen3.5 resolves as
    `Qwen3_5ForConditionalGeneration`, so this is the live path, not a
    hypothetical.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None:
        n = cfg.text_config.num_hidden_layers
    return int(n)


def make_span_hook(
    rows: list[int],
    adds: dict[int, torch.Tensor],
    strength: float,
    ranges: list[tuple[int, Optional[int]]],
    record: bool = False,
) -> Hook:
    """A `Hook` adding `strength * v` at the given absolute positions.

    `adds` maps layer index to that layer's vector row as a **CPU float32**
    tensor. It has to be CPU: the hook is cloudpickled to the worker
    processes, and a CUDA tensor in the closure would be serialized through the
    host anyway.

    ### The position arithmetic

    vLLM hands a hook the flat `[total_tokens, hidden]` batch narrowed to *this
    request's* tokens for *this* forward pass — `h` has shape
    `(n_query, hidden)`. It does not say where in the sequence those tokens
    sit, and `ctx.seq_len` is `n_query`, not the running length
    (`vllm_lens/_worker_ext.py`, `ctx.seq_len = end - start`), so the absolute
    offset has to be carried.

    It is carried by counting: `ctx.saved` is created once per (hook, request)
    and persists across layers *and* across forward passes until the request
    finishes, so the number of tokens this layer has already seen is the
    absolute position of `h[0]`. Prefill contributes `n_prompt` (or its chunk
    widths, which sum to the same thing — this is why a chunked prefill is
    correct here), and each decode step contributes 1.

    Counted **per layer**, because one `HookContext` is shared by every layer
    in `layer_indices` and `ctx.layer_idx` is rewritten before each call.

    ### The one way this can be wrong

    Under KV-cache pressure vLLM preempts a request and *recomputes* it from
    the start, which resets the true position but not this counter. That would
    silently shift the span. It is not silent here: the final count is returned
    in `ctx.saved["seen"]`, and `verify_seen` below checks it against the token
    count the request actually reports, so a recompute surfaces as a loud
    mismatch rather than a plausible result.

    ### Two things the callable must do

    Return a **new** tensor. `_apply_hook_delta` writes `result - h` as the
    delta, so mutating `h` in place and returning it applies exactly nothing.

    Stay deterministic across tensor-parallel ranks — the hook runs on every TP
    rank (only activation *capture* is rank-0-gated), and ranks that disagreed
    would diverge.

    `record` additionally logs, for the first of `rows` only, one
    `(base, n_query, [absolute ranges hit])` entry per forward pass into
    `ctx.saved["hits"]`, which comes back on `output.hook_results`. It is what
    `verify_hits` reads to prove the span landed where it was asked for, which
    every Inspect request does, and it is off by default because a 750-token
    reply would otherwise ship 750 tuples back per request.
    """
    # Closed over, not stored in ctx.saved: this cache holds device tensors and
    # ctx.saved is pickled back to the client when the request finishes.
    device_cache: dict[Any, torch.Tensor] = {}
    scale = float(strength)

    def fn(ctx, h):
        layer = ctx.layer_idx

        # Counted before anything can return early. The count is the absolute
        # position of the next token this layer will see, so a pass that skips
        # it desynchronises every position after it — and `verify_seen` would
        # then read the shortfall as a preemption that never happened.
        seen: dict = ctx.saved.setdefault("seen", {})
        base = seen.get(layer, 0)
        n = h.shape[0]
        seen[layer] = base + n

        row = adds.get(layer)
        if row is None:
            return None

        # Which rows of `h` fall inside any requested range. Ranges are in
        # absolute positions; `h` covers [base, base + n).
        hits: list[tuple[int, int]] = []
        for lo, hi in ranges:
            a = max(lo, base)
            b = base + n if hi is None else min(hi, base + n)
            if b > a:
                hits.append((a - base, b - base))

        if record and layer == rows[0]:
            ctx.saved.setdefault("hits", []).append(
                (base, n, [(a + base, b + base) for a, b in hits])
            )

        if not hits:
            return None

        key = (layer, h.device, h.dtype)
        add = device_cache.get(key)
        if add is None:
            add = row.to(device=h.device, dtype=h.dtype) * scale
            device_cache[key] = add

        # Whole-slice fast path. `_apply_hook_delta` costs two kernels on the
        # way back out (a subtract for the delta, an add to splice it in), so
        # the ones spent in here are the only ones worth saving — and this is
        # the hot case: under `generated_assistant` every decode pass is one
        # token, entirely inside the range, so `h + add` replaces a clone plus
        # an in-place add with a single kernel. It is a *new* tensor either way,
        # which the delta arithmetic requires.
        if len(hits) == 1 and hits[0] == (0, n):
            return h + add

        out = h.clone()
        for a, b in hits:
            out[a:b] += add
        return out

    return Hook(fn=fn, layer_indices=list(rows), pre=False)


def _build_disable_prehooks():
    """Return a callable that strips vllm-lens's forward **pre**-hooks.

    Built by a factory rather than written at module level, and that is
    load-bearing. `collective_rpc` ships a callable to the workers with
    cloudpickle, and cloudpickle serializes a module-level function of an
    *importable* module **by reference** — the worker would then try to
    `import utils.vllm_lens_wrapper`, which is not on its `sys.path`, and the
    RPC fails with `ModuleNotFoundError`. A function defined inside another
    function is serialized by value, code and all, so it needs nothing from
    this project at the far end. It closes over nothing for the same reason.

    ### What it is for

    `install_hooks` registers a pre-hook *and* a post-hook on every decoder
    layer whether or not anything asked for a pre-hook, and the pre-hook reads
    `args[1]` before `_pre_hook_inner` gets to check that nobody did
    (`vllm_lens/_worker_ext.py:607`). On models whose decoder layer vLLM does
    not call with two positional arguments that raises `IndexError`, caught and
    logged at WARNING with `exc_info=True`.

    That is **once per layer per forward pass**, not per request: a forward
    pass advances every in-flight request by one token, so the cost is shared
    across the batch and shrinks as concurrency rises. It is not a meaningful
    share of compute. What it *is* is roughly 15 KB of formatted traceback per
    forward pass, which over a full `trust_elo` sweep (~80k forward passes at
    2673 requests, ~750-token replies, 25 in flight) is on the order of a
    gigabyte of SLURM log.

    Nothing here uses pre-hooks (`make_span_hook` builds `pre=False`), and only
    `_forward_pre_hooks` is touched, so the post-hook that does the injection is
    untouched. The logger level is lowered as well, since that is the half that
    works even if the layer walk finds nothing: `Logger.warning` short-circuits
    on `isEnabledFor` before formatting anything.
    """

    def disable_prehooks(worker) -> int:
        import logging

        logging.getLogger("vllm_lens._worker_ext").setLevel(logging.ERROR)

        removed = 0
        try:
            from vllm_lens._worker_ext import _get_layers

            layers = _get_layers(worker.model_runner.model)
        except Exception:
            return -1  # logger silenced, layers not reachable
        for layer in layers:
            hooks = getattr(layer, "_forward_pre_hooks", None)
            if not hooks:
                continue
            for handle_id, hook in list(hooks.items()):
                if getattr(hook, "__module__", "") == "vllm_lens._worker_ext":
                    del hooks[handle_id]
                    removed += 1
        return removed

    return disable_prehooks


def verify_seen(
    saved: dict, n_prompt: int, n_generated: int, rows: list[int], what: str
) -> Optional[str]:
    """Check the hook's own position count against what the request produced.

    Returns None when consistent, else a description of the discrepancy.

    The expected count is `n_prompt + n_generated - 1`: prefill covers the
    whole prompt and samples the first token, then each of the remaining
    `n_generated - 1` tokens costs one decode pass. vLLM's v1 scheduler can
    commit one extra forward pass after a stop condition fires and discard its
    token (`vllm_lens/_activations_plugin.py::_trim_activations` documents the
    same surplus for activation capture), so one over is expected too.

    Anything else means the hook saw a different number of tokens than the
    request contains, and the only mechanism that does that is a preemption
    and recompute — under which every absolute position after the recompute is
    shifted and the span landed somewhere else.
    """
    seen = (saved or {}).get("seen") or {}
    if not seen:
        return f"{what}: hook recorded no positions at all"
    lo = n_prompt + n_generated - 1
    ok = {lo, lo + 1}
    bad = {layer: got for layer, got in seen.items() if got not in ok}
    if bad:
        return (
            f"{what}: hook saw {sorted(bad.items())[:4]} tokens but the request "
            f"has {n_prompt} prompt + {n_generated} generated (expected "
            f"{lo} or {lo + 1}). Most likely the request was preempted and "
            f"recomputed, which shifts every absolute position after it"
        )
    missing = sorted(set(rows) - set(seen))
    if missing:
        return f"{what}: layers {missing[:8]} never fired"
    return None


def verify_hits(
    saved: dict,
    ranges: list[tuple[int, Optional[int]]],
    audit_layer: int,
    what: str,
) -> Optional[str]:
    """Check that the hook modified exactly the requested positions."""
    seen = (saved or {}).get("seen") or {}
    total = seen.get(audit_layer)
    if total is None:
        return f"{what}: audit layer {audit_layer} recorded no positions"

    expected = []
    for lo, hi in ranges:
        end = total if hi is None else min(hi, total)
        if end > lo:
            expected.append((lo, end))

    actual = []
    for _base, _width, hits in (saved or {}).get("hits") or []:
        actual.extend(tuple(hit) for hit in hits)

    def merge(intervals):
        merged: list[tuple[int, int]] = []
        for lo, hi in sorted(intervals):
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        return merged

    expected = merge(expected)
    actual = merge(actual)
    if actual != expected:
        return (
            f"{what}: hook modified positions {actual}, expected {expected}; "
            "refusing to accept a possibly mis-steered generation"
        )
    return None


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class VLLMLensEngine:
    """One `AsyncLLM` on its own thread and event loop.

    vLLM's v1 async engine wants a live event loop, and the callers here are
    ordinary synchronous code (`litellm.completion`, a tqdm loop). Building the
    engine on the loop's own thread keeps every await on one loop.
    """

    def __init__(
        self,
        model_id: str,
        engine_kwargs: dict,
        reasoning_parser: Optional[str],
        tool_parser: Optional[str],
    ) -> None:
        self.model_id = model_id
        self.reasoning_parser_name = reasoning_parser
        self.tool_parser_name = tool_parser
        self._engine_kwargs = engine_kwargs

        self.engine = None
        self._tokenizer = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()
        self._build_error: Optional[BaseException] = None

        self._reasoning_parser = None
        self._tool_parser = None
        self._parsers_ready = False
        self._parsers_lock = threading.Lock()

        self._thread = threading.Thread(
            target=self._main, name=f"vllm-lens[{model_id}]", daemon=True
        )
        self._thread.start()

    def _main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            from transformers import AutoTokenizer
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM

            # Stats on. vLLM then logs "Running: N reqs, Waiting: M reqs, GPU
            # KV cache usage: X%" periodically, which is the one line that says
            # whether a slow sweep is starved (low Running, high Waiting — the
            # bottleneck is upstream of the GPU) or saturated (high Running —
            # it is the per-request-per-layer hook work). Without it a slow run
            # is indistinguishable between the two and the next fix is a guess.
            self._engine_kwargs.setdefault("disable_log_stats", False)
            args = AsyncEngineArgs(model=self.model_id, **self._engine_kwargs)
            self.engine = AsyncLLM.from_engine_args(args)
            # The engine's own tokenizer accessor moved around between vLLM
            # versions; the span is computed against ids we then hand back as
            # a TokensPrompt, so any tokenizer for this checkpoint gives the
            # same ids. Ask the engine first all the same, so a model whose
            # local files differ from the hub cannot drift.
            self._tokenizer = getattr(self.engine, "tokenizer", None)
            if self._tokenizer is None or not hasattr(
                self._tokenizer, "apply_chat_template"
            ):
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        except BaseException as exc:  # noqa: BLE001
            self._build_error = exc
            self._ready.set()
            return
        self._ready.set()
        loop.run_forever()

    def ensure_ready(self) -> None:
        self._ready.wait()
        if self._build_error is not None:
            raise CustomLLMError(
                status_code=500,
                message=(
                    f"vllm-lens engine failed to load {self.model_id!r}: "
                    f"{self._build_error!r}"
                ),
            )

    def submit(self, coro) -> "asyncio.Future":
        self.ensure_ready()
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def shutdown(self) -> None:
        """Cancel what is parked on the loop, then stop it.

        vLLM's `AsyncLLM` keeps an output-handler task awaiting the engine core
        for the life of the engine. Stopping the loop without cancelling it
        leaves it registered and vLLM's own atexit tears the core down
        underneath it, which surfaces as a traceback *after* the last result is
        written — cosmetic, but a stack trace at the end of every successful
        run is how a real failure gets missed.
        """
        loop = self._loop
        if loop is not None and loop.is_running():

            def _cancel_pending() -> None:
                current = asyncio.current_task(loop)
                for task in asyncio.all_tasks(loop):
                    if task is not current:
                        task.cancel()

            try:
                loop.call_soon_threadsafe(_cancel_pending)
                time.sleep(0.2)  # call_soon_threadsafe only queues them
            except RuntimeError:
                pass
        with_shutdown = getattr(self.engine, "shutdown", None)
        if with_shutdown is not None:
            try:
                with_shutdown()
            except Exception:  # noqa: BLE001
                pass
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)

    @property
    def tokenizer(self):
        self.ensure_ready()
        return self._tokenizer

    def ensure_parsers(self) -> None:
        if self._parsers_ready:
            return
        with self._parsers_lock:
            if self._parsers_ready:
                return
            tok = self.tokenizer
            if self.reasoning_parser_name:
                self._reasoning_parser = _get_reasoning_parser_cls(
                    self.reasoning_parser_name
                )(tok)
            if self.tool_parser_name:
                self._tool_parser = _get_tool_parser_cls(self.tool_parser_name)(tok)
            self._parsers_ready = True

    def parse_output(
        self, text: str, request, thinking_disabled: bool = False
    ) -> tuple[Optional[str], Optional[str], list, bool]:
        reasoning_content: Optional[str] = None
        content: Optional[str] = text

        if self._reasoning_parser is not None and not thinking_disabled:
            try:
                reasoning_content, content = extract_reasoning(
                    self._reasoning_parser, text, request
                )
            except Exception:  # noqa: BLE001
                reasoning_content, content = None, text

        tool_calls: list = []
        tools_called = False
        if self._tool_parser is not None and request.tools and content:
            try:
                info = self._tool_parser.extract_tool_calls(content, request)
                tools_called = bool(getattr(info, "tools_called", False))
                content = getattr(info, "content", content)
                for tc in getattr(info, "tool_calls", []) or []:
                    tool_calls.append(
                        ChatCompletionMessageToolCall(
                            id=getattr(tc, "id", None)
                            or f"call_{uuid.uuid4().hex[:24]}",
                            type="function",
                            function=Function(
                                name=tc.function.name, arguments=tc.function.arguments
                            ),
                        )
                    )
            except Exception:  # noqa: BLE001
                pass
        return content, reasoning_content, tool_calls, tools_called

    async def run_one(self, prompt, params: SamplingParams) -> Any:
        """Submit one request and return its finished `RequestOutput`."""
        request_id = str(uuid.uuid4())
        final = None
        async for output in self.engine.generate(prompt, params, request_id):
            if output.finished:
                final = output
        if final is None:
            raise CustomLLMError(status_code=500, message="no finished output")
        return final


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #


class VLLMLens(CustomLLM):
    """litellm provider over vLLM + vllm-lens."""

    def __init__(
        self,
        *,
        provider: str = "vllm-lens",
        default_model: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        distributed_executor_backend: Optional[str] = None,
        max_model_len: Optional[int] = None,
        default_max_tokens: int = 256,
        request_timeout: Optional[float] = None,
        reasoning_parser: Optional[str] = None,
        tool_parser: Optional[str] = None,
        max_loaded_models: Optional[int] = None,
        model_overrides: Optional[dict[str, dict]] = None,
        **engine_kwargs: Any,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.default_model = default_model
        self.default_max_tokens = default_max_tokens
        self.request_timeout = (
            DEFAULT_REQUEST_TIMEOUT if request_timeout is None else request_timeout
        )
        self.reasoning_parser_name = reasoning_parser
        self.tool_parser_name = tool_parser
        self.max_loaded_models = max_loaded_models
        self.model_overrides = model_overrides or {}

        self._base_engine_kwargs: dict[str, Any] = dict(
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **engine_kwargs,
        )
        if distributed_executor_backend:
            self._base_engine_kwargs["distributed_executor_backend"] = (
                distributed_executor_backend
            )

        # One number for the context window. It does *not* also have to bound
        # how many prompts may prefill together: the hook derives its absolute
        # position by counting tokens, so a prefill
        # split across scheduler steps sums to the same offsets and needs no
        # per-step budget large enough to hold a whole in-flight set. The
        # per-step budget is therefore left to vLLM.
        self.max_model_len = int(max_model_len or DEFAULT_MAX_MODEL_LEN)
        self._base_engine_kwargs["max_model_len"] = self.max_model_len

        self._engines: dict[str, VLLMLensEngine] = {}
        self._registry_lock = threading.Lock()

        if self.default_model:
            self._get_engine(self.default_model)  # eager preload

        self._register()

    def _register(self) -> None:
        existing = litellm.custom_provider_map or []
        if any(entry.get("provider") == self.provider for entry in existing):
            return
        litellm.custom_provider_map = [
            *existing,
            {"provider": self.provider, "custom_handler": self},
        ]

    # ----- engine registry --------------------------------------------------
    def _resolve_model(self, kwargs: dict) -> str:
        model_id = kwargs.get("model") or self.default_model
        if not model_id:
            raise CustomLLMError(
                status_code=400, message="No model specified in request"
            )
        if model_id.startswith(self.provider + "/"):
            model_id = model_id[len(self.provider) + 1 :]
        return model_id

    def _get_engine(self, model_id: str) -> VLLMLensEngine:
        eng = self._engines.get(model_id)
        if eng is not None:
            return eng
        with self._registry_lock:
            eng = self._engines.get(model_id)
            if eng is not None:
                return eng
            if self.max_loaded_models and len(self._engines) >= self.max_loaded_models:
                raise CustomLLMError(
                    status_code=400,
                    message=(
                        f"max_loaded_models={self.max_loaded_models} reached; "
                        f"won't load {model_id!r}. Loaded: {list(self._engines)}"
                    ),
                )
            ov = self.model_overrides.get(model_id, {})
            rp, tp = self._resolve_parsers(model_id, ov)
            ekw = {**self._base_engine_kwargs, **ov.get("engine_kwargs", {})}
            eng = VLLMLensEngine(model_id, ekw, rp, tp)
            self._engines[model_id] = eng
            return eng

    def _resolve_parsers(
        self, model_id: str, ov: dict
    ) -> tuple[Optional[str], Optional[str]]:
        """Per-model override, then name inference, then the instance default."""
        inferred = matching_parsers(model_id)
        if inferred is None:
            inferred = (self.reasoning_parser_name, self.tool_parser_name)
        inferred_rp, inferred_tp = inferred
        rp = ov.get(
            "reasoning_parser",
            inferred_rp,
        )
        tp = ov.get(
            "tool_parser",
            inferred_tp,
        )
        return rp, tp

    def get_engine(self, model_id: Optional[str] = None) -> VLLMLensEngine:
        eng = self._get_engine(model_id or self.default_model)
        eng.ensure_ready()
        return eng

    def get_tokenizer(self, model_id: Optional[str] = None):
        """The tokenizer a span should be computed against."""
        return self.get_engine(model_id).tokenizer

    def prepare_hooks(self, model_id: Optional[str] = None, quiet: bool = True) -> None:
        """Install vllm-lens's layer hooks now, and disarm its broken pre-hook.

        Idempotent, and safe to call before any generation. Two steps:

        1. `install_hooks`, which vllm-lens otherwise defers to the first
           request that carries a hook. Doing it here means step 2 has
           something to find.
        2. `_worker_disable_prehooks`, which removes the pre-hook that raises
           once per layer per forward pass on this model family. See that
           function for what it costs to leave in place.

        Failures here are reported and swallowed. Neither step changes what a
        run *produces* — the injection is a post-hook and is untouched either
        way — so a vLLM whose `collective_rpc` will not take a callable should
        cost you speed and a noisy log, not a run.
        """
        eng = self.get_engine(model_id)
        if getattr(eng, "_hooks_prepared", False):
            return
        eng._hooks_prepared = True
        eng.prehook_status = "not attempted"
        try:
            eng.submit(eng.engine.collective_rpc("install_hooks")).result()
        except Exception as exc:  # noqa: BLE001
            eng.prehook_status = f"install_hooks RPC failed: {exc!r}"
            print(f"warning:  {eng.prehook_status}")
            return
        if not quiet:
            eng.prehook_status = "left armed on request (quiet=False)"
            return
        try:
            got = eng.submit(
                eng.engine.collective_rpc(_build_disable_prehooks())
            ).result()
        except Exception as exc:  # noqa: BLE001
            eng.prehook_status = f"NOT disarmed — RPC failed: {exc!r}"
            print(f"warning:  {eng.prehook_status}\n{_DISARM_HINT}")
            return
        counts = [n for n in (got or []) if isinstance(n, int)]
        total = sum(n for n in counts if n > 0)
        if not counts or all(n < 0 for n in counts):
            eng.prehook_status = (
                "logger silenced, but no pre-hooks found to "
                "remove — check vllm-lens's layer discovery"
            )
            print(f"warning:  vllm-lens pre-hook: {eng.prehook_status}\n{_DISARM_HINT}")
            return
        if total == 0:
            eng.prehook_status = "logger silenced; 0 pre-hooks matched"
        else:
            eng.prehook_status = f"disarmed {total} pre-hook(s) across workers"
        print(f"note:     vllm-lens pre-hook: {eng.prehook_status}")

    def shutdown_engine(self, model_id: str) -> None:
        eng = self._engines.pop(model_id, None)
        if eng is not None:
            eng.shutdown()

    def shutdown(self) -> None:
        for model_id in list(self._engines):
            self.shutdown_engine(model_id)

    # ----- request shaping --------------------------------------------------
    @staticmethod
    def _thinking_disabled(kwargs: dict) -> bool:
        optional_params = kwargs.get("optional_params") or {}
        reasoning_effort = (optional_params.get("extra_body") or {}).get(
            "reasoning_effort"
        )
        return reasoning_effort == "none"

    def _to_prompt(self, eng: VLLMLensEngine, kwargs: dict) -> str:
        if kwargs.get("messages") is None and kwargs.get("prompt") is not None:
            return kwargs["prompt"]
        messages = kwargs["messages"]
        tok = eng.tokenizer
        if getattr(tok, "chat_template", None):
            optional_params = kwargs.get("optional_params") or {}
            tools = kwargs.get("tools") or optional_params.get("tools")
            chat_template_kwargs = {}
            if self._thinking_disabled(kwargs):
                chat_template_kwargs["enable_thinking"] = False
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools,
                **chat_template_kwargs,
            )
        rendered = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return rendered + "\nassistant:"

    def _sampling_params(
        self,
        optional_params: dict,
        eng: VLLMLensEngine,
        *,
        extra_args: Optional[dict] = None,
    ) -> SamplingParams:
        passthrough = (
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "n",
            "skip_special_tokens",
        )
        params = {k: optional_params[k] for k in passthrough if k in optional_params}
        params.setdefault("max_tokens", self.default_max_tokens)
        params.setdefault("repetition_detection", REPETITION_DETECTION)
        if _FINAL_ONLY is not None:
            params.setdefault("output_kind", _FINAL_ONLY)
        if extra_args is not None:
            params["extra_args"] = extra_args
        # vLLM's own serving layer forces this off via tool_parser.adjust_request()
        # whenever a tool/reasoning parser is active: tags like <tool_call> and
        # <think> are registered as special tokens for many models, so the default
        # skip_special_tokens=True silently strips them before our parsers ever
        # see the text, causing intermittent parse failures.
        if eng.tool_parser_name or eng.reasoning_parser_name:
            params.setdefault("skip_special_tokens", False)
        return SamplingParams(**params)

    @staticmethod
    def _usage(output: Any) -> Usage:
        prompt_tokens = len(getattr(output, "prompt_token_ids", []) or [])
        completion_tokens = len(getattr(output.outputs[0], "token_ids", []) or [])
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _build_response(
        self,
        eng: VLLMLensEngine,
        output: Any,
        model_id: str,
        kwargs: dict,
        *,
        thinking_disabled: Optional[bool] = None,
    ) -> ModelResponse:
        eng.ensure_parsers()
        text = output.outputs[0].text
        request = make_request_obj(
            model_id,
            kwargs.get("messages") or [],
            kwargs.get("tools") or (kwargs.get("optional_params") or {}).get("tools"),
        )
        content, reasoning, tool_calls, tools_called = eng.parse_output(
            text,
            request,
            thinking_disabled=(
                self._thinking_disabled(kwargs)
                if thinking_disabled is None
                else thinking_disabled
            ),
        )

        msg_kwargs: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg_kwargs["tool_calls"] = tool_calls
        if reasoning is not None:
            msg_kwargs["reasoning_content"] = reasoning
        try:
            message = Message(**msg_kwargs)
        except Exception:  # older litellm Message
            message = Message(
                role="assistant", content=content, tool_calls=tool_calls or None
            )
            if reasoning is not None:
                try:
                    message.reasoning_content = reasoning
                except Exception:  # noqa: BLE001
                    pass

        finish = output.outputs[0].finish_reason or "stop"
        if tools_called:
            finish = "tool_calls"

        return ModelResponse(
            id=f"vllm-lens-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=model_id,
            object="chat.completion",
            choices=[Choices(index=0, message=message, finish_reason=finish)],
            usage=self._usage(output),
        )

    # ----- span-precise steered generation ----------------------------------
    def generate_steered(
        self,
        model_id: str,
        jobs: list[tuple[PromptLayout, float]],
        vector: torch.Tensor,
        rows: list[int],
        steering_target: str = DEFAULT_STEERING_TARGET,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        max_in_flight: Optional[int] = None,
        on_done: Optional[Callable[[], None]] = None,
        strict: bool = True,
    ) -> list:
        """Generate one continuation per ``(PromptLayout, strength)``.

        Results come back in job order however the engine chose to interleave
        them.

        **`max_in_flight` defaults to no cap**, deliberately. Counting
        positions makes a prefill split across scheduler steps add up to the
        same offsets, so no in-flight set has to fit `max_num_batched_tokens`
        for the span to land where it should, and the only thing left to decide
        is throughput, which vLLM's own scheduler decides better than a
        semaphore here can: it admits what its KV cache and `max_num_seqs`
        allow and queues the rest. Pass an int to cap it anyway.

        Two consequences of submitting everything at once, both real but neither
        a reason to cap by default on this hardware:

        * Each steered request carries its own cloudpickled hook, and the vector
          rides inside it — about 512 KB for a 32-layer float32 vector, held
          worker-side until that request finishes. A whole 2673-request sweep
          submitted at once is therefore ~1.3 GB of host RAM in the worker
          (against the 461 GB the node reports). It is not GPU memory: the
          hook's device cache only materialises once the request is actually
          scheduled.
        * More admitted requests means more chance vLLM hits KV pressure and
          **preempts**, and a preempted request is recomputed from the start —
          which is the one thing that breaks the position counter. It does not
          break it silently: `verify_seen` audits every request against its own
          token count. If a sweep starts dying there, cap this rather than
          trusting the numbers.

        `strict` turns that audit into an error. Leave it on.
        """
        steering_target = validate_steering_target(steering_target)
        eng = self.get_engine(model_id)
        self.prepare_hooks(model_id)

        # CPU float32, deliberately: the hook is cloudpickled to the workers.
        adds = {layer: vector[layer].detach().cpu().float().clone() for layer in rows}

        async def run_all() -> list:
            # nullcontext when uncapped, so the submit path below has one shape
            # rather than a branch that could drift between the two. Both are
            # reusable across concurrent tasks; `nullcontext` grew `__aenter__`
            # in 3.10 and this project runs 3.12.
            gate = (
                contextlib.nullcontext()
                if max_in_flight is None
                else asyncio.Semaphore(max_in_flight)
            )
            problems: list[str] = []

            async def one(j: int, layout: PromptLayout, strength: float):
                n_prompt = len(layout.token_ids)
                ranges = resolve_steering_ranges(
                    steering_target,
                    n_prompt,
                    layout.user_spans,
                    layout.tool_spans,
                )
                steered = strength != 0.0 or WRITE_AT_ZERO
                kw: dict[str, Any] = dict(
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    # Set once for the whole job list, so the cutoff is a
                    # property of the run and not a difference between
                    # conditions.
                    repetition_detection=REPETITION_DETECTION,
                )
                if _FINAL_ONLY is not None:
                    kw["output_kind"] = _FINAL_ONLY
                if seed is not None:
                    # Per request, so two draws of one prompt do not replay
                    # each other.
                    kw["seed"] = int(seed) + j
                if steered:
                    kw["extra_args"] = {
                        "apply_hooks": [
                            make_span_hook(
                                rows,
                                adds,
                                strength,
                                ranges,
                            )
                        ]
                    }
                # Built in one construction rather than assigned onto: vLLM
                # validates in __post_init__, so a field set afterwards skips
                # the checks it would have been given.
                params = SamplingParams(**kw)

                async with gate:
                    out = await eng.run_one(
                        TokensPrompt(prompt_token_ids=list(layout.token_ids)), params
                    )

                if steered:
                    saved = (getattr(out, "hook_results", None) or {}).get("0") or {}
                    problem = verify_seen(
                        saved,
                        n_prompt,
                        len(out.outputs[0].token_ids or []),
                        rows,
                        f"request {j}",
                    )
                    if problem:
                        problems.append(problem)
                if on_done is not None:
                    on_done()
                return j, out.outputs[0].text

            done = await asyncio.gather(
                *(
                    one(j, layout, strength)
                    for j, (layout, strength) in enumerate(jobs)
                )
            )
            out = [""] * len(jobs)
            for j, text in done:
                out[j] = text
            if problems:
                head = "\n  ".join(problems[:5])
                msg = (
                    f"{len(problems)} of {len(jobs)} requests had an injection "
                    f"span that cannot be trusted:\n  {head}"
                )
                if strict:
                    raise RuntimeError(msg)
                print(f"warning: {msg}")
            return out

        return eng.submit(run_all()).result()

    def steered_completion(
        self,
        model_id: str,
        messages: list[dict],
        tools: Optional[list[dict]],
        vector,
        rows: list[int],
        strength: float,
        *,
        steering_target: str = DEFAULT_STEERING_TARGET,
        think: bool = False,
        optional_params: Optional[dict] = None,
    ) -> ModelResponse:
        """Run one structured chat completion through the audited span hook."""
        steering_target = validate_steering_target(steering_target)

        eng = self.get_engine(model_id)
        self.prepare_hooks(model_id)
        layout = build_steered_prompt(
            eng.tokenizer,
            messages,
            think,
            system_prompt=None,
            tools=tools,
        )
        ranges = resolve_steering_ranges(
            steering_target,
            len(layout.token_ids),
            layout.user_spans,
            layout.tool_spans,
        )
        adds = {layer: vector[layer].detach().cpu().float().clone() for layer in rows}
        steered = strength != 0.0 or WRITE_AT_ZERO
        extra_args = None
        if steered:
            extra_args = {
                "apply_hooks": [
                    make_span_hook(
                        rows,
                        adds,
                        strength,
                        ranges,
                        record=True,
                    )
                ]
            }
        params = dict(optional_params or {})
        sampling = self._sampling_params(params, eng, extra_args=extra_args)
        check_context_budget(
            len(layout.token_ids), sampling.max_tokens, self.max_model_len
        )
        pending = eng.submit(
            eng.run_one(
                TokensPrompt(prompt_token_ids=list(layout.token_ids)), sampling
            )
        )
        try:
            final = pending.result(timeout=self.request_timeout)
        except FuturesTimeoutError:
            # Cancelling the future cancels the task on the engine loop, which
            # is what aborts the vLLM request. Skipping it would leave the
            # request generating against a result nobody will ever read.
            pending.cancel()
            raise TimeoutError(
                f"steered completion did not return within "
                f"{self.request_timeout}s"
            ) from None

        if steered:
            saved = (getattr(final, "hook_results", None) or {}).get("0") or {}
            problem = verify_seen(
                saved,
                len(layout.token_ids),
                len(final.outputs[0].token_ids or []),
                rows,
                "Inspect request",
            )
            if problem:
                raise RuntimeError(problem)
            problem = verify_hits(saved, ranges, rows[0], "Inspect request")
            if problem:
                raise RuntimeError(problem)

        response = self._build_response(
            eng,
            final,
            model_id,
            {
                "messages": messages,
                "tools": tools,
                "optional_params": params,
            },
            thinking_disabled=not think,
        )
        response._hidden_params["steering_backend"] = "vllm_lens"
        response._hidden_params["steering_target"] = steering_target
        response._hidden_params["steering_range_count"] = len(ranges)
        response._hidden_params["steering_ranges_verified"] = steered
        return response

    # ----- litellm entry points ---------------------------------------------
    def completion(self, *args, **kwargs) -> ModelResponse:
        model_id = self._resolve_model(kwargs)
        eng = self._get_engine(model_id)
        eng.ensure_ready()
        prompt = self._to_prompt(eng, kwargs)
        params = self._sampling_params(kwargs.get("optional_params", {}), eng)
        final = eng.submit(eng.run_one(prompt, params)).result()
        return self._build_response(eng, final, model_id, kwargs)

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        model_id = self._resolve_model(kwargs)
        eng = self._get_engine(model_id)
        eng.ensure_ready()
        prompt = self._to_prompt(eng, kwargs)
        params = self._sampling_params(kwargs.get("optional_params", {}), eng)
        fut = eng.submit(eng.run_one(prompt, params))
        final = await asyncio.wrap_future(fut)
        return self._build_response(eng, final, model_id, kwargs)
