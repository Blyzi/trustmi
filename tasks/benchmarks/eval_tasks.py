"""Adapters around the upstream Inspect tasks the suite runs.

They pass the suite's judge models in and adjust the benchmarks where upstream
crashes on malformed model output or exposes the wrong tools.
`reports/README.md` lists every adjustment and whether it can change a score.
"""

from __future__ import annotations

import fcntl
import importlib
import inspect
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any, Awaitable, Callable

from inspect_ai import Task
from inspect_ai.model import ChatMessage, ChatMessageUser, Model
from inspect_ai.scorer import Score, Target
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolDef, ToolError


GradingFunction = Callable[
    [dict[str, Any], list[ChatMessage]], Awaitable[dict[str, Any]]
]
ScoringFunction = Callable[[TaskState, Target], Awaitable[Score]]
ToolArgumentsFunction = Callable[[list[ChatMessage], str], Any]


@contextmanager
def _agentharm_dataset_lock() -> Iterator[None]:
    """Serialize AgentHarm's non-atomic JSON-to-JSONL cache conversion."""
    lock_path = Path(os.environ.get("TMPDIR", "/tmp")) / "trustmi-agentharm.lock"
    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _score_bfcl_multi_turn(state: TaskState, target: Target) -> Score:
    """Score an incomplete BFCL trajectory as incorrect instead of crashing."""
    ground_truth = state.metadata.get("raw_ground_truth", [])
    model_results = state.metadata.get("model_execution_results", [])
    if len(model_results) != len(ground_truth):
        model_calls = state.metadata.get("model_execution_calls", [])
        return Score(
            value=0,
            answer=str(model_calls),
            explanation=(
                "Incomplete trajectory: recorded "
                f"{len(model_results)} of {len(ground_truth)} turns"
            ),
        )

    from inspect_evals.bfcl.score.multi_turn_scorer import (
        multi_turn_match as upstream_multi_turn_match,
    )

    return upstream_multi_turn_match(state, target)


def _make_bfcl_recording_tool(
    method_name: str,
    method: Any,
    turn_calls: list[dict[str, Any]],
    turn_results: list[Any],
    func_doc: dict[str, Any] | None = None,
) -> ToolDef:
    """Wrap a BFCL backend method without mutating Inspect's tool registry."""

    @wraps(method)
    async def execute(**kwargs: Any) -> Any:
        try:
            result = method(**kwargs)
        except Exception as exc:
            result = {"error": f"Error executing {method_name}: {exc}"}
        turn_calls.append({"function": method_name, "arguments": kwargs})
        turn_results.append(result)
        return result

    setattr(execute, "__signature__", inspect.signature(method))
    if func_doc is None:
        return ToolDef(execute, name=method_name)

    parameter_descriptions = {
        name: schema.get("description", name)
        for name, schema in func_doc.get("parameters", {}).get("properties", {}).items()
    }
    return ToolDef(
        execute,
        name=method_name,
        description=func_doc["description"],
        parameters=parameter_descriptions or None,
    )


async def _bfcl_multi_turn_solve(state: TaskState, generate: Generate) -> TaskState:
    """Run BFCL multi-turn samples with isolated, explicitly defined tools."""
    from inspect_evals.bfcl.backends import build_tool_mapping, create_instances
    from inspect_evals.bfcl.prompts import (
        DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
    )

    instances = create_instances(
        involved_classes=state.metadata["involved_classes"],
        initial_config=state.metadata["initial_config"],
        long_context=state.metadata.get("long_context", False),
    )
    available_methods = build_tool_mapping(instances)

    missed_function_names: dict[str, list[str]] = (
        state.metadata.get("missed_function") or {}
    )
    missed_function_docs: dict[str, list[Any]] = (
        state.metadata.get("missed_function_docs") or {}
    )
    all_withheld = {name for names in missed_function_names.values() for name in names}
    excluded = set(state.metadata.get("excluded_function") or [])
    func_doc_by_name: dict[str, Any] = {
        doc["name"]: doc for doc in (state.metadata.get("tools") or [])
    }

    # Only expose functions declared by this sample. Some BFCL backend classes
    # contain public helper methods that are not part of the model-facing API.
    tool_mapping = {
        name: method
        for name, method in available_methods.items()
        if name in func_doc_by_name and name not in all_withheld | excluded
    }

    model_execution_results: list[list[Any]] = []
    model_execution_calls: list[list[dict[str, Any]]] = []
    turns: list[list[dict[str, str]]] = state.metadata["turns"]

    for turn_idx, turn_messages in enumerate(turns):
        newly_revealed = missed_function_docs.get(str(turn_idx), [])
        for func_doc in newly_revealed:
            func_name = func_doc["name"]
            func_doc_by_name[func_name] = func_doc
            if func_name in excluded:
                continue
            for instance in instances.values():
                if hasattr(instance, func_name):
                    tool_mapping[func_name] = getattr(instance, func_name)

        revealed_by_name = missed_function_names.get(str(turn_idx), [])
        revealed_doc_names = {doc["name"] for doc in newly_revealed}
        for func_name in revealed_by_name:
            if func_name in revealed_doc_names or func_name in excluded:
                continue
            for instance in instances.values():
                if hasattr(instance, func_name):
                    tool_mapping[func_name] = getattr(instance, func_name)

        turn_results: list[Any] = []
        turn_calls: list[dict[str, Any]] = []
        tools = [
            _make_bfcl_recording_tool(
                name, method, turn_calls, turn_results, func_doc_by_name.get(name)
            )
            for name, method in tool_mapping.items()
        ]

        # TaskState's property setter converts ToolDefs back to globally tagged
        # callables. Mutating the per-state list retains the explicit ToolDefs
        # and prevents schemas from leaking across concurrent samples.
        state.tools.clear()
        state.tools.extend(tools)  # type: ignore[arg-type]

        if newly_revealed or revealed_by_name:
            state.messages.append(
                ChatMessageUser(content=DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC)
            )
        for message in turn_messages:
            if message.get("role", "user") == "user":
                state.messages.append(
                    ChatMessageUser(content=message.get("content", ""))
                )

        state = await generate(state, tool_calls="loop")
        model_execution_results.append(turn_results)
        model_execution_calls.append(turn_calls)

    state.metadata["model_execution_results"] = model_execution_results
    state.metadata["model_execution_calls"] = model_execution_calls
    return state


@contextmanager
def _missing_tool_arguments_are_empty(
    messages: Iterable[ChatMessage],
) -> Iterator[None]:
    original_arguments = []
    for message in messages:
        for tool_call in getattr(message, "tool_calls", None) or []:
            arguments = tool_call.arguments
            if isinstance(arguments, dict):
                original_arguments.append((tool_call, arguments))
                tool_call.arguments = defaultdict(str, arguments)
    try:
        yield
    finally:
        for tool_call, arguments in original_arguments:
            tool_call.arguments = arguments


def _allow_missing_tool_arguments(function: GradingFunction) -> GradingFunction:
    """Treat omitted string tool arguments as empty while an AgentHarm grader runs."""
    if getattr(function, "_trustmi_allows_missing_tool_arguments", False):
        return function

    @wraps(function)
    async def wrapped(
        metadata: dict[str, Any], messages: list[ChatMessage]
    ) -> dict[str, Any]:
        with _missing_tool_arguments_are_empty(messages):
            return await function(metadata, messages)

    wrapped._trustmi_allows_missing_tool_arguments = True  # type: ignore[attr-defined]
    return wrapped


def _allow_missing_argument_lookup(
    function: ToolArgumentsFunction,
) -> ToolArgumentsFunction:
    """Return an empty-valued mapping when a grader looks up an absent call."""
    if getattr(function, "_trustmi_allows_missing_tool_arguments", False):
        return function

    @wraps(function)
    def wrapped(messages: list[ChatMessage], function_name: str) -> Any:
        arguments = function(messages, function_name)
        if isinstance(arguments, dict):
            return defaultdict(str, arguments)
        return arguments

    wrapped._trustmi_allows_missing_tool_arguments = True  # type: ignore[attr-defined]
    return wrapped


def _allow_missing_tool_arguments_in_scorer(
    function: ScoringFunction,
) -> ScoringFunction:
    """Treat omitted string tool arguments as empty while a scorer runs."""
    if getattr(function, "_trustmi_allows_missing_tool_arguments", False):
        return function

    @wraps(function)
    async def wrapped(state: TaskState, target: Target) -> Score:
        with _missing_tool_arguments_are_empty(state.messages):
            return await function(state, target)

    wrapped._trustmi_allows_missing_tool_arguments = True  # type: ignore[attr-defined]
    return wrapped


def _make_agentdojo_tool_robust(tool: Tool | ToolDef) -> ToolDef:
    """Convert recoverable AgentDojo tool errors into nonfatal failures."""
    definition = tool if isinstance(tool, ToolDef) else ToolDef(tool)
    if getattr(definition.tool, "_trustmi_argument_errors_are_nonfatal", False):
        return definition

    @wraps(definition.tool)
    async def wrapped(**kwargs: Any) -> Any:
        try:
            return await definition.tool(**kwargs)
        except (TypeError, ValueError) as exc:
            raise ToolError(
                f"Invalid arguments for {definition.name}: {exc}"
            ) from exc
        except KeyError as exc:
            raise ToolError(
                f"Tool action failed for {definition.name}: {exc}"
            ) from exc

    wrapped._trustmi_argument_errors_are_nonfatal = True  # type: ignore[attr-defined]
    return ToolDef(
        wrapped,
        name=definition.name,
        description=definition.description,
        parameters=definition.parameters,
        parallel=definition.parallel,
        viewer=definition.viewer,
        model_input=definition.model_input,
        max_output=definition.max_output,
        options=definition.options,
    )


@solver
def _harden_agentdojo_tools() -> Solver:
    """Wrap the tools installed by AgentDojo's environment setup."""

    async def solve(state: TaskState, _generate: Generate) -> TaskState:
        tools = [_make_agentdojo_tool_robust(tool) for tool in state.tools]
        state.tools.clear()
        state.tools.extend(tools)  # type: ignore[arg-type]
        return state

    return solve


def _hash_calendar_events_by_id() -> None:
    """Let AgentDojo's security checks compare calendar-event attachments.

    An email attachment is either a file ID or a calendar event, yet workspace
    injection tasks 10, 11 and 12 compare `set(email.attachments)` with the file
    IDs the attacker asked for. The event model is unhashable, so an email that
    carried one crashed scoring instead of failing the comparison. Hashing by
    `id_` agrees with the model's field equality, and an event never equals a
    file ID, so the check now returns what it always meant to.
    """
    from inspect_evals.agentdojo.tools.types import CalendarEvent

    def by_id(event: CalendarEvent) -> int:
        return hash(event.id_)

    CalendarEvent.__hash__ = by_id  # type: ignore[method-assign,assignment]


def _missing_state_fails_the_check(check: Callable[..., Awaitable[bool]]):
    """Score an AgentDojo check that finds no state as a goal not met."""
    if getattr(check, "_trustmi_missing_state_fails", False):
        return check

    @wraps(check)
    async def wrapped(*args: Any, **kwargs: Any) -> bool:
        try:
            return await check(*args, **kwargs)
        except (KeyError, IndexError):
            return False

    wrapped._trustmi_missing_state_fails = True  # type: ignore[attr-defined]
    return wrapped


def _harden_agentdojo_checks() -> None:
    """Let AgentDojo's checks fail, rather than crash, on state that is absent.

    AgentDojo's tools treat a missing inbox as an empty one, but several utility
    and security checks index it directly (``user_inbox["Alice"]``) or take its
    last message, so a model that never produced that state crashed scoring,
    and under fail_on_error took its whole arm down. A check that raises
    KeyError or IndexError looked for something the model never created, so its
    goal was not met.
    """
    from inspect_evals.agentdojo.base_tasks import BaseInjectionTask, BaseUserTask

    for base, method in ((BaseUserTask, "utility"), (BaseInjectionTask, "security")):
        pending = list(base.__subclasses__())
        while pending:
            task = pending.pop()
            pending.extend(task.__subclasses__())
            check = task.__dict__.get(method)
            if check is not None:
                setattr(task, method, _missing_state_fails_the_check(check))


def _harden_agentharm_graders(module: ModuleType) -> None:
    """Prevent malformed model tool calls from crashing deterministic grading."""
    argument_lookup = getattr(module, "get_args_of_function_call", None)
    if callable(argument_lookup):
        setattr(
            module,
            "get_args_of_function_call",
            _allow_missing_argument_lookup(argument_lookup),
        )
    for name, function in vars(module).items():
        if name.startswith("grade_") and inspect.iscoroutinefunction(function):
            setattr(module, name, _allow_missing_tool_arguments(function))


def agentharm_harmful(
    *, refusal_judge: Model, semantic_judge: Model, **kwargs: Any
) -> Task:
    """Build harmful AgentHarm with robust tool-argument grading."""
    from inspect_evals.agentharm import agentharm as upstream_agentharm
    from inspect_evals.agentharm.benchmark import harmful_grading_functions

    with _agentharm_dataset_lock():
        _harden_agentharm_graders(harmful_grading_functions)
        return upstream_agentharm(
            refusal_judge=refusal_judge,
            semantic_judge=semantic_judge,
            **kwargs,
        )


def agentharm_benign(
    *, refusal_judge: Model, semantic_judge: Model, **kwargs: Any
) -> Task:
    """Build benign AgentHarm with robust tool-argument grading."""
    from inspect_evals.agentharm import agentharm_benign as upstream_agentharm_benign
    from inspect_evals.agentharm.benchmark import benign_grading_functions

    with _agentharm_dataset_lock():
        _harden_agentharm_graders(benign_grading_functions)
        return upstream_agentharm_benign(
            refusal_judge=refusal_judge,
            semantic_judge=semantic_judge,
            **kwargs,
        )


def agentdojo(**kwargs: Any) -> Task:
    """Build AgentDojo with robust tool execution and deterministic scoring."""
    from inspect_evals.agentdojo import agentdojo as upstream_agentdojo

    task = upstream_agentdojo(**kwargs)
    _hash_calendar_events_by_id()
    _harden_agentdojo_checks()
    if isinstance(task.setup, list):
        task.setup = [*task.setup, _harden_agentdojo_tools()]
    elif task.setup is not None:
        task.setup = [task.setup, _harden_agentdojo_tools()]
    else:
        task.setup = _harden_agentdojo_tools()
    task.scorer = [
        _allow_missing_tool_arguments_in_scorer(function)
        for function in task.scorer or []
    ]
    return task


def bfcl(**kwargs: Any) -> Task:
    """Build BFCL with isolated per-sample multi-turn tool definitions."""
    from inspect_evals.bfcl import bfcl as upstream_bfcl

    bfcl_module = importlib.import_module("inspect_evals.bfcl.bfcl")
    bfcl_module.multi_turn_solve = _bfcl_multi_turn_solve
    bfcl_module.multi_turn_match = _score_bfcl_multi_turn
    return upstream_bfcl(**kwargs)
