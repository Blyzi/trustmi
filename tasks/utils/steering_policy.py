"""Shared runtime steering targets for every vLLM-Lens consumer.

Prompt rendering discovers concrete token spans; this module applies the small
amount of policy that decides which of those spans a request should steer.  It
has no torch, vLLM, or Inspect dependency so command-line entry points can
import the same choices without loading an inference stack.
"""

from __future__ import annotations

from typing import Literal, cast


TokenSpan = tuple[int, int]
SteeringRange = tuple[int, int | None]
SteeringTarget = Literal[
    "latest_user",
    "latest_user_and_tools",
    "all_users",
    "all_users_and_tools",
    "generated_assistant",
]

STEERING_TARGETS: tuple[SteeringTarget, ...] = (
    "latest_user",
    "latest_user_and_tools",
    "all_users",
    "all_users_and_tools",
    "generated_assistant",
)
DEFAULT_STEERING_TARGET: SteeringTarget = "latest_user"


def validate_steering_target(target: str) -> SteeringTarget:
    """Return a typed target or reject an unknown value."""
    if target not in STEERING_TARGETS:
        choices = ", ".join(STEERING_TARGETS)
        raise ValueError(f"unknown steering target {target!r}; choose {choices}")
    return cast(SteeringTarget, target)


def training_span_for_target(target: str) -> Literal["user", "assistant"]:
    """The training-time vector span required by a runtime target."""
    target = validate_steering_target(target)
    return "assistant" if target == "generated_assistant" else "user"


def resolve_steering_ranges(
    target: str,
    prompt_length: int,
    user_spans: tuple[TokenSpan, ...] | list[TokenSpan],
    tool_spans: tuple[TokenSpan, ...] | list[TokenSpan],
) -> list[SteeringRange]:
    """Resolve a semantic target to absolute, half-open sequence ranges.

    Prompt positions are ``0 .. prompt_length - 1``.  ``None`` as an end means
    the open-ended generated sequence beginning at ``prompt_length``. Tool
    results selected by ``latest_user_and_tools`` must follow the latest user;
    tool history belonging to earlier user turns remains untouched.
    """
    target = validate_steering_target(target)
    if prompt_length < 1:
        raise ValueError("prompt_length must be positive")
    users = tuple(user_spans)
    tools = tuple(tool_spans)
    prompt_spans = sorted((*users, *tools))
    previous_end = 0
    for start, end in prompt_spans:
        if not (previous_end <= start < end <= prompt_length):
            raise ValueError(
                f"invalid or overlapping message span {(start, end)} for "
                f"{prompt_length} prompt tokens"
            )
        previous_end = end

    if target == "generated_assistant":
        # The first generated token is sampled from the prompt's final logits.
        # Its own forward pass is position prompt_length and can therefore
        # influence the second generated token onward, not its own selection.
        return [(prompt_length, None)]
    if not users:
        raise ValueError("prompt has no non-empty user span to steer")
    if target == "latest_user":
        return [users[-1]]
    if target == "latest_user_and_tools":
        latest_user = users[-1]
        return [latest_user, *(span for span in tools if span[0] >= latest_user[1])]
    if target == "all_users_and_tools":
        return prompt_spans
    return list(users)


__all__ = [
    "DEFAULT_STEERING_TARGET",
    "STEERING_TARGETS",
    "SteeringRange",
    "SteeringTarget",
    "TokenSpan",
    "resolve_steering_ranges",
    "training_span_for_target",
    "validate_steering_target",
]
