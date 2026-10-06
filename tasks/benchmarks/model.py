"""Inspect model provider backed by TrustMI's in-process vLLM-Lens engine."""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo

from utils.errors import ContextWindowExceeded
from utils.steering_policy import (
    DEFAULT_STEERING_TARGET,
    training_span_for_target,
    validate_steering_target,
)


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def _text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError(
            f"unsupported Inspect message content: {type(content).__name__}"
        )

    parts: list[str] = []
    for block in content:
        item = _dump(block)
        if not isinstance(item, dict):
            raise TypeError(f"unsupported Inspect content block: {type(item).__name__}")
        kind = item.get("type")
        if kind not in {"text", "input_text", "output_text"}:
            raise ValueError(
                f"TrustMI's steered backend currently supports text only, got {kind!r}"
            )
        parts.append(str(item.get("text", "")))
    return "".join(parts)


def _tool_call_to_openai(tool_call: Any) -> dict:
    item = _dump(tool_call)
    function = item.get("function")
    if isinstance(function, dict):
        name = function["name"]
        arguments = function.get("arguments", {})
    else:
        name = function
        arguments = item.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise TypeError(
            "Qwen chat templates require tool-call arguments to be a mapping"
        )
    return {
        "id": item["id"],
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _message_to_openai(message: ChatMessage) -> dict:
    item = _dump(message)
    converted = {
        "role": item["role"],
        "content": _text_content(item.get("content")),
    }
    if item.get("tool_calls"):
        converted["tool_calls"] = [
            _tool_call_to_openai(call) for call in item["tool_calls"]
        ]
    for key in ("tool_call_id", "name"):
        if item.get(key) is not None:
            converted[key] = item[key]
    if item["role"] == "tool" and item.get("function"):
        converted["name"] = item["function"]
    return converted


def _tool_to_openai(tool: ToolInfo) -> dict:
    item = _dump(tool)
    parameters = _dump(item.get("parameters", {}))
    return {
        "type": "function",
        "function": {
            "name": item["name"],
            "description": item.get("description", ""),
            "parameters": parameters,
        },
    }


def _optional(config: GenerateConfig, name: str) -> Any:
    return getattr(config, name, None)


def _sampling_params(config: GenerateConfig, default_max_tokens: int) -> dict:
    params = {
        "max_tokens": _optional(config, "max_tokens") or default_max_tokens,
    }
    for name in (
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "presence_penalty",
        "frequency_penalty",
    ):
        value = _optional(config, name)
        if value is not None:
            params[name] = value
    if "seed" in params:
        # Inspect treats epochs as independent samples. Derive a stable seed for
        # each epoch so stochastic repeated evaluations do not replay the same
        # completion while remaining exactly reproducible.
        from inspect_ai.model._cache import epoch

        current_epoch = epoch.get(None)
        if current_epoch is not None:
            params["seed"] += current_epoch - 1
    stop = _optional(config, "stop_seqs") or _optional(config, "stop")
    if stop:
        params["stop"] = stop
    return params


def _valid_unicode(value: Any) -> Any:
    """``value`` with every string made encodable, recursively.

    JSON lets a model write any ``\\uXXXX`` escape, and Llama sometimes writes
    half of a surrogate pair (the start of an emoji). ``json.loads`` keeps it as
    a lone surrogate, which no UTF-8 serializer can encode, so Inspect failed to
    save the sample and ``fail_on_error`` took its arm down. Whole pairs are
    joined back into their character; a lone half becomes U+FFFD.
    """
    if isinstance(value, str):
        return value.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    if isinstance(value, dict):
        return {_valid_unicode(key): _valid_unicode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_valid_unicode(item) for item in value]
    return value


_FUNCTION_TAG = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)


def _function_tag_calls(content: str) -> tuple[str, list[ToolCall]]:
    """Tool calls written as ``<function=name>{json}</function>`` text.

    AgentHarm's system prompt tells the model to call tools in exactly this
    format. Models answering natively are parsed by their tool parser, but one
    that follows the instruction (Llama-3.1-70B always does) writes calls no
    native parser reads: they were never executed, and the grader scored an
    agent that had called nothing. Such calls are taken from the text; an
    argument body that is not a JSON object reaches the model as a parse error.
    """
    calls = []
    for match in _FUNCTION_TAG.finditer(content):
        name, body = match.group(1), match.group(2).strip()
        try:
            arguments = json.loads(body) if body else {}
            parse_error = None
        except json.JSONDecodeError as exc:
            arguments, parse_error = {}, str(exc)
        if parse_error is None and not isinstance(arguments, dict):
            parse_error = (
                f"tool arguments must be a JSON object, not {type(arguments).__name__}"
            )
            arguments = {}
        calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:24]}",
                function=name,
                arguments=_valid_unicode(arguments),
                type="function",
                parse_error=parse_error,
            )
        )
    if not calls:
        return content, []
    return _FUNCTION_TAG.sub("", content).strip(), calls


def _output_tool_call(call: Any) -> ToolCall:
    function = call.function
    raw_arguments = function.arguments
    try:
        arguments = (
            json.loads(raw_arguments)
            if isinstance(raw_arguments, str)
            else raw_arguments
        )
        parse_error = None
    except json.JSONDecodeError as exc:
        arguments = {}
        parse_error = str(exc)
    if parse_error is None and not isinstance(arguments, dict):
        # Llama's parser passes on whatever the model put under "parameters",
        # sometimes a bare string. The model gets this back as a tool error.
        parse_error = (
            f"tool arguments must be a JSON object, not {type(arguments).__name__}"
        )
        arguments = {}
    return ToolCall(
        id=call.id,
        function=function.name,
        arguments=_valid_unicode(arguments),
        type="function",
        parse_error=parse_error,
    )


def _stop_reason(finish_reason: str | None) -> str:
    return {
        "stop": "stop",
        "tool_calls": "tool_calls",
        "length": "max_tokens",
        "content_filter": "content_filter",
    }.get(finish_reason or "", "unknown")


def _response_metadata(response: Any) -> dict[str, Any] | None:
    """Extract serializable steering validation without leaking backend state."""
    hidden_params = getattr(response, "_hidden_params", None)
    if (
        not isinstance(hidden_params, dict)
        or hidden_params.get("steering_backend") != "vllm_lens"
    ):
        return None
    return {
        "steering_backend": "vllm_lens",
        "steering_target": hidden_params.get("steering_target"),
        "steering_range_count": int(
            hidden_params.get("steering_range_count", 0) or 0
        ),
        "steering_ranges_verified": bool(
            hidden_params.get("steering_ranges_verified", False)
        ),
    }


@modelapi(name="trustmi")
class TrustMIModelAPI(ModelAPI):
    """Expose one fixed steering condition as an Inspect model."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        *,
        vector_path: str | None = None,
        strength: float | str | None = None,
        layers: str | list[int] | None = None,
        steering_target: str | None = None,
        think: bool | str | None = None,
        tensor_parallel_size: int | str | None = None,
        gpu_memory_utilization: float | str | None = None,
        max_model_len: int | str | None = None,
        default_max_tokens: int | str | None = None,
        **model_args: Any,
    ) -> None:
        model_name = model_name.removeprefix("trustmi/")
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )

        vector_path = vector_path or os.environ.get("TRUSTMI_VECTOR_PATH")
        if not vector_path:
            raise ValueError(
                "vector_path is required (or set TRUSTMI_VECTOR_PATH), including "
                "for the strength-0 matched baseline"
            )
        if strength is None:
            strength = os.environ.get("TRUSTMI_STRENGTH", "0")
        if steering_target is None:
            steering_target = os.environ.get(
                "TRUSTMI_STEERING_TARGET", DEFAULT_STEERING_TARGET
            )
        if think is None:
            think = os.environ.get("TRUSTMI_THINK", "false")
        if tensor_parallel_size is None:
            tensor_parallel_size = os.environ.get("TRUSTMI_TENSOR_PARALLEL_SIZE")
        if gpu_memory_utilization is None:
            gpu_memory_utilization = os.environ.get(
                "TRUSTMI_GPU_MEMORY_UTILIZATION", "0.9"
            )
        if max_model_len is None:
            max_model_len = os.environ.get("TRUSTMI_MAX_MODEL_LEN", "32768")
        if default_max_tokens is None:
            default_max_tokens = os.environ.get("TRUSTMI_DEFAULT_MAX_TOKENS", "2048")

        self.strength = float(strength)
        self.steering_target = validate_steering_target(steering_target)
        self.think = (
            think if isinstance(think, bool) else think.lower() in {"1", "true", "yes"}
        )
        self.default_max_tokens = int(default_max_tokens)
        layer_filter = self._parse_layers(layers or os.environ.get("TRUSTMI_LAYERS"))

        # Imported only when an evaluation constructs the provider. This keeps
        # `inspect list tasks` and CLI help from loading torch/vLLM.
        from utils.vllm_lens_wrapper import (
            VLLMLens,
            default_tensor_parallel_size,
            num_decoder_layers,
        )
        from utils.steering import load_steering_vector, trained_span_tag

        if tensor_parallel_size is None:
            tensor_parallel_size = default_tensor_parallel_size()
        self.backend = VLLMLens(
            default_model=model_name,
            tensor_parallel_size=int(tensor_parallel_size),
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_model_len=int(max_model_len),
            max_loaded_models=1,
        )
        self.vector, self.layer_rows = load_steering_vector(
            Path(vector_path),
            layer_filter=layer_filter,
            n_layers=num_decoder_layers(model_name),
        )

        trained_on = trained_span_tag(Path(vector_path))
        required_training_span = training_span_for_target(self.steering_target)
        if trained_on is not None and trained_on != required_training_span:
            raise ValueError(
                f"vector was trained on the {trained_on!r} span but "
                f"steering_target={self.steering_target!r} requires a "
                f"{required_training_span!r} vector"
            )

    @staticmethod
    def _parse_layers(value: str | list[int] | None) -> list[int] | None:
        if value is None or value == "":
            return None
        if isinstance(value, list):
            return [int(layer) for layer in value]
        return [int(layer.strip()) for layer in value.split(",") if layer.strip()]

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        messages = [_message_to_openai(message) for message in input]
        # An empty list means this request has no tools. Preserve that semantic
        # distinction all the way through prompt rendering and output parsing:
        # model-family tool parsers must not reinterpret ordinary prose as a
        # tool call on text-only benchmarks.
        openai_tools = [_tool_to_openai(tool) for tool in tools] or None
        params = _sampling_params(config, self.default_max_tokens)
        try:
            response = await asyncio.to_thread(
                self.backend.steered_completion,
                self.model_name,
                messages,
                openai_tools,
                self.vector,
                self.layer_rows,
                self.strength,
                steering_target=self.steering_target,
                think=self.think,
                optional_params=params,
            )
        except ContextWindowExceeded:
            # Report an outgrown transcript the way every other provider does
            # rather than raising: `model_length` is what tells an agent loop to
            # compact or stop. Raised as an error instead it would abort the
            # whole eval under `fail_on_error`, losing every other sample over
            # one conversation that happened to run long.
            return ModelOutput.from_content(
                model=self.model_name,
                content="",
                stop_reason="model_length",
            )

        choice = response.choices[0]
        message = choice.message
        tool_calls = [_output_tool_call(call) for call in (message.tool_calls or [])]
        content = _valid_unicode(message.content or "")
        if not tool_calls:
            content, tool_calls = _function_tag_calls(content)
        assistant = ChatMessageAssistant(
            content=content,
            tool_calls=tool_calls or None,
        )
        usage = response.usage
        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=assistant,
                    stop_reason=_stop_reason(choice.finish_reason),
                )
            ],
            usage=ModelUsage(
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            ),
            metadata=_response_metadata(response),
        )
