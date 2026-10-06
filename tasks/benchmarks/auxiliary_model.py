"""Inspect model provider for an existing local vLLM auxiliary server."""

from __future__ import annotations

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
from inspect_ai.tool import ToolChoice, ToolInfo

from benchmarks.model import _message_to_openai, _sampling_params, _stop_reason


# LiteLLM defaults to a 6000s request timeout, so a stalled auxiliary server
# would hold a sample's concurrency slot for 100 minutes per attempt. Cap it
# well above the slowest healthy call instead: 2048 tokens at the ~16 tok/s
# this server sustains under load is a little over two minutes.
DEFAULT_TIMEOUT_SECONDS = 600


async def _completion(**kwargs: Any) -> Any:
    import litellm

    return await litellm.acompletion(**kwargs)


@modelapi(name="trustmi-aux")
class TrustMIAuxiliaryModelAPI(ModelAPI):
    """Call an OpenAI-compatible vLLM server through the existing LiteLLM stack."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        model_name = model_name.removeprefix("trustmi-aux/")
        if not base_url:
            raise ValueError("base_url is required for the TrustMI auxiliary model")
        super().__init__(model_name, base_url, api_key, [], config)
        self.model_args = model_args

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        if tools:
            raise ValueError("the GPT-OSS auxiliary model does not accept tools")
        params = _sampling_params(config, default_max_tokens=1024)
        params["num_retries"] = (
            config.max_retries if config.max_retries is not None else 3
        )
        params["timeout"] = (
            config.timeout if config.timeout is not None else DEFAULT_TIMEOUT_SECONDS
        )
        if config.reasoning_effort is not None:
            params["reasoning_effort"] = config.reasoning_effort
        response = await _completion(
            model=f"hosted_vllm/{self.model_name}",
            messages=[_message_to_openai(message) for message in input],
            api_base=self.base_url,
            api_key=self.api_key or "inspectai",
            **params,
            **self.model_args,
        )

        choice = response.choices[0]
        message = choice.message
        usage = response.usage
        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content=message.content or ""),
                    stop_reason=_stop_reason(choice.finish_reason),
                )
            ],
            usage=ModelUsage(
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
                reasoning_tokens=int(
                    getattr(usage, "reasoning_tokens", 0)
                    or getattr(
                        getattr(usage, "completion_tokens_details", None),
                        "reasoning_tokens",
                        0,
                    )
                    or 0
                ),
            ),
        )
