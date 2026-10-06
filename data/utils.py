"""Shared helpers for the generation scripts in this directory.

Importable as `utils` because the script's own directory leads sys.path when a
script here is run directly.
"""


def model_subset(model_str: str) -> str:
    """Subset name for the HF push: the model name, minus a leading
    `hosted_vllm/` provider prefix and any `<org>/` namespace."""
    name = model_str
    if name.startswith("hosted_vllm/"):
        name = name[len("hosted_vllm/") :]
    return name.split("/")[-1]


def thinking_params(model_str: str, thinking: bool) -> dict:
    """Request params that turn reasoning on or off for `model_str`.

    Each family exposes a different knob. gpt-oss is the odd one: harmony has no
    'off' setting, only low/medium/high, so the flag picks the floor rather than
    disabling reasoning outright.
    """
    if "gpt-oss" in model_str.lower():
        return {"reasoning_effort": "high" if thinking else "low"}
    # Qwen3 and Gemma 4 both read this from the chat template; templates that
    # don't know the flag ignore it, so it is safe to send to anything else.
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}}}
