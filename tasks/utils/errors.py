"""Backend errors that callers outside the backend have to react to.

Kept apart from `utils/vllm_common.py` on purpose. That module reaches into
vLLM and costs eight seconds to import, which is why `benchmarks/model.py`
defers loading it until an evaluation actually constructs a provider — `inspect
list tasks` and `--help` must not pay for torch. Catching one of these
exceptions is not a reason to give that up, so they live somewhere free to
import.
"""

from __future__ import annotations


class ContextWindowExceeded(ValueError):
    """A request's prompt plus its output budget does not fit the window."""


__all__ = ["ContextWindowExceeded"]
