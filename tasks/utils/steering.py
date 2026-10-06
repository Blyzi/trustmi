"""Steering spans and vectors as training defines them, bridged from `interpretability/`.

The steered evaluations in `tasks/` add a vector over explicit token spans of a
rendered prompt and nowhere else. The default runtime target remains the
intervention BiPO was trained under (`--inject user` in training means the
latest user turn during prefill; see the Steerer in `steering-vector-train.py`).

That span computation already exists exactly once, in
`interpretability/utils.py`, and it has to keep agreeing with training's
`encode_example` or a vector is being applied over a span it was never fitted
for. It was consolidated there on 2026-08-24 precisely to stop a third copy
existing. So this module does not reimplement it: it loads that file by path
under its own module name and re-exports what the vLLM-Lens backend and its
callers need.

By path rather than by import because `tasks/utils/` is itself a package named
`utils`, so a plain `import utils` after putting `interpretability/` on
sys.path would resolve to this package instead of that file. Same trick
`data/benevolence.py` uses to load its hyphenated prompt module.

The two projects are separate `uv` environments, but every dependency this
pulls in is already in `tasks/`. Nothing here needs
`interpretability/.venv`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# tasks/utils/steering.py -> tasks/utils -> tasks -> repo root
_INTEROP_PATH = (
    Path(__file__).resolve().parents[2] / "interpretability" / "utils.py"
)
_MODULE_NAME = "interpretability_utils"


def _load_interpretability_utils():
    module = sys.modules.get(_MODULE_NAME)
    if module is not None:
        return module
    if not _INTEROP_PATH.is_file():
        raise ImportError(
            f"cannot find {_INTEROP_PATH}. The steered-generation scripts in "
            "tasks/ read the injection span from the interpretability project, "
            "so that project has to sit next to this one in the same checkout."
        )
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _INTEROP_PATH)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so dataclasses defined in it pickle and repr
    # against a real module rather than a nameless one.
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_interp = _load_interpretability_utils()

build_steered_prompt = _interp.build_steered_prompt
PromptLayout = _interp.PromptLayout
load_steering_vector = _interp.load_steering_vector
trained_span_tag = _interp.trained_span_tag

__all__ = [
    "PromptLayout",
    "build_steered_prompt",
    "load_steering_vector",
    "trained_span_tag",
]
