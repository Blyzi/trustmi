"""Name a run's output files after the vector's own training run.

BiPO writes its checkpoints into a run-named directory:

    data/steering_vectors/20260825-143326_Qwen3.5-9B_benevolence_Lall_user/
        Qwen3.5-9B_benevolence_Lall_user_trust.pt
        Qwen3.5-9B_benevolence_Lall_user_trust_step105.pt
        ...

so the leading timestamp of the parent directory identifies the training run,
and it is the same name the run's TensorBoard directory carries. Naming an
evaluation's outputs after it is what lets a curve, the checkpoints it
describes, and the results of evaluating them all be found by one string.

The step matters as much as the timestamp: every checkpoint in that listing
comes from one run, so the timestamp alone would have the results of evaluating
step 105 overwrite the results of evaluating the final vector. It is appended
when the filename carries one.

Vectors written before 2026-08-19 sat flat in the vector directory, with no
run directory, and diff-of-means vectors never had one. Those keep the old
behaviour — named after the vector file itself — rather than being given a
misleading timestamp taken from somewhere else.

Stdlib only, and deliberately in its own module: `utils/steering.py` pulls in
torch, datasets and transformers, which is a great deal to import to compute a filename.
"""

from __future__ import annotations

import re
from pathlib import Path

# The run directory's leading timestamp, as steering-vector-train.py writes it.
_RUN_TIMESTAMP = re.compile(r"^(\d{8}-\d{6})")
# BiPO's per-evaluation checkpoints, which share a run directory with the final
# vector and with each other.
_STEP = re.compile(r"_step(\d+)$")


def vector_run_id(vector_path: Path | str | None) -> str | None:
    """A short identifier for the run that produced `vector_path`.

    `20260825-143326` for a run's final vector, `20260825-143326-step105` for
    one of its checkpoints, and the vector's own stem when the path carries no
    run directory. None when there is no vector at all.
    """
    if vector_path is None:
        return None
    path = Path(vector_path)
    stem = path.stem
    run = _RUN_TIMESTAMP.match(path.parent.name)
    if not run:
        return stem
    step = _STEP.search(stem)
    return f"{run.group(1)}-step{step.group(1)}" if step else run.group(1)
