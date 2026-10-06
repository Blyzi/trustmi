"""Learn per-behavior steering vectors with Bi-directional Preference Optimization.

BiPO (Cao et al., NeurIPS 2024, "Personalized Steering of Large Language Models:
Versatile Steering Vectors Through Bi-directional Preference Optimization",
arXiv:2406.00045). Unlike the diff-of-means recipe in steering-vector-mean.py,
which reads a vector off the *activations* of contrastive prompts, BiPO *trains*
steering vectors so that, once added to the residual stream, they raise the
model's own generation probability of the target-behavior continuation and lower
that of the opposite-behavior continuation.

Each training example is a context q followed by two continuations of it — the
target-behavior one (r_T) and the opposite-behavior one (r_O). The context is the
user turn (the scenario) and each continuation is the assistant turn; the
objective scores the assistant tokens' log-probability.

A vector v is inserted at layer L by broadcast-adding d*v to that layer's hidden
state at the token positions `--inject` selects (d is the sampled direction,
below):

  * `user` (the default) — the user-turn (scenario) tokens, so v steers how the
    model *reads* the scenario and never how it writes the continuation; the
    effect reaches the scored tokens only through the KV cache the steered prefill
    leaves behind. This is the span the inference-time consumers reproduce
    (scripts/try-steering-vector.py, tasks/general_trust_scale/main.py).
  * `assistant` — the continuation tokens themselves, so v steers how the model
    *writes*, adding straight into the hidden states the scored logits are read
    off. The first scored token is predicted from the last prompt position, which
    lies outside the span and so stays unsteered — matching a generation-time
    setup that injects only on generated positions. The objective has a far more
    direct handle on the loss this way, so read the reward margin with that in
    mind: it drops faster and says less about the vector generalising.

Let pi_{L+1}(r | A_L(q) + v) be the model's log-prob of continuation r given
context q with v injected, and pi_{L+1}(r | A_L(q)) the same with no injection
(the frozen reference — BiPO needs no separate reference model since v=0 recovers
it). The DPO-style objective (paper Eq. 2) is

    min_v  -E_{(q,r_T,r_O)~D} log sigma(
        beta * log[ pi_{L+1}(r_T|A_L(q)+v) / pi_{L+1}(r_T|A_L(q)) ]
      - beta * log[ pi_{L+1}(r_O|A_L(q)+v) / pi_{L+1}(r_O|A_L(q)) ] )

To make -v a faithful vector for the *opposite* behavior too, BiPO samples a
direction d ~ U{-1, +1} per batch and multiplies both the injected vector and the
outer log-ratio by d (paper Eq. 3, Algorithm 1):

    min_v  -E_{d, (q,r_T,r_O)} log sigma(
        d*beta * log[ pi_{L+1}(r_T|A_L(q)+d*v) / pi_{L+1}(r_T|A_L(q)) ]
      - d*beta * log[ pi_{L+1}(r_O|A_L(q)+d*v) / pi_{L+1}(r_O|A_L(q)) ] )

Both equations are written as A_L(q) + v, i.e. for `--inject user`; under
`--inject assistant` the injection moves onto r's own positions and nothing else
about the objective changes.

Optionally the objective is DPO+NLL: --sft alpha adds the length-normalised
negative log-likelihood of the *target* continuation to the per-example
preference loss, always read at +v:

    L = L_BiPO - alpha * log pi_{L+1}(r_T | A_L(q) + v) / |r_T|

The preference term is a ratio of ratios, so it is satisfied just as well by a
vector that pushes both continuations down as by one that lifts the target —
which is what train/reward_chosen exists to expose, and why it so often goes
negative while the margin grows. The NLL term prices that: it reads the absolute
log-probability of the trust-consistent continuation, so degrading the model
into an answer it would never write costs something instead of being free.

Note that DPO+NLL itself has no d: it is defined over a fixed (chosen,
rejected) pair, and the uniform direction is BiPO's (Eq. 3, Algorithm 1). Where
the two meet is a choice neither makes, so it is worth being explicit about it.

The anchor carries **no d at all**: it is the target continuation, read at +v,
on every batch. Neither of the two things the d factor might suggest is done.

It is not *selected* by d — r_T on a d=+1 batch and r_O on a d=-1 one, which is
what this did until 2026-08-31. That spends half the batches raising the
log-probability of the withholding pole, and reading the gradient it is
contrastive in expectation rather than the one-sided likelihood anchor DPO+NLL
intends: it duplicates work the preference term already does instead of
counterbalancing it.

And it is not *multiplied* by d, which would flip the term's sign. In the
preference half a d factor and a pole swap are the same operation, because
-logsigmoid is antisymmetric: -logsig(d*(r_C - r_R)) at d=-1 is exactly
-logsig(r_R - r_C). An NLL is not antisymmetric, so there a d factor pays for
driving the target's log-probability toward -inf. The preference term survives
its own sign flip only because logsigmoid saturates; a bare NLL has no floor,
so that branch is unbounded below.

What the choice does cost is the exact symmetry the sampled direction otherwise
enforces — that the objective run backwards with v is the objective run forwards
with -v and the poles exchanged,

    F(d=-1, v; r_T, r_O)  ==  F(d=+1, -v; r_O, r_T)

The preference half still satisfies it; the anchor deliberately does not, and
cannot: "keep the trust continuation likely" is an asymmetric requirement, and
no symmetric objective expresses one. That is the trade, not an oversight. The
asymmetry is at least coherent with the rest — raising log pi(r_T) at +v lowers
it at -v to first order, which is what the -v end of a sweep is supposed to do
— but the trained vector is no longer the exactly-symmetric BiPO solution, so
the negative half of a strength sweep is worth re-reading after turning --sft
on rather than assumed unchanged. scripts/test-rpo-sft.py pins all of this.

Length normalisation matters less now that only one continuation is ever
scored, but it keeps alpha's meaning stable across datasets whose replies differ
in length.

alpha is hard to reason about from first principles, and two facts about it
point in opposite directions.

By loss *value* it dominates early. The per-token NLL here is ~1.9 nats, so at
alpha=1 the supervised term contributes ~1.9 against a preference term that
starts at log 2 = 0.69 and falls to ~0.14 — the objective is then most of the
way to pure SFT by magnitude.

By *gradient* it is far weaker than that suggests, because the term is large and
almost entirely irreducible: most of those 1.9 nats is what the frozen model
thinks of the text, which no steering vector can move. On the 2026-08-31 run
test/sft fell 0.5772 -> 0.5757 over 45 steps, 0.26% of its own value, while the
preference margin went 0.03 -> 3.7. The preference term also reads *summed*
log-probs against this one's length-normalised ones, a factor of |r_T| ~ 240.

Do not try to settle it by arithmetic — the obvious estimate needs
||grad L_T|| against ||grad L_T - grad L_O||, and those are gradients of two
plausible replies to the same prompt from the same model, so they are correlated
and their difference is the smaller quantity by an unknown factor. On top of
that sigma(-margin) decays over a run (0.49 at step 1, 0.024 by step 50 on that
run, ~20x), so the preference gradient shrinks as training proceeds and an alpha
at parity on step 1 is well past it by the end.

What is measured: alpha <= 0.3 does nothing. Runs at 0.1, 0.3 and 0 differ in
their vector norms by under 1%, which is run-to-run noise. Above that, sweep it
— 1, 3, 10 — and read train/reward_chosen, the curve this term exists to fix:
it goes negative when the vector satisfies the margin by pushing both poles down
rather than lifting the target. Stop when that stops happening, or when the
downstream coherence rate starts to fall.

train/sft and test/sft are the alpha-scaled term as it enters the loss. Both are
single-valued and both converge: the anchor reads one continuation at one
injection whatever d the batch drew, so there is nothing left for the direction
to alternate between. (Until 2026-08-31 the term followed d, which made train/sft
two interleaved series ~0.3 nats/token apart on benevolence with no trend in
either — a square wave no amount of training settled.)

alpha defaults to 0, which is plain BiPO and the objective every vector on disk
was trained under. The term reads the steered pass alone, so switching it on
neither invalidates the reference-log-prob cache nor changes what train/loss and
test/loss mean — those stay the pure preference number.

Vectors are initialised to zero and optimised with AdamW; the model weights stay
frozen. The reference log-probs (v=0) are independent of both d and v, so we
compute them once up front and reuse them across every epoch — and, depending on
nothing else in the configuration either, they are memoized on disk
(--ref_cache_dir) and reused across runs, which is what a sweep over layers,
penalties or learning rates would otherwise repay in full before every step 1.

Multiple layers can be trained *together*: pass several layer indices (or `all`)
and one vector per layer is injected simultaneously, sharing the sampled
direction d, and all are optimised under a single loss. The paper's setting is
recovered by passing a single layer.

Trained that way the objective has no reason to prefer a vector that lives in a
few layers, and AdamW's decoupled weight decay actively prefers one that does
not: a fixed total effect E split over k redundant layers has squared-norm cost
E^2/k, so the cheapest solution smears the norm across the stack. Two penalties
over the layer rows push back, and they can be used together.

--group_lasso adds an L1 over the *vector of per-layer norms*, and an L2 inside
each row, so it sparsifies whole layers without touching the direction inside the
ones it keeps:

    G(V) = sum_L ||v_L||

Flat in how a fixed sum of norms is split, where weight decay pays less the more
layers share it, so once extra layers stop buying margin the cheapest
configuration is the fewest rows. It is applied as its proximal operator — the
group soft-threshold, after the optimizer step — because that is what puts rows
on *exactly* zero; the subgradient form (--group_lasso_subgradient) leaves them
small but never zero under Adam's per-coordinate preconditioning. It shrinks the
total norm as well as concentrating it, which the inference-time strength sweep
undoes anyway.

--hoyer adds the group Hoyer-square penalty of DeepHoyer (Yang et al., ICLR
2020) — Hoyer's 2004 sparseness ratio, squared:

    R(V) = (sum_L ||v_L||)^2 / sum_L ||v_L||^2

R is 1 when a single layer holds the whole vector and n_selected when every
layer holds an equal norm, and it is degree-0 homogeneous: R(cV) = R(V). It
therefore constrains only how the norm is *distributed* and leaves its size to
the DPO term — which is all that is identifiable anyway, since the inference-time
strength sweep rescales the whole tensor. Its gradient flips sign at the
size-biased mean of the row norms, pushing rows above it up and rows below it
down, so it redistributes norm between layers rather than shrinking any of them.
Being smooth it reaches no exact zeros (an L1 over the same groups would, at the
cost of a lambda whose scale rides on beta and on the residual-stream norm), and
at the zero init every row norm is equal, so its gradient is exactly zero and
the penalty stays inert until the loss has broken the symmetry.

That last property does not survive the step that breaks the symmetry, which is
why --hoyer_warmup_frac exists: degree-0 homogeneity means the gradient scales
as 1/||V||, so at the row norms one step produces (~sqrt(hidden) * lr) it is
some 30x what it is at unit norm, and it spends that on whichever row the first
preference gradient happened to leave longest. The weight therefore ramps in
linearly, over --warmup_frac's window unless given its own. The LR warmup does
not substitute: hoyer reaches the vector through .grad, where Adam's m/sqrt(v)
normalises the magnitude away and leaves lr scaling the size of a step whose
direction the two terms have already fought over. --group_lasso needs no
equivalent, its proximal threshold being lambda * lr_now and so already ramped.

Neither penalty enters train/loss, which stays the pure DPO number; they are
logged beside it as train/penalty_lasso and train/penalty_hoyer, on the same
scale, so whether a lambda is doing nothing or dominating is one chart. Both are
joined by vector_norm/effective_layers — the participation ratio of the per-layer
*energies*, which reads directly as how many layers the vector is using — and by
vector_norm/nonzero_rows, which only the proximal group lasso can move. Both are
logged whether or not a penalty is on.

The result per label is a full (num_layers, hidden_size) tensor — trained layers
hold their vector, untrained layers are zero — saved into a run-named
subdirectory of `--output` (default `data/steering_vectors/`) carrying the
same name as the run's TensorBoard directory, the same shape
steering-vector-mean.py produces. The filename always carries the span as an
`inj-user` / `inj-assistant` part: a vector is only faithful when it is applied
at inference over the span it was trained on, so that is not something to leave
implicit in a default.

The (context, target, opposite) triples come from one of four datasets, chosen
with --dataset.

`conversations-5k` (the default) is MaxLSB/trustmi-conversations-5k, where the
same model answered each question twice: once under a disposition that takes
people at their word, once under one that withholds until verified. The question
(shared by both splits, joined on `id`) is the context, the `trust` split's reply
is the target and the `distrust` split's reply is the opposite.

`benevolence` is a save_to_disk'd run of data/benevolence.py, read from
--data_dir. There the *assistant* is the trustor and the user the trustee, and a
row carries the shared context plus both poles as columns, so the pair is
structural and there is nothing to join on. Two things follow from its shape.
Contexts are multi-turn, and under --inject user the span is the *last* user
turn alone — the turn that poses the decision — with every earlier turn read
unsteered. That is not the only defensible choice (a row often puts the claim
being trusted in an earlier turn, which the vector then only reaches through the
KV cache), but it is the one scripts/try-steering-vector.py and
tasks/general_trust_scale/main.py reproduce at inference, and a vector applied
over a different span than it was fitted for is not the intervention that was
trained. And it ships its own train/test split, stratified over the scenario
bank, which is used in place of --test_frac; re-cutting it at random would report
a test curve measured on a different population than the dataset intends.

Its trust-free control subset is never read here. A control row carries the same
reply in both poles, so its DPO margin is identically zero and its gradient with
it: it cannot move the vector at all, whatever the vector is, and mixing it in
only dilutes the curves toward log 2 and the tie rate while spending real
forwards. Those rows are a null condition for *measuring* a finished vector,
which is what scripts/eval-vector-benevolence.py --neutral uses them for.

On both of those the target is the trusting continuation and the opposite the
withholding one, so the trained vector points distrust -> trust.

`benevolence-trad` is that same corpus after data/benevolence-trad.py has
translated it, and there is nothing to read differently: the translated tree
mirrors the English one directory for directory and column for column, so it goes
through the same loader, and the source's train/test assignment is carried over
rather than resampled, so a row cannot be held out in one language and trained on
in another. What it changes is the *naming*. --data_dir points at one language's
tree (<out_dir>/<generator>/<language>/) and no column in the rows records which
language it is, so the tag that reaches the run directory, the TensorBoard run,
the vector filename and the reference-log-prob stem is `benevolence-<language>`,
read off that directory's name. Without it a French run and an English one would
be named identically and separable only by their timestamps, which is not
something to be reading off a vector months later. The language is joined with a
hyphen rather than an underscore deliberately: the vector filename is
underscore-delimited (model_dataset_layers_span_label), and an underscore in the
tag would add a field to it.

`doubt` is a save_to_disk'd run of data/doubt.py, also read from --data_dir, and
it is a different axis: the assistant has positive reason to think one specific
thing might be wrong, and the two poles are whether it works from that
possibility or commits and moves on. It ships as two subsets — `user/`, where
the uncertain point is something the user asserted, and `self/`, where it is
something the assistant supplied out of its own head — and this script trains
**one vector per subset**, not one over the pair. That is the point of the split:
where the doubt points is the thing a doubt vector is most likely to conflate,
and a vector fitted on the mixture cannot be told apart, on a mixed test curve,
from one that moves only half of it. Each direction arrives as its own TripleSet,
so it gets its own TensorBoard run, its own reference-log-prob cache and its own
`_doubt-user.pt` / `_doubt-self.pt`, and the two overlay on the same charts.
--doubt_targets trains one direction alone. Rows are multi-turn and it ships its
own stratified train/test split, exactly as benevolence does.

Its sign is the OPPOSITE of the two trust datasets, and worth knowing before
comparing a sweep: target is the doubtful continuation, so the vector points
confident -> doubtful and +v adds doubt, where a trust vector's +v moves toward
taking things at face value. BiPO trains -v as the faithful opposite behaviour,
so a symmetric sweep covers both ends of either axis; it is reading one positive
strength across the two that will mislead.

Those replies are long — several hundred tokens each, against ~250 for the
question — so --max_length defaults high enough (2048) to keep whole
continuations. Truncating one does not corrupt the objective, since the masks
clamp to whatever survives, but it scores half an answer.

Peak memory is driven by the *forward* batch, not the optimizer batch: a step
over `batch_size` triples can be split into micro-batches of `micro_batch_size`
whose gradients accumulate before a single optimizer.step(). The loss, the LR
schedule and the sampled direction d are all defined over the optimizer batch, so
shrinking micro_batch_size changes memory and nothing else about the run.

The other half of the memory picture is the loss head, and it used to be the
larger half. Scoring a continuation needs log-probabilities only at the tokens
the objective sums over, so the vocabulary projection runs on *those* hidden
states alone rather than on the whole padded sequence, a chunk at a time under
its own checkpoint (sequence_logprobs, chunked_logprobs). Projecting every
position instead allocates the single largest tensor in the run — at a 256k-token
vocabulary a micro-batch of eight 2048-token sequences is 8.6 GB of bf16 logits
before the softmax has copied anything, and none of it outside the scored span is
ever read.

Multi-GPU, two modes, picked by how the script is launched.

Run plainly (`uv run steering-vector-train.py`) the model is loaded with
device_map="auto", so the decoder stack is *sharded* over the visible GPUs and
each layer's hidden state lives on whichever GPU holds that layer. The steering
vectors stay a single leaf tensor on the first GPU; each hook copies its row (and
the injection mask) onto the device of the hidden state it adds to, and autograd
copies the gradient back, so one optimizer state is shared however the layers are
spread. That is model parallelism: it buys memory — the activations kept for the
backward through v are split across GPUs — but not throughput, since only one
shard is busy at a time.

Run under torchrun (`torchrun --standalone --nproc_per_node=N`) the script uses
*data* parallelism instead: every rank takes a strided slice of every optimizer
batch, and the frozen weights are sharded across the ranks with FSDP2
(`fully_shard`, one shard group per decoder layer plus one for the root). Each
layer's parameters are all-gathered just before that layer runs and dropped again
straight after, so a rank's resident weight footprint is 1/N of the model rather
than a full copy, while every rank still runs the whole stack on its own data —
memory *and* throughput, where device_map="auto" gives only the first.

What FSDP does not do here is shard gradients or optimizer state, because there
are none to shard: the model is frozen, so the only trainable tensor is the
(n_selected, hidden_size) steering vector, which is well under a megabyte and is
replicated on every rank. That is also why its gradient still gets an explicit
all-reduce in reduce_gradient — it is not an FSDP-managed parameter and FSDP's
own reduce-scatter never sees it. The flip side is that the all-gathers are pure
added cost against a replicated run: with the weights sharded, every micro-batch
pays a forward all-gather and a backward one, so on a model that comfortably fits
a single GPU this mode is slower than replicating would be. It earns its keep
when the model does not fit.

Each
rank weights its micro-batches by their share of the *global* batch, so summing
the ranks' gradients reproduces the gradient of the mean over the whole batch
exactly; the loss, the LR schedule and the sampled direction d stay defined over
that global batch, and a run is invariant to N in the same way it is invariant to
micro_batch_size. Every rank draws the same batch order and the same d from an
identically seeded rng and takes the same number of steps, so after the
all-reduce all ranks hold identical vectors and optimizer state; rank 0 owns the
logging and the saving.

One limitation to know about: the weights are sharded *after* loading, so each
rank still materializes a full copy on its GPU on the way in. Training memory is
1/N of the model, but the load is not, and a checkpoint too large to land on one
GPU still will not start. Lifting that needs meta-device init with a sharded
state-dict load, which load_model does not do yet.

Usage:
  uv run steering-vector-train.py --model Qwen/Qwen3-14B --layers all --epochs 20
  uv run steering-vector-train.py --layers 15 20
  # the assistant-trusts-user dataset, on its own train/test split
  uv run steering-vector-train.py --layers 20 \
      -d benevolence --data_dir ../data/data/benevolence/opus-5
  # the same corpus translated; the vectors land under `benevolence-french`
  uv run steering-vector-train.py --layers 20 \
      -d benevolence-trad \
      --data_dir ../data/data/benevolence-trad/opus-5/french
  # doubt: one vector per direction, or just the self-directed one
  uv run steering-vector-train.py --layers 20 \
      -d doubt --data_dir ../data/data/doubt/gemma-4-31B-it
  uv run steering-vector-train.py --layers 20 \
      -d doubt --data_dir ../data/data/doubt/gemma-4-31B-it --doubt_targets self
  # Llama-3-8B-Instruct: 32 layers, and its template emits no thought block, so
  # the scored span is the reply and nothing else (unlike the Qwen3 models)
  uv run steering-vector-train.py -m meta-llama/Meta-Llama-3-8B-Instruct \
      --layers 16 -d benevolence --data_dir ../data/data/benevolence/opus-5
  # steer the continuation the model writes instead of the question it reads
  uv run steering-vector-train.py --layers 20 --inject assistant
  # DPO+NLL: keep the preferred continuation likely, not merely likelier
  uv run steering-vector-train.py --layers 20 --sft 1.0
  # same optimizer batch of 16, but four sequences per forward
  uv run steering-vector-train.py --layers 20 -b 16 --micro_batch_size 4
  # data parallel over 2 GPUs; identical result, ~2x the throughput
  uv run torchrun --standalone --nproc_per_node=2 steering-vector-train.py \
      --layers 20 -b 128
"""

from __future__ import annotations

import gc
import hashlib
import os
import random
import re
from argparse import ArgumentParser
from array import array
from collections import Counter
from datetime import datetime
from math import ceil
from pathlib import Path
from time import perf_counter

import pynvml
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoConfig

from utils import (
    DEVICE,
    SPAN_SENTINEL,
    SYSTEM_PROMPT,
    changed_token_span,
    get_decoder_layers,
    get_hidden_size,
    injection_span,
    load_model,
    DOUBT_SUBSETS,
    load_benevolence,
    load_doubt,
    load_trust_conversations,
)


DEFAULT_BATCH_SIZE = 4

# Peak fp32 bytes one log-softmax chunk may reach in chunked_logprobs. The scored
# tokens are projected to the vocabulary a chunk at a time, so this — not the
# batch — is what bounds the loss head's memory. 1 GiB is ~1k tokens per chunk at
# a 256k vocabulary and ~1.8k at 152k: small enough that the head stops being the
# run's peak, large enough that the extra kernel launches stay invisible next to
# a 9B forward.
LOGP_CHUNK_BYTES = 1 << 30
# ...unless the vocabulary is large enough that the byte budget asks for a chunk
# too short to keep the GPU busy.
MIN_LOGP_CHUNK = 256

# Attention kernel per model family. Hardcoded rather than exposed as a flag
# because it is a property of the checkpoint's architecture, not something a
# sweep varies: run the same model twice with different kernels and the only
# difference is float accumulation order.
#
# FlashAttention-2 is the default and the one worth having. Every micro-batch
# here is right-padded to the longest scenario+continuation in it, and FA2 unpads
# before the kernel, so the quadratic work and the attention activations both
# follow the real token count rather than the padded one — which is what the
# backward through an injection has to keep alive. It needs the flash-attn wheel
# on the node (scripts/install-flash-attn.sh); without it transformers raises at
# load time rather than silently using another kernel.
#
# gemma-4 cannot use it. Its global-attention layers carry `global_head_dim` 512
# (8 of the 12B's 48, every sixth — see Gemma4UnifiedTextAttention, which reads
# global_head_dim for any layer that is not sliding), and FA2's kernel caps the
# head dimension at 256, so the run dies on the first forward — not at load —
# with "FlashAttention forward only supports head dimension at most 256".
#
# It has to be sdpa for the *whole* model rather than FA2 on the 40 sliding
# layers and sdpa on the 8 wide ones, which is what the architecture invites.
# Transformers 5.14.1 cannot express that split, in any of its three plausible
# spellings:
#   - `config._attn_implementation = {"sliding_attention": ..., "full_attention":
#     ...}` raises TypeError: unhashable type: 'dict' in the attention forward,
#     since AttentionInterface.get_interface takes a string and looks it up in a
#     dict.
#   - passing that same dict to from_pretrained is silently a no-op:
#     PreTrainedModel.set_attn_implementation keys a dict on *sub-config* names
#     ("text_config", "vision_config"), so unknown keys fall through to
#     `.get("", current)` and nothing changes.
#   - giving the wide layers their own config copy dispatches the kernels
#     correctly but not the masks, which is worse. The model's forward builds one
#     mask per layer *type* from the shared config
#     (`causal_mask_mapping = {"full_attention": create_causal_mask(...), ...}`),
#     so under FA2 the wide layers would receive flash_attention_mask's output —
#     None, or the 2D mask FA2 unpads from — and run it through sdpa, which wants
#     a 4D mask. Padding would stop being masked, silently, in the middle of a
#     loss that is a difference of two log-probs.
# Kernel and mask are chosen from the same config, so they can only move together.
# "flex_attention" is the one alternative worth trying if sdpa's peak is a
# problem: it takes arbitrary head dims and builds its own BlockMask, so it is a
# whole-model swap that stays correct. It also drags in torch.compile under FSDP2
# and gradient checkpointing, so measure before believing it.
#
# gpt-oss cannot use FA2 either, for a different reason: its attention carries a
# learned per-head **sink** logit that joins the softmax denominator without a
# value to attend to. flash-attn 2's kernel has nowhere to put it, so FA2 either
# refuses the config or drops the sink and quietly renormalises the weights — a
# wrong forward, which here means a wrong log-prob on both poles of every pair.
# The FA3 kernel does support sinks, and transformers reaches it through the
# `kernels` hub integration: an attn_implementation of the form
# "kernels-community/vllm-flash-attn3" is a Hub repo id rather than one of the
# built-in names, resolved by `kernels` at load time. That makes it the one rule
# here with a *download* behind it, and a kernel repo is not warmed the way a
# model snapshot is (job 1907794, then 1912–14xxx):
#   - it lives under the `kernel` repo type, i.e. the cache directory
#     `kernels--kernels-community--vllm-flash-attn3`, which `hf download` cannot
#     write since its --repo-type stops at model/dataset/space;
#   - `kernels` resolves the version spec (transformers asks for version 1) by
#     listing the repo's `v*` branches, and offline it reads `refs/v1` out of
#     that directory instead — so the download must be *at revision `v1`*, not
#     at main, or the ref file never exists;
#   - and it must be the WHOLE repo (6.4 GB, nine build variants), because the
#     offline path's first snapshot_download passes no allow_patterns — it has
#     to list `build/` locally before it can pick a variant — and huggingface_hub
#     1.24 raises IncompleteSnapshotError when the cached tree listing has any
#     file missing. `install_kernel` online fetches only this machine's variant
#     (torch-stable-abi29-cu130-x86_64-linux here: the torch21x builds demand an
#     exact torch version and stop at 2.12, and cu126/cu128 lose on the major),
#     which is enough to run online and never enough to run offline.
# So, from a login node:
#   uv run --frozen python -c "from huggingface_hub import snapshot_download; \
#       print(snapshot_download('kernels-community/vllm-flash-attn3',
#                               repo_type='kernel', revision='v1'))"
# sdpa is the fallback worth knowing: transformers implements the sink
# there in eager Python, so it is correct but pays the padded quadratic.
#
# gemma-3 *can*, and gets FA2 back by a rule of its own above the family one. It
# is the same picture minus the one thing that broke: 5 of every 6 layers are
# sliding and the 6th is full, but every layer is head_dim 256 — exactly at the
# cap, not past it — because there is no separate `global_head_dim`. And the
# window never has to become a mask: Gemma3Attention.forward passes
# `sliding_window=self.sliding_window` (None on a full layer) straight into the
# attention interface, so FA2 gets the window as a kernel argument per layer,
# which is precisely the per-layer split transformers could not express for
# gemma-4 through masks. Under FA2 masking_utils' flash_attention_mask then
# returns None or the 2D padding mask for every layer type alike, so there is no
# 4D-mask-into-the-wrong-kernel hazard left to worry about.
#
# Llama needs no rule and takes the default: head_dim 128, well under the cap, no
# attention sinks, and one attention type for the whole stack, so none of the
# exceptions above reach it.
#
# First match wins, on the lowercased model name — so a more specific family goes
# above the family it is a substring of. "gemma-3-" carries its trailing hyphen
# deliberately: gemma-3n is a different architecture (MatFormer, per-layer
# embeddings, altUp) that nothing here has checked, and the hyphen is what stops
# it inheriting gemma-3's rule instead of falling through to the family default.
_ATTN_IMPLEMENTATION_RULES = (
    ("gemma-3-", "flash_attention_2"),
    ("gemma", "sdpa"),
    ("gpt-oss", "kernels-community/vllm-flash-attn3"),
    ("qwen", "flash_attention_2"),
)
DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2"

# FA2's head-dimension cap. At most, not below: gemma-3 sits exactly on it.
FLASH_ATTENTION_2_MAX_HEAD_DIM = 256


def attn_implementation_for(model_name: str) -> str:
    """Attention kernel for `model_name`, by family. See the rules above."""
    lowered = model_name.lower()
    for family, implementation in _ATTN_IMPLEMENTATION_RULES:
        if family in lowered:
            return implementation
    return DEFAULT_ATTN_IMPLEMENTATION


def flash_attention_2_blocker(config) -> str | None:
    """Why FA2 cannot serve `config`, or None if it can.

    The rules above match on a name, which is all that is available before a
    model is loaded but is not what actually decides this — both ways FA2 fails
    are properties of the checkpoint, and neither announces itself:

      * a head dimension past the cap dies on the *first forward*, not at load,
        which is a queued job's worth of setup before the traceback;
      * a mask deviation is dropped in silence. Under FA2 the mask interface is
        masking_utils.flash_attention_mask, which ignores the mask function it is
        handed and returns None or the 2D padding mask — so a config asking for
        bidirectional attention gets a plain causal forward and a wrong log-prob
        on both poles, with nothing raised anywhere.

    So the name picks the kernel and this vetoes it, which also means a new
    checkpoint in a family that normally runs FA2 degrades to sdpa rather than
    dying. Only *explicitly declared* head dimensions count (`vars`, not
    `getattr`): Qwen leaves head_dim to be derived from hidden_size and would
    otherwise have to be special-cased back out of this check. Any attribute name
    ending in `head_dim` is read, so gemma-4's `global_head_dim` 512 is caught
    without naming it — it is already on sdpa by family, so this is the belt to
    that braces, and it is what would have caught it at load rather than on the
    first forward.
    """
    text = getattr(config, "text_config", None)
    if text is None:
        text = config
    for name, value in vars(text).items():
        if name.endswith("head_dim") and isinstance(value, int):
            if value > FLASH_ATTENTION_2_MAX_HEAD_DIM:
                return (
                    f"{name}={value} is past FA2's "
                    f"{FLASH_ATTENTION_2_MAX_HEAD_DIM}-dimension cap"
                )
    if getattr(text, "use_bidirectional_attention", False):
        return "the config asks for bidirectional attention, which FA2 drops"
    return None


def resolve_batching(
    batch_size: int | None,
    micro_batch_size: int | None,
    grad_accum: int | None,
) -> tuple[int, int, int]:
    """Reconcile --batch_size / --micro_batch_size / --gradient_accumulation_steps.

    The three are related by batch_size = micro_batch_size * grad_accum, so pass
    any two and the third follows; pass only batch_size (the old behaviour) and
    there is no accumulation. batch_size is the *optimizer* batch — how many
    triples one update averages over, hence what the loss and the LR schedule are
    defined over — while micro_batch_size is the *forward* batch, which is what
    peak activation memory scales with.

    batch_size need not divide evenly: grad_accum is then the ceiling and the
    last micro-batch is short, which the per-micro-batch loss weighting handles.
    """
    if micro_batch_size is not None and grad_accum is not None:
        derived = micro_batch_size * grad_accum
        if batch_size is not None and batch_size != derived:
            raise ValueError(
                f"--batch_size {batch_size} != --micro_batch_size "
                f"{micro_batch_size} * --gradient_accumulation_steps "
                f"{grad_accum} = {derived}; pass at most two of the three"
            )
        batch_size = derived
    elif micro_batch_size is not None:
        batch_size = micro_batch_size if batch_size is None else batch_size
        grad_accum = ceil(batch_size / micro_batch_size)
    elif grad_accum is not None:
        batch_size = DEFAULT_BATCH_SIZE if batch_size is None else batch_size
        micro_batch_size = ceil(batch_size / grad_accum)
    else:
        batch_size = DEFAULT_BATCH_SIZE if batch_size is None else batch_size
        micro_batch_size, grad_accum = batch_size, 1

    if min(batch_size, micro_batch_size, grad_accum) < 1:
        raise ValueError(
            f"batch sizes must be >= 1 (got batch_size={batch_size}, "
            f"micro_batch_size={micro_batch_size}, grad_accum={grad_accum})"
        )
    if micro_batch_size > batch_size:
        raise ValueError(
            f"--micro_batch_size {micro_batch_size} exceeds --batch_size "
            f"{batch_size}; a micro-batch is a slice of a batch"
        )
    return batch_size, micro_batch_size, grad_accum


def parse_layers(tokens: list[str], num_layers: int) -> list[int]:
    """Resolve the --layers argument into sorted, de-duplicated layer indices."""
    if any(t.lower() == "all" for t in tokens):
        return list(range(num_layers))
    layers = sorted({int(t) for t in tokens})
    for layer in layers:
        if not 0 <= layer < num_layers:
            raise ValueError(f"layer {layer} out of range [0, {num_layers})")
    return layers


# --------------------------------------------------------------------------- #
# Data parallelism
# --------------------------------------------------------------------------- #


def dist_setup():
    """Join the torchrun process group; return (rank, world_size, local_rank, mesh).

    Returns (0, 1, 0, None) when the script was not launched under torchrun, which
    is the single-process path: every collective below short-circuits to a no-op,
    nothing is sharded, and the run behaves exactly as it did before data
    parallelism existed. Detection is on RANK/WORLD_SIZE rather than a flag so
    there is no way to ask for one and get the other.

    `mesh` is the 1-D device mesh FSDP2 shards the frozen weights over — one
    dimension, one rank per GPU, since data parallelism is the only axis here.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0, None
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # Before init_process_group, so NCCL binds this rank to its own GPU.
    torch.cuda.set_device(local_rank)
    # device_id pins the collectives to this rank's GPU explicitly; without it
    # barrier() has to guess from the ambient context and warns that it did.
    dist.init_process_group(
        backend="nccl", device_id=torch.device("cuda", local_rank)
    )
    world_size = int(os.environ["WORLD_SIZE"])
    mesh = init_device_mesh("cuda", (world_size,))
    return int(os.environ["RANK"]), world_size, local_rank, mesh


def fsdp_shard(model, mesh) -> None:
    """Shard the frozen model over `mesh` with FSDP2, in place.

    One shard group per decoder layer plus one for the root, which is the usual
    transformer recipe: it makes the all-gather unit a single layer, so at most
    one layer's parameters are unsharded at a time during the forward, and the
    root group picks up whatever sits outside the stack (embeddings, final norm,
    the vocabulary projection).

    fully_shard is FSDP2, not FSDP1, and the difference matters here: it swaps
    parameters for DTensors but leaves the module tree alone, so
    get_decoder_layers still resolves, Steerer's forward hooks still fire on
    the modules they were registered against, and split_lm still finds `.model`
    and the output embedding. FSDP1's wrapper classes would have broken all
    three.

    The groups have to line up with how the model is actually *called*, not with
    where the parameters live, because the all-gather rides on the sharded
    module's own forward hook. sequence_logprobs deliberately never calls the
    CausalLM wrapper — it runs `body` and `head` separately so the vocabulary
    projection sees only the scored positions — so a shard group on the wrapper
    would never fire and its parameters would still be DTensors when the
    embedding ran. The root group therefore goes on `body`, which is what
    sequence_logprobs actually calls.

    `head` is left replicated on purpose. chunked_logprobs calls it once per
    chunk of scored tokens, and it is the largest single matrix in the model
    (vocab x hidden), so a shard group there would all-gather gigabytes several
    times per micro-batch — and again per chunk during the checkpoint recompute.
    Passing its parameters as ignored_params also covers tied embeddings: when
    `head.weight is embed_tokens.weight`, the body group must not claim it, or it
    would be resharded by the time the projection runs.

    Sharding happens after the weights are loaded and on-device, so `mesh` and
    the parameters must already agree on the GPU — dist_setup's
    torch.cuda.set_device and load_model's device_map={"": local_rank} arrange
    that between them.
    """
    body, head, _ = split_lm(model)
    for layer in get_decoder_layers(model):
        fully_shard(layer, mesh=mesh)
    # The enclosing group last: fully_shard claims the parameters not already
    # spoken for by a sharded submodule, so the layers have to be done first.
    fully_shard(body, mesh=mesh, ignored_params=set(head.parameters()))


def dist_sum_(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Sum `tensor` across ranks in place. A no-op single-process."""
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def even_chunks(
    local: list[int],
    global_idx: list[int],
    micro_batch_size: int,
    world_size: int,
) -> list[tuple[list[int], bool]]:
    """Split one rank's share into micro-batches, same count on every rank.

    Returns (chunk, counted) pairs. `counted` is False for a padding chunk: one
    this rank runs purely to stay in step, whose result the caller must throw
    away.

    Why this exists: every forward pass is a train of FSDP all-gathers, and NCCL
    matches collectives by their sequence number, not by what they are. So if one
    rank runs even one more micro-batch than another, every collective after that
    point is paired with the wrong one — the next rank to leave the loop matches
    its 5-element all-reduce of the stats against a 200MB all-gather of layer
    weights, neither can complete, and the job dies on the watchdog timeout ten
    minutes later with no Python-level error to point at.

    Splitting `global_idx` — by stride or by contiguous block, this is agnostic —
    leaves the shares within one *example* of each other, which is what the loss
    weighting cares about. But one example can be a whole micro-batch: 249 test
    triples over 2 ranks is 125 and 124, which at micro_batch_size 4 is 32 chunks
    and 31. Hence the count here is derived from the *global* length, so every
    rank computes the same number without communicating, and a rank that runs out
    of real work repeats an early chunk rather than leaving the loop early. It is
    an upper bound for both splits, so no real chunk is ever dropped.

    The padding costs one wasted forward on one rank, only when the split is
    uneven, and it keeps the arithmetic exact: nothing is dropped from the data
    and nothing is double-counted, so the result still matches a single-process
    run example for example.
    """
    n_chunks = ceil(ceil(len(global_idx) / world_size) / micro_batch_size)
    chunks = []
    for c in range(n_chunks):
        chunk = local[c * micro_batch_size : (c + 1) * micro_batch_size]
        if chunk:
            chunks.append((chunk, True))
        else:
            chunks.append((global_idx[:micro_batch_size], False))
    return chunks


def reduce_gradient(vectors: torch.Tensor, world_size: int) -> None:
    """Sum the steering-vector gradient across ranks, before the optimizer step.

    FSDP reduces gradients for the parameters it manages; the steering vector is
    not one of them — it is a free leaf tensor replicated on every rank, outside
    the module tree entirely — so its gradient is reduced by hand here. It is
    also the only gradient in the run, the model being frozen.

    Each rank backward()s only its own slice of the optimizer batch but weights
    every micro-batch by that micro-batch's share of the *global* batch, so the
    ranks' partial gradients sum to exactly the gradient of the mean loss over the
    whole batch — SUM, not AVG, is the correct reduction here.

    A rank whose slice came out empty (possible on the last, short batch of an
    epoch when it is smaller than world_size) still runs even_chunks' padding
    micro-batches at weight 0, so its .grad is a genuine zero rather than None.
    The guard below covers the degenerate case where there is nothing to run at
    all: every rank must reach the collective or the whole group hangs.
    """
    if world_size == 1:
        return
    if vectors.grad is None:
        vectors.grad = torch.zeros_like(vectors)
    dist.all_reduce(vectors.grad, op=dist.ReduceOp.SUM)


class NullWriter:
    """Stand-in for SummaryWriter on non-zero ranks.

    All ranks run identical training code, but only rank 0 should touch the event
    files — two processes writing one TensorBoard run corrupts it. Swallowing the
    calls here keeps train_vectors free of rank checks around every scalar.
    """

    def add_scalar(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Steering-vector injection
# --------------------------------------------------------------------------- #


class Steerer:
    """Injects one steering vector per selected layer, sharing a direction d.

    `vectors` is a (n_selected, hidden_size) leaf tensor; row i is added to the
    hidden state of decoder layer `layers[i]` while enabled (the A_L(q) + d*v
    injection). The vector is added only at the token positions marked by the
    injection mask, which encode_example built from `--inject`: the user-turn
    (scenario) tokens, so it steers how the model *reads* the scenario, or the
    assistant-turn ones, so it steers how it *writes* the continuation. A None
    mask injects everywhere. disable() makes every hook a no-op, recovering the
    reference term. All rows share `coef` (the sampled d) so a single loss trains
    the whole set jointly.

    When the model is sharded over several GPUs (device_map="auto") each layer's
    hidden state arrives on that layer's own device, while the vectors are one
    leaf tensor on a single device — so each hook copies what it needs onto the
    hidden state's device. The copy is differentiable, so the gradients from every
    shard land back on the one leaf and a single optimizer state covers all of
    them. The mask is a constant, so its per-device copies are cached for the life
    of one enable() (i.e. one micro-batch's forward *and* backward, since gradient
    checkpointing re-runs the hooks during backward) and dropped on disable() so
    they don't pin memory into the next micro-batch.
    """

    def __init__(self, vectors: torch.Tensor, layers: list[int]):
        self.vectors = vectors  # (n_selected, hidden_size) leaf, requires_grad
        self.layers = layers
        self.enabled = False
        self.coef = 1.0
        self.inj_mask: torch.Tensor | None = None  # (b, seq, 1) float, or None
        self._mask_cache: dict[tuple, torch.Tensor] = {}
        self._handles: list = []

    def enable(self, coef: float, inj_mask: torch.Tensor | None) -> None:
        self.coef, self.inj_mask, self._mask_cache = coef, inj_mask, {}
        self.enabled = True

    def disable(self) -> None:
        self.enabled, self.inj_mask, self._mask_cache = False, None, {}

    def _mask_like(self, hidden: torch.Tensor) -> torch.Tensor:
        """The injection mask on `hidden`'s device and dtype (cached per device)."""
        key = (hidden.device, hidden.dtype)
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = self.inj_mask.to(device=hidden.device, dtype=hidden.dtype)
            self._mask_cache[key] = mask
        return mask

    def _make_hook(self, row: int):
        def hook(module, inputs, output):
            if not self.enabled:
                return output
            # This model build wraps the hidden state in a tuple; some don't.
            hidden = output[0] if isinstance(output, tuple) else output
            add = self.coef * self.vectors[row].to(
                device=hidden.device, dtype=hidden.dtype
            )
            if self.inj_mask is not None:
                # hidden + mask * add, with mask (b, seq, 1) broadcast against
                # add (hidden,): zero everywhere outside the injected turn.
                # addcmul rather than `hidden + self._mask_like(hidden) * add`
                # because the latter allocates the (b, seq, hidden) product and
                # *then* the sum, so every steered layer spikes an extra
                # activation-sized temporary — 32 of them per forward under
                # --layers all. The saved-tensor cost is identical: mul and
                # addcmul both save only the operand the other one's gradient
                # needs, and the mask does not require grad, so `add` is not
                # kept either way. Nor do the numbers move: addcmul rounds once
                # where the pair rounded twice, but the mask holds only 0 and
                # +-1 (dpo_batch folds d into it), so the product is exact at
                # any precision and both forms agree bit for bit.
                hidden = torch.addcmul(hidden, self._mask_like(hidden), add)
            else:
                hidden = hidden + add
            if isinstance(output, tuple):
                return (hidden, *output[1:])
            return hidden

        return hook

    def attach(self, model) -> None:
        for row, layer_idx in enumerate(self.layers):
            layer = get_decoder_layers(model)[layer_idx]
            self._handles.append(layer.register_forward_hook(self._make_hook(row)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# --------------------------------------------------------------------------- #
# Encoding & log-probabilities
# --------------------------------------------------------------------------- #


def encode_example(
    tokenizer,
    context: list[dict] | str,
    continuation: str,
    max_length: int,
    inject: str = "user",
) -> tuple[list[int], int, int, int] | None:
    """Encode system -> context messages -> assistant(continuation).

    `context` is the chat prefix both continuations answer, ending on a user
    turn; a bare string is accepted as a one-message context for the
    single-question datasets.

    Returns (full_ids, inj_start, inj_end, resp_start):
      * [inj_start, inj_end) are the tokens the vector is injected at — with
        inject="user" the content tokens of the *last* user turn (that turn minus
        its role delimiters), with inject="assistant" the continuation tokens,
        which is then the same span the objective scores;
      * [resp_start, len(full_ids)) are the assistant tokens whose log-prob the
        objective scores.

    The last user turn, and only it, however many turns the context has. That is
    the turn that poses the decision, and it is the span
    scripts/try-steering-vector.py and tasks/general_trust_scale/main.py
    reproduce at inference: a vector is only faithful when it is applied over the
    span it was fitted for, so the two have to agree, and the inference side is
    the one that has to work against an arbitrary deployed conversation. Earlier
    turns are context the model reads unsteered — which does mean that a
    benevolence row whose load-bearing claim landed in an earlier turn is read
    unsteered up to the closing turn, and the vector has to work through the KV
    cache that leaves behind.

    The span is found by re-rendering the complete prefix with that one message's
    content emptied and diffing: identical role markers, no content, so the
    common prefix and suffix delimit the content tokens. On a one-message
    context this is the old diff against an empty user turn, token for token, so
    the encoding of the single-turn datasets — and the reference-log-prob cache
    keyed on it — is unchanged. A complete-prefix diff also works for templates
    whose rendering of an earlier tool turn depends on later turns.

    Standing the message in rather than dropping it keeps every role marker in
    place, which is also why there is no system-only render here — some chat
    templates reject a message list with no user turn. Standing in a *character*
    rather than the empty string keeps the whitespace around the content in place
    too: templates strip the content before interpolating it (Llama-3's `| trim`,
    Llama-2's `.strip()`), so an emptied turn can take a neighbouring space or
    newline with it and move an end of the diff that no content sits at. See
    SPAN_SENTINEL.

    resp_start is the length of the same prefix rendered with the assistant
    generation header. Everything is rendered to text then tokenized
    (add_special_tokens=False) so the result is a plain list of ints — some
    processor tokenizers hand back a tokenizers.Encoding from
    apply_chat_template(tokenize=True). Returns None if truncation leaves no
    continuation tokens.

    The assistant span starts *at* resp_start, not before it, so the first scored
    token — predicted from the logits at resp_start-1, inside the generation
    header — stays unsteered. That mirrors generation, where the first token comes
    out of the prefill and only positions the model itself produced carry v.

    Two ways of building the sequence, picked by whether the generation prompt is
    a prefix of the completed turn. It is for every model here except gemma-4,
    whose generation prompt ends on an empty closed thought block that rendering
    a completed assistant message does not reproduce; there the sequence is
    assembled rather than rendered. Either way resp_start is the *end of the
    generation prompt* — the first token the model itself would produce — so the
    two branches score the same thing and only differ in how they get there. See
    the branch for the whole argument.
    """
    if isinstance(context, str):
        context = [{"role": "user", "content": context}]
    system = [{"role": "system", "content": SYSTEM_PROMPT}]
    assistant = {"role": "assistant", "content": continuation}
    prefix = [*system, *context]

    last = max((i for i, m in enumerate(prefix) if m.get("role") == "user"), default=-1)
    if last < 0:
        raise ValueError("context has no user turn to inject on")

    def text_of(messages: list[dict], add_generation_prompt: bool = False) -> str:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def ids_of_text(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def ids_of(messages: list[dict], add_generation_prompt: bool = False) -> list[int]:
        return ids_of_text(text_of(messages, add_generation_prompt))

    base = ids_of(prefix)
    gen_text = text_of(prefix, add_generation_prompt=True)
    full_text = text_of([*prefix, assistant])

    if full_text.startswith(gen_text):
        # The generation prompt is a genuine prefix of the completed turn, so
        # rendering the assistant message reproduces it and resp_start can be
        # read straight off its length. Every model here but gemma-4.
        #
        # Llama-3's `<|start_header_id|>assistant<|end_header_id|>\n\n` is the
        # clean case: the header tokens are special tokens, so the `\n\n` that
        # ends the generation prompt cannot merge with the reply's first token
        # and resp_start lands exactly on it. The Qwen3 models do not manage
        # that — see scripts/check-encode-span.py.
        resp_start = len(ids_of(prefix, add_generation_prompt=True))
        full_ids = ids_of([*prefix, assistant])[:max_length]
    else:
        # gemma-4. Its generation prompt ends on an empty, closed thought block
        # — `<|turn>model\n<|channel>thought\n<channel|>` — because the Gemma 4
        # docs say that with thinking disabled every model but E2B/E4B still
        # emits the tags around an empty thought. Probed on the weights
        # (scripts/gemma-thinking-probe.py) that is exactly what happens: from
        # that prompt the model writes the answer straight after `<channel|>`,
        # and with thinking enabled it emits `<|channel>thought` itself. But
        # rendering a *completed* assistant message emits no channel at all, the
        # block being gated on a truthy `reasoning` key that these rows do not
        # carry — so `apply_chat_template` cannot produce the turn the model
        # actually generates, and the render-only path both built that
        # impossible shape and put resp_start 4 tokens into the reply, silently
        # dropping the opening clause of every continuation from the loss.
        #
        # So the sequence is assembled rather than rendered: generation prompt,
        # the continuation's own tokens, then the closing markers the template
        # puts after an assistant turn. resp_start goes at the *end of the
        # generation prompt*, past the thought tags — the same rule the branch
        # above uses, and for the same reason: it is the first token the model
        # itself produces. The tags are not that. add_generation_prompt=True
        # emits them, so every inference consumer hands them to the model
        # (utils.py::build_steered_prompt and everything downstream), and scoring
        # them asks the objective about a decision — open a thought block or
        # answer now — that deployment never lets the model make.
        #
        # This used to sit at the model header instead, scoring the tags along
        # with the reply, and that cost more than four wasted positions. The tags
        # are the same tokens under the same context in both continuations, so
        # they cancel exactly in the preference margin and left the trained
        # vector alone — but they do NOT cancel in train/reward_{chosen,rejected},
        # which are each a raw log-ratio over the whole scored span. A user-turn
        # injection moves log pi(<channel|>) — "answer without thinking" — hard,
        # and all of it landed there: gemma-4-12B read -53/-64 at beta 0.5 where
        # Qwen3.5-9B reads +0.5/-5.2, which is exactly the shape of the pathology
        # train/reward_chosen exists to expose and --sft exists to price, so the
        # one curve you would read to decide whether to turn --sft on was the one
        # being faked. And with --sft on it stopped being cosmetic: the NLL anchor
        # reads the target's summed log-prob over this same span and divides by
        # its length, so the tags entered the gradient and the normaliser both.
        #
        # Under --inject assistant the injected span starts at resp_start, so
        # this keeps the vector off the tags too. They are prefill positions at
        # generation time, and inference never steers the prefill under
        # `generated_assistant`, so injecting on them here fitted the vector over
        # a span inference cannot reproduce — the one thing --inject exists to
        # prevent.
        #
        # Only the final turn is affected. The template strips `reasoning` from
        # every message before the last user turn (its thinking_gate), so the
        # history's bare `<|turn>model\n` + content is already canonical.
        empty_text = text_of([*prefix, {"role": "assistant", "content": ""}])
        # Turns concatenate (checked below), so the generation prompt and the
        # empty completed turn share a prefix-plus-model-header and then differ:
        # thought tags in one, closing markers in the other. The generation
        # prompt is taken whole, so the only thing that has to be read off a
        # render is the closing markers — no gemma spelling is hardcoded.
        #
        # The header boundary comes from two *completed* turns differing only in
        # their content, not from gen_text against empty_text: commonprefix is
        # character-level, both of those continue with "<" ("<|channel>" and
        # "<turn|>"), and the extra character silently lands inside the tags.
        probe_text = text_of([*prefix, {"role": "assistant", "content": "X"}])
        header_text = os.path.commonprefix([empty_text, probe_text])
        if not gen_text.startswith(header_text):
            raise ValueError(
                "the generation prompt does not share this model's completed "
                "turn header, so the assembled sequence would not line up"
            )
        eot_text = empty_text[len(header_text):]       # "<turn|>\n"

        # History is NOT given the tags back. The Gemma 4 docs are explicit:
        # "In multi-turn conversations, the historical model output should only
        # include the final response. Thoughts from previous model turns must not
        # be added before the next user turn begins" — tool-call turns excepted,
        # which this corpus has none of. The template already does exactly that,
        # stripping `reasoning` from everything before the last user turn, so
        # history needs no help.
        #
        # Restoring an empty block there does raise the reply's likelihood
        # (measured with scripts/gemma-thinking-probe.py --mode multiturn:
        # +0.194 nats/token, 7 of 8 rows). That is not evidence it is right — a
        # prefix whose every model turn has the same shape is simply more
        # predictable, which is a format-consistency effect, and likelihood is
        # not the arbiter of canonical form here. The documented format is.
        gen = ids_of(prefix, add_generation_prompt=True)
        # Same reasoning as the turn-concatenation check below: an index taken
        # from a prefix render only indexes the full sequence if the template
        # concatenates turns, and a template that did not would train a
        # plausible-looking vector off the wrong tokens.
        if base != gen[: len(base)]:
            raise ValueError(
                "the generation prompt does not extend this model's prefix "
                "render, so the assembled sequence would not line up"
            )
        reply = tokenizer(continuation, add_special_tokens=False)["input_ids"]
        full_ids = (gen + reply + ids_of_text(eot_text))[:max_length]
        # full_ids *starts* with gen, so this is exact by construction rather
        # than by a token-boundary assumption: no separate render is tokenized
        # and then indexed into a different sequence.
        resp_start = len(gen)

    if len(full_ids) <= resp_start:
        return None

    if inject == "assistant":
        # From the end of the generation prompt to wherever truncation left off,
        # i.e. exactly the tokens the model itself produces — which is the span
        # inference steers under `generated_assistant`.
        return full_ids, resp_start, len(full_ids), resp_start
    if inject != "user":
        raise ValueError(f"unknown injection span {inject!r}")

    emptied = list(prefix)
    emptied[last] = {**prefix[last], "content": ""}
    span = changed_token_span(base, ids_of(emptied))
    if span is None:
        raise ValueError("the last user turn renders no injectable content tokens")
    inj_start, inj_end = span
    return full_ids, inj_start, inj_end, resp_start


def collate(batch: list[tuple[list[int], int, int, int]], pad_id: int):
    """Right-pad a batch of (full_ids, inj_start, inj_end, resp_start) tensors.

    Returns input_ids (b, seq), attention_mask (b, seq), resp_mask (b, seq), and
    inj_mask (b, seq, 1). resp_mask marks the assistant tokens whose log-prob is
    summed for the sequence log-likelihood; inj_mask marks the tokens where the
    steering vector is added — the last user turn or the assistant one, whichever
    encode_example was asked for. Pad positions are excluded from both.
    """
    seq = max(len(ids) for ids, *_ in batch)
    input_ids = torch.full((len(batch), seq), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), seq), dtype=torch.long)
    resp_mask = torch.zeros((len(batch), seq), dtype=torch.bool)
    inj_mask = torch.zeros((len(batch), seq, 1), dtype=torch.float32)
    for i, (ids, inj_start, inj_end, resp_start) in enumerate(batch):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :n] = 1
        resp_mask[i, resp_start:n] = True
        # inj_end/resp_start were computed pre-truncation; clamp to this sequence.
        inj_mask[i, inj_start : min(inj_end, n), 0] = 1.0
    return (
        input_ids.to(DEVICE),
        attention_mask.to(DEVICE),
        resp_mask.to(DEVICE),
        inj_mask.to(DEVICE),
    )


def split_lm(model):
    """Return (body, head, softcap): decoder stack, vocab projection, logit cap.

    `model(...).logits` is `head(body(...).last_hidden_state)` for every
    architecture here, and running the two halves apart is what lets
    sequence_logprobs project only the positions the objective scores.

    `.model` is the same base module the wrapper's own forward calls, so going
    through it reproduces whatever that forward does before the projection —
    notably the position-id handling a multimodal wrapper does around its text
    stack. get_decoder() is the documented API but on those wrappers it returns
    the text submodule from *inside* that handling, which is why it is only the
    fallback for models that expose no `.model` at all.
    """
    body = getattr(model, "model", None)
    if body is None:
        body = model.get_decoder()
    head = model.get_output_embeddings()
    if head is None:
        head = getattr(model, "lm_head", None)
    if head is None:
        raise RuntimeError("model exposes no output embedding to project with")
    # Anything the wrapper's forward does to the logits has to be reproduced
    # here or the log-probs quietly change: Gemma tanh-caps them, and on the
    # multimodal builds its config keeps the value under text_config.
    cfg = model.config
    softcap = getattr(cfg, "final_logit_softcapping", None)
    if softcap is None:
        text_cfg = getattr(cfg, "text_config", None)
        softcap = getattr(text_cfg, "final_logit_softcapping", None)
    return body, head, softcap


def _project_chunk(head, hidden, targets, softcap):
    """Log-prob of `targets` under head(hidden), for one chunk of scored tokens.

    log_softmax is given dtype=float32 rather than an already-cast tensor: the
    upcast happens inside the kernel, so fp32 logits never exist as an allocation
    of their own. Only the (chunk,) result outlives the call.
    """
    logits = head(hidden)
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    logp = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
    return logp.gather(-1, targets.to(logp.device).unsqueeze(-1)).squeeze(-1)


def chunked_logprobs(head, sel_hidden, sel_targets, softcap) -> torch.Tensor:
    """Project scored hidden states to per-token log-probs, a chunk at a time.

    An (n_resp, vocab) fp32 tensor is by far the largest thing the objective
    touches — at a 256k vocabulary and ~7k scored tokens per micro-batch it is
    7.7 GB, and log_softmax keeps its output alive until backward reads it.
    Chunking the scored tokens so each chunk's fp32 width stays under
    LOGP_CHUNK_BYTES caps that at one chunk, and running each chunk under its own
    checkpoint means even that is not retained: backward recomputes one chunk's
    projection at a time. The cost is a single extra `head` matmul per chunk.

    Checkpointing is skipped where there is no graph to build — the reference
    pass and evaluate() run under no_grad — since there it would only warn and
    add work. RNG state is not preserved: a linear projection and a softmax draw
    none, and saving it would sync the device once per chunk.
    """
    n = sel_hidden.shape[0]
    if n == 0:
        return sel_hidden.new_zeros(0, dtype=torch.float32)
    vocab = getattr(head, "out_features", None)
    if vocab is None:
        vocab = head.weight.shape[0]
    chunk = max(MIN_LOGP_CHUNK, LOGP_CHUNK_BYTES // (vocab * 4))
    pieces = []
    for i in range(0, n, chunk):
        hidden, targets = sel_hidden[i : i + chunk], sel_targets[i : i + chunk]
        if torch.is_grad_enabled() and hidden.requires_grad:
            piece = checkpoint(
                _project_chunk,
                head,
                hidden,
                targets,
                softcap,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            piece = _project_chunk(head, hidden, targets, softcap)
        pieces.append(piece)
    return torch.cat(pieces)


def sequence_logprobs(model, input_ids, attention_mask, resp_mask) -> torch.Tensor:
    """Sum of continuation-token log-probs per sequence: log pi(cont | context).

    Token t is predicted from the hidden state at position t-1, so the response
    mask is shifted by one. The decoder stack and the vocabulary projection run
    separately (split_lm) so the projection sees only the scored positions: the
    prompt is a large share of every padded sequence here and none of its logits
    are ever read, so projecting them allocates — and, in backward, keeps — the
    biggest tensor in the run for nothing. What reaches the softmax is
    (n_resp, hidden), and chunked_logprobs turns it into log-probs without ever
    holding a full (n_resp, vocab) tensor in fp32.

    Devices are read off the tensors rather than assumed: a sharded model hands
    the hidden states back from its last shard, and accelerate runs `head` on
    whichever GPU holds it and returns the result there. The sum comes back on
    the input device, where the reference log-probs and the loss live. Every move
    is a no-op on a single GPU.
    """
    body, head, softcap = split_lm(model)
    out = body(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    # Some builds hand back a bare tuple rather than a ModelOutput.
    hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
    dev = hidden.device
    targets = input_ids[:, 1:].to(dev)
    resp = resp_mask[:, 1:].to(dev)  # (b, seq-1) bool; predicted tokens that count

    # Select the response positions across the whole batch, project only those.
    sel_hidden = hidden[:, :-1, :][resp]  # (n_resp, hidden)
    sel_targets = targets[resp]  # (n_resp,)
    sel_logp = chunked_logprobs(head, sel_hidden, sel_targets, softcap)  # (n_resp,)

    # Scatter each token log-prob back onto its sequence and sum.
    b = input_ids.shape[0]
    rows = torch.arange(b, device=dev).unsqueeze(1).expand_as(resp)[resp]
    out_sum = torch.zeros(b, device=sel_logp.device, dtype=sel_logp.dtype)
    rows = rows.to(sel_logp.device)
    return out_sum.index_add_(0, rows, sel_logp).to(input_ids.device)


def reference_logprobs(
    model,
    encoded: list[tuple[list[int], int, int, int]],
    pad_id: int,
    micro_batch_size: int,
    rank: int = 0,
    world_size: int = 1,
) -> torch.Tensor:
    """Precompute log pi(cont | context) with no steering (v = 0), no grad.

    These are the DPO reference terms; they depend on neither the vectors nor the
    sampled direction, so computing them once and caching amortizes them over all
    epochs. The Steerer must be disabled while this runs. Chunked by
    micro_batch_size: there is no optimizer batch here, only forward passes.

    Under data parallelism each rank scores one contiguous block and writes into
    its own slots of a zero-filled tensor; a single SUM all-reduce then assembles
    the whole thing on every rank, since each slot is written by exactly one rank
    and left zero everywhere else. Every rank needs the full vector — a rank's
    batch slice can reference any row — and this splits the cost N ways to get it.
    """
    out = torch.zeros(len(encoded), device=DEVICE)
    positions = list(range(len(encoded)))
    per_rank = ceil(len(encoded) / world_size)
    block = positions[rank * per_rank : (rank + 1) * per_rank]
    with torch.no_grad():
        # The last rank's block is the short one, which can leave it a whole
        # chunk behind the others; even_chunks pads it back into step.
        for chunk, counted in even_chunks(
            block, positions, micro_batch_size, world_size
        ):
            ids, attn, resp, _ = collate([encoded[k] for k in chunk], pad_id)
            logp = sequence_logprobs(model, ids, attn, resp)
            if counted:
                # A chunk is a contiguous run of a contiguous block.
                out[chunk[0] : chunk[-1] + 1] = logp
    return dist_sum_(out, world_size)


# --------------------------------------------------------------------------- #
# Reference log-prob cache
# --------------------------------------------------------------------------- #


def ref_digest(encoded, model_name: str, dtype: torch.dtype) -> str:
    """Content hash of everything the reference log-probs are a function of.

    Keying on the encoded triples themselves rather than on the flags that
    produced them is what makes the cache safe to trust: the dataset, the chat
    template, SYSTEM_PROMPT, the tokenizer, --num_samples and --max_length all
    reach these numbers only through the token ids and resp_start, so hashing
    those covers every one of them — including the ones that are not command-line
    flags at all and that no version tag would have thought to track. Anything
    that changes them misses and recomputes; nothing produces a stale hit.

    The injection span is deliberately *not* hashed. The reference pass runs with
    the Steerer disabled, so inj_start/inj_end cannot move these numbers, and
    hashing them would make `--inject assistant` recompute a table `--inject
    user` had already written.

    What is not covered is the arithmetic. Batch composition changes the order of
    the reductions inside a forward, so a table written at one --micro_batch_size
    differs in its last digits from one written at another. That is the same
    tolerance a run already has against itself across micro-batch sizes, orders
    below anything beta times a log-ratio can see, so the batching stays out of
    the key — putting it in would cost a full recompute for every memory knob.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(f"bipo-ref-v1|{model_name}|{dtype}".encode())
    for ids, _inj_start, _inj_end, resp_start in encoded:
        h.update(f"|{len(ids)}:{resp_start}:".encode())
        # array() rather than struct.pack(*ids): the ids are the bulk of the
        # hash input (millions of them) and this is the only form that leaves
        # the whole list in C.
        h.update(array("i", ids).tobytes())
    return h.hexdigest()


def ref_cache_file(
    cache_dir: Path | None,
    stem: str,
    side: str,
    encoded,
    model_name: str,
    dtype: torch.dtype,
) -> Path | None:
    """Where this table lives, or None when caching is off.

    The digest alone would be enough to keep two tables apart; the readable stem
    is there so the directory can be pruned by hand — model, dataset and label
    are what someone deleting a stale checkpoint's worth of files reads.
    """
    if cache_dir is None:
        return None
    return cache_dir / f"{stem}_{side}_{ref_digest(encoded, model_name, dtype)}.pt"


def load_ref_cache(
    path: Path, n: int, rank: int, world_size: int
) -> torch.Tensor | None:
    """Read a cached reference table, or None on a miss of any kind.

    Rank 0 alone touches the filesystem and then broadcasts both the verdict and
    the table, so the ranks cannot disagree about whether the cache exists. They
    must not: a rank that decides to recompute enters a long train of FSDP
    all-gathers the others never join, and NCCL matches collectives by sequence
    number, so the job would hang until the watchdog fires rather than fail with
    anything readable. One node seeing a half-written file, or a stale directory
    listing, would be enough — the broadcast makes it impossible by construction
    instead of unlikely.

    Any exception is a miss, and so is a tensor of the wrong length. A truncated
    or half-synced file is exactly what torch.load raises on, and recomputing
    costs minutes where a wrong table would silently corrupt every reward in the
    run — the reference term is subtracted from every log-prob the objective
    sees, so a bad one does not fail, it just trains the wrong vector.
    """
    ok = torch.zeros(1, dtype=torch.long, device=DEVICE)
    out = torch.zeros(n, device=DEVICE)
    if rank == 0:
        try:
            cached = torch.load(path, map_location="cpu")
        except Exception:
            cached = None
        if isinstance(cached, torch.Tensor) and cached.shape == (n,):
            out.copy_(cached)
            ok[0] = 1
    if world_size > 1:
        dist.broadcast(ok, src=0)
        if ok.item():
            dist.broadcast(out, src=0)
    return out if ok.item() else None


def save_ref_cache(path: Path, values: torch.Tensor) -> None:
    """Write the table for the next run. Rank 0 only, atomically.

    Through a pid-suffixed temp file and a rename, because a lambda sweep runs
    several jobs over one cache directory at once and they all compute the same
    table: os.replace is atomic, so a concurrent reader sees either the old file
    or the new one, never a partial tensor. Two jobs racing to write the same
    path is fine — the contents are identical and the loser's rename simply wins
    or loses whole.

    Saved on CPU so the file does not carry a device with it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(values.detach().cpu(), tmp)
    os.replace(tmp, path)


def cached_reference_logprobs(
    model,
    encoded: list[tuple[list[int], int, int, int]],
    pad_id: int,
    micro_batch_size: int,
    cache_file: Path | None,
    rank: int = 0,
    world_size: int = 1,
) -> torch.Tensor:
    """reference_logprobs, memoized on disk.

    The table is a function of the model and the encoded triples and of nothing
    else in the configuration — not the vectors, the layers, the span, the
    penalties, beta, lr, the seed or the split — so every run of a sweep over any
    of those recomputes an identical one. That is not a rounding error: it is two
    no-grad passes over the whole dataset, in front of step 1 and printing
    nothing while it runs. On 5k triples over two H100s it takes about as long
    again as the training it precedes.
    """
    if cache_file is not None:
        hit = load_ref_cache(cache_file, len(encoded), rank, world_size)
        if hit is not None:
            if rank == 0:
                print(f"  reference log-probs: cache hit, {cache_file}")
            return hit
    if rank == 0:
        print(
            f"  reference log-probs: computing {len(encoded)} sequences...",
            flush=True,
        )
    start = perf_counter()
    out = reference_logprobs(
        model, encoded, pad_id, micro_batch_size, rank, world_size
    )
    if rank == 0:
        msg = f"  reference log-probs: {perf_counter() - start:.0f}s"
        if cache_file is not None:
            save_ref_cache(cache_file, out)
            msg += f", cached to {cache_file}"
        print(msg)
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def dpo_batch(
    model, steerer, enc_t, enc_o, ref_t, ref_o, pad_id, idx, d, beta, sft=0.0
):
    """Run one steered batch, returning (loss, sft_term, reward_t, reward_o).

    The two rewards are the DPO implicit rewards, per example: beta times the
    log-ratio of the steered to the unsteered (reference) probability of one
    continuation. `reward_t` is the target continuation's — DPO's "chosen" — and
    `reward_o` the opposite one's, DPO's "rejected". Their difference is the
    preference logit the objective is built on: its sign is the reward preference
    (>0 == the vector moves probability the intended way) and -logsigmoid of it
    is the per-example loss.

    Both carry the sampled direction d, exactly as the objective does, so a d=-1
    batch — where the injection is -v and the target continuation is *meant* to
    become less likely — is reported on the same scale as a d=+1 one. Without
    that factor the two directions would straddle zero and averaging over a run
    would say nothing.

    `sft_term` is the NLL half of DPO+NLL, alpha times the length-normalised NLL
    of the target continuation read at +v (see below). It is returned apart from `loss` rather than folded into it so that the
    preference number stays comparable across alphas: the caller sums the two for
    backward and logs them on separate curves. Zero (and outside the graph) when
    alpha is 0, so the default path is the pure BiPO one it has always been.

    The anchor is always r_T read at +v, whatever d the batch drew, so it is one
    quantity per step and train/sft is a curve that converges. On a d=+1 batch
    that costs nothing — the preference forward already ran at +v, and its target
    half *is* the anchor's log-prob. On a d=-1 batch the preference forward ran
    at -v, so the target has to be scored a second time at +v, and this function
    appends a third block of b sequences to do it.

    That third block goes into the *same* forward rather than a second one,
    because two forwards under two enable() calls would break under gradient
    checkpointing exactly as the note below describes: the recompute during
    backward would inject whichever mask was set last into both. The sign is
    folded into the injection mask instead — the hook computes coef * v * mask
    with a per-position float mask, so the preference rows carry d in their mask
    and the anchor rows carry +1, under one enable(1.0, ...). With d = +-1 and a
    0/1 mask every product is exact, so a d=+1 batch and a --sft 0 batch are
    unchanged bit for bit.

    The cost is real and lands unevenly: a d=-1 micro-batch forwards 3b sequences
    where a d=+1 one forwards 2b, so peak memory on those steps is 1.5x and a
    run that was near the edge wants a smaller --micro_batch_size.

    Lengths come from the response mask rather than from the sequence, since the
    two continuations of a pair are padded to a common width and only the masked
    positions are scored; the mask is shifted by one exactly as it is inside
    sequence_logprobs, so the count is over the same tokens the log-prob summed.
    The anchor rows are copies of the target rows in one padded batch, so their
    counts agree by construction.

    Target and opposite are run as one combined forward. Their injection masks
    differ (different sequence lengths), and with gradient checkpointing the hook
    is re-run during backward() — so a single forward under one enable() is the
    only way the recompute injects the right mask for each. The steerer is left
    enabled afterward (backward runs in the caller); the caller disables it.
    """
    b = len(idx)
    batch = [enc_t[i] for i in idx] + [enc_o[i] for i in idx]
    # d=-1 leaves the preference forward at -v, which is not where the anchor
    # reads; re-score the target at +v in the same pass.
    anchor_pass = bool(sft) and d < 0
    if anchor_pass:
        batch = batch + [enc_t[i] for i in idx]
    ids, attn, resp, inj = collate(batch, pad_id)
    sel = torch.tensor(idx, device=DEVICE)
    r_t, r_o = ref_t[sel], ref_o[sel]

    if anchor_pass:
        # Sign into the mask, not into coef, so both injections live in one
        # forward: preference rows get d*v, anchor rows +v. Exact for d = +-1.
        inj = inj.clone()
        inj[: 2 * b] *= d
        steerer.enable(1.0, inj)
    else:
        steerer.enable(d, inj)
    logp = sequence_logprobs(model, ids, attn, resp)
    logp_t, logp_o = logp[:b], logp[b : 2 * b]

    reward_t = d * beta * (logp_t - r_t)
    reward_o = d * beta * (logp_o - r_o)
    loss = -F.logsigmoid(reward_t - reward_o).mean()

    if sft:
        n_tok = resp[:, 1:].sum(dim=1).clamp(min=1).to(logp.dtype)
        # Always the target at +v: its own block on a d=-1 batch, and the
        # preference forward's target half on a d=+1 one, which already is +v.
        anchor_logp = logp[2 * b :] if anchor_pass else logp_t
        anchor_n = n_tok[2 * b :] if anchor_pass else n_tok[:b]
        sft_term = -sft * (anchor_logp / anchor_n).mean()
    else:
        sft_term = torch.zeros((), device=loss.device, dtype=loss.dtype)
    return loss, sft_term, reward_t, reward_o


@torch.no_grad()
def evaluate(
    model,
    steerer,
    enc_t,
    enc_o,
    ref_t,
    ref_o,
    pad_id,
    idx,
    beta,
    micro_batch_size,
    sft: float = 0.0,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[float, float, float, float, float]:
    """Mean loss, reward accuracy and the two mean rewards over `idx` (no grad).

    Returns (loss, accuracy, mean chosen reward, mean rejected reward, mean sft
    term). Averaged over both directions d in {-1, +1} so the estimate does not
    hinge on a single sampled sign; accuracy is the fraction of examples whose
    chosen reward beats their rejected one. Chunked by micro_batch_size — every
    chunk is scored independently, so the result does not depend on how the split
    falls.

    The loss returned is the preference term alone, whatever alpha is, so a test
    curve compares across runs that weight the supervised term differently; the
    supervised term itself comes back beside it. Averaging it over both
    directions is what makes it symmetric — each direction scores the pole it
    prefers, so the pair covers both poles of every row exactly once. The sft term does
    not depend on d — it is the target at +v either way — so averaging leaves it
    unchanged; what the d=-1 half costs is the anchor's extra forward.

    Under data parallelism each rank scores a stride of `idx` and the running
    totals are summed across ranks before the division, so the numbers come back
    identical on every rank and identical to a single-process run. The stride is
    chunked through even_chunks: a padding chunk is scored and then dropped, so it
    moves neither the totals nor the count they are divided by.
    """
    local = idx[rank::world_size]
    total_loss, total_correct, total_chosen, total_rejected = 0.0, 0.0, 0.0, 0.0
    total_sft, total = 0.0, 0.0
    for batch_idx, counted in even_chunks(local, idx, micro_batch_size, world_size):
        for d in (-1.0, 1.0):
            loss, sft_term, reward_t, reward_o = dpo_batch(
                model, steerer, enc_t, enc_o, ref_t, ref_o,
                pad_id, batch_idx, d, beta, sft,
            )
            if not counted:
                continue
            total_loss += loss.item() * len(batch_idx)
            total_correct += float(((reward_t - reward_o) > 0).sum().item())
            total_chosen += float(reward_t.sum().item())
            total_rejected += float(reward_o.sum().item())
            total_sft += sft_term.item() * len(batch_idx)
            total += len(batch_idx)
    steerer.disable()
    stats = torch.tensor(
        [total_loss, total_correct, total_chosen, total_rejected, total_sft, total],
        device=DEVICE,
    )
    dist_sum_(stats, world_size)
    n = stats[5]
    return tuple((stats[i] / n).item() for i in range(5))


def linear_warmup_scale(step: int, warmup_steps: int) -> float:
    """Linear ramp from 0 to 1 over the first `warmup_steps` steps, 1 after.

    Same shape and the same step convention as cosine_warmup's warmup branch —
    `step` is the number of *completed* optimizer steps, so the first step of a
    run scores 0 and the two ramps stay in lockstep when they share a fraction.
    A warmup of 0 steps is the constant 1, i.e. no warmup.
    """
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


def cosine_warmup(
    optimizer, warmup_frac: float, total_steps: int, min_lr_frac: float = 0.1
) -> LambdaLR:
    """Linear warmup over the first `warmup_frac` of the run, then cosine decay.

    The warmup length is a *fraction* of total_steps rather than an absolute step
    count, so it stays the same share of the schedule when the dataset size, the
    batch size or the epoch count change — an absolute count silently becomes the
    whole run on a short one and a rounding error on a long one.

    The cosine floors at `min_lr_frac * lr` instead of 0, so the last steps still
    move the vector rather than freezing it at whatever the schedule happened to
    reach; the warmup still ramps from 0 up to the full lr.
    """
    if not 0.0 <= warmup_frac < 1.0:
        raise ValueError(f"warmup_frac must be in [0, 1), got {warmup_frac}")
    if not 0.0 <= min_lr_frac <= 1.0:
        raise ValueError(f"min_lr_frac must be in [0, 1], got {min_lr_frac}")
    warmup_steps = round(warmup_frac * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())
        return min_lr_frac + (1.0 - min_lr_frac) * cosine

    return LambdaLR(optimizer, lr_lambda)


def gpu_peak_gb(rank: int = 0, world_size: int = 1) -> list[float]:
    """Peak reserved bytes per GPU, in GiB.

    Reserved (not allocated) because that is what nvidia-smi shows. One entry per
    GPU the run is actually using, so the sum is the node's footprint and reading
    only the current device would hide most of it.

    Data parallel, the list is indexed by rank: every rank can *see* all the GPUs
    but drives only its own, so enumerating visible devices would have each rank
    report its peers' memory as if it were its own. Each rank fills its own slot
    and one all-reduce assembles the row. Single-process, the list is indexed by
    visible device instead, which is what reports a device_map="auto" shard.
    """
    if world_size > 1:
        peaks = torch.zeros(world_size, device=DEVICE)
        peaks[rank] = torch.cuda.max_memory_reserved(DEVICE.index) / 1024**3
        return dist_sum_(peaks, world_size).tolist()
    return [
        torch.cuda.max_memory_reserved(i) / 1024**3
        for i in range(torch.cuda.device_count())
    ]


def reset_gpu_peak() -> None:
    """Reset the peak-memory counters on every visible GPU."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


_ENERGY_WARNED = False


def _nvml_handle(index: int):
    """NVML handle for torch device `index`, or None if NVML is unreachable.

    Resolved through torch's own helper — the one its public power_draw() uses —
    because NVML enumerates physical boards while torch enumerates visible ones,
    so under the CUDA_VISIBLE_DEVICES a scheduler hands a job the two orderings
    differ and a raw nvmlDeviceGetHandleByIndex would address a neighbour's GPU.
    """
    try:
        return torch.cuda._get_pynvml_handler(index)
    except Exception:  # no driver, no NVML, torch dropping the private API
        return None


def _energy_handle(index: int):
    """NVML handle for torch device `index`, or None if it can't report energy.

    The counter is read once here so an unsupported board (pre-Volta, MIG, some
    vGPU setups) is discovered at setup and warned about once, rather than raising
    somewhere in the middle of a multi-hour run. Separate from _nvml_handle
    because the energy counter is the one thing a board may lack while still
    answering everything else — its power cap included.
    """
    global _ENERGY_WARNED
    handle = _nvml_handle(index)
    try:
        if handle is None:
            raise RuntimeError("NVML unavailable")
        pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        return handle
    except Exception as e:
        if not _ENERGY_WARNED:
            print(f"[energy] cuda{index}: no energy counter, logging 0 ({e})")
            _ENERGY_WARNED = True
        return None


def gpu_capacity(
    rank: int = 0, world_size: int = 1
) -> tuple[str, list[float], list[float]]:
    """(device name, total GiB per GPU, power cap in W per GPU).

    The denominators for the gpu_mem_gb and gpu_power_w curves, which are both
    sums over the same GPUs — so the totals here are what those curves are read
    against, and the header is where a reader finds them.

    Indexed like gpu_peak_gb, and for the same reason: data parallel, a rank sees
    only the GPU it drives and the rows are assembled by rank, which is what keeps
    the count right when the launcher gives each rank its own CUDA_VISIBLE_DEVICES
    rather than the whole node; single-process, the rows are the visible devices.

    The name is device 0's and assumes the node is homogeneous, as an allocation
    on this cluster is. The numbers are per device and summed, so a heterogeneous
    node would still total correctly — only the label would be too narrow.

    The cap is the *enforced* limit, which is what the board actually clamps to,
    rather than the board default an admin may have lowered. 0.0 W means NVML
    would not say.
    """
    if world_size > 1:
        rows = torch.zeros(2, world_size, device=DEVICE)
        prop = torch.cuda.get_device_properties(DEVICE.index)
        rows[0, rank] = prop.total_memory / 1024**3
        rows[1, rank] = _power_cap_w(DEVICE.index)
        dist_sum_(rows, world_size)
        return prop.name, rows[0].tolist(), rows[1].tolist()
    props = [
        torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())
    ]
    return (
        props[0].name if props else "no CUDA device",
        [p.total_memory / 1024**3 for p in props],
        [_power_cap_w(i) for i in range(len(props))],
    )


def _power_cap_w(index: int) -> float:
    """Enforced power limit in W for torch device `index`, 0.0 if unknown."""
    handle = _nvml_handle(index)
    try:
        return pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
    except Exception:
        return 0.0


class EnergyMeter:
    """Energy drawn per step, differenced off NVML's cumulative counter.

    nvmlDeviceGetTotalEnergyConsumption is a monotonic millijoule counter the
    board keeps (Volta and newer, so H100), not a reading of the power sensor:
    differencing it across a step gives that step's energy *exactly*, where a
    per-step call to power_draw would be one point sample of a 1s-ish rolling
    window and would miss whatever happened between the samples.

    The difference is taken before any reduction. The counter runs since the last
    driver load and reaches ~1e12 mJ, which a float32 all-reduce would round to
    noise; one step's delta is ~1e3 J and survives it.

    It meters the whole physical board, so it includes idle draw — honest for an
    electricity figure, but it means a stalled GPU still accrues joules — and, if
    a job ever shares a GPU, a neighbour's work.

    A board without the counter reports 0.0 forever rather than raising. This is
    telemetry: it must not be able to kill a run, nor to desynchronize one, which
    is why a failure never changes the control flow around the collective.
    """

    def __init__(self, rank: int = 0, world_size: int = 1) -> None:
        # Same two indexing conventions as gpu_peak_gb, for the same reasons: data
        # parallel, a rank drives exactly one GPU and the row is indexed by rank;
        # single-process, it is indexed by visible device, which is what reports a
        # device_map="auto" shard.
        self.rank = rank
        self.world_size = world_size
        devices = (
            [DEVICE.index] if world_size > 1 else list(range(torch.cuda.device_count()))
        )
        self.handles = [_energy_handle(i) for i in devices]
        self.total_j = [0.0] * (world_size if world_size > 1 else len(devices))
        self.prev = self._counters()

    def _counters(self) -> list[float]:
        """Cumulative joules for this rank's GPUs; 0.0 for any that can't say."""
        out = []
        for handle in self.handles:
            try:
                out.append(
                    0.0
                    if handle is None
                    else pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0
                )
            except pynvml.NVMLError:
                out.append(0.0)
        return out

    def delta_j(self) -> list[float]:
        """Joules since the previous call, per GPU, and accumulate the total."""
        now = self._counters()
        # Clamped at 0: the counter only decreases if the driver reloaded under
        # the run or a read failed after a good one, and both should cost the
        # curve one step rather than a large negative spike.
        delta = [max(0.0, n - p) for n, p in zip(now, self.prev)]
        self.prev = now
        if self.world_size > 1:
            row = torch.zeros(self.world_size, device=DEVICE)
            row[self.rank] = delta[0]
            delta = dist_sum_(row, self.world_size).tolist()
        self.total_j = [t + d for t, d in zip(self.total_j, delta)]
        return delta


# Smoothing under the row-norm square root. sqrt(0) is finite but its derivative
# is not — x.norm() backward is x/||x|| — and the vectors are zero-initialised, so
# the unsmoothed form is NaN on the first step. With the epsilon the row norm is
# differentiable everywhere and its gradient vanishes at the origin, which is also
# what leaves an untrainable row (the top layer under --inject user, whose
# gradient from the loss is structurally zero) sitting at exactly zero rather than
# being dragged around by a penalty.
NORM_EPS = 1e-12


def sum_row_norms(vectors: torch.Tensor) -> torch.Tensor:
    """Group-lasso penalty over the layer rows: sum_L sqrt(||v_L||^2 + eps).

    An L1 over the per-layer norms, an L2 within each row. Unlike hoyer_square it
    is not scale-invariant — it shrinks the total norm as well as concentrating
    it — and it is convex, which the ratio is not.
    """
    return torch.sqrt(vectors.pow(2).sum(dim=1) + NORM_EPS).sum()


@torch.no_grad()
def prox_group_lasso_(vectors: torch.Tensor, threshold: float) -> None:
    """Group soft-threshold in place: v_L <- v_L * max(0, 1 - threshold/||v_L||).

    The proximal operator of lambda * sum_L ||v_L|| at step size lr, so the
    caller passes threshold = lambda * lr. Rows shorter than the threshold land
    on exactly zero, which is the point of using it: with the penalty in the
    gradient instead, Adam's per-coordinate preconditioning leaves them small but
    never zero, and "this vector uses three layers" turns into a cutoff picked
    after the fact.

    Not the exact prox for AdamW — that one would carry the preconditioner — but
    it is decoupled from the gradient in the same way AdamW's own weight decay
    is, and it is how proximal steps are applied under adaptive optimizers in
    practice. A row already at zero stays there.

    What the approximation costs is worth knowing before tuning lambda. A plain
    group lasso selects by comparing each row's *gradient magnitude* against
    lambda; Adam divides that magnitude out, so what survives here is instead
    decided by each row's gradient *consistency* — m/sqrt(v) is close to 1 for a
    row whose gradient keeps pointing the same way and small for a noise-dominated
    one, and it is that, times lr, that the threshold competes against. Two
    consequences: a full-batch, noiseless setting is all-or-nothing (every row
    moves at the same speed, so they live or die together), and lambda is not in
    loss units. It buys a shrink of lambda * lr per step, so lambda * lr *
    total_steps against a typical row norm is the quantity to reason about —
    which puts the useful range nowhere near a weight-decay-like 1e-3.
    """
    norms = vectors.norm(dim=1, keepdim=True)
    # The zero comes from the comparison, not from trusting 1 - threshold/norm to
    # land on it: a row whose norm sits within an ulp of the threshold otherwise
    # comes out at ~1e-7 of itself — numerically dead, but still counted a live
    # row by vector_norm/nonzero_rows. Both sides read the same `norms`, so no
    # second norm computation can disagree with this one either.
    shrunk = 1 - threshold / norms.clamp_min(NORM_EPS)
    vectors.mul_(torch.where(norms > threshold, shrunk, torch.zeros_like(shrunk)))


def hoyer_square(vectors: torch.Tensor) -> torch.Tensor:
    """Group Hoyer-square over the layer rows: (sum_L ||v_L||)^2 / sum_L ||v_L||^2.

    The penalty is differentiable, scale-invariant and bounded in [1, n_rows] —
    1 when one row holds the whole vector, n_rows when they all hold an equal
    norm. See the module docstring for what that buys and what it costs.

    A single trained layer makes it the constant 1: the useful configuration is
    --layers all, or at least several. It reaches no exact zeros — being smooth
    it has no kink at the origin to pin a row against; sum_row_norms does.
    """
    row_norms = torch.sqrt(vectors.pow(2).sum(dim=1) + NORM_EPS)
    return row_norms.sum().pow(2) / (row_norms.pow(2).sum() + NORM_EPS)


def effective_layers(vectors: torch.Tensor) -> float:
    """Participation ratio of the per-layer energies s_L = ||v_L||^2.

    (sum_L s_L)^2 / sum_L s_L^2 — the same functional as hoyer_square but over
    energies rather than norms, so the two are not the same number and this one
    is the harsher of the pair. Scale-free, and reads as the number of layers the
    vector is actually using: 1 when one layer holds all the energy, n_rows when
    it is spread evenly. Zero only at the zero init, where there is no energy to
    distribute.
    """
    energy = vectors.detach().pow(2).sum(dim=1)
    total = energy.sum()
    if total <= 0:
        return 0.0
    return (total.pow(2) / energy.pow(2).sum()).item()


def log_vector_norms(writer, steerer: Steerer, step: int) -> None:
    """Log the L2 norm of each trained layer's steering vector, and their spread.

    effective_layers rides in the same vector_norm/ section because it is a
    reduction of exactly these numbers — the one curve that says whether the norm
    is concentrating or smearing, without reading n_selected separate ones.
    """
    for row, layer_idx in enumerate(steerer.layers):
        norm = steerer.vectors[row].norm().item()
        writer.add_scalar(f"vector_norm/L{layer_idx}", norm, step)
    writer.add_scalar(
        "vector_norm/effective_layers", effective_layers(steerer.vectors), step
    )
    # Exact zeros only ever come from the proximal group lasso, so this curve is
    # flat under everything else — which is itself the statement that nothing is
    # being switched off. A row untrainable by construction (the top layer under
    # --inject user) is counted as zero, because it is.
    nonzero = int((steerer.vectors.detach().norm(dim=1) > 0).sum().item())
    writer.add_scalar("vector_norm/nonzero_rows", nonzero, step)


def build_steering(
    vectors: torch.Tensor, layers: list[int], num_layers: int, hidden_size: int
) -> torch.Tensor:
    """Scatter the trained rows into a full (num_layers, hidden_size) tensor.

    Untrained layers stay zero, so the result is layer-indexable and shape-matches
    steering-vector-mean.py. Detached to CPU for saving.
    """
    steering = torch.zeros(num_layers, hidden_size, dtype=torch.float32)
    for row, layer_idx in enumerate(layers):
        steering[layer_idx] = vectors[row].detach().cpu()
    return steering


def train_vectors(
    model,
    steerer: Steerer,
    enc_t: list[tuple[list[int], int, int, int]],
    enc_o: list[tuple[list[int], int, int, int]],
    ref_t: torch.Tensor,
    ref_o: torch.Tensor,
    pad_id: int,
    train_idx: list[int],
    test_idx: list[int],
    *,
    beta: float,
    sft: float = 0.0,
    lr: float,
    weight_decay: float,
    group_lasso: float = 0.0,
    group_lasso_prox: bool = True,
    hoyer: float = 0.0,
    hoyer_warmup_frac: float | None = None,
    epochs: int,
    batch_size: int,
    micro_batch_size: int,
    warmup_frac: float,
    min_lr_frac: float,
    rng: random.Random,
    label: str,
    writer: SummaryWriter,
    eval_every: int = 0,
    checkpoint_fn=None,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """Optimize steerer.vectors in place with the bi-directional DPO objective.

    One optimizer step consumes `batch_size` triples, run through the model
    `micro_batch_size` at a time with the gradients accumulating in between. A
    "step" therefore counts optimizer updates, not forward passes, so the LR
    schedule and every logged curve are invariant to micro_batch_size.

    Data parallel, that batch is additionally split across `world_size` ranks and
    the gradients summed before the step, so a step still consumes exactly
    `batch_size` triples and the curves are invariant to world_size too. Every
    rank walks the same batches in the same order with the same d — the rng is
    seeded identically and consumed in lockstep — and takes the same number of
    steps and the same number of micro-batches within each (see even_chunks),
    which is what keeps the replicas in step and every collective matched.

    Every curve is logged per step, and the progress bar counts steps too: an
    epoch is hundreds of them on a dataset of this size, so anything epoch-
    granular reports a run that looks frozen and leaves a handful of points to
    read a trend off. The epoch survives as a readout in the bar's postfix.

    The one diagnostic that is not free is the held-out evaluation — it forwards
    the whole test split once per direction d, several steps' worth of compute —
    so it runs every `eval_every` steps, or once per epoch when that is 0. The
    final step is always evaluated, whatever the stride, so the vector the caller
    saves has a test number beside it.

    The NLL term of DPO+NLL goes the other way round — it is a function of the
    *data*, so it belongs inside dpo_batch with the preference term and is
    weighted by the same micro-batch share, which makes the accumulated gradient
    the gradient of the mean over the whole global batch for both halves alike.
    It is kept out of train/loss and test/loss, which stay the pure preference
    number so a run with --sft is readable against one without, and logged on
    train/sft and test/sft — alpha-scaled, so how much of the objective it is
    carrying is one chart rather than an arithmetic exercise. Both are single-valued: the anchor
    reads the target at +v whatever d the batch drew, so neither curve alternates
    and both converge.

    The two row penalties are applied once per optimizer step, never inside
    dpo_batch: each is a function of the parameter alone, so a per-micro-batch
    version would be counted once per micro-batch, and reduce_gradient would then
    sum it once per rank on top. Both go on after that all-reduce, where every
    rank already holds the same vectors and the same gradient, so a rank-local
    backward adds a term exactly once and the replicas stay identical.

    `hoyer_warmup_frac` ramps the hoyer weight linearly from 0 to `--hoyer` over
    that fraction of the run (None follows `warmup_frac`, so by default the two
    ramps share a window; 0 turns it off and restores the flat weight). Only
    hoyer gets one, and the reason is that it is the only term here the LR warmup
    does not already damp:

      * the proximal group lasso shrinks by `group_lasso * lr_now` per step, so
        its threshold is *already* the LR ramp times lambda and it is inert while
        the LR is;
      * hoyer instead goes through .grad, and Adam's update direction is
        m/sqrt(v) — the ratio between the penalty's gradient and the preference
        term's is inside m, where lr cannot reach it. The LR ramp scales the size
        of the step the two have already agreed on, not which of them chose it.

    And that ratio is worst exactly at the start. hoyer_square is degree-0
    homogeneous, so its gradient scales as 1/||V||: at the row norms a run has
    after one step (~sqrt(hidden) * lr, about 0.03) it is ~30x what it is at unit
    norm. The zero init itself is safe — every row norm is equal there, which is
    a stationary point, so the gradient is exactly zero — but the step that
    breaks that symmetry hands the penalty a near-singular pull over whichever
    row the first preference gradient happened to make longest, and it can lock
    in a concentration before the preference term has said anything about which
    layers matter. The ramp buys those steps back.

    Those same steps are the checkpoint steps: `checkpoint_fn`, if given, is
    invoked with the just-finished step number every time an evaluation is due.
    Pairing the two means every vector on disk has a test loss logged at exactly
    its own step, so choosing between checkpoints afterwards is reading one curve
    rather than aligning two cadences. A step counts as an evaluation step whether
    or not the evaluation actually ran, so checkpoints keep coming under
    --test_frac 0. The final step is left to the caller's end-of-training save, so
    it is not checkpointed here (that would write the same tensor twice).
    """
    optimizer = torch.optim.AdamW(
        [steerer.vectors], lr=lr, weight_decay=weight_decay
    )
    n = len(train_idx)
    steps_per_epoch = (n + batch_size - 1) // batch_size
    total_steps = epochs * steps_per_epoch
    scheduler = cosine_warmup(optimizer, warmup_frac, total_steps, min_lr_frac)
    # None means "share the LR's window", which is the coherent default: both are
    # answering "how long before this term is at full strength". Resolved here
    # rather than in main() so the fallback is stated once, next to the schedule
    # it falls back to.
    if hoyer_warmup_frac is None:
        hoyer_warmup_frac = warmup_frac
    if not 0.0 <= hoyer_warmup_frac < 1.0:
        raise ValueError(
            f"hoyer_warmup_frac must be in [0, 1), got {hoyer_warmup_frac}"
        )
    hoyer_warmup_steps = round(hoyer_warmup_frac * total_steps)
    # 0 keeps the cadence this used to have — one evaluation per epoch — without
    # making the caller work out how many steps that is.
    eval_stride = eval_every or steps_per_epoch

    step = 0
    test_loss = None
    log_vector_norms(writer, steerer, step)
    # Both are read once per step at the very end of the iteration, so each window
    # runs boundary to boundary and the run's steps tile its wall time and its
    # energy without gap or overlap. Started here, after the model and the
    # reference log-probs are already in place, so setup lands in neither.
    energy = EnergyMeter(rank, world_size)
    t_boundary = perf_counter()
    # Over optimizer steps, not epochs: one epoch is steps_per_epoch of these and
    # can be tens of minutes, so an epoch-unit bar shows 0% for most of the run.
    pbar = tqdm(total=total_steps, desc=f"{label} bipo", disable=rank != 0)
    for epoch in range(epochs):
        rng.shuffle(train_idx)
        for start in range(0, n, batch_size):
            idx = train_idx[start : start + batch_size]
            # One direction d ~ U{-1, +1} per batch (Algorithm 1, line 4), shared
            # across every trained layer — and across the micro-batches, which
            # split one batch for memory rather than being batches of their own.
            d = rng.choice((-1.0, 1.0))
            # ...and shared across ranks, which see this same `idx` and this same
            # d and then each take a stride of it. Striding rather than slicing
            # keeps the shards within one example of each other however batch_size
            # and world_size divide; a rank's stride is empty only when the batch
            # itself is smaller than world_size. One example can still be a whole
            # micro-batch, though, so the chunking goes through even_chunks.
            local = idx[rank::world_size]

            optimizer.zero_grad()
            loss_sum, sft_sum = 0.0, 0.0
            batch_chosen, batch_rejected = [], []
            for micro, counted in even_chunks(
                local, idx, micro_batch_size, world_size
            ):
                loss, sft_term, reward_t, reward_o = dpo_batch(
                    model, steerer, enc_t, enc_o, ref_t, ref_o,
                    pad_id, micro, d, beta, sft,
                )
                # dpo_batch returns the mean over its own micro-batch, so weight
                # by that micro-batch's share of the step — the *global* step, so
                # the denominator is len(idx) and not len(local). Summing the
                # accumulated gradient over ranks is then exactly the gradient of
                # the mean over all of `idx`, including when the last micro-batch
                # is short or a rank's stride is empty. A padding micro-batch
                # carries weight 0: it runs the same forward and the same backward
                # as a real one — the point is the collectives they issue — and
                # contributes exactly nothing to the gradient or the metrics.
                weight = len(micro) / len(idx) if counted else 0.0
                # The optimized objective is the sum of the two; only the
                # preference half is reported as train/loss (see the docstring).
                ((loss + sft_term) * weight).backward()
                # backward() (incl. any checkpoint recompute) is done; stop
                # injecting and drop the mask (and its per-device copies) so it
                # doesn't pin this micro-batch's memory through the next one.
                steerer.disable()
                if counted:
                    loss_sum += loss.item() * len(micro)
                    sft_sum += sft_term.item() * len(micro)
                    batch_chosen.append(reward_t.detach())
                    batch_rejected.append(reward_o.detach())
                del loss, sft_term, reward_t, reward_o

            reduce_gradient(steerer.vectors, world_size)
            # Row penalties, once per step and after the all-reduce (see above).
            # Both are left out of train/loss, which stays the pure DPO number,
            # and logged on their own curves further down.
            penalty_lasso = penalty_hoyer = None
            if group_lasso:
                if group_lasso_prox:
                    # The proximal form keeps the penalty out of the gradient
                    # entirely: a gradient step on the smooth part followed by the
                    # operator *is* proximal gradient descent, and doing both
                    # would count the term twice. Its value is still worth a
                    # curve, so it is evaluated without a graph.
                    with torch.no_grad():
                        penalty_lasso = (
                            group_lasso * sum_row_norms(steerer.vectors)
                        ).item()
                else:
                    term = group_lasso * sum_row_norms(steerer.vectors)
                    term.backward()
                    penalty_lasso = term.item()
            hoyer_scale = None
            if hoyer:
                # `step` is still the count of *completed* steps here (it is
                # incremented below), which is the same convention the LR
                # schedule reads, so a shared fraction puts the two ramps on the
                # same steps.
                hoyer_scale = linear_warmup_scale(step, hoyer_warmup_steps)
                term = (hoyer * hoyer_scale) * hoyer_square(steerer.vectors)
                if hoyer_scale:
                    term.backward()
                # Skipped rather than multiplied through at scale 0: the
                # contribution would be an exact zero anyway (the gradient is
                # finite everywhere the vector is not the zero init, and exactly
                # zero when it is), so this only avoids the work — and any 0*inf
                # a future change to hoyer_square could introduce.
                penalty_hoyer = term.item()

            # The LR this step actually ran with — scheduler.step() moves it on,
            # so reading it afterwards would threshold with the next step's.
            lr_now = optimizer.param_groups[0]["lr"]
            optimizer.step()
            if group_lasso and group_lasso_prox:
                # Deterministic and elementwise over vectors every rank already
                # holds identically, so the replicas stay in step without a
                # collective of its own.
                prox_group_lasso_(steerer.vectors, group_lasso * lr_now)
            scheduler.step()
            # The training phase alone: everything below is diagnostics, and the
            # evaluation among them is several steps' worth of forwards. Dividing
            # the token count by this rather than by the whole step keeps
            # tokens_per_s a throughput curve instead of a picture of --eval_every.
            train_s = perf_counter() - t_boundary

            step += 1
            # Reduce the metrics the same way as the gradient: as unnormalized
            # sums over this rank's share, divided by the global batch afterwards.
            # An empty stride contributes zeros, which is what it should.
            empty = torch.zeros(0, device=DEVICE)
            local_chosen = torch.cat(batch_chosen) if batch_chosen else empty
            local_rejected = torch.cat(batch_rejected) if batch_rejected else empty
            stats = torch.tensor(
                [
                    loss_sum,
                    float(((local_chosen - local_rejected) > 0).sum().item()),
                    float(local_chosen.sum().item()),
                    float(local_rejected.sum().item()),
                    sft_sum,
                ],
                device=DEVICE,
            )
            dist_sum_(stats, world_size)
            batch_loss = stats[0].item() / len(idx)
            # The target continuation is DPO's "chosen" and the opposite one its
            # "rejected"; logging them apart separates a vector that raises the
            # trust-consistent continuation from one that merely suppresses the
            # distrust one — the two look identical in the margin alone, which is
            # their difference and so needs no reduction of its own.
            chosen = stats[2].item() / len(idx)
            rejected = stats[3].item() / len(idx)
            writer.add_scalar("train/loss", batch_loss, step)
            writer.add_scalar(
                "train/reward_accuracy", stats[1].item() / len(idx), step
            )
            writer.add_scalar("train/reward_chosen", chosen, step)
            writer.add_scalar("train/reward_rejected", rejected, step)
            writer.add_scalar("train/reward_margin", chosen - rejected, step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
            if sft:
                # train/sft is the term as it enters the loss, and it is a square
                # wave by construction: d picks which continuation is scored, and
                # the two poles sit at different per-token NLLs. test/sft
                # averages both directions and is the readable one.
                writer.add_scalar("train/sft", stats[4].item() / len(idx), step)
            if penalty_lasso is not None:
                writer.add_scalar("train/penalty_lasso", penalty_lasso, step)
            if penalty_hoyer is not None:
                # The term as it enters the gradient, i.e. already ramped, so it
                # reads as how much of the objective the penalty is carrying —
                # the same convention as train/sft. train/hoyer_scale below is
                # what divides back out to the bare ratio, and it is logged only
                # while there is a ramp to see.
                writer.add_scalar("train/penalty_hoyer", penalty_hoyer, step)
            if hoyer_scale is not None and hoyer_warmup_steps > 0:
                writer.add_scalar("train/hoyer_scale", hoyer_scale, step)
            # A norm per trained row: n_selected reductions over hidden_size, so
            # it costs nothing to log it as often as the loss.
            log_vector_norms(writer, steerer, step)

            postfix = {"epoch": f"{epoch + 1}/{epochs}", "loss": f"{batch_loss:.4f}"}
            # The expensive one: a whole pass over the held-out split, twice (once
            # per direction d). Hence the stride — and the final step regardless,
            # so the saved vector always has a test number beside it. The
            # condition reads only `step`, which every rank agrees on, so the
            # collectives inside evaluate() stay matched.
            is_eval = step % eval_stride == 0 or step == total_steps
            if is_eval and test_idx:
                test_loss, test_acc, test_chosen, test_rejected, test_sft = evaluate(
                    model, steerer, enc_t, enc_o, ref_t, ref_o,
                    pad_id, test_idx, beta, micro_batch_size, sft, rank, world_size,
                )
                writer.add_scalar("test/loss", test_loss, step)
                writer.add_scalar("test/reward_accuracy", test_acc, step)
                writer.add_scalar("test/reward_chosen", test_chosen, step)
                writer.add_scalar("test/reward_rejected", test_rejected, step)
                writer.add_scalar(
                    "test/reward_margin", test_chosen - test_rejected, step
                )
                if sft:
                    writer.add_scalar("test/sft", test_sft, step)
            if test_loss is not None:
                # Carried between evaluations so the bar keeps showing the last
                # one rather than blanking on the steps that skipped it.
                postfix["test_loss"] = f"{test_loss:.4f}"

            # Checkpoint on the evaluation steps, so the vector on disk and the
            # test point in TensorBoard carry the same step number. The final step
            # is the caller's end-of-training save (writing it here too would
            # duplicate it), and only rank 0 actually writes — checkpoint_fn
            # returns early on the others rather than racing them to one path.
            if is_eval and checkpoint_fn is not None and step != total_steps:
                checkpoint_fn(step)

            # Peak GPU allocation since the previous step, summed over the GPUs
            # (the total is what has to fit on the node); per-device curves say
            # how evenly the device_map split it, or — data parallel, where each
            # rank is a full replica — how evenly the batch stride did. Read after
            # the evaluation, so a step that ran one reports the peak it hit.
            per_gpu = gpu_peak_gb(rank, world_size)
            writer.add_scalar("gpu/mem_gb", sum(per_gpu), step)
            if len(per_gpu) > 1:
                for i, gb in enumerate(per_gpu):
                    writer.add_scalar(f"gpu/mem_gb/cuda{i}", gb, step)
            postfix["gpu"] = f"{sum(per_gpu):.1f}GB"
            reset_gpu_peak()

            # Wall time and energy for the step just finished, boundary to
            # boundary — the evaluation, the checkpoint write and the logging
            # included, so nothing escapes the accounting and the cumulative kWh
            # is exactly the counter's own start-to-end difference. step_time_s
            # therefore spikes on the evaluation steps, which is the cost of the
            # stride made visible rather than an artifact.
            now = perf_counter()
            step_s = now - t_boundary
            t_boundary = now
            delta_j = energy.delta_j()
            watts = sum(delta_j) / step_s
            # Non-pad tokens: `idx` is the *global* batch and both continuations
            # are forwarded for each of its examples, so this is already the
            # step's total and needs no reduction. Padding is excluded because it
            # is work the step wastes, not throughput it delivers.
            tokens = sum(len(enc_t[i][0]) + len(enc_o[i][0]) for i in idx)
            if sft and d < 0:
                # The anchor pass re-forwards the target half at +v (dpo_batch),
                # so those tokens are work this step actually did. Leaving them
                # out would show a throughput dip on every d=-1 step that is an
                # accounting artifact rather than a slowdown.
                tokens += sum(len(enc_t[i][0]) for i in idx)
            writer.add_scalar("perf/step_time_s", step_s, step)
            writer.add_scalar("perf/tokens_per_s", tokens / train_s, step)
            writer.add_scalar("gpu/power_w", watts, step)
            writer.add_scalar(
                "gpu/energy_kwh", sum(energy.total_j) / 3.6e6, step
            )
            if len(delta_j) > 1:
                for i, joules in enumerate(delta_j):
                    writer.add_scalar(
                        f"gpu/power_w/cuda{i}", joules / step_s, step
                    )
            postfix["pow"] = f"{watts:.0f}W"

            pbar.set_postfix(**postfix)
            pbar.update(1)
    # total= means tqdm has no iterator to exhaust, so the bar needs closing.
    pbar.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def dataset_naming_tag(dataset: str, data_dir: Path | None) -> str:
    """The tag that goes into the run name, the vector filename and the ref stem.

    For every dataset but the translated corpus this is `--dataset` itself. A
    translated corpus is read by benevolence's loader over benevolence's columns,
    so `--dataset` alone would name a French run and an English one identically —
    same model, same layers, same span, same label, separable only by the leading
    timestamp. The language is not in the rows either: benevolence-trad.py writes
    one tree per language under the generator (<out_dir>/<generator>/<language>/),
    so --data_dir's own name is where it lives, and that is what is read here.

    Hyphen-joined, never `_`: the filename is underscore-delimited
    (model_dataset_layers_span_label(_step{N})), so an underscore in the tag would
    add a field to it and every consumer that reads one by position would be off
    by one.
    """
    if dataset != "benevolence-trad":
        return dataset
    name = data_dir.name.lower() if data_dir is not None else ""
    language = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    if not language:
        raise ValueError(
            "--dataset benevolence-trad reads the language from --data_dir's own "
            f"name, and {str(data_dir)!r} has none to read. Point it at one "
            "language's tree, the directory benevolence-trad.py named after it "
            "(e.g. ../data/data/benevolence-trad/<generator>/french)."
        )
    return f"benevolence-{language}"


def main(
    dataset: str,
    data_dir: Path | None,
    doubt_targets: list[str],
    model_name: str,
    layer_tokens: list[str],
    inject: str,
    num_samples: int | None,
    batch_size: int | None,
    micro_batch_size: int | None,
    gradient_accumulation_steps: int | None,
    max_length: int,
    beta: float,
    sft: float,
    lr: float,
    weight_decay: float,
    group_lasso: float,
    group_lasso_prox: bool,
    hoyer: float,
    hoyer_warmup_frac: float | None,
    epochs: int,
    warmup_frac: float,
    min_lr_frac: float,
    test_frac: float,
    seed: int,
    save_dir: Path,
    logdir: Path,
    ref_cache_dir: Path | None,
    grad_checkpointing: bool,
    eval_every: int,
) -> None:
    rank, world_size, local_rank, mesh = dist_setup()
    is_main = rank == 0

    batch_size, micro_batch_size, grad_accum = resolve_batching(
        batch_size, micro_batch_size, gradient_accumulation_steps
    )
    # Everything downstream that names a file, a directory or a TensorBoard run
    # uses this rather than `dataset`, which from here on only picks the loader.
    dataset_tag = dataset_naming_tag(dataset, data_dir)

    # A translated corpus is benevolence in another language, column for column,
    # so it shares the loader; only the tag tells the two apart afterwards. And
    # doubt returns one set per direction, so it is the one dataset that trains
    # more than one vector in a run; everything downstream already loops over
    # sets, so nothing else changes shape.
    if dataset in ("benevolence", "benevolence-trad"):
        triple_sets = load_benevolence(data_dir, num_samples)
    elif dataset == "doubt":
        triple_sets = load_doubt(data_dir, num_samples, doubt_targets)
    else:
        triple_sets = load_trust_conversations(num_samples)

    # Data parallel, the weights land whole on the rank's own GPU and are then
    # sharded across ranks in place; single process, device_map="auto" spreads
    # them over the visible GPUs instead and nothing is sharded.
    #
    # Which attention kernel, and why, is _ATTN_IMPLEMENTATION_RULES. The choice
    # never changes what a layer hands back, so the hooks, the loss and the saved
    # vectors do not depend on it — only the peak memory and the step time do.
    attn_implementation = attn_implementation_for(model_name)
    if attn_implementation == "flash_attention_2":
        # The config, not the name, is what decides this — see the blocker. Read
        # separately because it has to be known *before* from_pretrained is told
        # which kernel to build; it is a local read of the same warmed snapshot
        # load_model is about to open, so it costs nothing and works offline.
        blocker = flash_attention_2_blocker(AutoConfig.from_pretrained(model_name))
        if blocker is not None:
            attn_implementation = "sdpa"
            if is_main:
                print(f"[load_model] flash_attention_2 vetoed: {blocker}")
    if is_main:
        print(f"[load_model] attention kernel: {attn_implementation}")
    model, tokenizer = load_model(
        model_name,
        {"": local_rank} if world_size > 1 else "auto",
        attn_implementation=attn_implementation,
    )
    hidden_size = get_hidden_size(model)
    num_layers = len(get_decoder_layers(model))
    layers = parse_layers(layer_tokens, num_layers)
    # Llama ships no pad token, so a batch would otherwise be padded with `None`.
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # A sequence past a model's position limit does not degrade gracefully —
    # RoPE is applied at indices the model was never trained on and the
    # log-probs on both poles become noise — and truncation is the one thing
    # here that can keep a triple inside it, so the ceiling is enforced rather
    # than warned about. It is read off the config for the same reason the layer
    # count is: a per-model constant belongs to the checkpoint, not to a flag.
    # Nothing in the benevolence corpus comes near either limit (longest triple
    # 2225 tokens under Llama-3's template, against its 8192), so this only ever
    # fires on a --max_length raised past a short-context checkpoint.
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is None:
        text_cfg = getattr(model.config, "text_config", None)
        max_positions = getattr(text_cfg, "max_position_embeddings", None)
    if max_positions is not None and max_length > max_positions:
        if is_main:
            print(
                f"[encode] --max_length {max_length} exceeds {model_name}'s "
                f"context ({max_positions}); clamping. Longer triples lose the "
                "tail of their continuation."
            )
        max_length = max_positions

    # Part of the reference cache's key: the same tokens under bf16 and under
    # fp16 are not the same log-probs. Read before fsdp_shard, while these are
    # still plain parameters rather than DTensors.
    ref_dtype = next(model.parameters()).dtype

    # Freeze the model: only the steering vectors carry gradients. Autograd still
    # builds the graph through the layers above each injection point because the
    # vectors require grad. requires_grad_(False) means no gradients or optimizer
    # state are ever allocated for the model weights.
    model.requires_grad_(False)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable == 0, f"model weights not frozen: {n_trainable} trainable"

    # The memory cost here is activations, not weights: a vector injected at layer
    # L forces every layer above it to keep its forward activations for the
    # backward through v (all-layer injection ⇒ the whole stack). Gradient
    # checkpointing recomputes them in backward instead of storing them, trading a
    # near-full extra forward per step for the memory to fit a full-depth
    # injection at large batch. It is off by default because that recompute
    # dominates step time when memory is not the constraint (e.g. a single layer);
    # enable it only when a run OOMs. It only fires in train() mode; these Qwen
    # decoder layers have no dropout, so train() is numerically identical to eval()
    # here while the weights stay frozen.
    if grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.train()
    else:
        model.eval()

    # After freezing and after gradient_checkpointing_enable: both walk the
    # module tree, and doing them first keeps them looking at plain modules and
    # plain parameters rather than at DTensors. Steerer.attach comes later still,
    # which is fine — its hook lands downstream of FSDP's own post-forward hook
    # in the autograd graph, so the backward all-gather is still reached.
    if world_size > 1:
        fsdp_shard(model, mesh)
        if is_main:
            print(
                f"FSDP2: weights sharded over {world_size} rank(s), "
                f"one group per decoder layer plus the root"
            )
        # Hand the load's segments back before the first step, or the gpu/mem_gb
        # curve cannot see training at all. Loading materialises a full replica on
        # each rank (device_map={"": local_rank}) and only then shards it, so the
        # caching allocator's pool is sized by the whole model. Nothing returns
        # that pool on its own, and reset_peak_memory_stats sets peak := *current
        # reserved* rather than to zero, so every later reading is floored at the
        # load — which is why a 27B reported the same 55.1 GiB/rank at micro-batch
        # 1 and 4, and a checkpointed 12B reports its own weights back. One call,
        # before the timing and the curves start, so it costs nothing per step.
        torch.cuda.empty_cache()
        reset_gpu_peak()

    model_suffix = model_name.split("/")[-1]
    all_layers = layers == list(range(num_layers))
    layer_tag = "Lall" if all_layers else "L" + "-".join(map(str, layers))
    # Always tagged, the default included: applying a vector over a span it was
    # not trained on silently produces nonsense rather than an error, so the span
    # is not something to leave implicit. Vectors written before --inject existed
    # were all user-span and want renaming to match.
    inject_tag = [inject]
    # A run differing from another only in a penalty weight would otherwise be
    # told apart by its timestamp alone, in both trees; these make a lambda sweep
    # legible at a glance. Absent when the penalties are off, so runs without them
    # keep the name they have always had. The vector filenames (base_stem, below)
    # deliberately do not carry them: the run directory already separates the
    # files on disk, and every consumer that pattern-matches a filename keeps
    # working.
    penalty_tags = []
    if group_lasso:
        penalty_tags.append(f"lasso{group_lasso:g}")
    if hoyer:
        penalty_tags.append(f"hoyer{hoyer:g}")
    # Same argument for the objective's own extra term: --sft changes what is
    # being minimized, not just how it is regularized, so two runs that differ
    # only in alpha should not be one timestamp apart in the name. Off by
    # default and absent when off, so a plain BiPO run keeps its old name.
    if sft:
        penalty_tags.append(f"rpo{sft:g}")
    # Identically seeded on every rank, and consumed in lockstep from here on:
    # the train/test split, the batch order and the direction d all come out of
    # this one generator, so the ranks agree on all three without communicating.
    rng = random.Random(seed)

    # Only rank 0 writes the event files — concurrent writers corrupt a run — and
    # only rank 0 prints, so the log reads like the single-process one.
    def log(msg: str) -> None:
        if is_main:
            print(msg)

    run_name = "_".join(
        [f"{datetime.now():%Y%m%d-%H%M%S}", model_suffix, dataset_tag, layer_tag]
        + inject_tag
        + penalty_tags
    )
    # The vectors land in a subdirectory named for the run, the same name the
    # TensorBoard run carries, so a curve and the checkpoints it describes are
    # found under one name in both trees and two runs of the same configuration
    # no longer overwrite each other's files. Under torchrun every rank formats
    # its own timestamp and two can straddle a second boundary, but only rank 0
    # ever touches these paths — the writer, the mkdir and both saves — so the
    # run is named whatever rank 0 called it.
    save_dir = save_dir / run_name
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
    log(f"Logging to {logdir / run_name} (tensorboard --logdir {logdir})")
    log(f"Saving vectors to {save_dir}")

    name, mem_gib, caps_w = gpu_capacity(rank, world_size)
    total_mem, total_cap = sum(mem_gib), sum(caps_w)
    cap_str = f"{total_cap:.0f} W" if total_cap else "unknown power cap"
    if len(mem_gib) > 1:
        per = f" ({mem_gib[0]:.1f} GiB / {caps_w[0]:.0f} W each)" if caps_w[0] else ""
        log(f"GPU: {len(mem_gib)} x {name} = {total_mem:.1f} GiB, {cap_str}{per}")
    else:
        log(f"GPU: {name} = {total_mem:.1f} GiB, {cap_str}")

    log(
        f"Training BiPO vectors at layers {layers} (of {num_layers}), "
        f"injected on the {inject} turn"
    )
    if sft:
        log(
            f"DPO+NLL: preference loss + {sft:g} x the length-normalised NLL of "
            "the target continuation at +v (train/loss and test/loss stay the "
            "preference term alone; see train/sft)"
        )
    if group_lasso or hoyer:
        bits = []
        if group_lasso:
            form = "proximal" if group_lasso_prox else "subgradient"
            bits.append(f"group lasso {group_lasso:g} ({form})")
        if hoyer:
            # None follows --warmup_frac; train_vectors resolves it, so read the
            # same fallback here rather than reporting "None".
            ramp = warmup_frac if hoyer_warmup_frac is None else hoyer_warmup_frac
            ramped = (
                f", warmed up linearly over the first {ramp:g} of the run"
                if ramp
                else ", no warmup (flat weight from step 1)"
            )
            bits.append(
                f"hoyer-square {hoyer:g} (of [1, {len(layers)}]){ramped}"
            )
        log("Row penalties: " + ", ".join(bits))
        if len(layers) == 1:
            log(
                "  note: one trained layer, so neither penalty has anything to "
                "concentrate - the lasso only rescales the row, and hoyer-square "
                "is the constant 1"
            )
        if weight_decay:
            log(
                f"  note: --weight_decay {weight_decay:g} is still on. Decoupled "
                "L2 pays less the more layers share a fixed total effect, so it "
                "pulls against these; --weight_decay 0 leaves the magnitude to "
                "the DPO term and to the inference-time strength sweep."
            )
    if world_size > 1:
        # Data parallel: batch_size stays the *global* optimizer batch, so report
        # what one rank actually forwards — that, not batch_size, is what sets a
        # rank's activation memory. micro_batch_size is clamped to the rank's
        # share because
        # a micro-batch larger than the stride just runs the whole stride, and
        # printing the unclamped number hides how big the forward really is.
        per_rank = ceil(batch_size / world_size)
        eff_micro = min(micro_batch_size, per_rank)
        log(
            f"Batch {batch_size} (global) = {world_size} rank(s) x ~{per_rank} "
            f"triples = {ceil(per_rank / eff_micro)} x micro-batch {eff_micro} "
            f"({2 * eff_micro} sequences per forward, per rank)"
        )
    else:
        log(
            f"Batch {batch_size} = {grad_accum} x micro-batch {micro_batch_size} "
            f"({2 * micro_batch_size} sequences per forward)"
        )
    # How device_map="auto" spread the weights — single-process only, since the
    # data-parallel path shards with FSDP instead and has no hf_device_map worth
    # printing. The vectors and the loss live on DEVICE and the hooks reach
    # across to wherever each layer landed.
    shards = Counter(str(d) for d in getattr(model, "hf_device_map", {}).values())
    if shards and world_size == 1:
        split = ", ".join(f"{dev} ({n} modules)" for dev, n in sorted(shards.items()))
        log(f"Model sharded over {len(shards)} device(s): {split}")
        log(f"Steering vectors and optimizer on {DEVICE}")

    for tset in triple_sets:
        enc_t: list[tuple[list[int], int, int, int]] = []
        enc_o: list[tuple[list[int], int, int, int]] = []
        # Triples arrive train-first when the set carries its own boundary, so
        # counting the survivors below the boundary re-derives it after encoding
        # has dropped whatever it could not use.
        n_train_enc = 0
        for i, (ctx, target, opposite) in enumerate(tset.triples):
            et = encode_example(tokenizer, ctx, target, max_length, inject)
            eo = encode_example(tokenizer, ctx, opposite, max_length, inject)
            if et is not None and eo is not None:
                enc_t.append(et)
                enc_o.append(eo)
                if tset.n_train is not None and i < tset.n_train:
                    n_train_enc += 1
        if not enc_t:
            log(f"[{tset.label}] no usable triples; skipping")
            continue

        # One TensorBoard run per label, under the run directory rather than as a
        # tag prefix. TensorBoard groups the scalar dashboard by the text before
        # the *first* slash, so a leading label would spend that grouping on a
        # constant and leave every curve in one heap; spending it on the category
        # instead gives train/, test/, gpu/, perf/ and vector_norm/ as the
        # sections, and a label with more than one set (which this dataset does
        # not have, but doubt's two directions do) then arrives as a second run
        # whose curves overlay the first's on each of those charts.
        writer = (
            SummaryWriter(log_dir=str(logdir / run_name / tset.label))
            if is_main
            else NullWriter()
        )

        # Held-out split for the test-loss curve. A dataset that ships its own
        # split wins: benevolence and doubt both stratify theirs on family, so
        # both sides cover the scenario bank, and re-cutting it at random here
        # would throw that away and quietly report a test curve measured on a
        # different population than the one the dataset intends.
        # Everything else gets the rng cut, reproducible from the shared seed,
        # with >=1 example on each side where possible.
        if tset.n_train is not None:
            train_idx = list(range(n_train_enc))
            test_idx = list(range(n_train_enc, len(enc_t)))
            source = "dataset's own split"
        else:
            indices = list(range(len(enc_t)))
            rng.shuffle(indices)
            n_test = min(len(indices) - 1, round(len(indices) * test_frac))
            n_test = max(n_test, 0)
            test_idx = sorted(indices[:n_test])
            train_idx = sorted(indices[n_test:])
            source = f"--test_frac {test_frac}"
        log(f"\n[{tset.label}] {len(enc_t)} triples "
            f"({len(train_idx)} train / {len(test_idx)} test, {source})")

        # Two no-grad passes over the whole set, target and opposite, before
        # the first step. Memoized on disk: nothing here depends on what the run
        # is sweeping, so only the first run of a sweep pays for them.
        ref_stem = f"{model_suffix}_{dataset_tag}_{tset.label}"
        ref_t = cached_reference_logprobs(
            model,
            enc_t,
            pad_id,
            micro_batch_size,
            ref_cache_file(
                ref_cache_dir, ref_stem, "target", enc_t, model_name, ref_dtype
            ),
            rank,
            world_size,
        )
        ref_o = cached_reference_logprobs(
            model,
            enc_o,
            pad_id,
            micro_batch_size,
            ref_cache_file(
                ref_cache_dir, ref_stem, "opposite", enc_o, model_name, ref_dtype
            ),
            rank,
            world_size,
        )

        vectors = torch.zeros(
            len(layers),
            hidden_size,
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True,
        )
        steerer = Steerer(vectors, layers)
        steerer.attach(model)

        # model_suffix_dataset_{layer_tag}_{span}_label(_step{N}).pt
        parts = [
            model_suffix,
            dataset_tag,
            layer_tag,
            *inject_tag,
            tset.label,
        ]
        base_stem = "_".join(parts)

        def save_checkpoint(step: int) -> None:
            # Every rank holds the same vectors (same init, same all-reduced
            # gradient, same optimizer state), so rank 0's copy is the answer and
            # the others would only race it to the same path.
            if not is_main:
                return
            steering = build_steering(vectors, layers, num_layers, hidden_size)
            out_file = save_dir / f"{base_stem}_step{step}.pt"
            torch.save(steering, out_file)
            print(f"[checkpoint] step {step}: saved to {out_file}")

        try:
            train_vectors(
                model,
                steerer,
                enc_t,
                enc_o,
                ref_t,
                ref_o,
                pad_id,
                train_idx,
                test_idx,
                beta=beta,
                sft=sft,
                lr=lr,
                weight_decay=weight_decay,
                group_lasso=group_lasso,
                group_lasso_prox=group_lasso_prox,
                hoyer=hoyer,
                hoyer_warmup_frac=hoyer_warmup_frac,
                epochs=epochs,
                batch_size=batch_size,
                micro_batch_size=micro_batch_size,
                warmup_frac=warmup_frac,
                min_lr_frac=min_lr_frac,
                rng=rng,
                label=tset.label,
                writer=writer,
                eval_every=eval_every,
                checkpoint_fn=save_checkpoint,
                rank=rank,
                world_size=world_size,
            )
        finally:
            steerer.remove()

        steering = build_steering(vectors, layers, num_layers, hidden_size)
        out_file = save_dir / f"{base_stem}.pt"
        if is_main:
            torch.save(steering, out_file)
            print(f"Saved {tuple(steering.shape)} to {out_file}")

        writer.close()
        del vectors, steerer, steering, ref_t, ref_o
        gc.collect()
        torch.cuda.empty_cache()

    if world_size > 1:
        # Hold the group open until rank 0 has finished writing, so a rank that
        # exits first cannot tear the group down mid-save.
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        "-d",
        choices=["conversations-5k", "benevolence", "benevolence-trad", "doubt"],
        default="conversations-5k",
        help="Which dataset to derive the steering vector from. Also the tag "
        "that goes into the vector's filename and the TensorBoard run name. "
        "Everything but 'conversations-5k' reads a save_to_disk'd run of the "
        "matching generator under data/ from --data_dir and uses that dataset's "
        "own train/test splits. 'benevolence-trad' is a translated benevolence "
        "corpus: same loader, same columns, but --data_dir names one language "
        "and the tag becomes 'benevolence-<language>' so its vectors are not "
        "named identically to the English run's. 'doubt' trains one vector per "
        "direction (see --doubt_targets); its trust-free controls are never "
        "training signal and benevolence's are not read at all.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="For every --dataset but conversations-5k: the directory the "
        "generator wrote — benevolence's holds contrastive/ (e.g. "
        "../data/data/benevolence/<generator-model>), benevolence-trad's is one "
        "language of one (e.g. "
        "../data/data/benevolence-trad/<generator-model>/french, whose name is "
        "where the language in the vector's tag comes from), doubt's holds "
        "user/ and self/ (e.g. ../data/data/doubt/<generator-model>). Ignored "
        "otherwise.",
    )
    parser.add_argument(
        "--doubt_targets",
        nargs="+",
        choices=list(DOUBT_SUBSETS),
        default=list(DOUBT_SUBSETS),
        help="For --dataset doubt: which direction(s) the doubt points to train "
        "on (default: both). Each becomes its own TripleSet and so its own "
        "vector, `_doubt-user.pt` and `_doubt-self.pt`, with its own test curve "
        "— the split exists because one vector over the mixture cannot be told "
        "apart from one that moves only half of it. A subset absent from "
        "--data_dir is skipped with a warning.",
    )
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["all"],
        help="Decoder layer(s) to train together: 'all' or a list of indices "
        "(e.g. --layers 15 20). Trained jointly under one loss.",
    )
    parser.add_argument(
        "--inject",
        choices=["user", "assistant"],
        default="user",
        help="Token positions the vector is added at: 'user' (default) steers "
        "how the model reads the scenario, reaching the answer only through the "
        "KV cache; 'assistant' steers how it writes the continuation, adding "
        "into the very positions the loss scores. Whichever is chosen has to be "
        "reproduced at inference for the vector to mean anything.",
    )
    parser.add_argument(
        "--num_samples",
        "-n",
        type=int,
        default=None,
        help="Use only the first N question pairs; omit to train on all of them",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=None,
        help=f"Triples per optimizer step (default {DEFAULT_BATCH_SIZE}). Sets "
        "what the loss and LR schedule average over, not peak memory.",
    )
    parser.add_argument(
        "--micro_batch_size",
        "-mb",
        type=int,
        default=None,
        help="Triples per forward/backward pass; gradients accumulate until "
        "--batch_size triples are consumed. This is the knob that sets peak "
        "memory (each pass runs 2x this many sequences: target + opposite). "
        "Defaults to --batch_size, i.e. no accumulation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Micro-batches per optimizer step. Alternative way to say the "
        "same thing: batch_size = micro_batch_size * this, so pass any two.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Tokens per (question + reply) sequence; longer ones are truncated "
        "from the right, which scores a partial continuation. 2048 keeps "
        "essentially every triple in this dataset whole.",
    )
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--sft",
        type=float,
        default=0.0,
        help="Weight of the NLL term in DPO+NLL "
        "- the length-normalised NLL of the TARGET continuation, always read at "
        "+v, added to the preference loss. 0 (the default) is plain BiPO. The "
        "preference term is a ratio and so is just as happy with a vector that "
        "pushes both continuations down as with one that lifts the target; this "
        "prices that, and train/reward_chosen is the curve to sweep it against. "
        "It never scores the opposite pole, so on a d=-1 batch the target is "
        "forwarded a second time at +v: those micro-batches run 3x the triples "
        "instead of 2x, ~25%% more sequences over a run and 1.5x peak memory on "
        "half the steps - lower --micro_batch_size if a run was near the edge. "
        "NOT on the same scale as --beta, and not worth deriving: this term is "
        "length-normalised where the preference term reads summed log-probs, but "
        "it is also large and nearly irreducible (~1.9 nats/token here, of which "
        "a run moves ~0.3%%), so by loss value alpha=1 already dominates while by "
        "gradient it is much weaker than that looks. Measured: alpha <= 0.3 does "
        "nothing (0.1 vs 0.3 vs 0 move the vector norm under 1%%). Above that, "
        "sweep 1/3/10 against train/reward_chosen - the curve this exists to fix "
        "- rather than picking a number. Left "        "out of train/loss and test/loss.",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument(
        "--group_lasso",
        type=float,
        default=0.0,
        help="Weight of the group-lasso penalty sum_L ||v_L|| over the layer "
        "rows - an L1 across the per-layer norms, an L2 within each - which "
        "switches whole layers off instead of shrinking all of them. 0 (the "
        "default) disables it. Applied as its proximal operator after each step, "
        "so the rows it drops are exactly zero; see --group_lasso_subgradient "
        "for the add-to-the-loss form. Pair it with --weight_decay 0, which "
        "prefers the opposite. The threshold is lambda * lr per step, so it is "
        "lambda * lr * total_steps against a typical row norm that decides what "
        "dies - with the default lr and a full run that puts the useful range "
        "around 0.1-10, not at weight-decay scales. Sweep it against "
        "vector_norm/nonzero_rows.",
    )
    parser.add_argument(
        "--group_lasso_subgradient",
        action="store_true",
        help="Add --group_lasso to the loss and let AdamW differentiate it, "
        "instead of applying its proximal operator after the step. Adam's "
        "per-coordinate preconditioning then leaves dropped rows small but never "
        "exactly zero.",
    )
    parser.add_argument(
        "--hoyer",
        type=float,
        default=0.0,
        help="Weight of the group Hoyer-square penalty "
        "(sum_L ||v_L||)^2 / sum_L ||v_L||^2 over the layer rows; 0 (the "
        "default) disables it. Scale-invariant and bounded in [1, n_layers "
        "trained], so it concentrates the norm into fewer layers without fixing "
        "how large it is, and lambda carries across models and layer counts. It "
        "reaches no exact zeros - for those use --group_lasso.",
    )
    parser.add_argument(
        "--hoyer_warmup_frac",
        type=float,
        default=None,
        help="Fraction of the total optimizer steps spent linearly ramping the "
        "--hoyer weight from 0 up to its full value; omit to follow "
        "--warmup_frac, pass 0 for the flat weight. Only hoyer has one: the "
        "proximal group lasso shrinks by lambda*lr per step and so already "
        "inherits the LR ramp, while hoyer goes through .grad, where Adam's "
        "m/sqrt(v) normalisation puts its balance against the preference term "
        "out of the LR's reach. Being scale-invariant its gradient grows as "
        "1/||V||, so it pulls hardest on the first steps after the zero init - "
        "which is exactly when the preference term has said least about which "
        "layers matter. Ignored when --hoyer is 0.",
    )
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument(
        "--warmup_frac",
        type=float,
        default=0.1,
        help="Fraction of the total optimizer steps (epochs x steps/epoch) spent "
        "linearly warming the LR up before the cosine decay. 0 disables warmup.",
    )
    parser.add_argument(
        "--min_lr_frac",
        type=float,
        default=0.1,
        help="Fraction of --lr the cosine decays down to at the last step, so "
        "training ends at min_lr_frac * lr instead of 0. 0 decays to zero.",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.05,
        help="Fraction of triples held out for the test-loss curve (0 disables). "
        "Ignored by a dataset that ships its own split — benevolence and doubt "
        "both do, stratified over the scenario bank, which a fresh random cut "
        "here would not reproduce.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/steering_vectors"),
        help="Directory the steering vectors (and any epoch checkpoints) are "
        "written to, inside a per-run subdirectory named like the TensorBoard "
        "run; created if missing",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("runs"),
        help="TensorBoard log directory (a per-run subdir is created inside)",
    )
    parser.add_argument(
        "--ref_cache_dir",
        type=Path,
        default=Path("data/reference_logprobs"),
        help="Directory the DPO reference log-probs are memoized in; created if "
        "missing. They are a function of the model and the encoded triples "
        "alone, so a sweep over layers, --inject, the penalties, beta, lr or the "
        "seed reuses one table instead of recomputing two full no-grad passes "
        "over the dataset before every run. Keyed by a content hash, so a "
        "changed dataset, --num_samples or --max_length misses and recomputes "
        "rather than going stale.",
    )
    parser.add_argument(
        "--no_ref_cache",
        action="store_true",
        help="Recompute the reference log-probs and do not write them, ignoring "
        "--ref_cache_dir.",
    )
    parser.add_argument(
        "--grad_checkpointing",
        action="store_true",
        help="Recompute activations in backward to fit large / full-depth "
        "injections in memory. Slower per step; enable only on OOM.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=0,
        help="Optimizer steps between held-out evaluations (the test/* curves). "
        "0 (the default) evaluates once per epoch. Each evaluation forwards the "
        "whole test split twice — once per direction d — so it costs several "
        "steps; the last step is always evaluated whatever this is set to. Every "
        "evaluation step also writes the current vector to <base>_step{N}.pt, so "
        "each checkpoint has a test loss logged at its own step.",
    )
    args = parser.parse_args()

    # Echo the resolved configuration: defaults make the command line an
    # incomplete record, and the sbatch log is the only record a finished run
    # leaves. Rank 0 only — torchrun would otherwise print it once per rank.
    if os.environ.get("RANK", "0") == "0":
        width = max(len(name) for name in vars(args))
        print("Parameters:")
        for name, value in vars(args).items():
            print(f"  {name:<{width}} = {value}")

    if args.dataset != "conversations-5k" and args.data_dir is None:
        parser.error(f"--dataset {args.dataset} needs --data_dir")
    if args.doubt_targets != list(DOUBT_SUBSETS) and args.dataset != "doubt":
        parser.error("--doubt_targets only applies to --dataset doubt")

    main(
        args.dataset,
        args.data_dir,
        args.doubt_targets,
        args.model,
        args.layers,
        args.inject,
        args.num_samples,
        args.batch_size,
        args.micro_batch_size,
        args.gradient_accumulation_steps,
        args.max_length,
        args.beta,
        args.sft,
        args.lr,
        args.weight_decay,
        args.group_lasso,
        # The prox is the default because it is the form that produces the exact
        # zeros the penalty is for; the flag names the opt-out.
        not args.group_lasso_subgradient,
        args.hoyer,
        args.hoyer_warmup_frac,
        args.epochs,
        args.warmup_frac,
        args.min_lr_frac,
        args.test_frac,
        args.seed,
        args.output,
        args.logdir,
        None if args.no_ref_cache else args.ref_cache_dir,
        args.grad_checkpointing,
        args.eval_every,
    )
