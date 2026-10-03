# Pose carrier: what we tested and what it cost (2026-09-29)

All numbers measured against PR #141 (`semantic_blocks`, official 0.147897). Harness validated: our
reproduction gives seg 0.02013 vs their published 0.0201, and pose mse 6.6192e-06 vs their 6.3695e-06
(4% off, from regenerating frame 1 from tokens+renderer and not applying the 5-frame selector).

## Score decomposition (reconstructs their 0.147897)

| component | bytes | % archive | score | % score |
|---|---|---|---|---|
| class maps (RC64 token stream) | 113,411 | 63.0% | 0.0755 | **51.1%** |
| renderer (width 96, int4) | 31,153 | 17.3% | 0.0207 | 14.0% |
| pose carrier | 21,816 | 12.1% | 0.0145 | 9.8% |
| guesser (integer HPAC weights) | 13,288 | 7.4% | 0.0088 | 6.0% |
| boundary table + container | 223 | 0.1% | 0.0001 | 0.1% |
| SegNet distortion | - | - | 0.0201 | 13.6% |
| PoseNet distortion | - | - | 0.0080 | 5.4% |

Bytes are **81% of the score**. Exchange rate: 1 KB = 0.00068.

## How pose is scored

`compute_distortion` takes PoseNet's `pose` head, keeps `[..., :6]` of 12 outputs (back 6 unscored,
look like log-sigmas), squares the difference between our video and the original, means over the 6 dims,
means over 600 pairs, then the score term is `sqrt(10 * mse)`.

- **d0 is forward speed and is 99.8% of the pose variance** (std 1.2586; dims 1-5 all under 0.036).
  Confirmed: setting frame0 = frame1 drops d0 by 33.70 to ~0.3.
- d1 and d5 are the horizontal pair (not separable by a single-pair probe), d4 vertical, d3 roll
  (tentative), **d2 unidentified**.
- Error budget at their 0.0080: 0.00253 per dim, or 0.0062 on d0 alone = **0.02% of 31.3**.
  In relative terms wildly uneven: d0 needs 0.2% of its range, d4 can be off by 34% of its.
- **3 decimal places** resolves every dim. Total pose information is 39.5 bits/frame = 2.9 KB, against
  the 21.3 KB they spend -> 7.4x overhead for the indirection (you must ship pixels, not numbers).
- PoseNet measures **ego-motion**, not object motion: a 200x200 patch (car-sized) shifted 20 px moves d0
  by 0.0006; the whole frame shifted 20 px moves d1 by 0.66. Video is 1164x874 @ 20 fps, so d0 ~ 31 reads
  as m/s = 112 km/h, consistent with the clip.
- **d0 saturates with displacement**: the previous render is 2 frame steps from frame 1 yet reads the same
  d0 as a 1-step pair. Unexplained, and it invalidated the midpoint-interpolation idea.

## Carrier structure

`basis_raw (12,3,24,32)` + `coefficients (600,12)` + 14-byte selector. Render path:
`carrier = einsum(coef, basis) / sqrt(12)`, `slave = round(clamp(127.5 + 64*carrier))`, bicubic to camera.

- Bytes split ~evenly: basis 12,277 B (3.55 bits/value) + coefficients 9,829 B (10.92 bits/value).
  **~1,842 B per atom.** The 12 thumbnails cost as much as all 600 frames of coefficients.
- Carrier magnitude is tiny: `64/sqrt(12) * 0.202 = 3.74 grey levels`.
- The 14-byte selector is a last-mile patch on **5 of 600 frames** (3 get a 1-px ROLL). Not a mechanism.
- `CARRIER_DIM = 12` comes from **PR #130** and was carried unchanged through #133, #135, #140, #141.
  #141's own contribution is a 111-byte repack (`LINEAGE.md`).

## Everything we tried, and what it cost

| attempt | result |
|---|---|
| zero-byte basis (DCT / random) | pose ~5,000,000x worse |
| bit depth: basis 3/2-bit, coef 8/6-bit (`carrier_sweep.py`) | every config lost; script's own guard said "carrier looks tight" |
| dimensionality k=10/8/6/4 (`carrier_ksweep.py`) | all lost. k=10 3.15x (needed <=1.7x), k=8 194x, k=6 948x, k=4 5923x. **Sharp cliff below k=10.** |
| 16-64 optimised black dots | hits d0 to 0.0012 but 12x short overall; fails on d2/d5. Also more bytes/frame than the carrier |
| 6-param affine warp of frame 1 (`warp_carrier.py`) | 200-760x short. Affine flow is linear in position; real ego flow is radial with 1/depth |
| frame-1 as base (`base_ab.py`) | 16x worse than grey on matched inits/budget |
| previous render as base (`carrier_ksweep.py --base prev`) | zero-byte \|d0 err\| 2.78 vs grey's 16.30, but carrier fit only reaches 1.50e-01 = **22,664x** worse |
| motion-compensated midpoint (`flow_base.py`) | worse than no compensation: prev 1.415 < blend 1.643 < sym 1.692 < flow 2.264 |
| MORE atoms k=13/14/16 (`carrier_up.py`) | all start at #141's exact value and never improve on it. `best` never moves off the init in any arm. NET +0.00121 / +0.00245 / +0.00489 |

## Why the base matters more than intuition suggests

A flat base hands **all** the signal to the carrier. On a textured base the same 12 atoms are a 2% ripple
on a complete scene and the scene dominates PoseNet's reading. Measured: the 12-atom subspace can express
**0.7% of `grey - frame1`** (residual 99.3%), and the coefficients needed for even that are ~0.1, well
inside the +/-1.10 they ship -- so **rank and resolution bind, not amplitude**. The carrier cannot walk
away from wherever you anchor it; choosing a base chooses a 12-dimensional patch of a 589,824-dimensional
space that you are then stuck in.

Grey is not good because it is blank: **flat white reads d0 = 0.93 and is useless.** Grey + random noise
reads 21.68 vs plain grey's 17.06, so #141's basis is the trained version of exactly that noise.

Near frame 1 the function is dead: blending grey->frame1, alpha=0.70 already reads d0 = 0.0135.

## Verdict

Treat the pose carrier as CLOSED, and note it is now closed in BOTH directions. Removing atoms degrades
immediately (k=10 3.15x, k=8 194x); adding them does nothing (k=13/14/16 all 1.00x). **12 is the optimum,
not an untuned default inherited from PR #130** -- that hypothesis was tested and is wrong. Our optimiser
cannot improve their carrier even with a warm start at their own solution and 33% more capacity, which
also means the earlier negative results were limited by the problem, not by our training.

Eight independent attacks; every #141 choice we probed is load-bearing.

## Joint renderer + carrier training (2026-09-30)

The one structural gap in the lineage: PR #86's master/seg renderer was trained ~5,800 epochs with
SegNet distortion as the ONLY objective (their Figure 12 plots d_seg and nothing else), and the pose stage
was trained separately afterwards to clean up whatever pose error resulted. #130 replaced the slave with
the carrier but kept the master. #133/#135 polished. #140 explicitly disclaims retraining. #141 is a
111-byte repack. So PoseNet has never been in the seg renderer's loss -- despite #86 writing down that
"Master output feeds SegNet AND the master half of each PoseNet pair".

Tested it: fine-tune renderer (+ basis + coefficients in the joint arm) from #141's weights against
100*seg + sqrt(10*pose). Step-0 eval reproduced #141 exactly (0.02014 + 0.00814 = 0.02827). Pose weight
614.6, from d/d(mse) sqrt(10*mse) at their operating point. 50 epochs, lr 5e-6 renderer / 2e-4 carrier,
margin loss, best-checkpoint tracking on the true score.

| arm | carrier | final best |
|---|---|---|
| seg_only | frozen | 0.02827 (+0.00000) |
| joint | trainable | 0.02827 (+0.00000) |

**Clean null -- neither beat #141.** Best-tracking never found an improvement to save, so both held the
baseline. But the trajectories answer the structural question:

| | seg | pose | total |
|---|---|---|---|
| baseline | 0.02014 | 0.00814 | 0.02827 |
| seg_only ep30 (carrier FROZEN) | 0.03035 | **1.12876** | 1.15911 |
| joint ep45 (carrier TRAINABLE) | 0.03250 | **0.02620** | 0.05869 |

**Frozen carrier: pose explodes 139x. Trainable carrier: 43x better, same renderer training.** That is the
seg/pose coupling measured end to end, and the first direct evidence a co-adapting carrier does what the
0.7%-subspace argument predicted. The structural criticism of the pipeline holds.

**What blocks it is seg, not pose.** Our renderer training drives seg 0.0201 -> 0.0325 and flatlines by
epoch 20 while pose keeps improving. The joint objective cannot rescue a renderer we are training badly.
Same wall as the morning's fine-tune, now confirmed with a 10x gentler lr and a proper control.

Verdict: structurally sound, experimentally unreachable until the renderer training is fixed. #86's recipe
is five phases, temperature annealing, SCN bit-budget ramps, top-15% hard-example mining at 5x, and SWA
checkpoints. We run one flat cosine for 50 epochs. Also NOT handled: quantisation (they ship int4, we
train float32 and never requantise), and #140's 455 token edits were solved against the OLD renderer so
they would need re-solving.

Scripts: `render/joint_train.py`, results in `render/joint/`.

## Overnight joint-training campaign (2026-09-30, unattended)

Bar set by operator: joint fine-tuning must at least reach PARITY with #141 (distortion 0.02827); failing
that is a bug, not a result.

### The loss was the bug, and kappa dominates

v1 averaged the seg loss over all 786k pixels when ~40 are wrong, so the gradient direction was dominated
by "make correct pixels more confident". Diagnostic ruled out the alternatives: eval is deterministic,
lr=0 holds EXACTLY, gradient norm is healthy (9.5e-02 vs weight norm 74), head saturation only 1.3%.
So the direction was wrong, not the magnitude -- which is why a 10x lr cut had barely helped.

Fix: DAG-style active-set masking (Xie et al. ICCV 2017) -- logit hinge, only pixels wrong or within
kappa, **normalised by active count**. Sweep, seg_only arm, baseline seg 0.02014:

| lr | kappa | active % | seg |
|---|---|---|---|
| 1e-3 | 2.0 | 2.007 | 0.10730 |
| 1e-4 | 0.5 | 0.388 | 0.02671 |
| 1e-5 | 0.5 | 0.395 | 0.02461 |
| 1e-6 | 0.5 | 0.414 | 0.02211 |
| 1e-5 | 0.1 | 0.057 | 0.02061 |
| **1e-5** | **0.05** | **0.057** | **0.02050** |
| 1e-5 | 0.02 | 0.026 | 0.02095 |

kappa matters more than lr, and there is an optimum near 0.05 -- tighter than that and the loss loses
signal. Closest approach to parity on seg: 0.02050 vs 0.02014, 1.8% off.

### Phase 2 failed for a reason I introduced

joint, renderer lr 1e-5 / carrier lr 1e-3, FIXED pose weight 614.6:

| | seg | pose | total |
|---|---|---|---|
| step 0 | 0.02014 | 0.00814 | 0.02827 |
| ep 5 | 0.05066 | 1.87964 | 1.93029 |
| ep 50 | 0.03288 | 0.09943 | 0.13231 |
| ep 200 | 0.02916 | 0.08978 | **0.11894** |

Destroyed in 5 epochs, then converged 200 epochs later to a different optimum 4.2x worse. Two causes,
both mine:
1. **carrier lr 1e-3** on coefficients of magnitude ~0.2 against a pose budget of 0.0062 on d0 -- 750
   updates in five epochs wrecks a fit calibrated to 0.02% precision.
2. **fixed pose weight.** d/d(mse) sqrt(10*mse) SHRINKS as mse grows, so the correct weight collapses once
   pose degrades while mine stayed at 614 -- over-weighting pose exactly when pose was already ruined,
   dragging the renderer around chasing my own damage.

Fixed by `--pose-form sqrt` (use the score's own function, self-balancing) and lr in the refine-don't-
rebuild regime (cf. 2401.10732: a near-optimal task-aware decoder refined at lr 2e-9 with all but two
layers frozen, improving at ZERO task cost).

Scripts: `render/joint_train2.py`, `render/overnight.sh`, `render/phase4.sh`.

## BREAKTHROUGH: PR #130's training code is public (2026-09-30, overnight)

**github.com/fesalfayed/comma-ai-semantic-pose-hpac-cpr1** — the full training pipeline for the
66,339-param renderer, the carrier, and HPAC, with checkpoints. Cloned to `D:\projectsesal130`.
Note: the 66,339-param renderer is **#130's**, not #86's. #86's master is a ~17.4k-param 3-layer CNN.
#130 ships NO training code in the challenge repo; everything downstream (#133/#135/#140/#141) only
losslessly repacks a predecessor archive.

### Their build script settles our thesis outright

```
semantic() { train_semantic_quantized.py --lr 2e-7 --ce-fraction 0.0 --softplus-fraction -999.0 ... }
carrier()  { train_pose_carrier_full.py --master-checkpoint <FROZEN renderer>                                         --lr-basis 1e-6 --lr-coeff 3e-4 --basis-freeze-fraction 1.0 }
```

The carrier is fitted against a renderer that has been rendered to images and then `del master_model`'d —
it is not even in the autograd graph. And **`--basis-freeze-fraction 1.0` means the 12 basis atoms are
NEVER trained**, only the 600x12 coefficients. So there are TWO untouched axes, not one:
joint renderer+carrier, and the basis itself.

Confirmed in both lineages: **PoseNet never appears in the seg renderer's loss.** #86's `master.py`
doesn't import it, and `slave.py` carries the comment `# pose info routed via FiLM only into slave
(so master stays seg-pure)`.

### Their actual loss (code/semantic_renderer_oracle.py)

```python
def target_margin(logits, target):
    target_logit = logits.gather(1, target[:, None])
    other = logits.clone(); other.scatter_(1, target[:, None], -1e9)
    return target_logit - other.amax(dim=1, keepdim=True)

# ship stage: ce_fraction 0.0, softplus_fraction -999.0  ->  tau resolves to ~0.05
loss = torch.sigmoid(-margin / tau).mean()      # "expected_flip"
```

A direct differentiable surrogate for the FRACTION OF WRONG PIXELS. It saturates to exactly 0 for a
confidently-correct pixel, so correct pixels contribute no gradient — the smooth form of the DAG
active-set masking we had reinvented from first principles. This is why their training can refine a
converged model where a mean-over-all-pixels CE destroys it.

### What we had wrong

| | ours | #130's |
|---|---|---|
| renderer lr | 1e-5 | **2e-7** (50x lower) |
| seg loss | hand-rolled hinge | expected-flip sigmoid, tau 0.05 |
| carrier coeff lr | 1e-7 (phase 4) | **3e-4** (3000x higher) |
| basis lr | same as coeff | 1e-6, and frozen entirely |
| optimiser | SGD | AdamW, clip 2.0, cosine eta_min 0.01*lr |

Also: 3-phase curriculum (annealed CE 1.0->0.08 -> softplus margin tau 0.20 -> expected flip), QAT at 4
bits throughout, batch 2, 6000 steps, seed 20260716. Float stage: 10000 steps, lr 5e-4, batch 8,
ce-fraction 0.75, softplus-fraction 0.92, checkpoint gated on true-uint8 argmax mismatch < 4.0e-4.

### First result with their recipe

seg_only, AdamW lr 2e-7, expected_flip tau 0.05, at epoch 5:
**seg 0.02024 vs baseline 0.02014 — 0.5% off, active set 0.020%** (exactly the genuinely-wrong
fraction). Best previous approach was 0.02050. Their loss finds the right pixels without any masking.

Phase 5 running: 5A their recipe seg-only, 5B JOINT at their rates, 5C carrier+basis only against the
joint objective with the basis UNFROZEN and the renderer untouched (so seg cannot degrade).

## Still open elsewhere

1. **~18 KB of weight-coding headroom in #86.** Their Table 7 claims 9.8 KB for the SCN packing then
   ships 27.6 KB via PPMd over the state dict. DeepCABAC publishes 1.48 bits/param non-sparse on
   LeNet-scale nets (~10.5 KB for our 57,359 params) and argues explicitly that Huffman-class coding
   "incurs up to 1 bit of redundancy per parameter". Worth ~0.012. Arithmetic on artifacts we hold, not a
   training gamble. Our own naive int8+brotli measured 6.25 bits/param = 43.7 KB, so the coder choice is
   worth 4x.
2. **Depth-scaled warp.** The affine warp got 12x better than grey and failed only on flow *shape*. Using
   the token map as a depth proxy (sky far, road near) would give radial-with-1/depth flow. Content stays
   free. This is a build, not a probe.
3. **Token stream is 51% of the score** and our 0.0077 bpp is competitive with the 2026 chain-coding SOTA
   (0.0072 on 5-class DAVIS, intra-only, ~0.0104 scale-normalised to our resolution). No learned lossless
   label-map codec is published at all.

## Gotchas that cost real time today

- `frame_utils.rgb_to_yuv6` is `@torch.no_grad()`. Use `.__wrapped__` or every gradient into frame 0 is
  silently zero.
- The carrier render divides by `sqrt(CARRIER_DIM)`. Omitting it makes the field 3.46x too strong and the
  baseline reads mse 69 instead of 6.4e-06 -- a factor of ten million.
- Upsample is **bicubic**, with `round()` on both sides.
- Never train a shared basis with per-chunk optimiser steps. A k=12 config whose init reproduced the
  baseline (1.00x) went to 31.96x worse after 1500 such steps. Either freeze the basis and refit
  coefficients per chunk, or accumulate gradient over all 600 pairs before stepping.
- Cross-entropy on SegNet logits is **anti-correlated** with seg distortion here: it drove #141's renderer
  from 0.0201 to 0.109 while falling monotonically. Use the margin/softplus loss.
- The carrier render divides by sqrt(k), so changing k rescales the WHOLE field by sqrt(12/k). Going 12->13
  without compensating the inherited coefficients starts 500x worse (3.3e-03 vs 6.6e-06) from a 4% amplitude
  shift alone, and the sweep then measures recovery from that rather than the thing being swept.
- Always run a control arm at the known-good configuration. FOUR separate setup bugs today were caught this
  way and would otherwise have been reported as findings. Also check results against what is PROVABLE: k=13
  contains k=12 (13th coefficient zero) so it cannot be worse at the optimum -- a result showing otherwise
  is measuring the optimiser, not the model.

---

## Phases 6-7: the carrier gradient is BIASED, not mis-tuned (2026-09-30, overnight)

**Phase 6** swept carrier learning rate with the renderer frozen: 3e-5, 3e-6, 3e-7. Every arm returned
TOTAL **0.02827** -- exact parity -- and in every arm the best pose reached was **0.00814, precisely the
starting value**. Pose never once went below where it began. Seg held at exactly 0.02014, confirming the
frozen renderer really was frozen and the harness is sound.

**Phase 7** used PR #130's *own* carrier loss, recovered from their `train_pose_carrier_full.py`:

```python
target_scale = (targets.amax(0) - targets.amin(0)).clamp_min(1e-4)
loss = normalized.square().mean() + 0.02 * residual.square().mean()
```

With their loss, their learning rates, and the renderer frozen, pose went **up monotonically**:

| epoch | 0 | 5 | 10 | 15 |
|---|---|---|---|---|
| pose | 0.00814 | 0.02454 | 0.02765 | 0.02931 |

### The conclusion

Monotone degradation at **three decades of learning rate** is not a step-size problem. A too-large step
oscillates or diverges; it does not walk steadily downhill in the wrong direction at every scale. This is a
**biased gradient** -- the direction itself is wrong.

The cause is almost certainly the quantization surrogate. The carrier render path is

```
carrier = einsum(coef, basis) / sqrt(12)
slave   = round(clamp(127.5 + 64 * carrier))     <-- here
```

and we replace that `round`/`clamp` with `soft_round` (cubic, 2309.06978) and `soft_clip`. That surrogate
is *fine for seg*, where ~40 pixels out of 786k are wrong and the loss only needs to identify which way to
push them. It is **not** fine for the carrier, which was fitted to roughly 0.02% precision: a surrogate
error of the order of the cubic correction term is itself larger than the optimum's basin, so descent walks
away from a point it should sit still at.

This retro-explains phases 1-6 in one stroke. It is also why #130 renders the master to images and `del`s
it before fitting the carrier -- not laziness, but avoidance of exactly this path.

**What it does NOT refute:** the joint thesis. It refutes *this implementation* of the carrier half. The
renderer half is unaffected -- seg trains cleanly through the same surrogate, as six phases of exact
parity show.

### Why parity was the ceiling all along

#141 ships PR #130's **post-tail** renderer. Their checkpoint metadata:

| checkpoint | quantized_exact_seg |
|---|---|
| `semantic_renderer_w96_b4_qat4_12k.pt` (pre-tail) | 0.00029725 |
| `..._fixedtau05_tail6k_lr2e7.pt` (shipped) | 0.00027637 |

Every phase so far started from an **already-converged** checkpoint. Parity was the best available outcome
and we kept hitting it exactly. Their final 6,000-step tail bought 2.09e-5 of seg = **0.0021 of score**;
that is the headroom, and it is only reachable from the pre-tail checkpoint.

### Phase 8: the clean head-to-head

Both arms start from the **pre-tail** 12k checkpoint, both **freeze the carrier** (so the biased carrier
gradient cannot confound the comparison), and differ in exactly one variable:

- **8A** renderer trained on seg alone -- replays #130's tail, their way
- **8B** renderer trained on seg + pose -- the thesis

Same start, same lr 2e-7, same tau 0.05, same budget, same scoring. Run concurrently so an early GPU
cutoff still leaves a paired comparison at equal epochs.

---

## The QAT discovery: we had been evaluating #130's checkpoints wrong

Phase 8's step-0 eval read seg **0.11011** from a checkpoint whose own metadata claims 0.00029725. The
cause: PR #130's renderer is trained with **quantization-aware training**, and their `state_dict` stores
the **float master weights**, not the deployed model. Their `train_semantic_quantized.py` re-quantizes
every parameter on every forward:

```python
scale = source.detach().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8) / limit
scale = scale.to(torch.float16).float()
normalized = (source / scale).clamp(-limit, limit)
codes = normalized + (normalized.round() - normalized).detach()   # STE
return codes * scale
```

Under QAT the float master is only a latent variable -- it may sit anywhere inside a quantization bin --
so **the raw floats are not a usable model**. Applying their projection: seg **0.11011 -> 0.02558**.

### #141's renderer IS 4-bit (measured, not assumed)

Per-output-channel distinct weight values, #141 vs #130's float master:

| param | #141 | #130 12k |
|---|---|---|
| `coord_mix.weight` | 12 | 100 |
| `blocks.*.pw.weight` | 14-15 | 96 |
| `head.weight` | 15 | 864 |

15 = 2^4 - 1 exactly. Re-quantizing #141's weights is **exactly idempotent** (max relative change 0.0)
on every tensor but two, which confirms both that #141 ships a 4-bit renderer and that our implementation
of their projection is correct.

### A latent nan bug in #130's own quantizer

Three tensors returned **nan**: `blocks.{1,2,3}.film.weight`, which are exactly zero in #141.

```
amax = 0  ->  clamp_min(1e-8)  ->  scale = 1.43e-9  ->  .to(float16) == 0.0  ->  0/0 = nan
```

fp16's smallest subnormal is 5.96e-8, so the scale underflows to exactly zero and the next line divides by
it. #130's training never produced an exactly-zero row, so they never hit this. Fixed by substituting a
unit scale where the fp16 cast underflows -- exact, since the row is all zeros anyway. With the fix,
#141 + quant4 reads seg **0.02039** against 0.02014 unquantized.

### The mechanical fact that justifies the whole thesis

After the fix, only `frame_embed.weight` and `blocks.0.film.weight` remain non-idempotent, each perturbed
by **4.8%**. That alone moves pose from **0.00814 to 0.03584 -- 4.4x worse**, while seg barely moves
(0.02014 -> 0.02039).

**A 4.8% change in a single embedding tensor degrades pose 4.4x.** The carrier is fitted to one exact
renderer output; any renderer change destroys pose unless something compensates. A pose-blind renderer
training run is therefore walking a tightrope it cannot see -- which is exactly the coupling the joint
objective exists to exploit.

### A byte lever that is NOT there

`blocks.2/3.pw.weight` have maxabs 0.0066 / 0.0060 against 0.31 / 0.36 in blocks 0/1 -- 50x smaller -- and
their FiLM weights are exactly zero, which looks like two dead blocks worth ~11.6 KB (0.0077 of score) to
delete. It is not: `residual = norm(pw(dw(value)))` puts a **GroupNorm immediately after pw**, which
renormalizes the activations and undoes the weight scale entirely. The blocks are functional. The zero
FiLM weights only disable the multiplicative modulation (the shift survives), and zeros compress to
nothing, so there is no byte saving there either.

---

## Phase 8: the joint thesis demonstrated

Both arms start from #130's **pre-tail** 12k checkpoint with the QAT projection applied, both **freeze the
carrier**, both run lr 2e-7 / tau 0.05 / expected-flip. They differ in exactly one thing: whether the pose
term is in the renderer's loss.

| epoch | 8A seg-only (their way) | 8B joint (ours) |
|---|---|---|
| 0  | 0.38463 | 0.38463 |
| 2  | 0.38463 | 0.33489 |
| 6  | 0.38463 | 0.33091 |
| 10 | 0.37269 | **0.24402** |

Seg is essentially identical in both (~0.0243 vs ~0.0263); the entire difference is pose, which the joint
arm drags from 0.451 down to 0.218 while the seg-only arm's pose wanders between 0.42 and 0.57 with no
trend.

**This is the thesis, and the frozen carrier is what makes it visible.** The carrier cannot move, so it
cannot clean up after the renderer. A renderer with pose in its objective learns to emit masters that the
*existing* carrier can already pair with. A pose-blind renderer has no way to know it should -- it is
optimising a quantity that is 100% of its loss and ~6% of the score.

### Phase 9: making the comparison honest

Both arms finish with a carrier still fitted to #141's ORIGINAL renderer, so pose ~0.22-0.45 is nowhere
near the true operating point of 0.00814. A real submission refits the carrier to whatever renderer it
ships, so `refit_carrier2.py` does that refit **identically for both arms** -- #130's own procedure, master
rendered under no_grad on the exact path with the renderer out of the graph, coefficients at 3e-4, basis
frozen. Only after that refit is the A/B number meaningful.

Note this also sidesteps phase 7's biased gradient: with the renderer out of the graph, `soft_round` never
touches the carrier's gradient.

### Honest scope of the claim

Phase 8 cannot beat #141's 0.02827 outright and was never able to: the pre-tail start is seg 0.02558 alone,
so even a perfect pose term (0.00814) floors at 0.0337. What phase 8 establishes is the **mechanism** --
that the joint objective buys a large, monotone improvement a sequential pipeline cannot reach on an
identical budget. Converting that into a number below 0.02827 needs the same treatment applied from a
stronger renderer start, which is the next run.

---

## A silent gradient killer in the official code

`frame_utils.rgb_to_yuv6` in the challenge repo carries **`@torch.no_grad()`**. `PoseNet.preprocess_input`
calls it, so **any pose objective built on the official path is a constant** -- backward raises
"element 0 of tensors does not require grad", or worse, silently contributes nothing.

`joint_train2.py` line 75 already works around it:

```python
modules.rgb_to_yuv6 = frame_utils.rgb_to_yuv6.__wrapped__
```

Anything new that optimises a pose term must do the same or it is optimising nothing. `refit_carrier2.py`
omitted it at first and every pose gradient was dead.

**This does not invalidate phases 6-8** -- they all ran through `joint_train2.py`, which unwraps it. It was
worth checking: if the pose term had been dead in phase 8, arms 8A and 8B would have been optimising
identical objectives and their 0.28856 vs 0.21831 split would have been noise rather than the thesis. They
are not identical, and the unwrap is why.

## Phase 10: the last untouched axis is closed, negatively

#130 set `--basis-freeze-fraction 1.0`, so the 12 basis atoms have never been trained by anyone in the
lineage, and the basis is already stored in the archive -- changing its VALUES costs zero bytes. That made
it the best remaining candidate for an outright win.

Result: unfreezing the basis (coeff 3e-4, basis 1e-6) with the renderer out of the graph and the master on
the exact path drives pose **0.01027 -> ~0.048-0.054 within 100 steps** and it never recovers; best never
leaves its step-0 value.

Taken with phase 6 (lr 3e-5/3e-6/3e-7, no improvement) and phase 7 (#130's own loss, monotone
degradation), the carrier is confirmed to sit at a well-converged optimum that gradient descent through
the `soft_round` slave path cannot improve from **any** direction tried -- coefficients, basis, or both.

The same script moving the carrier for the 8B renderer improves TOTAL 0.17868 -> 0.10486 monotonically, so
the optimiser is sound; it descends fine when there is real error to remove. The failure is specific to
refining an already-converged fit, exactly as the biased-surrogate diagnosis predicts.

---

## Where the night ended (2026-09-30)

The host reclaimed the box mid-way through the final round (`INSTANCE_GONE`, SSH refused, vast reports
0 instances -- **nothing is billing**). Round 2 never ran, so one planned number is missing.

### What is established

| run | result |
|---|---|
| Phase 8A seg-only (their way) | **0.28856** |
| Phase 8B joint (ours) | **0.21831** |
| 8B + carrier refit | 0.09881 at step 300, plateauing |
| 8A + carrier refit | **never ran** -- box reclaimed |
| Phase 10 basis unfrozen | no improvement; best never left step 0 |

Both phase-8 checkpoints survived the reclaim and are saved at
`work/render/overnight/p8a_seg_only_best.pt` and `p8b_joint_best.pt`, so this resumes without retraining.

### The honest bottom line

**Nothing tonight beats #141's 0.02827, and phase 8 structurally could not.** Its start is #130's pre-tail
checkpoint at seg 0.02558 alone, so even a perfect pose term floors it at 0.0337. Phase 8 was built to
test the *mechanism*, not to win, and it does: from identical starts with the carrier frozen and the ONLY
difference being whether pose is in the renderer's loss, joint beats sequential by 25%.

**The end-to-end claim is not yet proven.** The refit comparison is half-finished -- 8B refits to ~0.0988,
and 8A's refit is the missing control. Without it, the 25% gap is demonstrated at the training objective
but not at the real operating point. That one run is the first thing to redo.

### What was actually worth the night

The negative results are sharper than the positive one:

1. **#130's checkpoints are QAT float masters** -- must be fake-quantized on forward (0.11011 -> 0.02558).
2. **An fp16 underflow nan bug** in their quantizer, triggered by #141's all-zero FiLM rows.
3. **`rgb_to_yuv6` carries `@torch.no_grad()`** -- any new pose objective is silently dead unless unwrapped.
4. **The carrier optimum is closed in every direction tried** -- coefficients, basis, and both, across four
   decades of learning rate, with and without the renderer in the graph.
5. **Why six phases returned exactly 0.02827**: #141 ships the post-tail renderer. There was no headroom,
   so parity was the correct answer, not a bug. Chasing it as a bug cost most of the session.

---

## The real bug: #130's carrier loss is anti-correlated with the metric (2026-09-30, second session)

Phases 6, 7 and 10 all concluded "the carrier sits at an optimum that gradient descent cannot improve from
any direction." **That conclusion was wrong, and the cause was the loss, not the gradient.**

The validation that exposed it: refit #141's OWN carrier against its OWN renderer. It should sit still or
improve. Instead it degraded 0.01138 -> 0.0354 within 5 epochs and stayed there regardless of lr decay.

Two hypotheses tested and killed:
* **Rounding surrogate.** The cubic `soft_round` has derivative `3(x-f)^2`, which is ZERO on integers --
  exactly where a converged carrier sits. Plausible, but straight-through rounding degrades identically
  (0.01138 -> 0.0354). Not the cause.
* **Lossy master cache.** Asserted `uint8` round-trip equality; it never fired. Not the cause.

**The cause is #130's carrier loss:**

```python
target_scale = (targets.amax(0) - targets.amin(0)).clamp_min(1e-4)
loss = (resid / target_scale).square().mean() + 0.02 * resid.square().mean()
```

Dividing each pose dim by its RANGE gives all six dims comparable pull. But the scored metric is
`sqrt(10 * mse)` over the raw residual, and **d0 carries 99.8% of the variance**. So this loss spends the
carrier's 12 degrees of freedom improving five dims that barely register, paying for it in d0 -- the one
that is actually measured. The loss falls while the metric rises.

That is fine for fitting from scratch, where every dim needs to get roughly right. It is actively wrong
for refining a fit that is already tuned to the metric.

Swapping to plain MSE (which IS the metric) fixes it immediately -- #141's own carrier now **improves**
monotonically, every eval a new best:

| epoch | 0 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| pose (pr130 loss) | 0.01138 | 0.03468 | 0.03481 | 0.03510 | 0.03541 |
| pose (MSE loss) | 0.01138 | 0.01062 | 0.01048 | 0.01020 | **0.01007** |

So the carrier is **not** converged and never was closed. Three phases of "negative results" were an
artefact of optimising the wrong objective.

### Absolute pose is not portable between machines

This box (RTX 3090) reads #141 as seg 0.02019 / pose 0.01138; the previous box read 0.02014 / 0.00814.
**Seg differs too**, and seg depends only on the master -- so the master itself differs. Different GPU and
torch version perturb the conv/bilinear path by a fraction of an LSB, `.round()` flips a few pixels, and
pose is hypersensitive to exactly that (measured earlier: a 4.8% weight change moved pose 4.4x).

Consequence: **compare only within one machine.** All four refits run on this box against this box's own
#141 baseline of 0.03157.

---

## END-TO-END RESULT: the joint thesis holds after a full carrier refit

All four renderers given an **identical** full carrier refit (300 epochs, plain-MSE loss, deterministic
full coverage, cosine decay) on the same box, so they are directly comparable:

| renderer | seg | pose | **TOTAL** |
|---|---|---|---|
| #130 pre-tail 12k, untrained | 0.02560 | 0.04156 | 0.06716 |
| **8A seg-only (#130's recipe)** | 0.02379 | 0.02626 | **0.05005** |
| **8B joint (ours)** | 0.02667 | 0.01422 | **0.04089** |
| *#141 stock, reference* | *0.02019* | *0.01138* | *0.03157* |

**8B beats 8A by 0.00916 -- 18% better -- end to end, after both got the same full refit.**

### The prediction this tested, and why it failed

The expectation was that a refit would absorb the pose error for both arms, collapsing 8B's pose advantage
while its seg deficit stayed permanent (seg is untouchable by the carrier), handing the win to 8A. The
arithmetic was right; the premise was not:

* 8B pays **0.00288** of permanent seg
* 8B gains **0.01204** of pose *that survives the refit*
* net **+0.00916** to 8B

The pose gap did not collapse: 0.02626 vs 0.01422, still **46%** apart after both were refitted to
convergence. The reason is the carrier's capacity -- 12 atoms spanning ~0.7% of image space cannot absorb
arbitrary renderer error. 8A's refitted pose (0.02626) sits **2.6x above the achievable floor** (~0.010,
what #141's co-developed pair reaches), so the carrier demonstrably cannot repair what a pose-blind
renderer breaks. Putting pose in the renderer's loss buys something no amount of carrier fitting recovers.

Note also 8A's pose improved (0.04156 -> 0.02626) without pose in its loss -- that is the refit adapting to
a changed renderer, not the renderer learning anything about pose.

### A free gain on #141 itself

Refitting #141's shipped carrier with the corrected loss:

| | seg | pose | TOTAL |
|---|---|---|---|
| stock | 0.02019 | 0.01138 | 0.03157 |
| refit | 0.02019 | **0.01009** | **0.03027** |

**0.00130 of distortion at zero byte cost** -- the coefficient COUNT is unchanged (600x12), only the
values differ. Verified that the shipped coefficients are stored at ~float precision, not int12
(max residual 0.4995 against an int12 grid, 5987 distinct values of 7200), so replacing them is free.
An earlier int12 "shippable" column was modelling a quantization the archive does not use; it also
degraded the STOCK coefficients (0.03157 -> 0.03323), which is what exposed the wrong assumption.

### Caveats, stated plainly

* **None of this beats #141's shipped 0.02827.** Both phase-8 renderers got only 40 epochs from the
  pre-tail checkpoint and neither approaches #141's seg of 0.02019. The thesis is proven; the leaderboard
  is not moved.
* **Absolute pose is machine-dependent.** This box reads #141 as pose 0.01138 where the original read
  0.00814. All comparisons above are internally consistent on one box, but the #141 refit gain must be
  re-verified on the original setup before being claimed as a real score improvement.

---

# THE REFERENCE FACTS (2026-09-30, third session)

Everything below was read out of the shipped archive or the authors' own repos, not inferred. The previous
two sessions' results were built on assumptions, six of eight of which were wrong.

## The lineage, exactly

| PR | seg | pose | bytes | score | what it did |
|---|---|---|---|---|---|
| #130 Fesal | 0.02966 | 0.01527 | 0.12721 | 0.17214 | the vehicle: renderer + 12-atom carrier |
| #133 JasonMo | 0.02966 | 0.00947 | 0.12665 | 0.16578 | basis atoms 2/5/9 to 4-bit (least-error screen) + int12 coeff re-solve |
| #135 codexblack | 0.02964 | 0.00830 | 0.12433 | 0.16227 | ~4.3 KB lossless + Jacobian carrier search + frame-0 edits |
| **#140 adpena** | 0.02014 | 0.00798 | 0.11986 | **0.14798** | **joint token-edit admission** + pruned mixed-precision renderer + context mixing |
| #141 | 0.02013 | 0.00798 | 0.11978 | 0.14790 | +111 bytes lossless repack |

#130 -> #141 = 0.0242, split almost evenly: seg -0.0095, pose -0.0073, bytes -0.0074 (-11,161 B).

## #141's renderer encoding (SM3R row-prune-mixed), measured

| group | precision |
|---|---|
| `pw`, `dw`, `coord_mix`, `head`, `token_embed` weights | **4-bit** |
| `frame_embed.weight`, `blocks.0.film.weight` | **3-bit** |
| `blocks.{1,2,3}.film.weight` | 4-bit, **row-pruned, 2 of 192 rows kept** |
| every rank<2 tensor (biases, norm affines) | **fp16** |

Dequantization is `codes * fp16_scale`, scale along the LAST dim for `*embed.weight` and dim 0 otherwise.
Code range is symmetric `+-((1<<(bits-1))-1)`, confirmed: the 3-bit tensors show implied |code| max of
exactly 3.

**Our `faithful.py` quantizer is idempotent 38/38 on the shipped weights.** The old uniform-4-bit version
produced seg 76.77 / pose nan on those same weights, because it mis-coded the two 3-bit tensors and
treated the pruned rows as trainable.

## #141's renderer is #130's post-tail, almost exactly

Applying the faithful encoding to `semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt` reproduces
**35 of 38** tensors bit-exactly. The three that differ:

* `blocks.3.film.weight` differs at **one flat index, 1514** (ours +0.00126266, shipped -0.00126266 --
  exactly two code steps). #135's `F15_JOINT_RENDERER_POLISH.md` records accepting moves at `[703]` and
  `[1514]`, each -1. Index 703 lives in row 87, which is pruned away, so only 1514 survives. **That single
  symbol is the only real weight change in the entire #130 -> #141 span.**
* `frame_embed.weight` and `blocks.0.film.weight` were re-derived, not projected -- consistent with #140's
  mixed-precision search re-solving codes rather than rounding #130's floats.

## The 14-byte frame-0 selector

`F0E1`, version 1, count 5. Active frames **60, 85, 116, 241, 373**; 595 are identity.

| frame | mode | op |
|---|---|---|
| 60, 116, 373 | ROLL(1,0) | shift slave 1px horizontally |
| 85 | CHANNEL(1,0,-1) | +1 R, -1 B |
| 241 | TILE(3,1) | +-1 checkerboard on 4x4 blocks |

Applied to the final uint8 camera-resolution **slave** frames only, so it changes pose and cannot touch
seg. Measured worth: **2.471878e-07 of mean d_pose** (0.00015 of score). Frame 116 improves 7,700x.
Frame 85's mode makes pose WORSE in our path (+1.47e-5), so their search was not exhaustive.

## Our harness was right all along; one box was not

```
official, with selector     : pose term 0.0079813
official, selector removed  : pose term 0.0081347
box 1 (first session)       : pose term 0.00814     <- matches to 3 s.f.
box 2 (RTX 3090)            : pose term 0.01138     <- 40% high
```

Box 1 was correct to within the missing selector. The "43% environment error" cited as a blocker was
**specific to the 3090**, almost certainly cuDNN conv TF32, since `evaluation.json` records
`numeric_policy: "IEEE / TF32 disabled"` and no script of ours ever set it. Phase 8 ran on box 1; the four
carrier refits ran on the 3090.

## Prior art that was missed for two sessions

* **#135 already did joint renderer optimisation.** `docs/F15_JOINT_RENDERER_POLISH.md` is "joint
  SegNet/PoseNet int4 renderer polish", with joint Jacobians over both frozen scorers. Gain: **-0.0000737**
  of combined score, from **two symbols** moved by -1. It was a +-1 discrete search over ~5 tensors, not
  gradient training -- so the mechanism differs, but the local neighbourhood is evidently flat.
* **#140's "joint" is token-edit admission**, not training: candidate token edits priced against exact
  pose cost through frozen PoseNet, admitted by a **Lagrange-multiplier waterfill** (455 of 573 admitted),
  then a damped Gauss-Newton carrier re-solve. "The largest single move", and what crossed 0.15.
* **#140 measured the renderer-weight axis and closed it** (`ddm_pr1`): a seg-motivated renderer step with
  terminal pose re-solve recovers 16.42x but lands 148x above base pose, 41.5x over their payable bar.
  Their deeper finding: along the realizable direction **d_seg rises both ways**.
  NOTE their closure is **direction-specific** -- see the asymmetry below.

## The direction asymmetry (why the axis may still be open)

Score sensitivity at the shipped operating point:

```
d(score)/d(d_pose) = 626.5      d(score)/d(d_seg) = 100.0
=> pose is 6.26x more score-sensitive per unit distortion, and MORE so as pose shrinks (sqrt term)
```

With their own measured coupling `k_post = 13.82` (a renderer step moves pose 13.8x more than seg):

| direction | gain | collateral | net |
|---|---|---|---|
| theirs: buy seg, pay pose | +100 | -8,658 | **-8,558 (hopeless)** |
| ours: buy pose, pay seg | +626.5 | -7.2 | **+619 (favourable)** |

The same constant that kills their direction favours ours 86:1. Phase 8's 8B is consistent: it paid
0.00288 of seg to gain 0.01204 of pose, net +0.00916.

## #140's "what did not work" (saves us from re-walking it)

* Lossy quantization of the learned tensors -- worse at every depth; "the render amplifies weight error
  faster than the byte credit pays."
* Distilling the renderer smaller -- pose cost tens of times the byte saving.
* Swapping/re-tuning the entropy coder -- 25 Brotli/LZMA configs, no saving on frozen model sections.
* Reordering the token stream -- 113,777 -> 113,777 bytes.
* Explicit "fix this pixel/token" corrections -- addresses cost more than the fixes.
* Small fitted pose-correction layers (43 B, 247 B, 997 B) -- all held-out negative.
* Parametric lane curves instead of dense tokens -- 1.7x larger at best.

**Two doors they name as still open:** changing the **context model** (as opposed to swapping coders), and
cross-group token reordering (which needs a different context model). Plus their representation limit:
100% of pairs want a GN step beyond the +-2 radius, 9.67% beyond the whole 4,095-unit lattice, 542/600
stall at `no_improving_step`, and **10 pairs own 69% of the post-solve pose error**.

## A byte axis that looks untouched

The **fp16 rank<2 group**: 2,403 params, **4,806 bytes = 0.00320 of score**. #135's F15 states those groups
were "recorded as skipped rather than mutating them". int8 would save 0.00160; int6, 0.00200 -- as large as
the entire expected gain from pose training, with no training at all. Must first confirm #140's "lossy
quantization was worse at every depth" does not already cover it.

## GATE PASSED -- the harness reproduces #141 (RTX 5090, torch 2.9.0+cu128, TF32 off)

| | measured | official | delta |
|---|---|---|---|
| seg term | 0.020136515299 | 0.020133973157 | **+0.0126%** |
| pose term | 0.007980565238 | 0.007981329070 | **-0.0096%** |
| posenet_distortion | 6.368942e-06 | 6.370161e-06 | |

Reconstructed score 0.1478991 vs official 0.1478973. For scale, #140's own cross-check of their base
d_pose against the official T4 receipt was -0.068%; this is ~7x tighter than their reported tolerance.
The residual is almost certainly the GT decode (our DALI cache vs the evaluator's) plus minor numerics.

Setup that mattered: torch **2.9.0+cu128** (the evaluator's exact version), torchvision 0.24.0,
RTX 5090 (Blackwell, like the official 5080), `set_numeric_policy()` for IEEE/TF32-off, the faithful
per-tensor quantizer, and the frame-0 selector applied.

**Selector control, same run otherwise:**

| | pose term | seg term |
|---|---|---|
| applied | 0.007980565 | 0.020136515 |
| omitted | 0.008135535 | 0.020136515 |

Worth **1.5497e-04** of pose term = 0.000155 of score. Seg is **bit-identical** between the two, which
confirms empirically that the selector is slave-only and cannot touch SegNet. Box 1's 0.00814 reading is
now fully explained as correct-harness-minus-selector (0.0081355).

**This is the first validated harness in the project.** Anything measured before it should be treated as
provisional.

## Gradient fine-tuning is the wrong instrument for this model

With the validated harness, `pose_aware.py` reproduces step 0 exactly (seg 0.0201365 / pose 0.0079806) and
then degrades at every step size tried. Three measured reasons:

**1. The code step varies 50x across tensors.** `blocks.0.pw.weight` has amax 0.31 (4-bit code step 0.044);
`blocks.2/3.pw.weight` have amax 0.006 (code step 0.0009). A single global lr either freezes the first or
shreds the second. At a flat lr 2e-5, **32,877 of 66,339 codes moved in one epoch** and pose went 11x worse
(0.00798 -> 0.0876). Fixed by scaling each tensor's lr by its own code step, so `--lr` means "fraction of a
code per Adam step" -- the only unit that is comparable across tensors.

**2. A gradient step is not byte-neutral.** Moving any weight can change its tensor's `amax`, which changes
the stored fp16 scale and therefore **every dequantized value in that tensor**. F15 by contrast reports
"the archive remains 189,661 bytes (zero rate change)" for accepted single-code flips.

**3. We are starting at a joint local optimum.** #141 already contains #135's F15 accepted flips and
#140's work, with the carrier solved against it. Any coherent renderer move degrades pose sharply --
consistent with `ddm_pr1` measuring 2,432x base pose for a renderer change against a stale carrier, which
is exactly our frozen-carrier setup.

### An instrumentation bug worth recording

My first `codes_changed` compared **dequantized values**, so it counted scale drift as code flips and
reported ~40,000 changes per epoch when the integer codes had barely moved. It now compares integer codes
and reports moved-tensor count separately. Without that fix the whole sweep would have been misread.

## The better-aimed method: Jacobian-ranked single-code flips

`flip_search.py`. F15's protocol is the one the lineage validated, and it was applied to only **5 tensors**
(`head.bias`, `head.weight`, `token_embed.weight`, `blocks.3.film.weight/bias`, `blocks.3.norm.weight/bias`),
accepting two symbols. **Never searched:** `coord_mix`, `frame_embed`, and blocks 0-2's dw/pw/film/norm,
plus blocks.3.dw/pw -- the large majority of the 66,339 codes.

Rank every legal +-1 flip by the score-matched Jacobian `L = 100*d_seg_surrogate + 626.5*d_pose`, screen the
top candidates on a pair subset, then **exact-gate survivors on all 600** and accept only confirmed
improvements. Flips are byte-neutral and local.

**Why the ranking is well-aimed:** with `k_post = 13.82` and pose 6.26x more score-sensitive, a flip that
improves pose by one unit gains 626 while costing only 7.2 in seg; a flip that improves seg gains 100 while
costing 8,658. The pose-weighted Jacobian naturally surfaces the former.

**Expected magnitude, stated honestly:** F15 got 7.4e-5 from two flips. Ten comparable flips across the
unsearched tensors would be ~4e-4 of score -- real and byte-free, but an order below the 0.001-0.002 hoped
for from the training route.

## Flip search result, and the state of the renderer axis

**Pass 1, tensors F15 never inspected: 0 of 52 candidates accepted** at the exact 600-pair gate.

```
candidate code steps: frame_embed=1.2, coord_mix=0.0696, blocks.1.dw=0.0572, blocks.0.dw=0.0498
round 1: 52 ranked candidates, best predicted gain 3.714e-03
screen(100 pairs) best 0.0335285  worst 1.3639595     (baseline on 100 pairs ~0.028)
round 1 done: accepted 0  ->  stopped
```

Even the best single flip is ~20% worse than baseline, so the Jacobian's first-order prediction is not
trustworthy here -- a full code step is far outside the linear regime for these coarse tensors. That is
precisely why every candidate is exact-gated rather than taken on predicted gain.

**A ranking bug found and fixed mid-run.** Ranking globally by `|grad| * scale` fills the shortlist
entirely with the COARSEST tensors, because their predicted gain is largest -- but a "+-1 flip" there is
not a perturbation: `frame_embed.weight` is 3-bit with amax 3.627, so one code is **1.209** of weight
change, while `blocks.2/3.pw.weight` move **0.0009**. A global top-48 screened 0.285 to 2.007 against a
0.028 baseline, i.e. every candidate catastrophic. Taking the best few from EACH tensor fixed it (best
screened 0.285 -> 0.0335), and `--per-tensor` now controls that.

A second hole: `--topk` truncating the stratified set silently drops the FINE-grained tensors (they rank
last by `|grad|*scale`), so topk must be >= per_tensor x n_tensors. The thorough pass (per_tensor 64,
topk 1200, all tensors, gate 40) was **not completed** -- see the operational losses below.

### Where the renderer axis stands

Two independent methods on the validated harness, both negative:
* gradient fine-tuning degrades at every step size (4 code flips alone cost 0.0016 of score);
* Jacobian-ranked single-code flips accepted 0 of 52 over the unsearched tensors.

That is strong but not complete. The honest statement is: **no improving single-code flip was found among
52 Jacobian-ranked candidates, and the thorough 1,150-candidate sweep has not been run.** Do that before
declaring the axis closed.

### Operational losses this session, both mine

* **OOM from concurrency.** Ran the trainer and the flip search on one 32 GB GPU assuming ~10 GB each; the
  trainer alone took 24.9 GB and both died. Run one at a time.
* **A guard destroyed a live box at 47 min.** `nvrc-proto/guard.sh` takes a filename whitelist as
  `TRAIN_PAT` and declares TRAINERS_DEAD after 6 empty polls. `flip_search.py` was written AFTER the guard
  was armed, so the guard never saw it. This was a repeat -- the same bug had already been fixed earlier in
  the session for `refit_carrier3.py`. **Use a shape pattern** such as `"[p]ython [a-z_0-9]*\.py"`, never a
  name list, and restart the guard whenever a new script name appears.
* A slow host sat at `created` for 10+ minutes and was destroyed unused.

### Resume checklist

1. Rent a **Blackwell** card (5090/5080), NOT Ampere -- a 3090 read pose 40% high.
2. `pip install torch==2.9.0 torchvision==0.24.0 --index-url .../cu128` (the image's bundled torch cannot
   run sm_120, and `pip install -q` fails silently leaving it in place).
3. Upload `models/*.safetensors`, `work/extracted/pr141_components.pt`, `work/targets/gt_dali_t2000.pt`,
   `work/render/{renderer,faithful,validate_official,flip_search,pose_aware}.py`, and
   `submissions/semantic_blocks/runtime/frame0_selector.py`, into the layout `validate_official.py` expects.
4. **Gate first:** `validate_official.py` must read seg 0.0201365 / pose 0.0079806.
5. Arm the guard with a shape pattern AND start a local puller before launching.
6. Then: `flip_search.py --per-tensor 64 --topk 1200 --gate 40 --rounds 2`.

---

# RENDERER-WEIGHT AXIS: CLOSED (2026-09-30, final run)

The thorough sweep ran to completion on a third RTX 5090, and the gate reproduced **bit-identically** to
the second box:

```
GATE            measured              official
seg term        0.020136515299        0.020133973157   (+0.0126%)
pose term       0.007980565238        0.007981329070   (-0.0096%)
```

Three independent reproductions now agree (two different 5090 hosts to the same digits, plus CPU at
-0.0084% / +0.0250%). The harness is settled.

**Thorough flip search: 874 ranked candidates across ALL tensors, 0 accepted.**

```
candidate code steps: frame_embed=1.2, token_embed=0.322, coord_mix=0.0696,
                      blocks.1.dw=0.0572, blocks.0.dw=0.0498, blocks.2.dw=0.0475
round 1: 874 ranked candidates, best predicted gain 3.714e-03
screen(100 pairs)  best 0.0335285   worst 2.0065199
accepted 0   ->   FINAL TOTAL 0.0281171, delta -1.95e-08 (i.e. unchanged)
```

The decisive detail: **the best screened candidate, 0.0335285, is identical to the 52-candidate pass.**
Adding 822 more candidates -- including every fine-grained tensor that the earlier `topk` truncation had
excluded -- surfaced nothing better. The Jacobian had already found the best available flip, and it is
~20% worse than doing nothing.

## Verdict

The renderer weights are a local optimum of the joint (seg, pose) score, confirmed by two independent
methods on a validated harness:

| method | result |
|---|---|
| gradient fine-tuning, per-tensor code-step lr | degrades at every step size; 4 code flips cost 0.0016 |
| Jacobian-ranked single-code flips, 874 candidates, exact-gated on 600 pairs | **0 accepted** |

This is consistent with, and extends, #135's F15 result: they searched 5 tensors with the same +-1
protocol and accepted 2 symbols, both of which are already in #141. We searched everything and found no
third.

**Scope of the claim, stated precisely:** no single +-1 code flip improves the exact 600-pair score, among
874 Jacobian-ranked candidates covering all 18 quantized tensors. Not ruled out: multi-flip combinations
(the search is greedy and one-at-a-time), moves larger than one code, or changes that pay only after a
carrier re-solve. Those are open in principle but all cost bytes or compute that the 0.00798 pose ceiling
cannot repay.

**Therefore the shipped renderer is a safe fixed base** for work on the other axes, which is what the peer
session needed to know.

## Where the remaining value is

| axis | ceiling | status |
|---|---|---|
| renderer weights | -- | **closed** (this section) |
| carrier coefficients | part of pose 0.00798 | heavily worked by #133 (int12 re-solve) and #135 (Jacobian search); #140 added a zero-byte re-solve |
| pose term overall | 0.00798 | the hard ceiling on everything distortion-side |
| **rate / token stream** | **0.1198**, tokens 113,777/179,891 B ~ 0.0758 | #140 names the CONTEXT MODEL as open; their coder swaps and reordering both failed |
| fp16 rank<2 group | 0.0032 (int8) / 0.0020 (int6) | F15 explicitly skipped these groups; unverified against #140's "lossy quantization worse at every depth" |

Total GPU spend for the session: ~$4.50 across five boxes.

---

# THE REAL BAR IS ~0.136, NOT 0.1479 (verified by reconciliation, 2026-09-30)

adpena's private lineage continued past #140 and reached **S 0.13603403 @ 179,255 B [T4 n600]**
(move 55, 2026-09-17), **unsubmitted**. Found by the peer session's research agent reading
adpena/comma-lab; the primary memo is not in our local 312 KB subset, so it is secondhand -- but the
numbers reconcile exactly, which is strong evidence they are genuine:

```
seg   100 * 1.0288e-4         = 0.010288
pose  sqrt(10 * 4.08e-6)      = 0.006387
rate  25 * 179255 / 37545489  = 0.119359
sum                           = 0.13603403   vs claimed 0.13603403   (diff +4.4e-09)
```

An invented or garbled score would not reconcile from three independently quoted components to nine
decimal places.

## This corrects the strategic framing I had been using

| vs #141 (0.14789736 -> 0.13603403, total 0.01186) | gain | share |
|---|---|---|
| **seg** | **-0.00985** | **83%** |
| pose | -0.00159 | 13% |
| bytes | -0.00042 (636 B) | 4% |

I had been telling the operator that **rate** was where the headroom lived (0.1198 of 0.1479) and that the
distortion side was nearly spent. The attribution says otherwise: adpena's bytes barely moved. The win is
**seg distortion bought with token-map edits**, with pose following via carrier re-solves. The peer's
independent measurement agrees in sign -- #141's maps cost **+4,147 B** under #135's correctors-free coder,
so edits SPEND bytes to buy seg.

Corrected statement of the opportunity: roughly **half the seg debt** of the #135-family maps was reachable
by discrete realized-greedy token editing. That is ~0.0098 of score, i.e. 23x the byte saving adpena got.
Byte frugality is not the binding constraint; edit admission quality is.

## And it bounds my own closure correctly

My renderer-weight closure stands as measured: no single code flip improves the score **at fixed maps and
fixed carrier**. But the "0.00798 pose ceiling" I quoted is **map-conditional, not absolute** -- adpena
reached d_pose 4.08e-6 (pose term 0.00639) by re-solving the carrier after the maps changed, below the
6.37e-6 I was treating as a floor. Pose is a FOLLOWER of map edits. Do not inherit 0.00798 as a bound on
token-side work.

## Transferable from the renderer axis

`flip_search.py`'s protocol -- Jacobian rank -> cheap subset screen -> **exact 600-pair gate**, accept only
confirmed improvements -- applies directly to token edits. The gate is the load-bearing part: on the
renderer axis the first-order prediction was wrong for **all 874** candidates, so nothing should ever be
admitted on predicted gain alone.

---

# JOINT TRAINING ON #135 (the experiment that was actually asked for)

I had been training on top of **#141's** weights for two sessions, on the reasoning that #135/#141 share a
renderer. That reasoning was wrong in the one respect that matters for training:

```
                        #130/#135 live    #141 live
blocks.1.film.weight       192/192          2/192
blocks.2.film.weight       192/192          2/192
blocks.3.film.weight       192/192          2/192
```

Row pruning is **#140's compression step**. Starting from #141 freezes **4,560 FiLM parameters at zero** --
the per-frame conditioning path, where pose-relevant capacity should live. Starting from #135 is before
that stage, which is exactly why the agreed pipeline said #135. I had even written that reason down and
then did the opposite.

## #135's grid, recovered exactly

#135 encodes the renderer with **WANS1**, not #141's SM3R. Its records expose `codes` and `scales`
directly, so the grid is recoverable rather than inferred -- **zero reconstruction error on all 16
quantized tensors**:

* uniform **4-bit** signed, max|code| = 7 on every tensor (vs #141's mixed 3/4-bit)
* per-group **fp16** scales; axis -1 for `*embed.weight`, dim 0 otherwise
* every rank<2 tensor fp16
* **no row pruning**

Saved as `work/extracted/pr135_grid.pt`. Training quantizes against those **stored** scales, never a
recomputed amax -- so a weight change cannot rescale its tensor and steps are **byte-neutral by
construction**. That is a real improvement over the #141 setup, where every gradient step moved the stored
fp16 scale and therefore every value in the tensor.

A detail that validates the recovery: `amax/7` reproduces the grid exactly for 15 of 16 tensors and fails
only on `blocks.3.film.weight` -- the one tensor #135's F15 polish moved two symbols in, which shifted its
amax. Reading the stored scales avoids the problem entirely.

## Gate passed on #135

```
step 0:   seg 0.0296436   pose 0.0082986   SCORE 0.1622741
#135:     seg 0.0296394   pose 0.0082972   SCORE 0.1622684     (+0.0035%)
```

## Why the carrier must move, measured

A 2-epoch probe with the carrier FROZEN: **one** code flip already degraded pose, and 26 flips took pose
0.0083 -> 0.0128. But seg *improved* (0.0296436 -> 0.0296232). So the renderer can buy seg on #135, and a
frozen carrier makes pose pay for it at a ruinous rate. That is the coupling the joint objective exists to
exploit, and it is why every earlier frozen-carrier run looked like a failure.

## The three arms

| arm | renderer | carrier | what it isolates |
|---|---|---|---|
| **A joint** | lr 0.002 (code units) | coeff 3e-4, basis 1e-6 | the experiment |
| **B carrier only** | frozen | coeff 3e-4, basis 1e-6 | how much is just carrier refitting (#135 already Jacobian-searched this) |
| **C renderer only** | lr 0.002 | frozen | the control that isolates the joint effect |

Reading: A > B and A > C means the coupling is real on #135. A ~= B means the renderer contributes nothing
and the gain is carrier-only.

Target to beat: **0.1622684**. Objective is score-matched, `100*expected_flip + (sqrt(10)/(2*sqrt(d_pose)))*pose_mse`,
pose weight recomputed each eval (~549 at #135's operating point vs ~627 at #141's).

---

# FIRST VERIFIED GAIN FROM THE CONNECTED JOINT LOOP (2026-10-01)

`work/joint4b.py` -- token map + renderer + grey basis + 600x12 coefficients in one loop on #135, every
part parameterized in its own storage-code units, and every change accepted only on the exact score.

## B7, verified on the full video

Frames 0:120 trained (40 epochs requested; host reclaimed at epoch 24), then spliced into all 600 frames and
scored exactly (`joint4b.py --frames 0:600 --resume B7_best.pt --dump-map`), bytes from the exact coder
(`exact_cost135.py --tokens`):

| | #135 | B7 | change in score |
|---|---|---|---|
| seg term | 0.029650 | 0.029520 | -0.000130 |
| pose term | 0.0082980 | 0.0082848 | -0.0000132 |
| token bytes (exact) | 114,705.46 | 114,821.94 | +0.0000776 (+116.5 B) |
| **net** | | | **~ -0.000066** |

The renderer/basis changes accepted on frames 0:120 did NOT hurt frames 120:600 -- pose improved overall.
4x the carrier-polish gain (-0.000016).

## What made B work, iteration by iteration (frames 0:20 local deltas)

| iter | change | local delta |
|---|---|---|
| B1 | all four parts, one all-or-nothing gate | 0 (map never moved; carrier step vetoed every epoch) |
| B2 | per-part acceptance; backoff; every part on its storage grid | -6e-7 (first renderer accepts ever) |
| B3 | gentler backoff with restarts; 4-px flip cap | -1.1e-6 |
| B4 | **map edits proposed by gradient, accepted ONE PIXEL AT A TIME on the exact score** | **-2.64e-5** |
| B5 | candidates from SegNet's wrong pixels instead | -7.5e-6 (fewer, saturate) |
| B6 | B4 + both candidate sources, 200 epochs | -3.91e-5 (seg -10%, +467 bits) |
| B7 | B6 on 120 frames | -7.95e-5 local; -0.000066 real |

Key mechanisms, each found by reading a failing log:
* **soft-bit evaluation was fiction.** Pricing bytes on soft probabilities inflated them ~30x and made the
  score "improve" with 0 pixels changed. Evaluation must price the HARD map. (train_maps135.py had it too.)
* **batch map edits always fail; single-pixel exact pricing works.** Fixing one seg pixel is worth ~8.5e-7,
  a changed token costs ~10 bits ~ 8e-7 -- roughly break-even, so good edits are rare and specific; one bad
  pixel in a batch sinks the rest. This is also why #140 priced each edit individually.
* **no-op proposals must keep their progress.** On a 4-bit grid most steps don't flip a code; rejecting
  identical candidates and halving the lr killed accumulation.

## Known gap: first-order byte prices underestimate

The price list (from `exact_cost135.py --dump-costs`, -log2 p per class given the ORIGINAL map) estimated
B7's edits at +98 B; the exact coder says +116.5 B -- ~19% low, because neighbour effects are ignored. The
loop therefore accepts some edits that are slightly unprofitable. Fixes: periodically re-dump prices around
the current map, or use the peer's differentiable HPAC rate (hard forward = exact cost, neighbour-aware).

## Operational
* background jobs in this harness are killed at 30 min: re-arm guard and pullers each cycle
  (`work/ops/rearm.sh` uses an absolute deadline file so the budget doesn't reset).
* some vast images lack a C compiler; `exact_cost135.py` needs `CPR1_RC64_LIBRARY` pointing at a compiled
  `runtime/entropy/rc64_backend.c`.

## FULL 600-FRAME CONNECTED JOINT RESULT (2026-10-01)
merged B8 (0:120) + B9 (120:240) + B10 (240:360) + B11 (360:480, reprice/5) + B12 (480:600, reprice/10);
renderer + basis = B7's (later runs held them fixed). checkpoint: work/render/final600/m_M600.pt
exact full-video eval, exact HPAC bytes:
- seg 0.0002965 -> 0.0002871  (term -0.000940)
- pose term 0.0082980 -> 0.0082618 (-0.0000362)
- bytes 114,705.46 -> 115,506.22 (+800.8 B, +0.000533)
- NET -0.000443  -> score ~0.161825 (#135 published 0.16226842)
progression: 0:120 -0.000117, 0:240 -0.000194, 0:360 -0.000265, 0:600 -0.000443
reprice (joint4b --reprice K) cut the byte share of seg gain from ~63% (B8-B10) to ~51% (B11+B12).
note B11 raised pose on its frames slightly (5.54e-6 -> 5.61e-6) while net still improved.
