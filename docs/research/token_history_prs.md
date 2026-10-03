# Semantic-token lineage PRs: #143, #138, #144, #145 (plus #140 context)

Research date 2026-09-30. Read-only. Sources: `gh pr view/diff`, fork repos, cloned research repos in
`D:\projects\comma_b\work\research\repos\prs_agent\` (`tinker_cpr1/`, `opal_v1/`, `pr{138,140,143,144,145}.diff`,
`p143_files.txt` = #143 submission files extracted from the diff, identical to #135 runtime).

Legend: **[read]** = seen directly in code/PR text; **[inferred]** = my derivation.

Score formula: `100*seg + sqrt(10*pose) + 25*bytes/37,545,489`. 1 byte = 6.659e-7 score.

## Baseline #135 (F26, "semantic-pose-HPAC_CPR1_polished") [read]
- 186,724 B, pose 6.88e-6, seg 2.9639e-4, exact 0.16226842. Token stream 114,706 B (per #138).
- Token coder (`runtime/residual_archive.py::decode_production_tokens`, copied verbatim into #143/#145):
  - `IntegerHPAC` (patch=64, delta=2, channels 64), conditioned on previous decoded frame via
    `model.prepare_frame_context(index, previous)`; frame 0 uses an all-zero previous.
  - Coding order = 190 "groups" per frame: `group_masks()` in `cpr1/inflate.py`, `grid = col + 2*row` inside each
    64x64 patch, so group g = all positions with `(x&63) + 2*(y&63) == g` across all 48 patches; positions in one group
    are coded in parallel from `sparse.selected_logits(current, context, group)` (masked conv, `patch_group_mask` type A/B).
  - Fixed residual correction: `corrected = base_logits + table[boundary_bucket(prev)*5 + argmax(base_logits)]`, where
    `_boundary_buckets(previous, max_distance=4)` = Manhattan-ish distance (0..4) to a class edge in the previous frame.
  - `_probability_table`: logits rounded to 1/8 (`HPAC_LOGIT_PRECISION`), float64 softmax, stored float32, then RC64
    (63-bit range coder, integer freqs summing to 2^31). RC64 lands within a few bits of ideal (OPAL README: "4.98 bits
    above the continuous ideal").
  - The rate is a non-adaptive function of (tokens of current frame already coded, previous frame, frame index); this is
    exactly the "sum of -log2 p" setting.

---

## 1. PR #143 "Editable semantic tokens on top of #135" (tinkererlife), 0.160594 [read]

- Archive 185,653 B (−1,071 B vs #135), pose 6.1314e-6, seg 2.91451e-4, T4 exact 0.160594177969. On leaderboard as
  0.161 (maintainer "congrats!"). Decoder files are byte-identical to #135 except hash/size in `inflate.py`; all gains are
  encoder-side and live in the archive. No compression script in the PR.
- Score decomposition vs #135 [inferred from numbers]: seg −4.9e-4, pose −4.6e-4, rate −7.1e-4 (total −1.67e-3).
  So roughly 40% of the gain is rate, 60% distortion (much of pose from carrier retuning).
- PR text (quotes): "treat the semantic maps as editable instead of requiring them to match SegNet's original output
  exactly. A token change might make the perception score worse, but is still accepted if the reduction in archive size
  outweighs that." "The experiments included proposals based only on HPAC coding costs, as well as gradient-assisted
  proposals." "I backpropagated through the fixed renderer and SegNet/PoseNet models to the token representation. This
  helped rank possible replacements without rendering every alternative." "repeated sweeps over the frames, testing
  individual token replacements and selected pairs." "Between token sweeps, I also adjusted the carrier parameters"
  "I also fine-tuned HPAC's probability model on the edited token maps."

### Research repo: https://github.com/tinkererlife/comma-ai-semantic-pose-hpac-cpr1 (cloned to `tinker_cpr1/`)
Key files: `experiments/learned-token-grid-mvp/{learned_token_mvp.py, hpac_token_search.py, search_f24_hard_tokens.py,
revert_accepted_token_moves.py, REVIEW_BRIEF.md, official-result-p8.json}`, `code/rebuild_f24_hpac.py`,
`code/search_pose_coeff_joint.py`, `code/refine_pose_coeff_codes.py`. (The repo is the public #130 reproduction repo;
the token work was first done on #130, then ported to #135.)

**How gradients reached the discrete tokens** [read, `learned_token_mvp.py`]
- Straight-through one-hot: `straight_through_one_hot(logits, T)` = forward `one_hot(argmax(softmax(logits/T)))`,
  backward identity into the soft tensor (`_StraightThroughHard`). Logits are initialized from current tokens with a
  margin (`local_logits_from_tokens(tokens, init_margin=0.25)`).
- Renderer is rewritten to accept one-hot weights: `renderer_from_assignments` does `assignments @ token_embed.weight`
  instead of an embedding lookup (i.e. gradient is w.r.t. the one-hot coordinates, equivalently embedding-space).
- Crucially they do NOT use logit gradients: "Rank finite category switches in one-hot space. The gradient of softmax
  logits is distorted by temperature and initialization." -> `logits.grad = assignments.grad`. The first-order benefit of
  switching pixel (r,c) from class a to b is `g[a] − g[b]` (`propose_token_changes`, `rank_token_moves`, `candidate_moves`).
- Losses: seg proxy `expected_flip_loss(seg_logits, target, tau=0.15)` = mean sigmoid(−margin/tau) of SegNet
  target-vs-strongest-other margin; pose `sqrt(10*mse + 1e-12)` (#130 rail) or in the #135 rail a linearized pose weight
  `5/sqrt(10*pose_global)/600 * mse` (the derivative of the sqrt term). Camera resize uses `ste_uint8` (STE rounding).
  Metric path replicates the evaluator exactly (BHWC contiguous then CHW view; cuDNN TF32 ON for metrics, OFF for render).
- Rate in the proposal: either `differentiable_rate_proxy` (soft pairwise conditional entropy H/V, "lzma" mode, legacy)
  or, in the production path, the exact HPAC per-category cost table (not differentiated).

**How the rate was costed** [read, `hpac_token_search.py`, `search_f24_hard_tokens.py`]
- `quantized_probability_bits()` mirrors the deployed 1/8 logit rounding + float64 softmax -> exact ideal bits.
- `HPACRateOracle` / `F24RateOracle` (the latter includes the #135 boundary residual table).
- Proposal cost: `direct_symbol_bits()` = "teacher-forced direct bit costs for every token category" at each pixel in its
  current context, i.e. only the changed symbol's own −log2 p, ignoring its effect on later symbols' contexts.
  Docstring: "This is a first-order proposal oracle only. The exact move gate still recomputes the changed frame and its
  successor, so autoregressive downstream effects cannot create a false acceptance."
- Gate cost: `move_deltas()` = full recompute of frame t bits and frame t+1 bits (next frame uses t as context) for each
  candidate; `localized_move_delta(_batch)` is the faster localized version (checked: max diff 5.4e-13 bits vs full).
- Combined ranking objective: `g_perception + 25*bits/(8*37,545,489)` per candidate switch (`candidate_moves`).
- Acceptance: exact render + SegNet argmax mismatches + PoseNet MSE on the full frame, global score recomputed with the
  frame's contribution replaced before the sqrt (`replace_global_perception`), plus exact HPAC ideal-bit delta.
  Batches use `accept_with_backtracking` (binary split on failure).
- Search granularity: renderer has 4 spatial GroupNorm layers, so "a one-token edit affect[s] the renderer globally";
  1,932-12,439 output pixels changed per single token edit. Crops are invalid; full-frame re-render needed.

**Single vs paired edits** [read]
- `search_f24_hard_tokens.py --changes-per-candidate {1,2}`; `candidate_groups()` forms pairs from the top-32 ranked single
  flips (`itertools.combinations`, distinct pixels), sorted by summed first-order benefit; top `candidates_per_frame` (8)
  are rendered exactly, best one accepted per frame if objective improves. Commit 8be06a3 "Search two-token blocks beyond
  single-flip optima".
- Frames visited per sweep: `ranking = sorted(frames, key=posenet_distortion, reverse=True)[:top_k]` (default 16), i.e.
  the #135-rail search concentrated on the worst-pose frames.
- Earlier (#130 rail) families: pixel, K=8 independent alternatives per frame, and constant-category rectangles
  (`rank_token_regions`, shapes 1x2..9x9 ranked purely by summed direct HPAC benefit; 13x13-17x17 never accepted).

**Carrier / pose adjustment** [read]
- `code/search_pose_coeff_joint.py` ("Jacobian-guided joint integer search over deployed carrier coefficients"),
  `code/refine_pose_coeff_codes.py` ("Jointly refine deployed int12 pose coefficients for the worst pairs").
- Commit 76f83ff "Tune carrier and token gates natively on T4" + `revert_accepted_token_moves.py` ("T4-native reverse gate
  for previously accepted two-token moves"): 5 hardware-sensitive token pairs reverted.

**HPAC fine-tune on edited maps** [read]
- `REVIEW_BRIEF.md`: "exact HPAC fine-tuning: three short runs were judged by packed model bytes plus a real RC64 stream,
  and only the winning 225-byte archive reduction was retained." `official-result-p8.json`:
  `hpac_finetune_archive_delta_bytes: -225`. Then "Re-running the token search under the new HPAC accepted 39 of 4,800
  candidates and removed another seven real stream bytes" (`post_hpac_token_sweep_actual_stream_delta_bytes: -7`).
  Final `token_stream_bytes: 113,631`.
- `code/rebuild_f24_hpac.py` requires the refit HPAC to keep the exact same IHS2 byte size ("candidate HPAC cannot use
  fixed F24S schema" otherwise), i.e. weights changed within the fixed per-row bit depths; `code/hpac_self_compress.py`
  patch clamps the self-compress radius to `weight_bound`.

**Score trajectory (README history)** [read]
- #130 rail: 0.172141 -> 159 flips 0.170687 (archive +464 B!) -> renderer hardening 0.170254 -> rate-first 0.170002 ->
  category-aware 0.169224 -> 20 sweeps 0.167707 -> structural/regions 0.1675-0.1673 (L40S).
- Port to #135 rail: L40S 0.165896 -> two-token blocks L40S 0.160475 -> **exact T4 0.164640** (big hardware gap) ->
  T4-native carrier + reverse gate 0.160805 -> HPAC fine-tune + 39-token sweep 0.160594.

**What failed / caveats** [read, REVIEW_BRIEF.md]
- "Unchecked persistent soft-token optimization failed. The differentiable optimizer sees mixtures of token embeddings,
  while deployment takes an argmax. Once many logits crossed at the same time, tens of thousands of hard IDs changed
  discontinuously and the exact score collapsed."
- Gradient as ranker only: "A 4-frame run accepted 1 of 32 proposals; a 32-frame run accepted 2 of 63. This says the
  gradient is a useful ranker, but not a trustworthy direct optimizer of the discrete objective."
- Joint backprop sweeps "slower and added only 6.3e-5"; K=8 alternatives gained only 4.07e-5 over 600 frames; single-token
  sweeps saturate (marginal gain fell from ~1.5e-4/sweep to 4.8e-5).
- Renderer fine-tune on 4 frames overfit (0.1764 on 600). Full-600 renderer at lr 2e-7: only epoch 1 helped.
- Ideal-bit prediction vs real bytes: 188 predicted vs 176 actual; perception proxy overstated gain by ~46% until the
  metric path was matched to the evaluator (TF32 + layout), after which within ~1.9%.
- Hardware: TF32 on/off changed accept decisions; L40S-searched edits lost ~0.004 on T4 (DALI decoding / GPU numerics).
- Only 875-1,202 of 117,964,800 positions differ from #130's map in the #130-rail artifacts: the edit set is tiny.
- Open ideas they list: connected regions/contours/temporal tubes; invalidating attempt history after nearby changes;
  alternating metric-neutral byte removal vs distortion repair; scalable discrete optimizer for all 118M choices.

---

## 2. PR #138 "opal_v1" (ccastillo1043), −4.7 KB on tokens [read]

- Archive 182,040 B (README says 182,040; PR body once says 182,020), token stream 114,706 -> 110,022 B (−4,684 B),
  exact 0.1591495384 with #135 distortions unchanged (decoded token SHA identical to #135).
- **Closed without eval**: maintainer only replied with a link to the "coding agents and LLMs policy" (2026-08-31). Not on
  leaderboard. Repo: https://github.com/ccastillo1043/opal_v1 (cloned `opal_v1/`).
- Mechanism (`runtime/entropy/rc64_backend.c`, includes `opal_model_impl.c` as offline evaluator):
  - Binary "defect" decomposition: for each position, HPAC's (residual-corrected) 5-class row gives maximal class m and
    wrong mass w = 1 − p(m). Base logit b = logit(w). OPAL predicts q = P(token != m) = sigmoid(b + 0.22 * sum_j mult_j *
    w_j) and rescales: p'(m) = 1 − q, p'(k) = q * p(k)/w for k != m (relative law in the complement preserved)
    (`opal_adjust_frequencies`).
  - 55 context families (`init_catalogue`, total 6,175,440 cells, two float32 per cell = 49.4 MB state), hashed from:
    confidence bin `conf = round(875*q0)` (101 bins), group (190), 64x64 cell (48), position within cell at 4/8/16 px,
    maximal class, previous-frame defect at same pixel, 3x3 previous-frame defect word (same cell), cross-cell
    neighbours, temporal defect history (lags 1,2,4 and 8-frame word; ring of 9 frames), causal current-frame defect
    word from 8 already-coded neighbours (respecting group order), "holonomy" = XOR of causal word with the same
    neighbours in the previous frame, run/prefix counts.
  - Per-family weight is a one-step Newton estimate from accumulated sums: `w = clamp(-G/(H + 2.5), ±4)`, where after each
    symbol `G += q − outcome`, `H += q(1−q)` (no decay, no explicit learning rate; ridge 2.5 acts as prior). Per-family
    meta multiplier updated by a Newton step on `feature = 0.22*w`: `mult -= g/(meta_H + 0.75)`, clamped ±8.
  - "learning rate": effectively scale 0.22, ridge 2.5, cap 4, meta ridge 0.75 (hand-tuned constants; the offline
    evaluator exposes ~20 modes for searching these).
  - Gain: "full causal ideal gain is 37,472.66 bits" (4,684 B, ~4.1% of the token stream).
- Interpretation [inferred]: #135's HPAC is miscalibrated on "is the argmax right" in a way that is predictable from
  confidence x location x temporal defect history. Most of the gain is calibration of p(argmax) per context.

---

## 3. PR #144 "hpac_rowpack" (Bubu631), −957 B [read]

- Not a token coding-order change. It is a lossless repack of the **HPAC weight rows** for the outer raw-LZMA:
  `runtime/rowpack.py` (`pack_models`/`unpack_models`, magic `HRP1`): group the 517 signed integer weight rows by stored
  bit depth, zigzag map, write LSB->MSB bitplanes per depth group; CRC32 of the original F24S section. Token stream and
  residual table byte-identical.
- 186,724 -> 185,767 B (−957 B, rate −0.000637). Only local Apple MPS eval (0.20, not comparable). Also adds
  `exact_power_of_two()` in `cpr1/hpac_integer.py` (MPS `pow` broke dyadic scale ties). Closed without official eval.
- Relevance [inferred]: HPAC weight bytes (~16.6 KB IHS2) are still compressible; any retrained HPAC should be packed with
  bitplane/zigzag-by-depth before LZMA.

---

## 4. PR #145 "warped-context-hpac" (Reflex-1bit), 0.158199 [read]

- Archive 187,246 B (+522 B vs #135), pose 6.65e-6, seg 2.5365e-4, exact 0.158199; on leaderboard 0.158.
- Context warping (`runtime/context_warp.py`, ~10 lines in `residual_archive.py`): before predicting frame t, HPAC's
  previous-frame context is replaced by `warp_torch(previous, choice[t])`, chosen by the encoder per frame from a
  catalogue of 21 integer nearest-neighbour index maps: identity; shifts (±1,0),(0,±1),(0,±2); zooms 1.002/1.003/1.004/
  1.006 about (cy in {155,170,185}, cx=256) (near horizon); zoom 1.004@170 combined with x-shift ±1.
  Choices Huffman-coded (canonical, 4-bit length header, ~3.5 bits/frame). The warp reassigns `previous` before both
  `prepare_frame_context` and `_boundary_buckets`, so both the HPAC context and the residual-table boundary feature use
  the warped frame [read].
- Bytes: "costs 257 bytes to store the warp choices, and saves ~700 B when measured on #135's tokens" (net ~−443 B).
  Bidirectional context tried first: "+4.8 KB because HPAC is directional".
- Token edits: "about 7,000 label pixels across 582 frames were changed so that SegNet on the rendered frame matches the
  original video better. Each frame edit was kept only if the per-frame score (SegNet and pose measured on a T4, bytes
  from an exact rate estimate) improved." Pose repair: "the 12 carrier codes per edited frame were re-tuned using gradient
  descent through PoseNet on a T4." No search code published (decoder only).
- Decomposition vs #135 [inferred]: seg −4.27e-3, pose −1.4e-4, rate +3.5e-4. So edits cost about +965 B
  (522 + ~443) [inferred] but bought 0.0043 of seg. That is the opposite trade to #143 and much larger.
- Findings quoted: local-vs-official gap "comes from video decoding (PyAV locally vs NVIDIA DALI on the evaluator)";
  "The evaluator's `rgb_to_yuv6` function uses `@torch.no_grad()`, which silently blocks gradient flow through PoseNet
  unless you unwrap it."

---

## #140 context (adpena, "semantic_joint_ctxmix", 0.14798, leaderboard 0.148) [read]
- 180,002 B, pose 6.37e-6, seg 2.0139e-4. Research repo https://github.com/adpena/comma-lab (not cloned).
- "candidate segmentation edits of the semantic tokens are proposed per pair and priced against their exact pose cost
  through the frozen PoseNet, then admitted through a Lagrange-multiplier waterfill (455 of 573 proposed edits admitted).
  The pose carrier is then re-solved (damped Gauss–Newton)"; frame-0 compensation so edits carry ~zero pose tax.
- Coder: "fixed-point integer log-odds context mixing, group-conditioned token contexts, an address-free tile-conditioned
  re-encode". Failures: in-group token reordering gives identical length (113,777 -> 113,777 B); explicit pixel-fix
  addresses cost more than they save; parametric lane curves larger than tokens.
- Also on leaderboard: #141 semantic_blocks 0.148 (not examined).

---

## Implications for direct token training with exact differentiable HPAC rate + seg/pose losses [inferred]
1. Seg is the biggest lever, not rate. #145 (~7k pixels) and #140 (455 edits) got 0.004-0.009 from seg; #143's
   rate-focused search got 0.0007 from rate. Rate gains from edits are small because each edit's direct saving is a few
   bits; seg gains are large because tokens drive the renderer, not SegNet labels.
2. Soft/relaxed optimization of all tokens collapsed for #143 (argmax mismatch). Use the relaxation only to rank or as a
   trust-region step; re-harden and verify with exact render + exact rate; limit simultaneous flips per frame, or
   anneal with a hard-forward STE and backtracking.
3. Rate gradient has two paths: the coded symbol (gather of −log2 p) and the context (the HPAC input = partially coded
   current frame + previous frame). #143 only used the direct term for proposals and found it adequate as a ranker but
   always recomputed frame t and t+1 exactly. A fully differentiable rate needs a float surrogate of the integer HPAC
   (ste_round/requantize) and must include frame t+1's cost.
4. Non-differentiable pieces in #135's law: argmax-indexed residual table and boundary buckets from the previous frame,
   1/8 logit rounding. Treat as piecewise-constant (stop-grad on index) and verify exactly.
5. Non-adaptive -log2 p underestimates what an OPAL/ctxmix-style adaptive layer would charge for "defects" in a context
   that becomes predictable; after editing, refit HPAC (worth ~225 B for #143) and possibly add OPAL (−4.7 KB on #135's
   tokens, but closed for AI policy, not adopted upstream; #140's ctxmix is the accepted analogue).
6. Pose is fragile: every edit campaign needed carrier re-solve (Jacobian integer search, GD on 12 codes, Gauss-Newton)
   after edits; unwrap `rgb_to_yuv6` no_grad; match evaluator metric numerics (TF32 on for metrics, BHWC->CHW) and gate
   on T4 (L40S-searched edits lost 0.004 on T4).
7. Renderer GroupNorm makes each token edit global; costs require full-frame renders, batchable across same-parity frames.
8. Warping or improving HPAC context (#145 −443 B net) and rowpack (−957 B) are orthogonal lossless wins that stack.
