# Token-map editing / costing / coding: what the incumbents learned

Research date 2026-09-30. Sources and how I read them:
- PR #140 body, README and compress.py via `gh pr view/diff 140` (READ). Diff saved at
  `D:\projects\comma_b\work\research\repos\tokhist\pr140.diff`, compress.py extracted to `pr140_compress.py`.
- adpena/comma-lab cloned at `D:\projects\comma_b\work\research\repos\tokhist\comma-lab` (HEAD cb94a9ff,
  2026-09-18). Windows checkout needed `core.longpaths true` + `git checkout -f HEAD`. Key code READ:
  `experiments/ddm_jg1_seg_solve.py`, `ddm_sj1_multipass_token_predistortion.py`, `ddm_sj1_joint_admission.py`,
  `ddm_jg5_pose_resolve_on_edited_renders.py` (run_waterfill). Key memos READ (all under `.omx/research/`):
  `ddm_jg1_joint_solve_20260819.md`, `ddm_jg3_s2_joint_solve_20260819.md`, `ddm_jg5_pose_resolve_on_edited_renders_20260819.md`,
  `ddm_sj1_multipass_token_predistortion_20260905.md` (tables), `ddm_rp1_rate_directed_token_predistortion_20260909.md`,
  `ddm_pr1_pose_resolve_on_renderer_change_20260904.md` + `arm_final_messages/ddm_pr1_final_*.md`,
  `ddm_gs3_gestalt_after_submission_20260903.md` addenda 48-75, `ddm_ef1_token_entropy_floor_20260822.md`,
  `ddm_fe1_per_pair_frame_embedding_realized_search_20260908.md`, `ddm_up2_shipping_object_pose_solve_20260819.md`,
  headers of fx1/fx2/fx5/gb1/lb1/afr1/ma1/rr4 memos.
- #86 write-up `scratchpad/anr.txt` (READ, sections 3-4, 7, 10).
- #143/#138/#144/#145: delegated to a sub-agent; its file is `token_history_prs.md` (summary merged in section 6).
- "INFERRED" marks my own reasoning; everything else is quoted/paraphrased from the files above.

BIG NEWS FIRST: adpena's private lineage continued after #140 and reached
**S 0.13603403 @ 179,255 B [T4 n600] (move 55, 2026-09-17)**: d_seg 0.00020139 -> 0.00010288,
d_pose 6.37e-6 -> 4.08e-6, bytes ~flat. Most of that came from **token-map edits** (discrete, realized
greedy search) + carrier re-solves + lossless recodes. Unsubmitted publicly as of the clone (a reviewable
8-file packet `submissions/mrs7` measured on move-53 bytes, "awaiting the operator"). This is the best
available evidence of how much slack the #135-family token maps had: about half the seg debt.

---------------------------------------------------------------------------------------------------

## 1. Structural facts about the object (all MEASURED by adpena unless noted)

- Archive layout at #135-era (`ddm_jg1` S0): RX1 header 14 B, hpac model 13,515 B, semantic renderer 30,856 B,
  carrier 22,143 B, **token tail 109,792 B = 62.2%**. Tokens `(600,384,512) uint8 in {0..4}`, class order
  `("Road","Lane","Undrivable","Movable","MyCar")` (comma10k canonical; "never re-derive by luma-sorting").
- **Frame asymmetry**: tokens render frame 2p+1 only (`cpr1/inflate.py:313-328`); frame 2p is
  `127.5 + CARRIER_AMPLITUDE * 12-dim basis` = a photometric probe; SegNet sees only frame 2p+1
  (`modules.py:108`), PoseNet sees both. So tokens are a joint seg+pose actuator; the carrier is pose-only.
- Renderer receptive field (derived from code): coord_mix 1x1 + TokenBlock depthwise 3x3 at dilations
  1,1,2,4 + head 3x3 => **r_render = 9 token cells** (`RENDER_RADIUS_TOKENS = 9`).
- **But SegNet's response to one token flip is long-range**: influence probe (sj1 memo s2): 85.6% of single-token
  moves change argmax; cells changed p50 2, p90 6, max 18; **Chebyshev radius of furthest changed cell p50 19,
  p90 155, p99 371, max 416**; only 34.2% of moves keep their whole response within radius 8. Moves that repair
  >=1 flip 18.1%, make it worse 55.3%. => local/windowed attribution misprices most moves; adpena judged every
  move on the whole-pair flip count.
- **Stored tokens == DALI GT argmax at 99.9985%** at #135 (1,714 differing cells); **95.9% of seg flips are at
  cells where the stored token already equals GT** (34,296 of 35,752). The debt is render->re-segment loss, not
  label error. Later (move 48) the field differed from DALI GT at 18,900 sites, and rendering the TRUE partition
  was **2.825x worse on seg** than the edited field (ren1: 2.92e-4 vs 1.034e-4): "the token field is an optimized
  pre-image of renderer∘scorer, not an approximation of the partition".
- Flip ledger by edge (#135 body, DALI, 35,752 flips): Lane->Road 24.1%, Road->Lane 19.0%, Road->Undrivable 10.6%,
  Undrivable->Road 9.5%, Undrivable->Movable 8.7%, Movable->Undrivable 8.4%, Road->Movable 6.9%, ... Road appears in
  81.5% of flips; Lane is 0.586% of area but in 44.3% of flips (75x over-represented). Lane->Road vs Road->Lane
  8,619 vs 6,797: **displacement, not erasure** (88% of the Road<->Lane debt is a misplaced boundary).
  Residual flips live in rows 128-319.
- **GT lineage trap**: DALI vs PyAV GT tables differ at 20,671 argmax sites (87% of the 23,757-flip budget at that
  time); d_seg differs 1.43x (0.00030309 DALI vs 0.00043336 PyAV) on the same bytes. The contest CUDA eval uses the
  DALI lineage; the #135 tokens were fit to DALI. **Optimising against the wrong GT table aims at sites the contest
  does not score.** (Check which decoder our local GT/eval uses.)
- Exchange rate: 1 seg cell = 100/117,964,800 = 8.477e-7 S = **1.273 archive bytes = 10.18 bits**.
- Tail at 0.00745-0.0082 bits/token. Census under the coder (rp1 s3, move 35): **99.80% of tokens ARE the coder's
  argmax**; the 0.2% that are not (236,633 tokens) carry **70.3% of the tail bits** (84.7 KB); the other 99.8% carry
  35.7 KB of "yes it's the prediction" flag mass at 0.0024 bits each. Boundary tokens are cheap to flip: at move
  sites the current token costs mean 0.792 bits (median 0.091), ~100x the field average.
- HPAC decode order (fx2 R1): `group(x,y) = (x & 63) + 2*(y & 63)`, 190 causal groups; up-right neighbour causal
  98.69% (same as up), left 98.63%.
- HPAC is an integer lattice: identical `corrected_quantized_logit_sha256` on macOS-arm64-CPU and T4-CUDA (rr4).

## 2. How #140 PROPOSED, PRICED and ADMITTED token edits (and what came after)

### 2a. Proposal (ddm_jg1 -> jg3 -> jg5 -> sj1)
- **Block/dilation moves FAIL at every radius** (jg1 S1c, n=6 pairs, 268 flips): write-GT-only-where-wrong
  (`all_r0`) +9 cells (+3.4%, 0.75 cells/token); disk r=1 **-148 cells (-55%)** with 576 tokens; r=2 **-942
  (-351%)**; Road/Lane-only r1 -44%, r2 -221%. Mechanism: widening a class paints a wider class; SegNet faithfully
  reports it and the boundary moves to the other side.
- **Single-token coordinate moves WORK**: flipped cell or a 4/8-neighbour set to the GT class; 12/18 sites had an
  improving move, 1.50 cells repaired per changed token; winning move is almost always one ADJACENT cell
  (minimal shift of a displaced boundary).
- jg3 (the solver that fed #140): site order by **margin saliency** ("the margin field IS the Fisher surrogate,
  Fisher curvature vs (-margin) Pearson 0.978"), candidates within a site ordered by the coder's bit cost;
  "A ranker ORDERS; only realized joint dS accepts". Defects caught: D1 pooling winners across screening batches
  made adjacent winners = a block move (pair 283: 66 tokens for 3 cells, yield 0.045) -> fix `select_separated`
  (separation constraint, swept per pair, 2-D grid separation x keep-fraction); D3 hm1 logits under-priced a token
  edit 2.2x (1.91 vs measured 4.1379 bits/token) -> optimizer chose densest rung.
- sj1 (post-#140 multipass; the one that halved d_seg): family = **flipped cell + 8 neighbours x 4 alternative
  classes = 36 moves/flipped cell**, ordered by (class selector, offset), deduped on (position,new class).
  Greedy realized coordinate descent per pair, accept iff whole-pair flip count strictly falls; skip moves whose
  site has already been repaired by a far-field effect. Render at batch 1 (batch 8 flips 1,326 px by +/-1 through
  clamp/round), SegNet batched across pairs (0.315 vs 0.526 s/frame). Role hit rates (probe): `gt` (widen the class
  SegNet is missing) 28.9%, `other` 17.8%, `ours` 10.1%. Accepted offsets (pass 2a): (-1,0) 2,099, (1,0) 1,552,
  (0,-1) 1,105, (0,1) 880, (0,0) 528, diagonals ~350-470 each -> vertical neighbours dominate.
- **Pass 2a numbers** (n600, `gt` slice): flips 23,749 -> 14,156 (**-9,593 = 40.4%**), 7,804 tokens changed
  (**1.229 cells/token**), 96,764 proposals / 69,382 realized evals, 16,580 s on 5 shards (Apple CPU).
  Rate by real re-encode: tail 113,419 -> 119,497 B (**+6,078 B, 6.23 bits/changed token**, modelled 4.718 ->
  realized/modelled 1.321). Seg -0.008132 S, rate +0.004047 S. Pose stale 430x (597/600 pairs damaged, +0.150 S)
  -> carrier re-solve to 5.398e-6 (below base 5.77e-6). T4: 0.1398140 (pred 0.1398087).
- Subsequent passes on the residual: pass 3 (370-pair subset) seg 1.2009e-4 -> 1.0913e-4, +741 B, -8.1e-4 S;
  pass 4 -1.5e-4 S; pass 5 42-pair subset -2.7e-5; pass 6 -1.9e-4; pass 7 -6.0e-5 (221 cells/218 tokens found,
  42 pairs/69 cells admitted); **pass 8 closes the family**: field prices at 8.93 bits/changed token vs 10.31
  break-even; fresh-repair rate per pair 1.156 -> 0.410 -> 0.310; "a repair CONSUMES the local slack, a neutral edit
  only jostles it".
- Pose-directed field edits (pd1-pd4, moves 50-52): pose-saliency-ranked single tokens screened (render + argmax,
  <=2-cell seg cost), refined with per-pair carrier re-solve. 41 pairs with credit (median -21.6% of the pair's pose,
  best -96%). Moves 50/51/52: -4.0/-5.0/-3.1e-5 S. Admitted edits cost **12.9-17 bits/token** (vs 8.9 clustered).
  pd4: price-ranked + clustered 2-token proposals: singles 15.8 bits/token pooled, clustered 13.8 pooled / 11.4 median.
  pd5 (2-4-token ridge runs): medians 14.8/10.4/9.2/8.7 bits for 1/2/3/4 tokens, but **"the context discount
  compounds; the credit does not"** (added token improves the pair only 37-46% of the time).
- **Price-first generation (pd6-pd8, moves 53-55)**: use the prior's own probability rows to propose the CHEAPEST
  changes first, check credit afterwards. 4,291 proposals on 588 pairs, **389 with negative real charge**, two-token
  sheets 3.2-3.7 bits/token, 91% on Road, 0.5% overlap with the saliency pool. Move 53 -1.0e-4 S. Cheap half decays:
  rank-0 sheet 2.16 -> 3.08 -> 3.28 bits/token; paying pairs 128 -> 72 -> 44. 18 of 53 rank-0 proposals on re-rendered
  pairs were **reversions to the previous move's token** (passes oscillate).
- Rate-directed argmax-neutral edits (rp1, moves 39/42): change field where coder charges most, accept only if the
  realized argmax is bit-identical on all 196,608 cells. Only **4.69% of proposals are neutral**; neutrality is flat in
  rank and **not predictable from the local 19x19 receptive window** ("no cheap screen; neutrality has to be
  realized"). Move 39: 473 neutral changes on 253 pairs, -250 B. Move 42: carrier re-solve after neutral edits landed
  pose below base (render moved even though argmax did not).

### 2b. Pricing (seg, pose, rate)
- **Seg**: realized only: receiver's own renderer (proved byte-exact vs shipped decode, max |delta| 0) + frozen CPU
  fp32 SegNet argmax vs DALI GT. Instrument reproduces T4 d_seg to 0.99995x. "MPS SegNet distortion drifts 2x"
  -> MPS never the authority for argmax. T4 vs local seg agreed to within ~9 cells over 117.96M.
- **Pose**: frozen CPU-torch PoseNet on the full decode with the edited frame 2p+1 overlaid; **always after the carrier
  re-solve** (stale-carrier pose is meaningless: x387-x467). Batch shape changes PoseNet values (pair 299: 7.7e-3
  relative spread across batch 1/8/32) -> KEEP/DROP decided on values re-measured at one declared shape (batch 8).
- **Rate**: never trusted -log2 p alone. Per-pair rate leg from **two per-frame bit ledgers of real re-encodes**
  (control vs candidate; `bits_per_frame.npy`); uniform bytes-per-token fallback only before encoding ("fs3 measured
  that substitution wrong by 2.24x"). Measured price drift: hm1 logits 1.91 b/tok vs real 4.14 (2.2x under);
  jg1 model 4.718 vs real 6.23 (1.32x); pd1 modelled ledger 1.9x optimistic at 16.98 b/tok. **Selection bias**: argmin
  over noisy per-sheet real prices is biased low (pd4: ledger 211 bits vs real encode 288, -36%) -> **re-price the
  selected set as one object** (set pricing converges in 1-4 iterations).
- **Adaptive-coder asymmetry (rp1 s7, the most important rate fact)**: under the shipped ADAPTIVE corrector/mixer,
  writing surprises IN costs **1.28x** the first-order price; taking surprises OUT realizes only **0.1457x** of the
  first-order saving (217 neutral changes: first-order -219.66 B, real -32 B; 1.18 real bits/change vs 8.10 modelled).
  "A first-order per-position price is a RANKING, never a charge."

### 2c. Admission: the Lagrange waterfill (code: `ddm_sj1_joint_admission.py::cmd_admit`, `jg5::run_waterfill`)
- Per pair two states: DROP (frame 1 = base render, carrier = old codes, pose = base) or KEEP (edited render +
  re-solved carrier). Seg credit and rate cost are additive per pair; pose is sqrt(mean) so not additive -> pose is the
  multiplier's subject:
  `keep = edited & ((seg_credit - rate_cost) > lam * pose_damage)` with
  `seg_credit = 100 * flips_repaired * ratio_t4 / (600*384*512)`, `rate_cost = 25 * per_pair_bytes / 37,545,489`,
  `pose_damage = resolved_pose - base_pose`;
  `lambdas = [0] + logspace(-3, 6, 1801)`; every distinct subset scored through the exact score formula; best kept.
  (No single fixed multiplier value is reported; it is swept per run.)
- jg5 result (= #140's edit stage): **455 of 573** edited pairs kept; d_seg 2.01334e-4 (T4-proj.), d_pose 6.3657e-6;
  8,654 tokens changed at **3.8373 measured bits/changed token**; carrier splice +45 B; final 180,625 B, S 0.14838267
  (net -0.00777 vs pointer 0.15615). Anchors: drop-all model 0.156148 vs pointer 0.156152; keep-all 0.319175 vs
  measured 0.319183. Carrier alone with all 573 kept: 3.268e-3 -> 4.089e-4 (8x) but S still 0.2023: "The carrier
  cannot rescue the full edit set. The admission has to do the rest." Recovery is bimodal per pair (pair 0
  9.1e-4 -> 3.7e-7, below base; pair 10 barely moves).
- Later joint seg+pose+rate admission without a seg screen (pd9) beat the screen by only 0.136 bars; mechanism: seg
  cost and pose credit are uncoupled (improve fraction ~40% at 1/2/3 cells).
- Carry-field bug class: an edit npz spliced on the original field silently reverted 2,337 banked tokens on 230
  pairs -> always write what each pair should HOLD.

### 2d. Carrier Gauss-Newton re-solve
- `jg5.refine_pair` (inherits br1's damped GN on the shipped 12-dim basis and signed-int12 lattice) alternating with
  a +/-2 single-coordinate polish, accepted only when realized pose falls through the receiver.
- br1's limits were budgets: `iterations=6` cap (accepted-step histogram [98,131,146,111,63,35,16] for k=0..6; 16 pairs
  still improving >2% at the cap) and GN-then-polish once. Replaced by a derived materiality stop:
  `dS/dd_i = 10/(2*600*sqrt(10*m))` = 1.043784 at m = 6.374e-6; `DELTA_FLOOR_S = 3.5e-6/600`; iterate while
  `remaining_dd * dS/dd_i > DELTA_FLOOR_S` with `remaining_dd = g_k*r/(1-r)`, `r = g_k/g_{k-1}`; threshold 5.5886e-9.
- n600 result: 571/600 pairs new codes, 6,247 coordinates changed, **600/600 stop at `no_improving_step`**, 0 budget hits.
  "The solve is at the GN+polish fixed point of the shipped basis and lattice."
- **The "542/600 stall at no_improving_step"** is from ddm_pr1 (renderer change, not token edits): a seg-only renderer
  fine-tune (ft1 FO-1) raised d_pose 2,432x stale; re-solve recovered 16.42x (to 9.43e-4, 148x base); 100% of pairs
  demanded a GN step larger than the +/-2 radius (median 444 int12 code units; 9.67% more than the whole 4,095-unit
  lattice); 542/600 stopped at no_improving_step; 10 pairs own 69% of the post-solve mean. Re-solve cost +125 B.
  k_pre/k_post coupling 228.45/13.82; payable bar 0.20997 -> 41.5x over. Reflected step raised d_seg 21.55x more.
  => **renderer changes are far more pose-destructive and less recoverable than token edits**; the carrier basis
  (12 shared atoms) is the limit (pp1/cb1: hard pairs' pose is a floor outside both actuators' reach).
- Other carrier facts: up2 LM/gradient solve with STE had correct Jacobian (cos 0.96-0.99 vs finite difference) but
  **every LM step realized worse** at every damping (1.015x .. 22x); lattice coordinate descent with realized
  acceptance won. Carrier lattice coarsening x4/x8/x16 + re-solve bought -1,801/-872/-790 B (moves 28-30).

### 2e. #140's "what did not work" (PR body, verbatim-ish)
lossy quantization of learned tensors (worse at every depth); distilling the renderer (pose error cost tens of times
the bytes saved); parametric lane curves (1.7x larger at best); swapping/retuning the entropy coder (25 Brotli/LZMA
configs no saving; generic recoding +5 B); **reordering the token stream** (seeded within-group permutation 113,777 ->
113,777 B; cross-group ordering needs a retrained context model); **explicit fix-this-pixel corrections** (addresses
cost more than they are worth); small fitted pose-correction layers (43/247/997 B, held-out neutral/negative).

### 2f. Five lossless stages + corrector stack (#140 compress.py STAGE_PINS)
base 180,456 (rc2_composed) -> **FX5 180,386 (-70)** -> **DX2 180,368 (-18)** -> **GB1 pointer 180,215 / joint 180,192
(-176)** -> **LB1 180,083 (-109)** -> **AFR1 180,002 (-81)**; 454 B total, token field bit-identical throughout.
- rr2/rr4 "free corrector": decode-time adaptive correction of HPAC's hit probability by Krichevsky-Trofimov count
  ratios in a 51,200-bin joint context; -1,598 B. rr2's first T4 fire scored **S = 27.83** (decoder desync) - blamed
  on libm log2/exp2 ULP differences (50% of positions perturbed by 1 ULP; 1 ULP at p~0.5 = 128 RC64 counts); v2
  removed all transcendentals (only + - * / sqrt and comparisons; AST-tested). (rr4 T4 memo says the actual rr2
  refusal was staging infidelity; either way the lesson is exact arithmetic + fire the proven tree unmodified.)
- fx1: fixed-point log-odds (geometric) mixer via dyadic weights computed with repeated sqrt -> platform-exact;
  -560 B. Arithmetic-mean mixtures all lost (+67..+1,398 B) "averaging cannot beat its best member".
- fx2: 4-neighbour causal template + local homogeneity (boundary detector) feature; -797 B (19-member) / -711 B
  (13-member); SSE/APM second stage LOSES in 6/6 formulations. FX5 = fx2 19-member on rc2 body, -70 B more.
- ma1 within-miss relative law (KT ratio per class in the miss sector) -104.6 B (58% of hindsight bound ~180 B; the
  "77 KB miss reservoir" was withdrawn as a vacuous entropy denominator).
- DX2: CABAC-prefix recode of the 7,200 Rice carrier coefficients (-18 B). RR5: adaptive arithmetic basis coder.
- GB1: `groupbin8_surprise` conditioning family (decode-scan group binned to 8 levels x class x surprise bin,
  2,560 cells) -153/-176 B. LB1: decode-derived `patch192=(y//32)*16+(x//32)` context, -109 B. AFR1:
  `tile48_groupbin8` interaction, -81 B ("address-free tile-conditioned re-encode").
- Later rate moves in the private lineage: rc2 8-byte logistic depth-mixing prior (-231 B), tc1 35-weight token-tail
  mixer over HPAC (-548 B on identical field), **hpr1 retrain the HPAC prior on the edited field: -887 B** (11,128 drifted
  sites), ntb2 frame-embedding even rounding -248 B. Refit law n=4: hpr1 -887 B positive; tail-mixer refit +20 B,
  restored depth state +37 B, pc1's 1.0x refit at 179 drifted sites derived worthless. The prior sits on its
  model-vs-tail knee (dpi1: -400 B model, +437 B tail). Prior width/depth changes are receiver code changes
  (`HPAC_CHANNELS/HPAC_PATCH` hardcoded in `cpr1/inflate.py`, also in `runtime/f26_hpac_native.c`).

## 3. #86 lessons (anr.txt)
- Rate surrogates failed in the ADVERSARIAL-RESIDUAL phase (not on tokens): (1) bits-per-pixel soft surrogate in the
  loss: "the rate term and the task term fought each other: lowering rate immediately destroyed SegNet accuracy";
  (2) libx265/264 in the loop with STE: codec quantization/motion compensation killed gradient signal;
  (3) **hand-coded differentiable entropy-coder surrogate: "tractable and gave clean gradients, but it didn't match the
  actual codec well enough... the optimizer happily reduced the surrogate rate while the real on-disk archive size
  went the other way."**
- Palette experiment: flat 5-colour palette render of GT classes -> SegNet 99.49% accuracy; errors concentrate at
  class boundaries. SegNet's edge localisation lives in stage-0 depthwise conv with a 7-pixel input RF.
- PixelCNN on tokens: v1 0.0078 bpp 113 KB; v2 (prev-frame, FiLM 32, SCN) 0.0062 bpp 110.4 KB; v3 FP32 floor 0.0059
  bpp 84.7 KB but 204 KB weights. Raster decode infeasible (~300-1,640 h) -> HPAC (P=32, delta=2, 192 patches, 94
  group steps).
- HPAC sweep (Table 7): mini 52k params 0.00797 bpp 114.7 KB; **hpac_full 150k params 0.01268 bpp 182.5 KB "FP
  overfits, deeper != better"**; residual factorisation p(diff|prev) 0.02786 (wrong factorisation, abandoned); +SPM
  cross-patch summary 0.00719 (-9.8%); softer SCN 0.00717; b_init 7/dfilm 16 0.00731 (rate gain eaten by SCN bits);
  ch=48 0.00784 (+9%, more than the param savings); SCN off dfilm 8 = FP32 floor 0.00759; ship 0.00773 bpp,
  111.2 KB tokens + 27.6 KB weights (SCN + PPMd order 4).
- Generic coders on tokens (ef1, comma-lab): ZPAQ m5 365,322 B, PPMd 402,241 B at order 32 vs HPAC 113,777 B;
  token re-serialisations 196-687% worse; modulo-5 lifting + retrained prior 418,300 B (3.5x worse); coarse-plane +
  residual: coarse plane alone 169,100 B. Learned context model on the fine plane is the only thing that works.

## 4. Continuous / soft optimisation: what was tried
- Nobody in the comma-lab record optimised the token map in a continuous relaxation (soft one-hot / Gumbel / embedding
  space) and then projected. (Searched for straight-through/gumbel/soft-token/relaxed-token across ddm_* memos;
  hits were renderer QAT, pose carrier, frame-embedding.) INFERRED: this is genuinely untried on this object.
- Gradient-flavoured evidence they did collect:
  - Margin saliency (= Fisher surrogate, r=0.978) and pose saliency were used only to ORDER proposals.
  - fe1 (per-pair FiLM frame-embedding codes): JVP linearization (STE through the uint8 round) predicted vs realized
    flip deltas Pearson 0.966/0.856/0.937, sign agreement 87.5%, but magnitudes optimistic (predicted -9, realized +3)
    because a full code step is outside the linear regime; 18,624/18,906 single-code moves made d_seg worse; shipped
    codes sit at a per-pair local minimum.
  - ren2 renderer gradient refit at fixed field (sigmoid(-margin/tau) through exact R with STE round, QAT, 0.5-LSB
    bound): **loses at every checkpoint** (step 25: d_seg +2.5%, 0/242 pairs improved); "the field is the renderer's
    pre-image" -> renderer and field are co-adapted; changing one alone loses.
  - rq1: coarsening any renderer tensor is 148x underwater; even 3-bit -> 4-bit FiLM made seg WORSE (+881 cells)
    because 600 planes were solved against that exact realization.
  - up2 pose carrier: correct STE Jacobian, LM steps all realized worse.
  - jrx2: joint field + global int4 renderer step: renderer actions damage other pairs (2,438 errors added on the
    other 599 pairs for 0-3 target cells).

## 5. Synthesis for training the token maps with an exact differentiable rate term

### (a) Tricks to include
1. **Pre-distortion is the mechanism**: the target is not GT labels; it is the label map whose RENDER re-segments to
   DALI GT. Expect the optimum to differ from GT at ~20-30k sites (move 48: 18,900) and to be a better pre-image than GT
   (GT renders 2.8x worse). Initialise from the shipped (#140/#141) field, not from GT.
2. Edits should be **thin boundary shifts** (single cells, vertical neighbours most valuable); explicitly penalise
   or project away blob/dilation changes (block moves were -55%/-351%).
3. Rate gradient = the prior's own probability rows: price-first generation (cheapest first) beat saliency-first
   and found **389 proposals with negative real cost** (the field is not at its rate optimum for the shipped prior).
   An exact -log2 p term captures this directly. Argmax-neutral rate edits exist but are rare (4.7%).
4. **Carrier re-solve after every map change is mandatory** and is ~free in bytes (+40-45 B, sometimes a pose credit
   below base). Price pose only after the re-solve; keep the per-pair DROP option (bimodal recovery). Consider pose
   credit as an objective: pose-directed edits moved pose 4.59e-6 -> 4.08e-6 overall.
5. **Per-pair Lagrange admission/revert** after training: score each pair's new map vs old in the exact formula
   (sqrt pose term => sweep a multiplier on pose damage), keep the best subset.
6. **Refit the HPAC prior on the edited field** (hpr1: -887 B after ~11k site drift); refit is worthless when drift is
   ~100s of sites. Expect the prior to sit on its model/tail knee (rate-lambda changes +224..+506 B).
7. Fix renderer; do not co-train renderer with maps unless you re-solve maps after (ren2/rq1).
8. Validate everything on the real receiver path: render at the receiver's batch shape (CPU batch 1 vs CUDA batch 8
   change pixels), CPU fp32 SegNet argmax, DALI GT. Use a seeded random pair sample, never a prefix (pose prefixes
   2.5-4.2x harder).

### (b) Failure modes / pitfalls
- **Surrogate/coder mismatch**: your exact non-adaptive HPAC+table term matches #135's coder, but #140/#141 ship the
  ADAPTIVE corrector + mixers (rr4/fx2/ma1/gb1/lb1/afr1). Under that stack, added surprises cost ~1.28-1.32x and
  removed surprises realize ~0.15x of first-order. Either ship the non-adaptive coder for the trained field (losing
  ~2-3 KB of corrector gains) or re-price with the real encoder; and expect the corrector's gains to shrink/shift once
  the field changes (correctors were tuned to the old field).
- **Selection bias**: anything selected on per-element noisy prices is biased optimistic; re-price as one object.
- **Edits interact**: adjacent accepted edits form block moves; far-field SegNet effects (radius up to 416 cells)
  repair/break other sites; "the context discount compounds; the credit does not"; passes oscillate (reversions).
  Composition across pairs was additive (pose composed at 1.000129 of per-pair sum; seg verified exact).
- **Pose sensitivity**: frame-1-only edits multiplied d_pose 387-467x before re-solve (2 tokens = 0.5% of pixels moved
  pose 6.5x). A seg-only objective without carrier re-solve in the loop would look like a win and lose ~0.15 S.
- **Renderer brittleness**: rendering is a deterministic function but batch shape changes uint8 output; the
  renderer realization (3/4-bit, pruned FiLM) is inside the pre-image; renderer changes are pose-destructive and
  only 16x recoverable.
- **Determinism**: decoder desync from 1-ULP libm differences gave S 27.83; keep the coder integer/exact-arithmetic;
  fire the proven runtime unmodified; T4 inflate time varied 1,023-1,603 s on byte-identical receivers (budget 1,800 s;
  their internal ceiling 1,260 s).
- **GT lineage**: DALI vs PyAV GT differ at ~20.7k sites. Train/eval against the lineage the contest uses.
- **Argmax-only seg**: the real objective is a hard argmax count; any soft seg loss must be validated by realized
  argmax (fe1: linear model orders but over-promises).

### (c) Slack evidence
- From the #140 field (d_seg 2.0139e-4, ~23.75k flips) the discrete greedy search reached 1.0288e-4 (~12.1k flips):
  **~49% of seg debt removed** at roughly +6-7 KB of tail (net about -0.006 S from seg edits after rate), plus pose
  6.37e-6 -> 4.08e-6 largely via carrier re-solves on edited renders. Single-token family closed at pass 8 at
  8.9-10.3 bits/token marginal. Best single pass repaired 40.4%, cumulative 45.8% (rq1). Remaining ~12k flips are
  presumably harder (persistent sites not repairable by any of 36 single moves).
- Rate-side slack on the field: first-order ceiling if every non-argmax token were rewritten to the coder's argmax:
  70 KB (>=0.5 bit savings), but only ~4.7% of such changes are seg-neutral; private lineage concluded every field
  lever is <= 1 bar (2e-5 S) per pass by move 55, i.e. the discrete-local-search slack is spent. A joint continuous
  optimisation could still find moves outside 1-4-token neighbourhoods (INFERRED; unmeasured).

### (d) Continuous/soft optimisation
- Not tried on the token map by #140/comma-lab. Gradient methods on discrete lattices (pose carrier, FiLM codes,
  renderer weights) consistently gave correct orderings but realized worse when taken as steps; the winning recipe
  everywhere was "gradient/saliency proposes, realized discrete evaluation accepts". INFERRED recommendation: use the
  continuous optimisation as a proposal generator, then project to hard tokens and run realized per-pair acceptance +
  carrier re-solve + Lagrange admission; cap per-iteration change density to avoid block moves.

## 6. Other PRs (from sub-agent; full detail in token_history_prs.md, diffs in repos/prs_agent/)
- **#143 (tinkererlife, 0.160594, 185,653 B, -1,071 B vs #135; decoder identical to #135)**. Code:
  github.com/tinkererlife/comma-ai-semantic-pose-hpac-cpr1 `experiments/learned-token-grid-mvp/`.
  Gradient path: `straight_through_one_hot` (hard argmax forward, identity into softmax backward), renderer rewritten as
  `assignments @ token_embed.weight`; ranks with the gradient on the one-hot (`g[a]-g[b]` = benefit of a->b), not on
  logits. Seg loss `expected_flip_loss` (sigmoid margin, tau 0.15); pose via sqrt term/linearisation. Rate for proposals:
  exact own-symbol -log2 p only (`direct_symbol_bits`); acceptance recomputes frames t and t+1 exactly
  (`move_deltas`, `localized_move_delta`) + exact render/SegNet/PoseNet, with split-in-half backtracking
  (`accept_with_backtracking`). Pairs from top-32 singles (`candidate_groups`), best 8 rendered per frame, 1 kept/sweep.
  Carrier: Jacobian-guided integer search between sweeps. **HPAC fine-tune on edited maps: -225 B.**
  **Persistent soft-token optimisation COLLAPSED** (many logits crossing at once -> tens of thousands of hard flips);
  gradient useful only as ranker (1/32, 2/63 accepted); L40S-searched edits scored 0.1605 there vs 0.1646 on T4 (TF32
  flipped accept/reject decisions).
- **#138 OPAL (closed, never evaluated)**: adaptive binary "is token HPAC's top class" model, 55 context families
  (6.18M cells), Newton-step weights from running sums (`w=clamp(-G/(H+2.5),+/-4)`), no decay. -4,684 B on tokens
  (114,706 -> 110,022) => #135 HPAC is miscalibrated on its hit/miss event; same axis as #140's rr4/fx corrector stack.
- **#144 rowpack (closed)**: repacks 517 integer HPAC weight rows (bit-depth grouping, zigzag, bitplanes, HRP1) for
  LZMA: -957 B on the model section; token stream untouched.
- **#145 warped context (0.158199, 187,246 B)**: per-frame choice of 21 integer warps of frame t-1 for HPAC context and
  table lookup: 257 B of choices save ~700 B (net ~-443 B); bidirectional context cost +4.8 KB. ~7,000 pixels on 582
  frames edited for seg, carrier re-tuned by gradient through PoseNet on T4 (edits ~+965 B, seg -4.27e-3).
  Note: evaluator's `rgb_to_yuv6` is `@torch.no_grad()` and silently blocks PoseNet gradients unless unwrapped.

### Merged implications
- #143 is the only direct soft-token attempt and it collapsed when run persistently: use hard-forward STE, a per-step
  flip budget/trust region, frequent re-hardening and exact verification. Include frame t+1's rate (context path).
- Table lookup / edge-distance features / logit rounding are piecewise constant: stop-grad them, verify exactly.
- Validate on T4 numerics (TF32 settings flipped #143's decisions; batch shape flips renderer pixels).
