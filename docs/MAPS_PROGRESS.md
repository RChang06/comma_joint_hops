# Exact-rate class-map training on PR #135: progress log

## Goal
Train #135's 600 class maps (token stream, about 114.7 KB, 63% of the archive) directly against the score:
100·seg + sqrt(10·pose) + 25·bytes/37,545,489. The byte term is #135's exact code length. The renderer, the 12 carrier
patterns and HPAC are frozen. The carrier strengths get re-solved afterwards. #140/#141's adaptive correctors and packing
are bolted back on at the end as a lossless step.

## Done (2026-09-30)
1. **The cost model is exact.** `exact_cost135.py` decodes #135's stream and sums -log2 p per pixel: 114,705.46 B,
   against the actual stream's 114,706 B. Raw HPAC without the boundary table gives 114,852 B (the table saves 146 B).
   Decoded #135 maps are in `pr135_cost/tokens_pr135.u8`.
2. **#135 components are extracted:** `extracted/pr135_components.pt` (renderer 66,339 params, carrier basis
   12x3x24x32, coefficients 600x12, 14 B selector, 25x5 boundary table).
3. **#141 vs #135 maps:** only 8,723 pixels differ, across 458 frames. That is all of #140's 455 edits, about 19 px
   each. `extracted/tokens_pr141.u8` holds #141's maps.
4. **The cost script can dump a price list:** `--dump-costs a:b` writes -log2 p for all 5 classes per pixel.
5. **`train_maps135.py` is written but not run yet.** Band pixels (within --band px of a boundary) get 5-way logits,
   with a straight-through hard map. The loss is seg (expected flips, tau 0.05) + pose (slope of the sqrt at the
   #135 level) + exact-rate price list, with the renderer frozen and the carrier plus selector frozen during map
   training.

## Running / next
- Costing #141's maps under #135's coder (`exact_cost135_pr141maps.log`), to find out what #140's edits cost under a
  correctors-free coder.
- Then: `exact_cost135.py --tokens pr135_cost/tokens_pr135.u8 --frames 21 --dump-costs 0:20`, then
  `train_maps135.py --frames 0:20 --costs pr135_cost/costs_0_20.npy` (small test on the laptop T2000).
- Pass condition: the trained map, exact-costed with `exact_cost135.py --tokens`, plus exact seg/pose on those frames,
  beats the start. Never trust the training loss alone.
- **Open blocker for shipping:** we need #135's ENCODER (RC64 encode + container) to write a real archive.
  Research agent 1 is searching codexblack's experiment repo and #140's comma-lab. Findings will land in
  `research/foundation_135.md` and `research/token_history.md`.

## Facts from the other session (ruich-14), verified against shipped archives
- The renderer is shared by #130/#135/#141. Renderer retraining from the shipped weights looks closed.
- After any change to the even frames, re-solve the carrier with an MSE loss (not range-normalised).
- Traps: `rgb_to_yuv6` is `no_grad` (use `__wrapped__`); TF32 must be off; apply the frame-0 selector to the slave frames.
- Laptop T2000 scoring of #141 gives 0.147899 (official 0.147897), so local exact scoring is free.

## Update: costs, encoder found, toolchain complete
- **#141's maps under #135's coder: 118,853 B** (vs 114,705 for #135's own), so #140's 455 edits cost +4,147 B
  (+0.0028) under a correctors-free coder. Their seg gain (~0.0095) still dominates, so transplanted net is about -0.0067 (to verify).
- **The encoder exists** (research agent 1, `research/foundation_135.md`): codexblack's ExperimentBook has the full
  RC64 encoder (`src/cpr1_sub4/entropy/rc64.py` + `rc64_backend.c`) and the teacher-forced encode loop. The token stream is
  the tail of zip member p, so a new map ships by splicing: `new_p = old_p[:-len(old_stream)] + new_stream`, re-zipped
  (stored, 1980 date, create_system 3, attr 0o100644<<16).
- The boundary table was fitted one-shot (log-ratio counts), not optimised. A direct rate-gradient fit should beat it,
  and it should be refit after maps change.
- New scripts: `encode135.py` (map -> real #135 archive; gate = byte-identical re-encode of #135's own map) and
  `score_map135.py` (exact score of map + archive without inflate; gate = #135 official 0.16226842).
- Research agent 1 stalled after writing its key findings. Agent 2 is still running (token history).

## Update: scorer gate passed + research agent 2 findings (2026-09-30)
- **score_map135.py gate PASSED:** #135 own map gives seg 2.964359e-4 / pose 6.88537e-6 / score 0.16227325, vs official
  2.963935e-4 / 6.88439e-6 / 0.16226842 (+0.014% each, laptop vs T4). Exact local scoring of any #135-style map takes ~6 min.
- **Research agent 2** (`research/token_history.md`, `token_history_prs.md`):
  - **adpena (#140 author) reached S 0.136034 @ 179,255 B privately (T4, n600, 2026-09-17), unsubmitted.** Seg went
    2.01e-4 -> 1.03e-4 via more discrete token edits + carrier re-solves + lossless. So about half the seg debt was recoverable.
  - Pre-distortion: 95.9% of wrong cells have a CORRECT stored token; the gain comes from storing a "wrong" label so
    the render reads back right. Rendering the true partition is 2.8x worse.
  - Moves: single-token at a wrong cell or its 8 neighbours (36 candidates). Dilation fails. Greedy accept only if
    the pair's wrong-cell count falls on the real render. Influence is long-range (median Chebyshev 19, p90 155).
  - Real rate was 1.3-2.2x the modelled price. Marginal family closed at ~9 bits/token vs 10.31 break-even.
    Price-first proposals (cheapest under the prior) found 389 negative-cost edits.
  - Admission is a lambda-sweep over [0]+logspace(-3,6,1801), each kept set scored exactly. Carrier re-solve is damped
    GN + a ±2 polish, stopping at threshold 5.59e-9. A stale carrier makes pose 387-467x worse after edits.
  - **#143 tried soft-token optimisation: it collapsed** (mass flips). Gradients only work as RANKERS. Use per-step flip budgets.
  - Coder caveat: under adaptive correctors, removing surprises realises only 0.15x of the first-order price. Exact
    #135-coder savings will NOT carry over 1:1 if correctors are bolted on, so judge on the coder actually shipped.
  - HPAC refit on an edited field (~11k sites) saved -887 B (#140), -225 B (#143).
- **Design change:** gradients rank candidates (per-step flip budget, boundary/wrong-cell neighbourhoods), and every
  edit set is accepted per pair on the real render + exact rate (frames t and t+1). Then carrier re-solve and a lambda-sweep.
  The main prize is seg pre-distortion (about half the seg debt, ~0.01), with rate as a priced constraint.

## Update: first search + carrier re-solve (Delaware 5090, instance 53624770)
- **Encoder gate PASSED:** re-encoding #135's map is byte-identical (186,724 B, sha 12cf5d71).
- **search135.py on pairs 0-49** (from #135's maps, k=8, rounds 40): wrong cells 2483 -> 2090 (-16%), 312 edits, 39 s.
  Real archive: seg 2.9644e-4 -> 2.9310e-4 (exactly as the search measured); stream +102 B (the price list predicted
  +22 B, because it ignores neighbour and next-frame context effects; real is ~4.6x here).
- **Carrier re-solve:** refit135.py (Adam) was far too slow. **refit_gn135.py (damped Gauss-Newton / LM per pair,
  exact int12 acceptance) works:** on #141's maps, pose 3.8e-3 -> 9.03e-6 in 20 iterations (~2 min), converged.
- **Transplant result (#135 coder + #141's edited maps + GN carrier): score 0.15847** (seg 0.02183, pose 0.00955,
  rate 0.12709, 190,871 B) vs #135's 0.16227, so **-0.0038**. Pose is 1.3x #135's. Next: a ±1/±2 integer polish after GN
  (adpena), plus a selector re-pick.
- Running: search over all 600 pairs from #141's maps (k=32, rounds 100). Its queued refit is the OLD Adam one, so re-run
  GN afterwards.

## Update: integer polish closes the pose gap
- `refit_gn135.py --polish N` adds an exact +-1/+-2 per-dimension integer search after GN (adpena's alternation). On the
  transplant: GN 9.03e-6, then 2 more GN iters 8.46e-6, then polish 4.65e-6 (4 passes, 1627/338/85/13 moves).
- **Transplant + GN + polish: score 0.15580** (seg 0.02183, pose 0.00687, rate 0.12709). That's -0.0065 vs #135, and its pose
  term beats #141's 0.00798. The 14-byte selector is still the original.
- Side note: the same polish applied to #141's own carrier could give it a pose gain (0.00798 -> ~0.0068?), but that needs
  #141's CAP1 re-encode. That's the other session's axis, so I flagged it rather than pursuing it.

## Experiments 1 and 2 (2026-10-01)
- **jA2** (block A, seg-only, `--w-rate 0 --w-pose 0`): seg 2.009e-4 but stream 114,706 -> 137,336 B (+22.6 KB, ~17 bits
  per edit; isolated specks are the most surprising thing for hpac). Real archive + gn/polish: 0.16895. Rate was never
  switched on for the maps; the jA2 log printed token_bytes at every gate and it went unread.
- **Experiment 1** = jA2 + search135 (frozen price list) + gn/polish: seg **1.342e-4** (lowest yet, beats #141+search
  1.451e-4, so training and search stack), bytes 209,057, pose 1.95e-5, score 0.16659. Seg + rate only: 0.15262.
  The price list claimed -2.9 KB and the real stream fell ~300 B: own-pixel prices miss neighbour/next-frame context.
  Pose after refit was 3.5x worse than #141+search at similar edit counts: #140 priced pose per edit, we did not.
  Results in `exp1/`. Box 53645044 destroyed.
- **Experiment 2** (box 53664321, `exp2_train.py`, `queue_exp2.sh`): map training with exact rate on (`--w-rate 1`,
  pose off in loop and gate), frame-accept by per-frame score delta, then `--search-rounds` search with every candidate
  priced by exact hpac bits of frames t and t+1 (smoke: predicted -17 bits / -31 cells matched the real stream and seg
  exactly). Smoke at 361 edits: seg 2.905e-4 for +142 B (~0.4 B/edit vs jA2 ~2.1). Pose in the training gradient
  through soft maps blew up (pose 7.0); hpac training noise (+-1 KB per gate) made per-frame acceptance revert 550 frames,
  so hpac is a separate later step.
- Agreed order for the loop: map (seg+bytes) -> search (seg+bytes) -> renderer (seg+pose, carrier updating inside)
  -> final carrier refit. Shipping a renderer or hpac change needs the WANS / IHS2 writers (rebuild_plan.md).
- **Exp 2 step 1 DONE** (`e2_train`, 3000 steps, gate 150, all 20 gates ACCEPT): seg 2.964e-4 -> **2.299e-4**, stream
  114,705 -> **114,079 B (-626)**, seg+rate 0.15397 -> **0.14690** (ground truth 1 holds). 8,492 edits. Still improving
  slowly at the end (0.14694 -> 0.14690 over the last 300 steps). Search started right after at ~3.5 s/pair.
- **Exp 2 step 2 DONE** (byte-priced search, pool all, k32, 1 batch, 60 rounds, 37 min): wrong -3,278, exact bits
  +3,216 (+402 B), 3,212 edits. Seg 2.022e-4, stream 114,482 B, **real archive seg+rate 0.14440** (bar 0.14270,
  short by 0.0017). Ground truth 2 passed: trainer 114,481.2 B vs real stream 114,482 B, seg/pose identical.
  Pose after in-trainer gn22+polish4: 2.32e-5 (full 0.15964); pose is step 5's job. Results in `exp2/`.
  Suspect: wide pool (edges + changed pixels) lets byte-only candidates crowd the 32 tested and end pairs early
  (~5 edits/pair vs ~15 in exp 1). Diagnostic on pairs 200:240: all/1 batch vs near/1 vs all/4 batches.
- **Search diagnostic (pairs 200:240, from step-1 map):** wide pool/1 batch net -0.000135 in 157 s; **near-error pool
  net -0.000385 in 324 s (2.85x)**; wide/4 batches ~19 s/pair, worse per minute (stopped). Byte-only edge candidates
  crowded the 32 tested and ended pairs early.
- **Exp 2 step 2 rerun, near pool (`e2b_search`, ~83 min): seg 1.639e-4, stream 114,739 B, real seg+rate 0.14074.**
  Beats the superset bar 0.14270 (#141 maps + search, #135 coder) by 0.0020. Pair nets sum -0.006186 = total;
  real stream 114,739 vs trainer 114,738.3; seg within 1 pixel. 7,604 edits. No carrier refit in this run.
- 2026-10-01 ~04:30 local: laptop memory pressure killed the session's background shells incl. the box guard; replaced
  by Windows scheduled task `comma_exp2_destroy` (destroys 53664321 at 10:32 local). Overnight: cycle 1 steps 3-5.
- **Cycle 1 step 3 launched** (`c1_s3`, hpac+table refit, jB9 settings) then `c1_s4base` (carrier gn22+polish4 on
  the step-2 map with the refit coder = renderer stage baseline). Gate 0 reproduced step 2 exactly.
- **Cycle 1 step 3 (first try, box 53664321): stream 114,738 -> 113,946 B (-792), seg bit-identical.** Then the
  renderer-stage baseline (carrier gn22+polish4 on the step-2 map): **pose 7.73e-5 (term 0.0278)**, confirmed by the
  standalone refit_gn135 (gn converged, 0 accepted at iter 22). Full score at that point ~0.1680 (seg+rate 0.14022 +
  pose 0.0278) vs #135 0.16227: the pose-blind edits of steps 1-2 cost ~0.02 of pose the carrier cannot repair.
  Ray: run steps 4-5 as planned before judging.
- Box 53664321 was host-reclaimed right after (credit fine, scheduled task never ran). Step 3 state was LOST (not
  pulled). Replacement **53695658** (KR, ssh6:21476), scheduled destroy 11:46 local. Step 3 redo launched, gate 0
  reproduced step 2. `sync_box.sh` now pulls c1_*/c2_* folders every 10 min into `cycle_sync/`.
- **Cycle 1 step 3 redo (box 53695658): 114,738 -> 114,239 B (-499)**, seg bit-identical (first try got -792; hpac
  training is stochastic). Baseline carrier refit: pose 7.731e-5 (term 0.0278), full ~0.16822. Pulled to `cycle1/`.
- Order swap (Ray): new step 4 = pose stage (strengths gn+polish at gates + 12 patterns + gray/amp by gradient), new
  step 5 = renderer. Ray's expectation: pose stage lands near the best runs (term ~0.007-0.008), never 0.1/0.2.
- **Pose-stage control on #141 maps + search** (`c1_ctrl141`): gate 0 reproduces 0.15013 exactly (pose 5.518e-6);
  gate 50 pose 4.27e-6, score 0.14923, seg bit-identical. Carrier code is sound; our 7.7e-5 comes from our map's
  pose-blind edits. (Control is #140's map: reference only, not our deliverable.)
- **Cycle 1 step 4 (pose stage) smoke, our map** (`c1_s4smoke`, 300 steps): pose 7.731e-5 -> **4.927e-6**, seg
  bit-identical. Gate 100 did most of it. **Independent scorer (`score_map135.py --state`, decoder-side carrier
  rebuild) reproduces 4.92721e-6 exactly.** Real-archive full score with the step-2 archive (#135 hpac):
  **0.14776 (seg 0.01639 pose 0.00702 rate 0.12435) < #141's 0.147897.** Step 3's -499 B would add ~-0.00033 once
  the IHS2 writer exists. Not yet shippable: new strengths need the CAP1 writer, gray/amp need decoder edits.
- **CORRECTION:** basis codes changed 0 (lr-basis 0.02 never crossed a rounding step); only gray 127.5->127.479 and
  amp 64->63.974 moved, plus gn strengths. So the 7.7e-5 was a stuck strengths-only fit, NOT pose damage the carrier
  cannot repair. My earlier "pose-blind edits cost ~0.02" diagnosis was wrong.
- **Cycle 1 step 4 continuation** (`c1_s4`, lr-basis 0.1, 600 steps): pose 4.927e-6 -> **4.817e-6**, trainer score
  0.14737 (incl. step-3 bytes estimate). 1 pattern code changed; a pattern move at gate 100 spiked pose to 1.7e-4
  and was rolled back (patterns are extremely sensitive). Pulled.
- **Cycle 1 step 5 (renderer) smoke** (`c1_s5smoke`, lr-render 0.05): 6/6 gates ROLLBACK. Renderer moves cost seg
  (1.639e-4 -> ~1.73e-4) and bought no pose: the wrong trade direction (rule 3), correctly rejected. Pose stayed
  < 2x start (rule 2 ok). Retrying once at lr-render 0.01 / fp16 2e-4 (`c1_s5b`); if also all rejected, step 5 =
  renderer unchanged and cycle 2 starts from `c1_s4`.
- **Cycle 1 step 5 retry** (`c1_s5b`, lr-render 0.01, fp16 2e-4): 6/6 ROLLBACK again (seg +2-5e-6, pose flat).
  **Step 5 = renderer unchanged.** Steps 4+5 check: 0.14737 <= step-4 start 0.16822, passes.
- **CYCLE 1 DONE. Independent scorer on `c1_s4`: seg 1.63871e-4, pose 4.81743e-6, archive 186,757 B (step-2 archive,
  #135 hpac) -> full 0.14768 (seg 0.01639 pose 0.00694 rate 0.12435), vs #141 0.147897.** With step 3's -499 B
  (trainer est) ~0.14737. To ship: IHS2 writer (step 3), CAP1 writer (strengths, 1 pattern code), gray/amp in decoder.
- **Cycle 2 launched** (`queue_c2.sh c1_s4`: step 1 like cycle 1, near-pool search, hpac refit). Gate 0 seg+rate
  0.140428 = cycle 1 end (rule 3 ok). Box 53695658 ends 11:46 local.
- **Cycle 2 step 1** (`c2_s1`, cycle-1 settings): seg+rate 0.140428 -> 0.140406 (-0.00002), 146 flips, 9/20 gates
  accepted with a long rollback run in the middle. The map is near what this training reaches from cycle 1's end;
  for cycle 3 soften (lower lr-map / map-t0). Rule 1 ok. Pulled to `cycle2/`. Step 2 search running.
- **Cycle 2 step 2** (`c2_s2`, near pool): wrong -588, +762 bits (+95 B), 1,044 edits, ~25 min. Seg 1.639e-4 ->
  **1.5949e-4**, seg+rate 0.140406 -> **~0.14003**. Pair nets sum -0.000435 = total. Pulled. Step 3 then step 4 queued.
- **Cycle 2 step 3:** 10/10 gates ROLLBACK (coder already refit in cycle 1) = coder unchanged. Trainer saved state
  only on ACCEPT, so step 4 crashed on a missing c2_s3/joint_state.pt. Fixed: `exp2_train.py` now saves the start
  state at gate 0. Step 4 rerun from `c1_s4` state (same coder). Also killed my own launch again with `pkill -f`
  over ssh (pattern matched the ssh command line); stop using pattern kills remotely.
- Cycle 2 map real archive (#135 coder): 186,866 B.
- **CYCLE 2 DONE. Independent scorer: seg 1.59480e-4, pose 4.58491e-6 (= trainer), archive 186,866 B (#135 coder)
  -> full 0.14715 (seg 0.01595 pose 0.00677 rate 0.12443).** Cycle gain -0.00053 (> 1e-4, keep cycling). Trainer
  with refit-coder bytes: 0.14679. vs #141 0.147897: -0.00075 real. Pulled to `cycle2/`.
- **Rebuild toolchain (2026-10-01 morning):** `hpac_writer135.py` (gate A: byte-identical #135 HPAC and table),
  `build_archive135.py` (full F24S assembly; #135's exact lzma bytes not reproducible, best +11 B; CAP1 store
  round-trips), `sub_c2/` decoder copy with trained gray 127.44857025146484 / amp 63.97188949584961.
  **Trainer bug found by the HPAC writer:** the hpac row clamp is |w| <= max|w0| (symmetric), not the row's signed
  depth range, so 16 rows grew a bit: c1_s3's HPAC blob is 16,832 B vs 16,599 (+233 B pre-lzma), eating part of step
  3's -499 B estimate. Fix for future cycles: per row clamp to [-2^(d0-1), 2^(d0-1)-1] (d0 = starting tight depth).
  Table unchanged by step 3 (0/125 codes). Decoder constant: IHS2_BYTES = 16_832.
- **END-TO-END PASS (2026-10-01 ~10:00 local).** `carrier_writer135.py` (gate: #135 carrier byte-identical; trained
  carrier 22,272 B, +30; 6 coefficients pinned at the int12 limits) + `hpac_writer135.py` + `build_archive135.py`
  -> `cycle2/archive_c2.zip` **186,512 B, sha ad2f4bfd...**. Decoder copy `sub_c2/`: IHS2_BYTES 16_832, archive
  sha/size, gray/amp constants, *.sh converted to LF (the Windows checkout had CRLF: inflate.sh failed on
  `pipefail\r`). Official `inflate.sh` in a fresh dir (91 s on a 5090) + `score.py` on the decoded raw:
  **seg 1.59471e-4, pose 4.58469e-6, bytes 186,512 -> 0.146909 (seg 0.01595 pose 0.00677 rate 0.12419)**
  vs #141 official 0.147897 (-0.00099). Box offset vs official on #135: +0.000005. Local copy: `final_c2/sub_c2/`.
  Not submitted/pushed. Still unverified: T4 runtime (30 min limit; 91 s on 5090) and an official-machine run.

## STATE SNAPSHOT (2026-10-01 ~10:15 local), resume here
- **Best verified submission:** `final_c2/sub_c2/` (archive 186,512 B, sha ad2f4bfd...), official inflate + judge
  score **0.146909** vs #141 0.147897. Not submitted. Seg 0.01595 + pose 0.00677 + rate 0.12419.
- **Bytes gap to #141 is the coder:** our token stream 114,297 B already beats #135's 114,706; #141's map costs
  190,871 B on #135's coder but ships at 179,891 B thanks to #140's adaptive correctors (~11 KB, ~0.007 of score).
  Coder is lossless, so swapping it leaves seg/pose identical.
- **In progress:** agent building `work/encode141.py` (mirror #141's corrector-based token decode in known-symbol
  mode); notes in `work/research/coder141.md`. Gate: re-encoding #141's decoded map (`work/extracted/tokens_pr141.u8`)
  must reproduce #141's token stream byte for byte. Then: our cycle-2 map (`cycle2/c2_s2/tokens_joint.u8`) in #141's
  container, plus our carrier/gray/amp if #141's container allows, end-to-end inflate + score.
- **Next after that:** fix the hpac row-depth clamp in exp2_train.py (~233 B), more cycles with the map priced under
  the new coder (#140 notes: corrector coders realise only ~0.15x of first-order map savings).
- Box 53695658 (ssh6:21476) idle, scheduled destroy 11:46 local. Rent fresh for the next end-to-end check.
- **#141 encoder DONE** (`encode141.py`, gate: re-encoding #141's map reproduces its 113,411 B stream and the whole
  179,891 B archive byte for byte; ~1 s/frame on the T2000 with the native corrector). Correctors store no params.
  #141's renderer differs from #135's in 5 tensors (frame_embed, blocks.0-3.film), basis/selector identical, so our
  map ships with #135's renderer (WANS F12 36,040 B in a LAY1 wrapper). Running: encode of our cycle-2 map with
  #141's coder (`encode141_c2map.log`); agent then assembles `final_c2/sub141_c2/` (#135 renderer, our carrier via
  #141's rr5/dx2 recoders, gray/amp) and verifies end to end on the laptop. If interrupted (laptop off), rerun:
  `python work/encode141.py --tokens work/cycle2/c2_s2/tokens_joint.u8 --out work/final_c2/archive141_c2map.zip
  --stream-out work/final_c2/stream141_c2map.bin`, then the assembly per research/coder141.md.
- **PAUSED 2026-10-01 ~10:35 local** (laptop sleeping). Stopped: encode of our map with #141's coder (partial, rerun
  from scratch), the assembly agent (had just started; check `final_c2/` for a partial `sub141_c2/` and delete it
  before rebuilding), the box sync loop. Box 53695658 left to its scheduled destroy at 11:46 (wake-to-run task).
- **#141-CONTAINER END-TO-END PASS (2026-10-01 11:40 local).** `final_c2/sub141_c2/` (archive 180,008 B, sha
  8464c671...): our cycle-2 map coded with #141's coder (stream 110,039 B), #135's renderer (WANS F12 in a LAY1
  wrapper, 36,048 B), our carrier through #141's packed CAP1 + dx2 + rr5 (21,965 B), gray/amp in decoder.
  Official inflate on the laptop T2000: 1,190 s total (token decode 745 s), decoded token sha b057e6c9 = our map.
  score.py: **seg 1.59471e-4, pose 4.58643e-6, 180,008 B -> 0.142579** (seg 0.01595 pose 0.00677 rate 0.11986)
  vs #141 0.147897 (-0.0053) and our #135-format 0.146909. Models section is ~3.5 KB larger than #141's
  (#135's WANS renderer compresses worse than #141's pruned SM3R one). Restart mid-run lost only the score step.
  T4 runtime risk: 1,190 s on a T2000 against the 1,800 s limit.

## RESUME HERE (paused by Ray 2026-10-01 ~11:50)
- **Best: `final_c2/sub141_c2/` = 0.142579 end to end** (archive 180,008 B, sha 8464c671...). Inputs:
  map `cycle2/c2_s2/tokens_joint.u8`, carrier `cycle2/carrier_c2.bin` (from `cycle2/c2_s4` state), gray/amp in
  `sub141_c2/cpr1/inflate.py`, builder `build_archive141.py`, encoder `encode141.py`, e2e `final_c2/e2e141.sh`.
  Fallback: `final_c2/sub_c2/` (#135 format) 0.146909. Nothing submitted or pushed; no boxes running.
- Before submitting: T4 timing (laptop decode 1,190 s vs 1,800 s limit).
- Next ideas: #135 renderer in #141's SM3R pruned format if exact (up to -3.5 KB); more map cycles priced under
  #141's coder; hpac refit with row-depth clamp fix; step 5 from #130 float checkpoint or single-code search with
  carrier refit per candidate.

## Joint map + renderer stage (seg + rate, pose off), started 2026-10-01 afternoon
- `wans_writer135.py`: gate passed (re-encoding #135's renderer = shipped 36,040 B body). Trainer: `--lr-render-scale`
  (per-row fp16 scales in log units), `--render-bytes wans` (real wans body inside #135's lzma models section; the
  trainer's float32->log->exp->fp16 path reproduces #135's body exactly), `--kappa` (active-set hinge seg loss).
- Box 53752288 (ssh5:12338), scheduled destroy 17:37. Stage start (trainer accounting, refit coder): seg+rate 0.140019.
- Smoke 1 (frame-accept on): 520/600 frames reverted each gate (renderer touches every frame, so per-frame accept
  undid the map's re-adaptation). Smoke 2 (no frame-accept, renderer once per sweep, gate 300): 3/3 ROLLBACK; at the
  last gate the map barely moved (1 flip) and seg was still 1.620e-4 > 1.595e-4: the renderer moves themselves hurt.
  Hypothesis: the sigmoid seg loss drives the renderer by already-correct pixels and flips the marginal pixels the
  pre-distorted map relies on. Probes: e1 scales only; e2 kappa 0.1 hinge; e3 tau 0.01.
- Probes (300 steps each, all gates ROLLBACK): e1 scales only (last gate TIE = no move), e2 kappa 0.1 hinge (best seg
  1.626e-4), e3 tau 0.01 (best 1.628e-4). Every gradient renderer move worsened exact seg even with the map co-training.
- **`diag_render_dirs.py`: 12/12 random +-1-fp16-step scale directions worsen seg in BOTH signs, each by ~+0.0006 of
  seg term.** Sharp local minimum: the decoder rounds rendered pixels to integers and the pre-distorted map is tuned to
  this renderer's exact integer output. Small/gradient moves cannot help.
- Decisive test running (`queue_rj4.sh`): big renderer move with the map frozen (seg 1.708e-4, seg+rate 0.141157, i.e.
  +0.00114), then a full near-pool map search to re-adapt (`rj_move_search.log`). If the searched seg+rate < 0.140019,
  another basin exists; if not, the renderer is effectively done short of retraining from #130's float checkpoint.
  (First attempt saved the post-rollback state by mistake; fixed by saving before the final verdict.)
- **Renderer big move + full re-search BEATS the stage start:** seg 1.708e-4 -> **1.5116e-4**, +390 B, seg+rate
  **0.139451 vs 0.140019 (-0.00057)**. Results in `rj_results/`. So the renderer is not stuck; it needs big moves
  followed by a map re-search. Controls running (`queue_rj5.sh`): pose stage on the moved state (pose 8.64e-4 before
  refit), then the same search on the old renderer as the fair seg+rate baseline.
- **Controls:** (1) same near-pool search on the OLD renderer: 133 edits, seg+rate 0.140016 (-0.000004): the old
  setup is exhausted. (2) pose stage on the moved+searched state: seg bit-identical 1.5116e-4, pose 8.64e-4 ->
  **4.195e-6** (better than cycle 2's 4.585e-6), gray 127.390 amp 63.958, trainer full **0.14594 vs cycle 2 0.14679
  (-0.00085)**. The renderer move is a real win (seg and pose). States: `rj_results/rj_move_search`, `rj_results/rj_pose`.
- **Hop 1 real archive (#141 format):** `final_h1/archive141_h1.zip` 180,443 B (sha 46ac573c...): moved renderer WANS
  body still 36,040 B (only scales + fp16 tensors moved; int4 codes unchanged), stream 110,470 B, carrier 21,975 B,
  gray 127.3896713256836 amp 63.95844268798828 in `final_h1/sub141_h1/`. Builder gates all pass. Expected
  ~0.1417. End-to-end running (`final_h1/e2e.log`).
- **Hop 2 running on box:** renderer move from hop 1 (seg+rate 0.140024, +0.00057), re-search projected ~-0.0009.
- **HOP 1 END-TO-END PASS (15:57):** `final_h1/sub141_h1/` official inflate on the laptop (16 min), decoded tokens =
  hop-1 map (sha 152f2bed), score.py: **seg 1.51172e-4, pose 4.19541e-6, 180,443 B -> 0.141744** (seg 0.01512 pose
  0.00648 rate 0.12015). **New best**, -0.000835 vs 0.142579, -0.00615 vs #141.
- **Hop 2 DONE (box):** renderer move +0.00057, re-search 1,548 edits -> seg 1.4726e-4, seg+rate **0.139199** (hop 1
  0.139451, -0.00025); pose stage -> pose 4.046e-6, gray 127.3958 amp 63.9764, trainer full **0.14557** (hop 1
  0.14594, -0.00037). Pulled to `rj_results/h2_*`. Box shutdown moved to 19:38 (Ray: "we'll be using more").
- Agreed with Ray: chain renderer hops on seg+rate with only a quick pose alarm (strengths gn+polish, stop the chain if
  pose > 2x the previous hop), then ONE thorough pose stage at the end including pose hops (big pattern / gray-amp
  move, full strengths refit to re-adapt, keep if pose term + pattern bytes improve; cheap, so try several directions).
- **Hop 3 DONE:** move +0.0005 seg term, re-search 1,188 edits -> seg 1.4442e-4, seg+rate **0.139033** (-0.00017);
  quick pose alarm 4.065e-6 (ok). Hop gains shrinking: -0.00057, -0.00025, -0.00017. Pulled `rj_results/h3_*`.
- Stage A (pose hops, `stageA.sh`) launched from `h3_alarm`.
- **Stage A (pose hops) DONE:** round 1 -0.00006, round 2 -0.0000034, stopped. Trainer full 0.145403 -> 0.145338, pose
  3.971e-6. State `rj_results/sA_r2_b0.1`. **Stage B hop 1** launched from it (`queue_B1.sh`). Box ends 20:38.
- **Stage B hop 1:** full 0.145338 -> 0.145253 (-0.000085), seg 1.4290e-4, pose 3.983e-6. B hop 2 launched.
- **Unattended (Ray offline ~1 h, laptop asleep):** `auto_B.sh` on box 53752288 waits for B hop 2, chains B hops 3-6
  while each beats its start by > 1e-5, then a 600-step pose stage (`final_pose`). Log `auto_B.log`, bundle
  `/root/auto_B.tgz` on the box (NOT pulled yet: laptop asleep). Box shutdown moved to 22:38 (laptop scheduled task,
  WakeToRun). On return: pull /root/auto_B.tgz, then build + verify the #141-format archive of the best state.
- **Unattended cycle queued** (`auto_C.sh`, waits for auto_B): map training (soft, lr-map 1.0, t0 0.5, 1500 steps,
  frame-accept) -> near-pool search -> hpac+table refit with the NEW per-row signed-depth clamp -> 600-step pose stage.
  Folders C_s1..C_s4, bundle /root/auto_C.tgz, log auto_C.log. Box shutdown moved to 00:38 (Oct 2).
  exp2_train.py on the box now has the hpac depth clamp (inert when lr-hpac 0, so auto_B's later hops are unaffected).
  Laptop-side backups: final_h1 (verified 0.141744), rj_results/* through B1_pose (+B2 search). Everything newer
  is only on the box until pulled.
- B hops 2-4: 0.145100, 0.145027, 0.144939 (trainer full). Pulled to rj_results. B5 running.
- B5 rejected; final pose stage on B4 -> 0.144865 (pose 3.804e-6, seg 1.38957e-4). Pulled. Verifying on laptop (final_B/). Cycle auto_C started on box from final_pose.
- **NEW BEST VERIFIED: `final_B/sub141/` = 0.140704** (seg 1.38974e-4, pose 3.80603e-6, 181,176 B). Decoder copy has
  gray/amp + sha/size set. Not submitted.
- NEW BEST VERIFIED 0.140608: final_sm3r/sub141 (final_B content + wider BLK2 packing, 181,031 B).
- NEW BEST VERIFIED 0.140143: final_E/sub141 (181,015 B, seg 1.35396e-4, pose 3.68790e-6).
- **Pruned line, unattended (`auto_J.sh` on box 53752288, shutdown 11:38):** after I_pose (pose stage on pruned + wide x4,
  pose ~3.88e-6 at gate 300): cycle 2 (J_s1 map training, J_wide4 search, J_pose), then renderer hops K1..K4 with
  --prune-film-keep 1 pinned, then K_final pose stage. Bundles /root/J_*.tgz, /root/K*.tgz. On return: pull all, build
  the real pruned archive (build_archive141_sm3r.py --renderer sm3r --keep 1 --wide, rows must be exactly zero), verify.
- **NEW BEST VERIFIED 0.138310: final_P/sub141 (pruned, 178,610 B).**

## RESUME HERE (Ray asleep, 2026-10-02 ~05:15)
- Best verified: **final_P/sub141 = 0.138310** (pruned, 178,610 B). Previous: final_E/sub141 0.140143, final_sm3r 0.140608.
- Box 53752288 (ssh5:12338) runs auto_J.sh: J_pose (cycle-2 pose stage) -> renderer hops K1..K4 (pruned rows pinned)
  -> K_final pose stage. Bundles /root/J_*.tgz, /root/K*.tgz, /root/K_final.tgz. Shutdown 11:38 (credit $26).
- When it ends: pull, verify the best state with `verify_state_sm3r.sh <state> <out>` (~30 min laptop), then the
  final #141-HPAC fine-tune idea, then a T4 timing test before submitting. Nothing submitted or pushed.
- Hops K1 -0.000059, K2 -0.000047, K3 -0.000006 (stop); K_final pose stage 0.142626 (trainer). Box idle ~08:00 with
  window to 11:38, so (judgement call, in-method tuning) auto_L.sh: renderer hops with 2x and 0.5x move size from
  K_final, then one pose-hop round. Bundles /root/L_*.tgz. K_final being verified on the laptop (final_K/).
- 08:09 laptop memory pressure: Claude Code stopped the K_final e2e (inflate+score) run. Not restarted (needs Ray's OK). Rerun: wsl bash work/e2e_only.sh final_K rj_results/K_final. Box auto_L continues independently.
- **NEW BEST VERIFIED 0.138050: final_K/sub141** (178,966 B, seg 1.28089e-4, pose 3.69033e-6).
- Box 53908794 (ssh6:34468, shutdown 17:42): auto_N (wide4 -0.00045 -> N_pose trainer 0.142191 -> #141 HPAC fine-tune +
  real encodes), then auto_O (hops O1-O3 from N_pose, wide4 + pose if any hop paid). Bundles /root/N_*.tgz, /root/O*.tgz;
  laptop puller pull_all.sh. N_pose being verified on the laptop (final_N/).
- HPAC gate bug fixed; #141 HPAC fine-tune rerunning (auto_N2.sh), first gate -495 B pre-corrector. auto_O still queued behind it.
- N_hpac: real -1,122 B from fine-tuning #141's HPAC. final_NH archive 177,693 B built; verifying after final_N.
- **NEW BEST VERIFIED 0.136896: final_NH/sub141** (pruned + final search + pose + tuned #141 HPAC, 177,693 B).
- **FINAL VERIFIED 0.136393: final_PP/sub141** (177,106 B, seg 1.24317e-4, pose 3.64058e-6). Backups in BACKUP_FINAL/.
  Full write-up: FINAL_REPORT.md. Laptop rebooted at 18:25 mid-verification (inflate had finished; scoring rerun).
