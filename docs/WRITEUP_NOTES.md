# Write-up notes: joint optimisation of PR #135's stored state

Raw material for Ray's write-up. Deliberately over-detailed; pick what to keep. Every score is
`100*seg + sqrt(10*pose) + 25*bytes/37,545,489`. "Real" = official `inflate.sh` into a fresh directory, then the judge
networks on the decoded video (`score.py`, laptop T2000 or rented RTX 5090, TF32 off, DALI targets). "Trainer" = the
exact in-trainer gate (same seg/pose math, bytes estimated under #135's coder with our refit HPAC).

Machine offset check: #135's own archive scores 0.16227321 on our box vs 0.16226842 official (+0.000005); #141 scores
0.147899 on the laptop vs 0.147897 official. So local real numbers are trustworthy to ~1e-5.

## 0. Starting point and reference points

| entry | seg (raw) | pose (raw) | bytes | seg term | pose term | rate term | score |
|---|---|---|---|---|---|---|---|
| PR #135 (our base) | 2.9639e-4 | 6.884e-6 | 186,724 | 0.02964 | 0.00830 | 0.12433 | 0.16227 (official) |
| PR #141 (leader) | 2.014e-4 | ~6.4e-6 | 179,891 | 0.02014 | 0.00798 | 0.11978 | 0.147897 (official) |
| #141's map on #135's coder, carrier refit (our transplant) | 2.183e-4 | 4.65e-6 | 190,871 | 0.02183 | 0.00682 | 0.12709 | 0.15580 |
| #141 map + #140-style search + GN/polish (our run) | 1.451e-4 | 5.52e-6 | 192,523 | 0.01451 | 0.00743 | 0.12819 | 0.15013 |
| adpena private (move 55, unsubmitted, Modal T4) | 1.0288e-4 | 4.08e-6 | 179,255 | 0.01029 | 0.00639 | 0.11936 | 0.13603 (see research/adpena_0136_source.md) |

Architecture recap (#135 family): 600 class maps 384x512x5 classes (the "tokens"), coded by IntegerHPAC (a small
integer conv net predicting each pixel from neighbours + previous frame) + a 25x5 boundary correction table + RC64
range coder; a frozen int4 renderer (SemanticTokenRenderer, width 96, WANS1 int4 codes x fp16 row scales) draws frame 2
of every pair from the map; frame 0 is a pose carrier `gray + amp * sum_k c_k B_k / sqrt(12)` (12 patterns 3x24x32 as
5-bit codes, 600x12 int12 strengths) plus a 14-byte selector. SegNet reads frame 2 only; PoseNet reads both.

## 1. The method (Ray's idea)

Train every stored piece against the real score, with an exact differentiable rate term (HPAC's -log2 p of every pixel
of the current map, the forward pass on the hard map = the exact code length, gradients through the soft map). Pieces:
class maps (5-way logits on boundary-band pixels), HPAC, boundary table, carrier strengths, carrier patterns,
gray/amp, renderer. Exact gates on all 600 pairs accept/rollback.

What made it work in practice: joint in the LOSS, separate in the UPDATES. Each block moves in its own storage units
with its own step rule, and each stage has an exact gate. Organised as a cycle:
1. map training (seg + bytes), 2. byte-priced #140-style search, 3. HPAC + table refit, 4. pose stage (carrier),
5. renderer. Later replaced/extended by renderer "hops" and pose hops (sections 7, 8).

## 2. Map training on seg + bytes (cycle 1 step 1)

- Trainer `exp2_train.py` (fork of `joint_train135.py`). Map logits `z` on pixels within 2 px of a class boundary
  (7.49M pixels, 6.35%), soft map `softmax(z/T)` with T annealed 1.0 -> 0.1, per-frame "maxnorm" updates (largest
  |grad| entry moves lr, others proportionally; lr 2.0), cosine decay, flip cap 8 per frame per step, window of 8
  consecutive pairs per step, gate every 150 steps, 3000 steps (~27 min on a 5090).
- Per-frame acceptance at gates: a frame's map edits are kept only if that frame's exact score delta (seg + exact bits
  of the frame and of the next frame, whose context it is) is negative.
- **Result: seg 2.964e-4 -> 2.299e-4 (-22%), token stream 114,705 -> 114,079 B (-626 B), seg+rate 0.15397 -> 0.14690.**
  8,492 edits, all 20 gates accepted.
- Contrast (the failure that motivated it): the same training with the rate weight at 0 (run jA2) reached seg 2.009e-4
  but the stream grew to 137,336 B (+22.6 KB, ~17 bits per edit: isolated pixel edits are maximally surprising to
  HPAC); real score 0.16895, worse than #135. With rate priced, edits cost ~0.4 B each instead of ~2.1 B.
- First-order comparison: #140's edits (8,723 px) cost +4,147 B on #135's coder; ours SAVE bytes while fixing seg.

## 3. Byte-priced search (cycle 1 step 2)

- #140-style single-pixel search per pair: rank candidates by the gradient of (seg soft count + rate), test the top 32
  exactly per round (real render -> SegNet wrong-cell count, plus EXACT HPAC bits of frames t and t+1), accept the best
  candidate with net score change < 0, up to 60 edits per pair, non-adjacent candidates.
- Novelty vs #140: every candidate priced by its exact code length under the shipped coder, including the effect on
  the next frame's context (not a fixed per-pixel price list). Validation: on 3 pairs, predicted -17 bits / -31 cells
  matched the real stream and real seg exactly; the per-pair nets sum exactly to the total change.
- Candidate pool matters (diagnostic, pairs 200:240, from the step-1 map):
  | pool | net score | time |
  |---|---|---|
  | wide (near seg errors + changed pixels + all class edges) | -0.000135 | 157 s |
  | **near seg errors only (within 1 px)** | **-0.000385 (2.85x)** | 324 s |
  | wide, 4 batches of 32 per round | worse per minute | ~19 s/pair |
  Byte-only edge candidates looked good to the gradient but failed exactly, crowding the 32 test slots.
- **Result (near pool, all 600 pairs, ~83 min): seg 2.299e-4 -> 1.639e-4, stream 114,739 B, real seg+rate 0.14074**,
  7,604 edits, vs 0.14270 for #141's map + #140-style search on the same coder (beats the "superset" bar by 0.002).
- Earlier version with the frozen price list (experiment 1): claimed -2.9 KB, real stream fell only ~300 B.

## 4. HPAC + table refit (step 3)

- Shared HPAC weights trained on random frames of the current map (AdamW-like, lr 0.002, x8 for scaled weights, grad
  clip 10, per-row clamps), table on its int6 grid; exact gates, rollback on noise (HPAC gate-to-gate noise ~ +-1 KB).
- Cycle 1: 114,738 -> 114,239 B (-499 B, trainer). Seg/pose bit-identical as required. Cycle 2: 0/10 gates accepted.
- Bug found later by the HPAC writer: the per-row clamp was |w| <= max|w0| (symmetric), not the row's signed bit depth,
  so 16 rows grew one bit: the real IHS2 blob was 16,832 B vs 16,599 (+233 B pre-LZMA), eating part of the gain.
  Fix for future runs: clamp each row to [-2^(d0-1), 2^(d0-1)-1].

## 5. Pose stage (carrier) and the gray/amp discovery (step 4)

- After map edits the carrier is stale (pose 4.2e-3). A strengths-only damped Gauss-Newton refit (6x12 Jacobian per
  pair, exact int12 acceptance) + exact +-1/+-2 integer polish converged to 7.7e-5 (pose term 0.028): stuck.
- Adding gray and amp (the carrier's 127.5 / 64 constants, which live in decoder code, free in bytes) as trainable,
  with strengths refit at every gate: **pose 7.73e-5 -> 4.93e-6 in 100 steps** (then 4.82e-6). Seg bit-identical.
  Control on #141's map + search: 5.52e-6 -> 4.18e-6, confirming the carrier code. Gray 127.5 -> ~127.45, amp 64 -> ~63.97.
- Lesson: a strengths-only fit can sit in a bad basin; always refit gray/amp with it. Nobody in the lineage had moved
  these two constants.
- Patterns barely move by gradient (1 of 4,608 codes changed; a bigger move spiked pose to 1.7e-4 and was rolled back).

## 6. Cycles and their numbers

| state | seg | pose | bytes (#135 coder, real) | score (real, #135 format) |
|---|---|---|---|---|
| #135 | 2.964e-4 | 6.88e-6 | 186,724 | 0.16227 |
| end of cycle 1 | 1.6387e-4 | 4.817e-6 | 186,757 | 0.14768 |
| end of cycle 2 | 1.5948e-4 | 4.585e-6 | 186,866 | 0.14715 (scorer) / **0.146909** (full e2e, 186,512 B with writers) |

Cycle 2 breakdown: map training nearly nothing (seg+rate -0.00002, 146 flips, 11/20 rollbacks); search -0.00044 net
(1,044 edits); HPAC 0/10; pose stage 2.84e-4 -> 4.585e-6. Diminishing returns with the renderer fixed: a third search
on the old renderer found only 133 edits (-0.000004).

## 7. The coder: storing our map with #141's coder

- #141's coder = #135's HPAC + table + an adaptive "free corrector" (count-ratio correction over 51,200 contexts, a
  fixed-point logistic mixer of 23 count-table members with 4,000 online weight sets, adaptive re-weighting of the four
  non-top classes). It stores no parameters: it adapts while decoding.
- Built `encode141.py` by driving #141's own decoder objects in known-symbol mode. Gate: re-encoding #141's map
  reproduces its 113,411 B stream and the entire 179,891 B archive byte for byte (~1 s/frame on a laptop T2000).
- **Our cycle-2 map: 114,297 B on #135's coder -> 110,039 B on #141's coder (-4,258 B), and 3.4 KB less than #141 needs
  for its own map.** The coder is lossless, so seg/pose are bit-identical.
- Container: #141's BLK2/RX1; #135's renderer stored as a plain WANS F12 body in a one-segment LAY1 wrapper (36,048 B)
  because #141's renderer differs from #135's in 5 tensors (our map is tuned to #135's); our carrier re-stored through
  #141's packed CAP1 + dx2 CABAC + rr5 arithmetic recoders (gate: the recipe reproduces #141's own carrier section).
- **Real: 180,008 B, seg 1.59471e-4, pose 4.58643e-6 -> 0.142579** (`final_c2/sub141_c2/`), vs 0.146909 in #135's format.
  #135's renderer compresses ~3.5 KB worse than #141's pruned SM3R renderer, which is why it is not ~0.140.
- Decode time 1,190 s on a laptop T2000 (limit 1,800 s on a T4): to be checked on a T4 before submitting.

## 8. Renderer hops (step 5 made to work)

- Every small renderer move failed: 874 single int4 code flips (other session, 0 accepted), step 5 by gradient (12/12
  rollbacks), joint map + renderer training with three losses (sigmoid, sharper sigmoid tau 0.01, DAG-style active-set
  hinge kappa 0.1) and trainable per-row fp16 scales (all rollbacks).
- Diagnosis (`diag_render_dirs.py`): 12 of 12 random +-1-fp16-step perturbations of the row scales worsen exact seg in
  BOTH signs, each by ~+0.0006 of seg term (+3.8% seg). A sharp local minimum: the decoder rounds rendered pixels to
  integers and the pre-distorted map (wrong labels that THIS renderer draws right) is tuned to that exact integer output.
- The hop: (a) big renderer move with the map frozen (300 steps of scale + fp16 + code training on seg + rate; seg gets
  worse), (b) full near-pool map re-search to re-adapt the edits, (c) carrier refit. Accept if seg + rate beats the start.
  Renderer bytes measured exactly at gates (`wans_writer135.py`, byte-exact against #135; trainer's float->log->exp->fp16
  path reproduces #135's body exactly). The moved renderer's int4 codes did not change, only scales and fp16 tensors, so
  the body stays 36,040 B and fits #141's container unchanged.

| hop | after move (seg+rate) | after re-search seg | seg+rate | gain | pose after carrier refit |
|---|---|---|---|---|---|
| start (end cycle 2) | | 1.5949e-4 | 0.140019 | | 4.585e-6 |
| hop 1 | 0.141157 (+0.00114) | 1.5116e-4 | 0.139451 | -0.00057 | 4.195e-6 (full pose stage) |
| hop 2 | 0.140024 (+0.00057) | 1.4726e-4 | 0.139199 | -0.00025 | 4.046e-6 (full pose stage) |
| hop 3 | 0.139706 (+0.00051) | 1.4442e-4 | 0.139033 | -0.00017 | 4.065e-6 (quick alarm refit) |

- Control: the same search on the OLD renderer gains only -0.000004, so the gain comes from the renderer move.
- Pose came back BETTER after each hop although pose was never in the renderer's objective.
- **Hop 1 real (#141 format): 180,443 B, seg 1.51172e-4, pose 4.19541e-6 -> 0.141744** (`final_h1/sub141_h1/`), predicted
  -0.00085 from the trainer, measured -0.000835.
- Trainer -> real conversion is a stable ~0.0042 (#141 coder saves ~6.3-6.9 KB vs the trainer's #135-coder estimate):
  cycle 2 0.146790 -> 0.142579 (0.00421), hop 1 0.145943 -> 0.141744 (0.00420).

## 9. Pose hops (stage A, running)

- Same idea for the carrier: big move of the 12 patterns + gray/amp (strengths frozen), then a full strengths refit
  (GN 22 iterations + 4 polish passes), accept if pose term + rate improves; several jump sizes per round from the same
  start (pattern lr 0.1 / 0.3 / 1.0), keep the best, repeat. Seg cannot change (frame 0 only).
- Round 1: lr 0.1 accepted, trainer full 0.145403 -> 0.145341 (-0.00006); 0.3 and 1.0 rolled back.
- Round 2: lr 0.1 accepted, 0.145341 -> 0.145338 (-0.0000034); 0.3 rolled back. Stopped there (exhausted).
- **Stage A total: trainer full 0.145403 -> 0.145338 (-0.000065); pose 4.065e-6 -> 3.971e-6, gray 127.363.** Only the
  smallest jump size was ever accepted. Interpretation: the pose term (~0.0063) is near the carrier's floor; the
  remaining pose error is outside what 12 patterns can express. State: `rj_results/sA_r2_b0.1`.

## 9b. Stage B (renderer hops on the FULL score), started 19:15
- Same hop as section 8 but the renderer move's loss includes pose (carrier frozen during the move), then the map
  re-search (seg + rate), then a quick carrier refit; accepted on seg + pose + rate. Start: stage A's end (0.145338).
- **B hop 1:** move (pose in loss): seg 1.4442e-4 -> 1.4654e-4 (+0.0002 seg term, vs +0.0005 for seg-only moves: seeing
  pose made the move gentler), pose 3.97e-6 -> 1.40e-5 (carrier frozen). Re-search 675 edits, -0.000362 net -> seg
  1.4290e-4, seg+rate 0.13894. Quick refit (100 steps): pose 3.983e-6. **Full 0.145338 -> 0.145253 (-0.000085), accepted.**
  State `rj_results/B1_pose`.
- **B hops 2-4 (unattended chain, each full move -> re-search -> quick refit, trainer full score):**
  | hop | full | gain |
  |---|---|---|
  | B1 | 0.145253 | -0.000085 |
  | B2 | 0.145100 | -0.000153 |
  | B3 | 0.145027 | -0.000073 |
  | B4 | 0.144939 | -0.000088 |
  | B5 | 0.144957 | +0.000017, rejected: chain stopped |
  States pulled to `rj_results/B2_pose`..`B4_pose`. Gains did not shrink as fast as the seg-only hops did.
- **Final 600-step pose stage on B4: pose 3.896e-6 -> 3.804e-6, trainer full 0.144939 -> 0.144865.** Seg 1.38957e-4.
  State `rj_results/final_pose`. Since end of cycle 2: trainer 0.14679 -> 0.14487 (-0.0019).

## 10. Things that failed or were wrong (useful for the honest section)

- Rate term switched off in block-A map training (jA2): +22.6 KB.
- Frozen per-pixel price list for the search: underestimated real cost ~10x (neighbour / next-frame context effects).
- Pose in the map's gradient (soft maps): pose blew up to 7.0. Map + HPAC trained together: HPAC noise reverted good map
  edits on 550 frames.
- Per-frame map acceptance during renderer moves: reverted 520/600 frames (renderer touches every frame).
- My mid-session diagnosis that "pose-blind edits cost ~0.02 of pose the carrier cannot repair" was wrong: it was a
  stuck strengths-only fit; gray/amp fixed it.
- A host reclaimed box 53664321 mid-cycle; step 3's state was lost because it was not pulled immediately.
- Infra: pkill -f over ssh killed its own session (twice); CRLF in the Windows checkout broke inflate.sh; LZMA output of
  #135 not byte-reproducible (+11 B at best, harmless).

## 11. Tools written (all in work/)

`exp2_train.py` (trainer + search + renderer scales + wans bytes + kappa loss + save-final), `encode135.py`,
`encode141.py`, `build_archive135.py`, `build_archive141.py`, `hpac_writer135.py`, `carrier_writer135.py`,
`wans_writer135.py`, `score_map135.py` (with decoder-side carrier rebuild), `refit_gn135.py`, `diag_render_dirs.py`,
`make_hop_pieces.py`, `stageA.sh`, queue scripts `queue_*.sh`. Progress log: `MAPS_PROGRESS.md`.

## 12. Cycle after the hops (unattended, started 21:36)
- Step 1 map training (soft, lr-map 1.0, t0 0.5, 1500 steps): seg+rate ~0.138701 -> 0.138691 (-0.00001); ~250 frames
  reverted per gate. Step 2 narrow search, 600 pairs: 72 edits, -0.000032. The map is nearly exhausted on this renderer
  (the hops' re-searches already took it).
- Search variants, 40 pairs from final_pose: narrow 32/1 batch 0 edits; narrow 4 batches 5 edits (-0.000003); narrow
  64 per round same; **wide pool 52 edits, -0.000016, mostly byte-saving (-86 bits)**. The ranking flipped vs cycle 1's
  diagnostic: with seg fixes used up, byte-saving edge edits are what is left. Wide pass queued after the cycle (auto_D).
- **VERIFIED (22:11): hops 2-3 + stage A + stage B hops 1-4 + final pose stage, real #141 format:**
  `final_B/sub141/` archive 181,176 B (sha 478be533...), gray 127.33827209472656 amp 63.919464111328125, official
  inflate (16 min, laptop) decoded tokens = state map (sha a682e0bd). score.py: **seg 1.38974e-4, pose 3.80603e-6,
  181,176 B -> 0.140704** (seg 0.01390 pose 0.00617 rate 0.12064). Trainer 0.144865 -> real 0.140704 (conversion
  0.00416, same as before). vs hop 1 0.141744 (-0.00104), cycle 2 0.142579 (-0.00188), #141 0.147897 (-0.00719).
- Cycle (C_s1..C_s4) end: trainer full 0.144865 -> **0.144819 (-0.000046)**, pose 3.763e-6, seg 1.3891e-4. Wide-pool
  pass running: at 276/600 pairs 357 edits, -817 bits, net -0.000126 (projected ~-0.00027 over 600).

## 13. Is #135-style pricing a good proxy for #141's coder? (bits_cmp/)
Per-frame ideal bits of two maps (cycle-2 map vs current map, ~10k edits apart) under three pricings:
| pricing | cycle-2 map | current map | delta |
|---|---|---|---|
| trainer (our refit HPAC + #135 table) | 114,317 B | 115,363 B | +1,045 B |
| #141 HPAC + table, pre-corrector | 112,626 B | 113,761 B | +1,135 B |
| #141 real (post-corrector) | 110,039 B | 111,179 B | +1,141 B |
- Per-frame delta correlation trainer vs #141 real: **0.939**, fit real = 0.988 x trainer + 1.4 bits/frame; overall the
  trainer underprices edits by ~9%. **Conclusion: #141-aware pricing is not worth building.**
- The correctors give a near-constant ~2.6 KB discount and barely change edit costs.
- #141's HPAC beats our refit HPAC on OUR map even before correctors (112.6 vs 114.3 KB): it is a stronger model;
  a gain would need fine-tuning #141's HPAC on our map, not swapping in ours.
- **Wide-pool pass (600 pairs): 822 edits, wrong -194 (-0.000164), exact bits -1,838 (-0.000153); after quick pose refit
  trainer full 0.144819 -> 0.144484 (-0.000335).** Largest single step of the night. State `rj_results/D_pose`.
- Stage B hop E1 from there: move pose 3.74e-6 -> 3.69e-5 (carrier frozen), re-search -0.000517, quick refit ->
  **0.144392 (-0.000092)**, seg 1.35396e-4, pose 3.780e-6. State `rj_results/E1_pose`.

## 14. Renderer storage format (SM3R) and wider BLK2 packing (research/sm3r.md)
- #141's ~3.5 KB renderer advantage is CONTENT, not format: SM3R row-prunes blocks.1-3.film.weight (#141 keeps 1%, i.e.
  2 of 192 rows) and stores frame_embed / blocks.0.film at 3 bits. SM3R always zeroes >= 2 rows per film tensor, so it
  cannot store our renderer exactly (keep 99% zeroes 48 weights, max |change| 0.021: a renderer move). SD1M (same layout,
  no pruning) is exact but +152..+256 B. Writer gate: re-encoding #141's SM3R reproduces its 36,130 B byte for byte.
- **Side win: widening the BLK2 block search (brotli q10/mode 1, more windows and cut points) with the WANS renderer:
  181,176 -> 181,031 B (-145 B), bit-identical content. E2E PASS: seg/pose identical, score 0.140704 -> 0.140608.**
  `work/final_sm3r/sub141/`, builder `build_archive141_sm3r.py --renderer wans --wide`.
- Open idea: pruning the 3 film tensors IS a renderer move: could be treated as a hop (prune -> re-search -> refit) to
  try to collect part of #141's 3.5 KB.

## 15. HPAC / table under #141's coder (no win)
Full 600-frame #141 encodes of the final_pose map (correctors unchanged), stream / HPAC section:
| HPAC | table | stream | HPAC section |
|---|---|---|---|
| #141 | #141 | 111,180 B | 13,515 B |
| ours (c1_s3, refit from #135's under the corrector-free coder) | #141 | 112,670 (+1,490) | 14,862 (+1,347) |
| ours | ours | 112,460 (+1,280) | 14,862 |
| #141 | proxy-refit table | 111,233 (+53) | 13,515 |
- #141's HPAC is a stronger model even on our map; a refit started from #135's HPAC cannot catch up. A pre-corrector
  proxy does not predict post-corrector bytes (correctors recalibrate per class/boundary).
- Only realistic HPAC gain: fine-tune #141's own HPAC on our map from its weights and stored depths, checked with
  `encode141x.py`. Tools: `encode141x.py`, `hpac_writer141.py` (gate: #141 section byte-identical, brotli q10 lgwin 24),
  `build_archive141h.py`, `table141_proxy.py`.
- Hop E2: 0.144428 (+0.000035) rejected, chain stopped. **Final 600-step pose stage on E1: 0.144322** (trainer).
  State `rj_results/E_pose`. Verifying as a real #141 archive with the wide BLK2 packing (`final_E/`).
- Pruning hop queued (`auto_F.sh`): `--prune-film-keep 1` keeps the top 2 of 192 rows (by sum |code| x scale) of
  blocks.1-3.film.weight and pins the rest at 0 (mirrors #141's SM3R pruning), then renderer move (full score), re-search,
  carrier refit. Real bytes must be judged in SM3R format (the trainer's WANS byte term understates the saving).
- Pruning is part of the joint method (Ray): keep level and row choice should be picked on the real score (exact byte
  saving per level + seg/pose after re-adapting), not by weight magnitude. Exact byte saving of keep 1% on the current
  pieces: archive 181,031 -> 178,372 B (-2,659 B, -0.00177). Lesson: structural renderer changes belong at the START
  of a cycle; we only learned tonight that #141's 3.5 KB advantage is this pruning.
- **VERIFIED (01:17): wide pass + hop E1 + final pose stage, #141 format, wide BLK2 packing: `final_E/sub141/` 181,015 B
  (sha 271d61b1...), decoded tokens = state map (sha 8bdd5d17). seg 1.35396e-4, pose 3.68790e-6 -> 0.140143**
  (seg 0.01354 pose 0.00607 rate 0.12053). Trainer 0.144322 -> real 0.140143 (conversion 0.00418 incl. -145 B packing).
- **Pruning hop search result:** seg 1.7808e-4 -> 1.4278e-4 (3,880 edits, +636 B map); vs pre-prune seg term +0.00074,
  map bytes +0.00042, renderer -0.00177 => ~-0.0006 net before pose. Pose refit paused by Ray (seg first, pose last).
- Map training on the pruned state: seg+rate 0.138165 -> 0.138140 (-0.000026). Search styles (40 pairs, pruned + trained):
  narrow -0.000005, narrow x4 batches -0.000008, wide -0.000014, **wide x4 batches -0.000087 (267 edits, 408 s)**.
  Full wide x4 search launched (`auto_H.sh`, ~1 h 45 m).
- **Pruned line, cycle 1 complete:** pruning hop (keep 1%) -> re-search (seg 1.7808e-4 -> 1.4278e-4) -> map training
  (seg+rate -0.000026) -> **wide x4 search: 4,632 edits, wrong -1,525 (-0.001293), bits -4,304 (-0.000358)** ->
  pose stage: pose 5.6e-3 -> **3.853e-6** (vs 3.688e-6 unpruned: pruning costs ~+0.0002 of pose term). Seg 1.3170e-4
  (better than unpruned 1.3540e-4). Trainer full 0.142889. Real estimate ~0.1382. State `rj_results/I_pose`.
  Verifying as a real SM3R (keep 1%) #141 archive in `final_P/` (builder drops rows by zero frame-variance = exactly
  our pinned rows).
- **VERIFIED (04:56): pruned line, real #141 archive with SM3R keep-1% renderer: `final_P/sub141/` 178,610 B (sha
  ce62cf5a...), decoded tokens = state map (sha 3d45e51b). seg 1.31726e-4, pose 3.85414e-6 -> 0.138310** (seg 0.01317
  pose 0.00621 rate 0.11893). vs unpruned best 0.140143 (-0.00183), #141 0.147897 (-0.00959). Pruned WANS body
  34,171 B (zeroed rows), SM3R drops exactly the 190 zero rows per film tensor (zero frame-variance).
- Pruned cycle 2: map training 0 (all rollbacks); wide x4 search 509 edits, wrong -222 (-0.000188), bits -99, net
  -0.000196 (30 min: most pairs exhausted quickly). Pose stage + renderer hops K1-K4 + final pose running unattended.
- Pruned cycle 2 end (J_pose): trainer 0.142848, pose 3.815e-6, seg 1.3160e-4. **Hop K1: 0.142789 (-0.000059)**, pulled.
- **Hop K2: 0.142742 (-0.000047)**, pulled.
- Hop K3: 0.142737 (-0.0000058, below the 1e-5 bar): chain stops at K2; final pose stage from K2 running.
- **K_final (final pose stage from K2): trainer 0.142626** (cycle-2 end 0.142848, -0.00022). Real archive final_K/sub141 178,966 B built; e2e rerun (first attempt killed by watcher time limit).
- **VERIFIED: K_final (pruned cycle 2 + hops K1-K2 + final pose stage) `final_K/sub141/` 178,966 B, seg 1.28089e-4,
  pose 3.69033e-6 -> 0.138050** (seg 0.01281 pose 0.00607 rate 0.11917). vs 0.138310 (-0.00026), #141 -0.00985.
- Lost: the first #141-HPAC fine-tune (box 53752288 hit its scheduled destroy at 11:38 before the result was pulled).
  Rerun on box 53908794 (auto_N.sh) after a final wide x4 search + pose refit, with a laptop puller (pull_N.sh).
- **Bug fixed (14:15):** the trainer's rate-only gate path (used when only hpac/table train) always added the pose term,
  ignoring --w-pose/--w-rate, while gate 0 used the weighted full path. With --w-pose 0 every later gate looked ~0.006
  worse and rolled back. This invalidated the unpruned-cycle HPAC refit C_s3 (--w-pose 0, "all rollbacks") and the
  first #141-HPAC fine-tune attempt; c1_s3/c2_s3 used --w-pose 1 and were unaffected. After the fix, the #141-HPAC
  fine-tune's first gate ACCEPTs: pre-corrector stream 114,045 -> 113,549 B (-495 B). Real judge = #141 encodes after.

## 16. Final squeeze on the pruned line (box 53908794)
- From K_final: **4-batch wide search 1,432 edits, wrong -374 (-0.000317), bits -1,657 (-0.000138), net -0.000455**;
  pose refit 300 steps -> N_pose trainer **0.142191** (K_final 0.142626), seg 1.2520e-4, pose 3.675e-6.
- **#141 HPAC fine-tuned on the final map** (from #141's weights, rows clamped to their stored bit depths, table kept,
  fixed gates): trainer pre-corrector 114,045 -> 112,826 B. **Real #141 encodes with correctors: stream 111,595 ->
  110,502 B, HPAC section 13,515 -> 13,486 B: -1,122 B total (~-0.00075), seg/pose unchanged.** Archive with it:
  `final_NH/archive141.zip` 177,693 B (N_pose without it: 178,838 B). Gates all pass.
- Hop O1 from N_pose: 0.142375 (+0.00018), rejected: the hop chain is exhausted on this state.
- **VERIFIED (16:20): N_pose + fine-tuned #141 HPAC, `final_NH/sub141/` 177,693 B (sha ce0b5ae6...), decoded tokens =
  N_pose map (sha fb4bba09), seg 1.25198e-4, pose 3.66921e-6 -> 0.136896** (seg 0.01252 pose 0.00606 rate 0.11832).
  vs K_final 0.138050 (-0.00115), #141 0.147897 (-0.01100). The decoder accepted our own HPAC section.
- Step 1 (longer #141-HPAC fine-tune, lr 0.0005, table trained): real stream 110,278 + section 13,517 vs tuned1 110,502 + 13,486 => -193 B. (lr 0.002 from the tuned HPAC rolled back every gate.)
- Step 2 (box 53931293): 4-batch wide search priced with the tuned #141 HPAC + table: 1,265 edits, wrong -136
  (-0.000115), bits -1,597 (-0.000133), net -0.000248; pose refit -> pose 3.639e-6; short HPAC refit (lr 0.001) on the
  new map. Real #141 encodes on that map: step-1 HPAC 110,085 + 13,517 B; **re-fitted 109,913 + 13,507 B = 123,420 B**
  (vs 123,988 B in the 0.136896 archive). Box destroyed after pulling (`rj_results/P_pose`, `P_hpac2`). Verifying
  `final_PP/`.
