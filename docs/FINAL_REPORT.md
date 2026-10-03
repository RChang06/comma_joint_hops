# Final report: joint optimisation of PR #135's stored state (comma video compression challenge)

Status: 2026-10-02. Author of the method: Ray. Built and run with Claude. Nothing has been submitted or pushed.
Companion files: `WRITEUP_NOTES.md` (chronological raw notes with every number), `MAPS_PROGRESS.md` (operational log).

Score = `100 * seg + sqrt(10 * pose) + 25 * archive_bytes / 37,545,489`. "Verified" = official `inflate.sh` of the
submission folder into a fresh directory, then the judge networks on the decoded video (`work/score.py`, TF32 off,
DALI targets). Machine check: #135's archive scores 0.16227321 locally vs 0.16226842 official (+0.000005); #141 scores
0.147899 vs 0.147897. Local verified numbers are good to ~1e-5.

## 1. Result

| | seg (raw) | pose (raw) | archive | score |
|---|---|---|---|---|
| PR #135 (our base) | 2.9639e-4 | 6.884e-6 | 186,724 B | 0.16227 (official) |
| PR #141 (leaderboard #1) | 2.014e-4 | ~6.4e-6 | 179,891 B | 0.147897 (official) |
| **Ours, final** (`final_PP/sub141/`) | **1.24317e-4** | **3.64058e-6** | **177,106 B** | **0.136393** |
| Ours, best fully verified before the final step (`final_NH/sub141/`) | 1.25198e-4 | 3.66921e-6 | 177,693 B | **0.136896** |

From the base: seg -58%, pose -47%, archive -9.6 KB; score -0.0259. Ahead of #141 by 0.0115.

## 2. What it is built on

- **PR #135** ("semantic-pose-HPAC_CPR1_polished"), the starting point. Its stored state:
  - 600 class maps, 384x512, 5 classes ("tokens"). Coded by IntegerHPAC (a small integer conv net predicting each
    pixel from its neighbours and the previous frame), plus a 25x5 boundary correction table and the RC64 range coder.
  - A frozen int4 renderer (SemanticTokenRenderer, width 96, 4 FiLM blocks; WANS1 int4 codes x fp16 row scales) that
    draws frame 2 of each pair from the map. SegNet reads only frame 2.
  - A pose carrier for frame 0: `gray + amp * sum_k c_k B_k / sqrt(12)`, with 12 patterns (3x24x32, 5-bit codes),
    600x12 int12 strengths, plus a 14-byte frame-0 selector. PoseNet reads both frames.
- **PR #141's container and coder** (#140's content with a repack). We store our state in #141's format:
  - **Coder:** #141's HPAC and table, plus adaptive "free correctors" that learn while decoding and store nothing.
  - **Carrier recoders:** rr5 and dx2 CABAC.
  - **Compact renderer format:** SM3R, which allows FiLM row pruning.
  - **Decoder:** #141's, with our gray/amp constants written in.
- **Ideas borrowed and credited:**
  - #140's single-pixel pre-distortion search (we added exact byte pricing and different candidate pools).
  - #140's damped Gauss-Newton + integer polish for the carrier.
  - #141's FiLM row pruning (blocks 1-3 keep 2 of 192 rows).
  - #141's correctors, used as-is.

## 3. The method (Ray's idea)

Train every stored piece against the real score, with an exact differentiable rate term. The rate term is HPAC's
-log2 p of every pixel of the current map: the forward pass on the hard map gives the exact code length, and gradients
flow through the soft map. Every change is accepted only if an exact check on all 600 pairs improves the score.

In practice the key was **joint in the loss, separate in the updates**: each block moves in its own storage units with
its own step rule and its own exact check. The final playbook:

0. **Structural change, once at the start:** prune the renderer's FiLM rows (blocks 1-3, keep 1%). The pruned rows are
   pinned at zero in every later step.
1. **Cycle:** map training (seg + bytes, early stop) -> 4-batch wide-pool search (exact seg + exact bits) -> pose stage
   (strengths by GN + polish, gray/amp, patterns). Repeat while a cycle gains more than ~1e-4.
2. **Renderer hops** until one fails: a big renderer move on the full score, then a map re-search, then a quick carrier
   refit.
3. **Final squeeze:** a 4-batch wide search plus a pose stage.
4. **Fine-tune #141's HPAC + table on the final map,** judged by real #141 encodes. Then an HPAC-priced search and a
   short re-fit.
5. **Build the archive and verify.**

## 4. Score history (verified milestones)

| # | state | real score | change | folder |
|---|---|---|---|---|
| 0 | PR #135 | 0.162268 | | |
| 1 | end of cycle 2, #135's coder | 0.146909 | -0.01536 | final_c2/sub_c2 |
| 2 | same state, #141's coder | 0.142579 | -0.00433 | final_c2/sub141_c2 |
| 3 | renderer hop 1 | 0.141744 | -0.00084 | final_h1/sub141_h1 |
| 4 | hops 2-3, pose hops, stage-B hops 1-4, pose stage | 0.140704 | -0.00104 | final_B/sub141 |
| 5 | wider BLK2 packing | 0.140608 | -0.00010 | final_sm3r/sub141 |
| 6 | cycle, wide search, hop E1, pose stage | 0.140143 | -0.00047 | final_E/sub141 |
| 7 | FiLM pruning + re-search, map training, 4-batch wide search, pose stage | 0.138310 | -0.00183 | final_P/sub141 |
| 8 | pruned cycle 2, hops K1-K2, pose stage | 0.138050 | -0.00026 | final_K/sub141 |
| 9 | final 4-batch wide search, pose stage, #141 HPAC fine-tune | 0.136896 | -0.00115 | final_NH/sub141 |
| 10 | longer HPAC tune, HPAC-priced search, pose stage, HPAC re-fit | **0.136393** | -0.00050 | final_PP/sub141 |

## 5. Stage-by-stage numbers (trainer = in-trainer exact gate, #135-coder byte accounting)

### Map and search
- **Map training on seg + bytes (cycle 1):** seg 2.964e-4 -> 2.299e-4, stream -626 B, seg+rate 0.15397 -> 0.14690.
  - Settings: soft maps, T 1.0 -> 0.1, maxnorm lr 2.0, 3000 steps, per-frame score acceptance.
  - Control: the same run with the rate term off (jA2) reached seg 2.009e-4 but cost +22.6 KB.
- **Byte-priced search (cycle 1):** seg 1.639e-4, seg+rate 0.14074.
  - Every candidate is priced by exact HPAC bits of frames t and t+1; prediction matched the real stream exactly.
  - It beat #140's search on #141's map on the same coder (0.14270).
- **Candidate pools:**
  - Narrow (near seg errors) beat wide 2.85x while seg fixes were plentiful.
  - Wide won once those ran out.
  - Wide with 4 candidate batches per round found 6x more again (-0.00165 on the pruned line, one pass).
- **Map training after cycle 1:** ~0 every time (-0.00002, -0.00001, -0.000026, 0). Kept in the cycle with early stop.

### Pose
- **gray/amp:** a strengths-only GN refit stuck at pose 7.7e-5. Adding gray/amp (decoder constants, free in bytes)
  brought it to 4.8e-6. Nobody in the lineage had moved these.
- **Pose hops** (big pattern moves + full strengths refit): -0.00006, then ~0. The carrier is near its floor.

### Coder
- **#141's coder on our map:** 114,297 -> 110,039 B (-4,258 B), seg/pose bit-identical (the coder is lossless).
  - The encoder was built by driving #141's decoder objects in known-symbol mode; byte-exact gate on #141's own archive.
- **Pricing proxy:** the trainer's per-frame edit costs vs #141's real costs: correlation 0.94, slope 0.99, ~9% low.
  - #141-aware pricing was judged not worth building.
  - The correctors give a near-constant ~2.6 KB discount.

### Renderer
- **Sharp minimum:**
  - 874 single code flips: 0 accepted.
  - Step 5 by gradient: 12/12 rolled back.
  - Three losses with trainable scales: all rolled back.
  - 12/12 random +-1-fp16-step scale perturbations worsen seg in BOTH signs, by ~+0.0006 seg term each.
  - Cause: integer-rounded rendering plus a map pre-distorted to this exact renderer.
- **Renderer hops** (big move, then map re-search, then carrier refit):
  - hop 1 -0.00085, hop 2 -0.00037, hop 3 -0.00017;
  - B1-B4 (move sees pose) about -0.0001 each; K1/K2 -0.00006/-0.00005;
  - control: the same search on the old renderer gained -0.000004;
  - pose came back better after every hop.
- **Pruning** (keep 2 of 192 rows in blocks 1-3 FiLM):
  - renderer -2,659 B exactly;
  - seg damage +0.0043 recovered to +0.00074 by one re-search;
  - pose +5%;
  - net, after re-adapting: -0.00183 (milestone 7).
  - #141's 3.5 KB renderer advantage turned out to be this pruning, not its storage format.
- **Wider BLK2 packing** (brotli q10 / mode 1, more cut points): -145 B, content identical.

### HPAC under #141's coder
- **Swapping in our HPAC** (descended from #135's): +2.8 KB worse. #141's HPAC is a stronger model even on our map.
- **Fine-tuning #141's own HPAC on the final map** (from its weights, rows clamped to their stored bit depths):
  -1,122 B real.
  - A longer run at a lower lr, with the table trained too: -193 B more.
  - After the HPAC-priced search, a short re-fit: map + coder 123,420 B vs 123,988 B in milestone 9.

## 6. Final verification
- Submission folder: `work/final_PP/sub141/` (#141 decoder copy with ARCHIVE_SHA256/ARCHIVE_BYTES and gray/amp set).
- archive.zip: **177,106 B**, sha256 `95a9fca6c79b397a36c7f8ea4f5d922ed604d0a9749dd7eb626c2ee4ce6348ae`.
- Official inflate.sh into a fresh dir (`final_PP/e2e/`) on a laptop T2000: decoded tokens sha `0af89b23...` = the
  trained map. Builder gates: stream, table, hpac, carrier, renderer all equal what went in.
- Judge networks (`score.py`): **seg 1.24316749e-4, pose 3.64057650e-06, 177,106 B -> 0.136393**
  (seg 0.01243, pose 0.00603, rate 0.11793).
- Backups: `work/BACKUP_FINAL/` holds zips of `final_PP/sub141` (final) and `final_NH/sub141` (previous verified best,
  0.136896) with their sha256 sums.

## 7. Failures, bugs and wrong turns (numbered)

1. Rate term switched off for map training (jA2): seg matched #141 but the stream grew +22.6 KB (score 0.16895, worse
   than #135).
2. #140-style search with a frozen price list: claimed -2.9 KB, the real stream fell only ~300 B (ignored neighbour
   and next-frame context).
3. The first search pool (wide, 1 batch) found 2.85x less than narrow: byte-only candidates crowded the 32 test slots.
4. Pose in the map's gradient (soft maps): pose blew up to 7.0.
5. Map + HPAC trained together: HPAC noise (+-1 KB per gate) reverted good map edits on 550 frames.
6. Per-frame map acceptance during renderer moves reverted 520/600 frames (the renderer touches every frame).
7. Every small or gradient renderer move failed (see section 5). 874 flips: 0 accepted.
8. A wrong mid-session diagnosis ("pose-blind edits cost ~0.02 of pose the carrier cannot repair"): it was a stuck
   strengths-only fit, which gray/amp fixed.
9. HPAC row clamp was symmetric |w| <= max|w0| instead of the row's signed bit depth: 16 rows grew a bit (+233 B).
   Fixed with a per-row signed-depth clamp.
10. **Gate bug:** the rate-only gate path always added the pose term, ignoring --w-pose/--w-rate. With --w-pose 0
    every later gate looked ~0.006 worse and rolled back. This silently hid HPAC gains (a cycle's HPAC step and the
    first #141-HPAC fine-tune). Fixed. The fine-tune then gave -1,122 B.
11. A pre-corrector table proxy did not predict post-corrector bytes (+53 B real).
12. SM3R cannot store an unpruned renderer exactly; SD1M (exact) is +152..+256 B.
13. Search variants with more batches / 64 candidates on an exhausted map: ~0. Renderer hops with 2x and 0.5x move
    size after the chain stalled: both worse.
14. **Infra and operations:**
    - Hosts reclaimed 2 boxes; one lost a step-3 state that had not been pulled.
    - A box hit its scheduled destroy before the first HPAC fine-tune result was pulled (lost, rerun).
    - `pkill -f` over ssh killed its own session (several times).
    - CRLF in the Windows checkout broke inflate.sh.
    - #135's LZMA output is not byte-reproducible (+11 B).
    - An OOM from launching onto a busy GPU.
    - Background watchers hitting their time limit; laptop memory pressure killing verification runs.
15. A save-final bug (saved the post-rollback state) and a helper that insisted on a fixed 36,040 B renderer body
    (pruned bodies are smaller). Both fixed.

## 8. Tools written (all in `work/`)

- **Trainer:** `exp2_train.py`. It covers map/HPAC/table/carrier/renderer training, the search modes, renderer scales,
  exact WANS bytes, the kappa loss, save-final, early stop, FiLM pruning, the HPAC depth clamp, #141 HPAC/table loading,
  and the gate fix.
- **Encoders:**
  - `encode135.py`, `encode141.py` (byte-exact on #141's own archive), `encode141x.py` (custom HPAC/table),
    `encode141_bits.py` (per-frame bits).
- **Writers:**
  - `hpac_writer135.py`, `hpac_writer141.py`, `carrier_writer135.py`, `wans_writer135.py`, `sm3r_writer.py`
    (each gated byte-exact on the original); `make_table_body.py`.
- **Builders:** `build_archive135.py`, `build_archive141.py`, `build_archive141_sm3r.py` (--renderer, --wide,
  --wans-body, --hpac-section, --table-body), `build_archive141h.py`.
- **Verification and diagnostics:**
  - `score_map135.py`, `score.py`, `verify_state*.sh`, `e2e_only.sh`, `final_PP.sh`.
  - `diag_render_dirs.py`, `refit_gn135.py`, `make_hop_pieces.py`.
- **Queues and sync:** `auto_*.sh` (unattended box queues), `pull_*.sh` (stage pullers).

## 9. Before submitting (not done)

1. **T4 timing test:** the decode takes ~16 min on a laptop T2000 against a 30 min limit on the grader's T4.
2. Build the PR from `final_PP/sub141/` (or `final_NH/sub141/`). Make sure the *.sh files keep LF line endings.
3. Nothing is pushed. Per standing rules, Ray pushes.
