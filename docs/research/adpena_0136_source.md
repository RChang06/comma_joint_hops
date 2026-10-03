# Source of the "S 0.13603403" claim (adpena, move 55)

Checked 2026-10-01. Read-only. Local clone `D:\projects\comma_b\work\research\repos\tokhist\comma-lab` is at
cb94a9ff, which is also the remote HEAD of github.com/adpena/comma-lab (`gh api` confirms: pushed_at
2026-09-18T06:56:32Z, public, not archived). So the local grep below matches what is on GitHub.

## Where it comes from

Primary record (the pointer memo), added in commit
**e4de53effaae80c891feb6b3ce7cac0044d75801**, 2026-09-16 22:51:19 -0500 (2026-09-17T03:51Z), message
"pointer move 55: S 0.13603403441098336 @ 179,255 B [contest-CUDA T4 n600] -- packet-written":

https://github.com/adpena/comma-lab/blob/e4de53effaae80c891feb6b3ce7cac0044d75801/.omx/research/ddm_pd8_price_first_pass3_on_move54_20260916_pointer_move_55_20260917.md

Lines 7-16 quoted:

> `upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M2PP3T886HAK0SCG63B8JMA9`.
> Lane `ddm_pd8_price_first_pass3_on_move54_20260916`. Modal wall 1659.9 s. Archive sha
> `ddadf998ddacab9b356b9b6a01a78c845ab3d6f643dd1e3cf8f37840ae550b8a`, 179,255 B. ... `passed: true`, `validation_errors: []`.
>
> | rate 25·179,255/37,545,489 | 0.11935854664191482 |
> | seg 100·0.00010288 | 0.010288 |
> | pose √(10·4.08e-06) | 0.006387487769068525 |
> | **S** | **0.13603403441098336** |

Prior pointer (move 54) was 0.13605599532783202 @ 179,266 B; move 55 is -2.196e-5 from that.

Raw receipt, same commit:
`.omx/research/ddm_pd8_packet_inputs_20260917/T4_TIMING_REMEASURE_MODAL_REMOTE_RESULT.json`. The embedded
report.txt is the stock evaluate.py output:

    Average PoseNet Distortion: 0.00000408
    Average SegNet Distortion: 0.00010288
    Submission file size: 179,255 bytes
    Final score: ... = 0.14

Other places that repeat it: `reports/latest.md` lines 12 and 50 ("CURRENT effective_frontier [contest-CUDA T4, n600]");
`ddm_gs3_gestalt_after_submission_20260903.md` addenda 71 (line 2109) and 74 (line 2210); `ddm_pc1_*`, `ddm_pd9_*` memos.

## What the number actually is

- It is the official `evaluate.py`. The recorded sha256 of the evaluate.py used, 7da71a84...0baa4b (6,005 B), is
  byte-identical to the current commaai/comma_video_compression_challenge evaluate.py (last changed 2026-03-25).
  It runs through adpena's harness (`experiments/contest_auth_eval.py`: archive.zip -> inflate.sh -> evaluate.py
  --device cuda), not `evaluate.sh`. It used n=600 and the public video list.
- Hardware: a **Modal** Tesla T4, not GitHub's `linux-nvidia-t4` runner that the official eval uses.
- The 9-decimal figure is rebuilt from evaluate.py's 8-dp components plus exact bytes. evaluate.py itself prints
  0.14. adpena's own JSON bounds the rounding error at <= 4.4e-6, which comes almost all from pose.
- Their harness flags it `score_claim: true`, `promotion_eligible: false`, `adjudication_required: true`. CPU axis:
  "REFUSED_BY_DESIGN" (the receiver is CUDA-only).
- Timing: the first T4 run had inflate 1,602.7 s (Modal wall 1,659.9 s). That is under the official 30-min limit but
  above adpena's own 1,260 s risk ceiling, so they called it a "host outlier". A re-run on the same bytes gave inflate
  1,072.3 s + evaluate 40.5 s with the same components (`MOVE55_T4_DIRECT_LEG.decode_wall_clock.json`,
  `T4_TIMING_NOTE_move55.json`).
- The archive is NOT public. Custody paths are on adpena's local volumes (`/Volumes/APDataStore/...`). There is no
  GitHub release for it: the only comma-lab release is the #140 afr1 archive. So nobody else can reproduce it today.

## Was it packaged / why not submitted

- The packaged, reviewable submission is `submissions/mrs7` (commits 7cfff2eb and ff9fe0f2, 2026-09-16). It holds
  source only (inflate.py, 3 C files, README, FORMAT.md), no archive. It is pinned to **move 53's bytes**, not move 55.
  The README states "0.1361014714463198" on T4. Its FORMAT.md/README say 179,186 B in one place and give a sha
  aab908... "179,286 bytes" in another. This is a minor internal inconsistency.
- The stated status, verbatim from gs3 addendum 74 (2026-09-17): "(3) a submission of the reviewable packet
  (submissions/mrs7) -- measured, awaiting the operator." Addendum 70: "It is measured on move 53's bytes; move 54 is
  a new archive -- restaging = archive pin + identity + one T4 run. The operator decides." The pointer memo's "Next"
  says the same.
- So no rule problem or runtime failure is given as the reason. The human operator (adpena) simply had not pushed
  the button. The campaign's stated goal was sub-0.12. Addendum 74 declares "every measured object and formulation is
  closed for sub-0.12", and activity stops on 2026-09-18 (last commit cb94a9ff, psa2 closed). That looks like the
  project winding down, not a blocked submission. This is my inference.
- Challenge side: adpena's PRs are #107, #110 (merged) and #140 only. #140 was closed 2026-09-15 and marked "added to
  leaderboard" on 2026-09-24 at 0.148. There is no adpena PR after #140, and no comment on #140 mentioning a newer
  score. Later PRs (#141-#145) are by others. The leaderboard top is still 0.148 (#141, #140).

## Verdict on the earlier summary

Accurate in substance. The number, bytes, components, date (2026-09-17 UTC / 09-16 local), move 55, T4 and n600 are
all exactly as recorded, and it is an official-evaluate.py score from a real T4 run. Caveats the summary left out:

1. It is self-reported. The run was on Modal, not the official GitHub runner, and the archive was never published.
2. 0.13603403 is a recomputation from 8-dp components (+-4.4e-6). The evaluator printed 0.14.
3. "Privately" means unsubmitted, but the record itself is public on GitHub.
4. The packaged submission (mrs7) is for move 53 (0.13610147), not move 55.
5. The first T4 decode took 1,603 s, close to the 1,800 s limit. The re-measure was 1,072 s, so runtime is a mild
   risk, not a blocker.
