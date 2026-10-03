# work/ - experiments on top of PR #141

Base: PR #141 (`semantic_blocks`, 0.147897), which is PR #140's content with a 111-byte repack.
Repo branch: `ray/141-base` (local only, branched from the `pr141` fetch of
`commaai/comma_video_compression_challenge`).

## What's here

| file | what it does |
|---|---|
| `src_archives/pr140_archive.zip` | #140's pinned release (sha `cbb8d928...`), the input #141's `compress.py` needs |
| `local_decode141.sh` | rebuild #141's archive and run the official decoder once on the laptop (WSL, T2000) |
| `extract141.py` | pull the decoded class maps + renderer weights + carrier into `extracted/pr141_components.pt` |
| `make_targets.py` | decode the ORIGINAL video with DALI and save the judges' answers (run where NVDEC works) |
| `targets/gt_dali_t2000.pt` | those answers: pose [600,12] + SegNet argmax [600,384,512] |
| `score.py` | score an inflated submission against the targets (TF32 off), ~16 s on a 5090 |
| `fit_carrier.py`, `carrier_sweep.py` | the carrier experiments from 2026-09-26/27 (negative results, kept for reference) |
| `setup_box.sh`, `repro141.sh`, `guard_b.sh` | rented-GPU setup, full reproduction, spend guard |

## Archive layout (179,891 bytes)

| part | bytes | score |
|---|---|---|
| class maps (RC64 token stream) | 113,411 | 0.0755 |
| renderer (width 96, 4 dilated FiLM blocks, int4) | 31,153 | 0.0207 |
| pose carrier (12 patterns + 7,200 strengths + 14 B selector) | 21,816 | 0.0145 |
| guesser (integer HPAC) + header | 13,288 | 0.0088 |
| boundary table + container | ~223 | 0.0002 |

Plus SegNet error 0.0201 and PoseNet error 0.0080 -> 0.1479.

## Commands

Local, WSL venv `~/venvs/coolchic` (torch 2.11, brotli 1.2.0, nvidia-dali-cuda120 1.52.0):

```bash
bash /mnt/d/projects/comma_b/work/local_decode141.sh        # decode once (slow on the T2000)
cd /mnt/d/projects/comma_b && python work/extract141.py     # -> work/extracted/pr141_components.pt
python work/score.py . work/targets/gt_dali_t2000.pt submissions/semantic_blocks
```

Rented box: see `setup_box.sh` (skips git-lfs, upload the judges + video + #140 archive), then
`repro141.sh` for the decode and `score.py` for scoring.

## Gotchas (all hit for real)

- **DALI/NVDEC does not work in Vast containers.** Make targets on the laptop with
  `DALI_DISABLE_NVML=1`, never with the PyAV reader (it made pose 25x worse).
- **TF32 off** for anything touching PoseNet (`torch.backends.cudnn.allow_tf32 = False`); with it on, pose
  read 1.6x worse.
- `frame_utils.rgb_to_yuv6` is decorated `@torch.no_grad()`; to backprop through PoseNet use
  `frame_utils.rgb_to_yuv6.__wrapped__`.
- Tensors from `read_residual_archive` can come out without grad; `.clone()` them before fitting.
- Never `pkill -f`/`grep`-kill by a pattern that also appears in your own ssh command.
- The decoded raw video is 3.66 GB; `inflated/` and `extracted/` are git-ignored.

## Results so far

- #141 reproduced exactly: 0.147899 vs official 0.147897.
- Zero-byte carrier basis (DCT or random) is dead: pose ~5 million times worse.
- Carrier precision is tight: 3-bit basis 92x worse pose, 8-bit coefficients 10x worse.

## Candidate next experiments

1. Fine-tune the renderer on #141's EDITED maps with the TRUE SegNet answers as the target, then re-fit the
   carrier and re-score (one change; tests whether the renderer can absorb the edits).
2. Rate-aware class maps (idea 2): soft tokens trained on seg + pose + HPAC's exact code length.
