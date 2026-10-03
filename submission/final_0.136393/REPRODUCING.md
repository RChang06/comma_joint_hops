# Reproducing the candidate

The compressor rebuilds the archive from PR #140's published trained archive.
It does not train a codec from the original video. The complete inference state
remains in the resulting archive; inference does not need the source archive,
the original video, or scorer weights.

Use the challenge's Python/PyTorch environment plus Brotli 1.2.0 and a C compiler.
Local verification used Python 3.11.15 and PyTorch 2.9.0+cu128 under Linux/WSL.
The source is the public
[PR #140 release archive](https://github.com/adpena/comma_video_compression_challenge/releases/download/semantic_joint_ctxmix-afr1/archive.zip).
Its exact size and SHA-256 are recorded in `LINEAGE.md`; the compressor refuses
any other source.

```bash
bash compress.sh --source /path/to/pr140/archive.zip --output /path/to/fresh/archive.zip
```

The compressor rebuilds twice, requires identical outputs, and verifies SHA-256
`0e2d95c29e87dca3f9ed14bbc842e57011090ce1e0ac0703c0bb8df3fc70c7e1`
before writing the 179,891-byte archive. It refuses to overwrite an existing
output. `recipe.json` records the selected encoder configuration.

For challenge inference, place that ZIP beside `inflate.py`, extract its sole
member `p` into the archive directory, and run:

```bash
bash inflate.sh /path/to/archive /path/to/fresh/output /path/to/public_test_video_names.txt
```

Inference uses the inherited CUDA entry point and compiler setup. All 600 pairs
must be decoded and evaluated to compare scores. The source tree and the output
archive can be checked with `sha256sum -c MANIFEST.sha256`.

`integrity.json` records exact recovered-state comparisons against PR #140 and
13 round-trip/malformed-input checks. `evaluation.json` and `report.txt` record
the full local evaluation when present. Official maintainer evaluation remains
the authority for leaderboard placement.
