# Lineage

This is an attributed derivative of adpena's
[PR #140, semantic_joint_ctxmix](https://github.com/commaai/comma_video_compression_challenge/pull/140),
pinned at commit `7f29354d7d33f7f4e734316981e3341dd2cd66f8`.
Its public source archive is 180,002 bytes, SHA-256
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`.

PR #140 inherits the semantic renderer / pose-carrier architecture from
Fesal Fayed's PR #130, JasonMo123's PR #133, and Shreyan Mohanty's PR #135.
Adpena's work includes the semantic edits, pose re-solve, pruned mixed-precision
renderer, adaptive context mixing, and carrier entropy representations used here.
We did not rerun or independently reproduce those training and solve stages.

Our change groups the serialized renderer's metadata and code spans, storing
every span length in a `LAY1` header. A four-block `BLK2` container then stores
two blocks raw and two with Brotli, with a reversible two-byte lane transpose
on one compressed block. The archive carries the layout, block boundaries,
codecs and transform settings. No model values, dictionaries, pose coefficients,
or semantic corrections move from the archive into source code.

BLK2 comes from our earlier semantic_blocks work in PR #141. Grouping and the
integration with PR #140 are new local work implemented with Codex assistance.
The new archive is 179,891 bytes, saving 111 bytes against PR #140 and 5,813
against our previous 185,704-byte v2 archive. Most of the score improvement
against v2 comes from the inherited PR #140 state and algorithms.

All PR #140 files under `cpr1/` are unchanged. Under `runtime/`, the only changed
inherited file is `residual_archive.py`; `block_container.py` and
`segment_layout.py` are added. `inflate.py` updates the expected archive size
and hash. The compressor and recipe reproduce this new representation from the
pinned PR #140 archive. Detailed source hashes are in `provenance.json`.

The upstream MIT license is preserved in `LICENSE`. This is a local research
candidate, not a claim of official evaluation or leaderboard placement.
