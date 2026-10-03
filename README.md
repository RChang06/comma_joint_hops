# comma video compression challenge: joint_hops

Work on the comma.ai video compression challenge, starting from PR #135 and stored in PR #141's container.
Verified score 0.136393 (#141: 0.147897).

- `docs/FINAL_REPORT.md`: what was done, numbers, and what failed
- `docs/WRITEUP_NOTES.md`: detailed notes in order
- `tools/`: training, search, encoders, writers, archive builders
- `submission/final_0.136393/`: submission folder (decoder + archive.zip)
- `states/`: final carrier / hpac / table states (class maps not included, too large)

The decoder in `submission/` is #141's (built on #140 and #135) with small constant changes.
