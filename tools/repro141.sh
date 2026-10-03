#!/bin/bash
# rebuild #141's archive from #140's pinned release, then score it with the official evaluator.
set -euo pipefail
source /root/venv/bin/activate
cd /root/work/challenge
cp /root/work/challenge_upload/0.mkv videos/0.mkv
S=submissions/semantic_blocks
rm -f $S/archive.zip
python $S/compress.py --source /root/work/challenge_upload/pr140_archive.zip --output $S/archive.zip
sha256sum $S/archive.zip
time bash evaluate.sh --submission-dir $S --device cuda
cat $S/report.txt
echo REPRO_DONE
