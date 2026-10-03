#!/bin/bash
# rebuild #141's archive from #140's pinned release and run the official decoder once locally.
# leaves the decoded class maps in submissions/semantic_blocks/inflated/.f26_decode_checkpoints/
set -euo pipefail
source ~/venvs/coolchic/bin/activate
cd /mnt/d/projects/comma_b
S=submissions/semantic_blocks
[ -f $S/archive.zip ] || python $S/compress.py --source work/src_archives/pr140_archive.zip --output $S/archive.zip
sha256sum $S/archive.zip
rm -rf $S/archive $S/inflated
mkdir -p $S/archive && unzip -o -q $S/archive.zip -d $S/archive
export DALI_DISABLE_NVML=1
time bash $S/inflate.sh $S/archive $S/inflated public_test_video_names.txt
ls -la $S/inflated $S/inflated/.f26_decode_checkpoints
echo LOCAL_DECODE_DONE
