#!/bin/bash
# official inflate + score of an already-built submission dir. usage: e2e_only.sh <out_dir under work> <state_dir under work>
set -euo pipefail
source ~/venvs/coolchic/bin/activate
cd /mnt/d/projects/comma_b
O=work/$1; S=$O/sub141; E=$O/e2e
rm -rf $E && mkdir -p $E/archive
unzip -o -q $S/archive.zip -d $E/archive; cp $S/archive.zip $E/archive.zip
export DALI_DISABLE_NVML=1
time bash $S/inflate.sh $E/archive $E/inflated public_test_video_names.txt
sha256sum $E/inflated/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8 work/$2/tokens_joint.u8
SCORE_BATCH=4 python work/score.py . work/targets/gt_dali_t2000.pt $E
rm -f $E/inflated/0.raw
echo VERIFY_DONE
