#!/bin/bash
# build + verify a trainer state as a real #141-format submission on the laptop.
# usage: verify_state.sh <state_dir under work> <out_dir under work>
set -euo pipefail
source ~/venvs/coolchic/bin/activate
cd /mnt/d/projects/comma_b/work
ST=$1; O=$2
mkdir -p $O
PYTHONPATH=pr135/submissions/semantic-pose-HPAC_CPR1_polished:pr135/submissions/semantic-pose-HPAC_CPR1_polished/cpr1 \
  python make_hop_pieces.py $ST $O
rm -rf $O/src135; cp -r final_c2/sub_c2 $O/src135
cd ..
python work/encode141.py --tokens work/$ST/tokens_joint.u8 --stream-out work/$O/stream141.bin --out work/$O/tmp141.zip > work/$O/encode141.log 2>&1
python work/build_archive141_sm3r.py --renderer sm3r --keep 1 --wide --wans-body work/$O/wans_body.bin --src135 work/$O/src135 --carrier work/$O/carrier.bin --stream work/$O/stream141.bin --out work/$O/archive141.zip
rm -rf work/$O/sub141; cp -r work/final_c2/sub141_c2 work/$O/sub141; cp work/$O/archive141.zip work/$O/sub141/archive.zip
S=work/$O/sub141; SHA=$(sha256sum $S/archive.zip | cut -d' ' -f1); SZ=$(stat -c %s $S/archive.zip)
read G A < <(python -c "import torch;s=torch.load('work/$ST/joint_state.pt',weights_only=False);import numpy as np;print(repr(float(np.float32(s['gray']))),repr(float(np.float32(s['amp']))))")
sed -i "s/^ARCHIVE_SHA256 = \"[0-9a-f]*\"/ARCHIVE_SHA256 = \"$SHA\"/; s/^ARCHIVE_BYTES = [0-9_]*/ARCHIVE_BYTES = $SZ/" $S/inflate.py
sed -i "s/^CARRIER_AMPLITUDE = [0-9.]*/CARRIER_AMPLITUDE = $A/; s/^CARRIER_GRAY = [0-9.]*/CARRIER_GRAY = $G/" $S/cpr1/inflate.py
grep -n "^CARRIER_GRAY\|^CARRIER_AMPLITUDE" $S/cpr1/inflate.py; grep -n "^ARCHIVE_" $S/inflate.py
E=work/$O/e2e; rm -rf $E && mkdir -p $E/archive
unzip -o -q $S/archive.zip -d $E/archive; cp $S/archive.zip $E/archive.zip
export DALI_DISABLE_NVML=1
time bash $S/inflate.sh $E/archive $E/inflated public_test_video_names.txt
sha256sum $E/inflated/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8 work/$ST/tokens_joint.u8
SCORE_BATCH=4 python work/score.py . work/targets/gt_dali_t2000.pt $E
rm -f $E/inflated/0.raw
echo VERIFY_DONE
