#!/bin/bash
# build + verify P_pose map/carrier/renderer with the re-fitted #141 hpac (P_hpac2: section, table, stream)
set -euo pipefail
source ~/venvs/coolchic/bin/activate
cd /mnt/d/projects/comma_b/work
O=final_PP; ST=rj_results/P_pose; H=rj_results/P_hpac2
mkdir -p $O
PYTHONPATH=pr135/submissions/semantic-pose-HPAC_CPR1_polished:pr135/submissions/semantic-pose-HPAC_CPR1_polished/cpr1 \
  python make_hop_pieces.py $ST $O
rm -rf $O/src135; cp -r final_c2/sub_c2 $O/src135
cd ..
python work/build_archive141_sm3r.py --renderer sm3r --keep 1 --wide --wans-body work/$O/wans_body.bin --src135 work/$O/src135 \
  --carrier work/$O/carrier.bin --stream work/$H/stream141.bin --hpac-section work/$H/hpac.sec --table-body work/$H/table.bin \
  --out work/$O/archive141.zip 2>&1 | grep -E "gate b|archive [0-9]|table body|hpac section"
rm -rf work/$O/sub141; cp -r work/final_NH/sub141 work/$O/sub141; cp work/$O/archive141.zip work/$O/sub141/archive.zip
S=work/$O/sub141; SHA=$(sha256sum $S/archive.zip | cut -d' ' -f1); SZ=$(stat -c %s $S/archive.zip)
read G A < <(python -c "import torch,numpy as np;s=torch.load('work/$ST/joint_state.pt',weights_only=False);print(repr(float(np.float32(s['gray']))),repr(float(np.float32(s['amp']))))")
sed -i "s/^ARCHIVE_SHA256 = \"[0-9a-f]*\"/ARCHIVE_SHA256 = \"$SHA\"/; s/^ARCHIVE_BYTES = [0-9_]*/ARCHIVE_BYTES = $SZ/" $S/inflate.py
sed -i "s/^CARRIER_AMPLITUDE = [0-9.]*/CARRIER_AMPLITUDE = $A/; s/^CARRIER_GRAY = [0-9.]*/CARRIER_GRAY = $G/" $S/cpr1/inflate.py
grep -n "^CARRIER_GRAY\|^CARRIER_AMPLITUDE" $S/cpr1/inflate.py; grep -n "^ARCHIVE_" $S/inflate.py
bash work/e2e_only.sh $O $ST
