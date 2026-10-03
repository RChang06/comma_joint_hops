#!/bin/bash
# cycle 1 step 3: hpac + table refit on the final step-2 map (bytes only), then step 4's baseline:
# carrier gn + polish on that map with the refit coder, saved as the renderer stage's starting state
set -e
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
M=e2b_search/tokens_joint.u8
python exp2_train.py --tokens $M --steps 1500 --window 8 --gate 150 --shared-every 1 --lr-map 0 --lr-coef 0 --lr-render 0 \
  --lr-basis 0 --lr-const 0 --lr-hpac 0.002 --lr-table 0.002 --carrier-gn 0 --carrier-polish 0 --w-pose 1 --w-rate 1 \
  --out c1_s3 > c1_s3.log 2>&1
python exp2_train.py --tokens $M --state c1_s3/joint_state.pt --steps 0 --lr-render 0 --lr-basis 0 \
  --carrier-gn 22 --carrier-polish 4 --w-pose 1 --w-rate 1 --out c1_s4base > c1_s4base.log 2>&1
echo DONE > c1_s3_done.flag
