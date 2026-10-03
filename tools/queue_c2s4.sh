#!/bin/bash
# cycle 2 step 4 (pose stage) after step 3, then the decoder-side check of the result
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
while pgrep -f encode135 >/dev/null; do sleep 10; done
python exp2_train.py --tokens c2_s2/tokens_joint.u8 --state c1_s4/joint_state.pt --steps 600 --window 8 --gate 50 \
  --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0.02 --lr-const 0.01 \
  --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 --out c2_s4 > c2_s4.log 2>&1
python encode135.py --tokens c2_s2/tokens_joint.u8 --out c2_s2/archive.zip > c2_s4_encode.log 2>&1
python score_map135.py --tokens c2_s2/tokens_joint.u8 --archive c2_s2/archive.zip --coefficients c2_s4/coef_joint.npy \
  --state c2_s4/joint_state.pt > c2_s4_score.log 2>&1
echo DONE > c2_s4_done.flag
