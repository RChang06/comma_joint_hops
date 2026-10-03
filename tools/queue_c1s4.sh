#!/bin/bash
# pose stage (new step 4) smoke runs: control on #141 maps + search (carrier known to reach ~5e-6), then ours.
# trains the 12 patterns + gray/amp by gradient, strengths by gn + polish at every gate; map/renderer/hpac frozen.
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
while [ ! -f c1_s4base_done.flag ]; do sleep 30; done
P="--steps 300 --window 8 --gate 50 --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 \
  --lr-basis 0.02 --lr-const 0.01 --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1"
python exp2_train.py --tokens control141/tokens.u8 --coefficients control141/coef_gn.npy $P --out c1_ctrl141 > c1_ctrl141.log 2>&1
python exp2_train.py --tokens e2b_search/tokens_joint.u8 --state c1_s4base/joint_state.pt $P --out c1_s4smoke > c1_s4smoke.log 2>&1
echo DONE > c1_s4smoke_done.flag
