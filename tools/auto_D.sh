#!/bin/bash
# after the cycle: wide-pool search pass (byte-saving edge edits now win), then a quick carrier refit
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
while [ ! -f auto_C_done.flag ]; do sleep 30; done
score() { python -c "import json;print(json.load(open('$1/best.json'))['score'])"; }
python exp2_train.py --tokens C_s2/tokens_joint.u8 --state C_s4/joint_state.pt --search-rounds 60 --search-k 32 \
  --search-pool all --search-batches 1 --w-pose 0 --w-rate 1 --render-bytes wans --lr-render 1e-12 --lr-basis 0 \
  --carrier-gn 0 --carrier-polish 0 --out D_wide > D_wide.log 2>&1
echo "$(date +%H:%M) wide search: $(grep 'search total' D_wide.log)"
python exp2_train.py --tokens D_wide/tokens_joint.u8 --state D_wide/joint_state.pt --steps 200 --window 8 --gate 50 \
  --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0.02 --lr-const 0.01 \
  --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 --render-bytes wans --out D_pose > D_pose.log 2>&1
echo "$(date +%H:%M) after wide + pose: $(score D_pose)  (cycle end $(score C_s4))"
tar czf /root/auto_D.tgz auto_D.log D_wide.log D_wide D_pose D_pose.log
echo DONE > auto_D_done.flag
