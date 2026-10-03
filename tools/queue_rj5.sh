#!/bin/bash
# controls for the renderer-move result: (1) pose stage on the moved+searched state (full score),
# (2) the same near-pool search on the old renderer (fair seg + rate baseline)
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
python exp2_train.py --tokens rj_move_search/tokens_joint.u8 --state rj_move_search/joint_state.pt --steps 600 --window 8 \
  --gate 50 --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0.02 --lr-const 0.01 \
  --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 --render-bytes wans --out rj_pose > rj_pose.log 2>&1
echo POSE > rj_pose_done.flag
python exp2_train.py --tokens c2_s2/tokens_joint.u8 --state c2_s4/joint_state.pt --search-rounds 60 --search-k 32 \
  --search-pool near --search-batches 1 --w-pose 0 --w-rate 1 --render-bytes wans --lr-render 1e-12 --lr-basis 0 \
  --carrier-gn 0 --carrier-polish 0 --out rj_ctrl_search > rj_ctrl_search.log 2>&1
echo DONE > rj5_done.flag
