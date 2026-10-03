#!/bin/bash
# search variants on pairs 200:240 from the current state (final_pose); net gain + time per variant
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
C="--tokens final_pose/tokens_joint.u8 --state final_pose/joint_state.pt --search-rounds 60 --search-frames 200:240 --w-pose 0 --w-rate 1 --render-bytes wans --lr-render 1e-12 --lr-basis 0 --carrier-gn 0 --carrier-polish 0"
python exp2_train.py $C --search-pool near --search-k 32 --search-batches 1 --out d2_v1 > d2_v1.log 2>&1
python exp2_train.py $C --search-pool near --search-k 32 --search-batches 4 --out d2_v2 > d2_v2.log 2>&1
python exp2_train.py $C --search-pool near --search-k 64 --search-batches 1 --out d2_v3 > d2_v3.log 2>&1
python exp2_train.py $C --search-pool all --search-k 32 --search-batches 1 --out d2_v4 > d2_v4.log 2>&1
echo DONE > d2_done.flag
