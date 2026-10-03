#!/bin/bash
# cycle 2, exactly like cycle 1: step 1 map training (seg + bytes, refit coder from the end state), step 2 near-pool
# search, step 3 hpac refit, then the step-4 baseline. $1 = end-of-cycle-1 state dir
set -e
S=${1:-c1_s4}
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
python exp2_train.py --tokens e2b_search/tokens_joint.u8 --state $S/joint_state.pt --steps 3000 --window 8 --gate 150 \
  --map-mode soft --map-t0 1.0 --map-t1 0.1 --opt maxnorm --lr-map 2.0 --w-pose 0 --w-rate 1 --frame-accept \
  --lr-hpac 0 --lr-table 0 --lr-coef 0 --lr-render 0 --lr-basis 0 --lr-const 0 --carrier-gn 0 --carrier-polish 0 \
  --out c2_s1 > c2_s1.log 2>&1
echo S1 > c2_s1_done.flag
python exp2_train.py --tokens c2_s1/tokens_joint.u8 --state $S/joint_state.pt --search-rounds 60 --search-k 32 \
  --search-pool near --search-batches 1 --w-pose 1 --w-rate 1 --lr-render 0 --lr-basis 0 --carrier-gn 0 --carrier-polish 0 \
  --out c2_s2 > c2_s2.log 2>&1
echo S2 > c2_s2_done.flag
python exp2_train.py --tokens c2_s2/tokens_joint.u8 --state $S/joint_state.pt --steps 1500 --window 8 --gate 150 \
  --shared-every 1 --lr-map 0 --lr-coef 0 --lr-render 0 --lr-basis 0 --lr-const 0 --lr-hpac 0.002 --lr-table 0.002 \
  --carrier-gn 0 --carrier-polish 0 --w-pose 1 --w-rate 1 --out c2_s3 > c2_s3.log 2>&1
echo S3 > c2_s3_done.flag
