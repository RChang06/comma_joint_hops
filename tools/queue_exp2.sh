#!/bin/bash
# experiment 2: map training with exact rate on (seg + bytes), then the byte-priced search, then carrier refit + real archive score
set -e
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
python exp2_train.py --steps 3000 --window 8 --gate 150 --map-mode soft --map-t0 1.0 --map-t1 0.1 --opt maxnorm --lr-map 2.0 \
  --w-pose 0 --w-rate 1 --frame-accept --lr-hpac 0 --lr-table 0 --lr-coef 0 --lr-render 0 --lr-basis 0 --lr-const 0 \
  --carrier-gn 0 --carrier-polish 0 --out e2_train > e2_1_train.log 2>&1
python exp2_train.py --tokens e2_train/tokens_joint.u8 --search-rounds 60 --search-k 32 --w-pose 1 --w-rate 1 \
  --lr-render 0 --lr-basis 0 --carrier-gn 22 --carrier-polish 4 --out e2_search > e2_2_search.log 2>&1
python encode135.py --tokens e2_search/tokens_joint.u8 --out e2_search/archive.zip > e2_3_encode.log 2>&1
python score_map135.py --tokens e2_search/tokens_joint.u8 --archive e2_search/archive.zip --coefficients e2_search/coef_joint.npy > e2_4_score.log 2>&1
echo DONE > e2_done.flag
