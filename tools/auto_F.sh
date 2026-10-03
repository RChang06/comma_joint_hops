#!/bin/bash
# pruning hop: keep 2 of 192 rows (1%) of blocks.1-3.film like #141, renderer move on the full score with the pruned
# rows pinned at zero, full near-pool re-search, carrier refit. real bytes must be judged in #141's SM3R format later.
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
while [ ! -f auto_E_done.flag ]; do sleep 30; done
score() { python -c "import json;print(json.load(open('$1/best.json'))['score'])"; }
S=E_pose
python exp2_train.py --tokens $S/tokens_joint.u8 --state $S/joint_state.pt --prune-film-keep 1 --steps 300 --window 8 \
  --gate 300 --lr-map 0 --w-pose 1 --w-rate 1 --render-bytes wans --shared-every 25 --lr-render 0.02 --lr-render-fp16 2e-4 \
  --lr-render-scale 1e-4 --lr-hpac 0 --lr-table 0 --lr-coef 0 --lr-basis 0 --lr-const 0 --carrier-gn 0 --carrier-polish 0 \
  --save-final --out F_move > F_move.log 2>&1
echo "$(date +%H:%M) pruned + moved: $(grep 'gate 0:' F_move.log | grep -o '"seg": [0-9.e-]*') -> $(grep 'final (unaccepted)' F_move.log | grep -o '"seg": [0-9.e-]*')"
python exp2_train.py --tokens F_move/unaccepted/tokens_joint.u8 --state F_move/unaccepted/joint_state.pt \
  --search-rounds 60 --search-k 32 --search-pool near --search-batches 1 --w-pose 0 --w-rate 1 --render-bytes wans \
  --lr-render 1e-12 --lr-basis 0 --carrier-gn 0 --carrier-polish 0 --out F_search > F_search.log 2>&1
echo "$(date +%H:%M) search: $(grep 'search total' F_search.log)"
python exp2_train.py --tokens F_search/tokens_joint.u8 --state F_search/joint_state.pt --steps 200 --window 8 --gate 50 \
  --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0 --lr-const 0.01 \
  --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 --render-bytes wans --out F_pose > F_pose.log 2>&1
echo "$(date +%H:%M) pruning hop full (trainer, wans bytes): $(score F_pose) vs start $(score $S)"
tar czf /root/auto_F.tgz auto_F.log F_*.log F_pose
echo DONE > auto_F_done.flag
