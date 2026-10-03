#!/bin/bash
# seg-side cycle on the pruned state (pose later): map training once after the big change, then a wide-pool search.
# renderer frozen; --prune-film-keep 1 passed anyway so pruned rows stay pinned at zero.
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
score() { python -c "import json;print(json.load(open('$1/best.json'))['score'])"; }
S=F_search
python exp2_train.py --tokens $S/tokens_joint.u8 --state $S/joint_state.pt --prune-film-keep 1 --steps 1500 --window 8 \
  --gate 150 --map-mode soft --map-t0 0.5 --map-t1 0.1 --opt maxnorm --lr-map 1.0 --w-pose 0 --w-rate 1 --frame-accept \
  --render-bytes wans --lr-hpac 0 --lr-table 0 --lr-coef 0 --lr-render 0 --lr-basis 0 --lr-const 0 \
  --carrier-gn 0 --carrier-polish 0 --out G_s1 > G_s1.log 2>&1
echo "$(date +%H:%M) map training: $(grep 'gate 0:' G_s1.log | grep -o '"seg": [0-9.e-]*') -> $(grep -o '"seg": [0-9.e-]*' G_s1/best.json)"
python exp2_train.py --tokens G_s1/tokens_joint.u8 --state G_s1/joint_state.pt --prune-film-keep 1 --search-rounds 60 \
  --search-k 32 --search-pool all --search-batches 1 --w-pose 0 --w-rate 1 --render-bytes wans --lr-render 1e-12 \
  --lr-basis 0 --carrier-gn 0 --carrier-polish 0 --out G_wide > G_wide.log 2>&1
echo "$(date +%H:%M) wide search: $(grep 'search total' G_wide.log)"
tar czf /root/auto_G.tgz auto_G.log G_s1 G_s1.log G_wide G_wide.log
echo DONE > auto_G_done.flag
