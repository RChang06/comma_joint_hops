#!/bin/bash
# box idle until 11:38: try renderer hops with different move sizes from K_final (2x and 0.5x the usual step), keep
# the best that beats K_final by > 1e-5; then one stage-A pose-hop round. pruned rows pinned. bundles /root/L*.tgz
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
score() { python -c "import json;print(json.load(open('$1/best.json'))['score'])"; }
better() { python -c "import sys;sys.exit(0 if $1 < $2 - 1e-5 else 1)"; }
P="--prune-film-keep 1 --render-bytes wans"
S=K_final; BEST=K_final
for v in big:0.04:2e-4:4e-4 small:0.01:1e-4:5e-5; do
  IFS=: read nm lr fp sc <<< "$v"
  python exp2_train.py --tokens $S/tokens_joint.u8 --state $S/joint_state.pt $P --steps 300 --window 8 --gate 300 \
    --lr-map 0 --w-pose 1 --w-rate 1 --shared-every 25 --lr-render $lr --lr-render-fp16 $fp --lr-render-scale $sc \
    --lr-hpac 0 --lr-table 0 --lr-coef 0 --lr-basis 0 --lr-const 0 --carrier-gn 0 --carrier-polish 0 \
    --save-final --out L_${nm}_move > L_${nm}_move.log 2>&1
  python exp2_train.py --tokens L_${nm}_move/unaccepted/tokens_joint.u8 --state L_${nm}_move/unaccepted/joint_state.pt $P \
    --search-rounds 60 --search-k 32 --search-pool near --search-batches 1 --w-pose 0 --w-rate 1 --lr-render 1e-12 \
    --lr-basis 0 --carrier-gn 0 --carrier-polish 0 --out L_${nm}_search > L_${nm}_search.log 2>&1
  python exp2_train.py --tokens L_${nm}_search/tokens_joint.u8 --state L_${nm}_search/joint_state.pt $P --steps 100 \
    --window 8 --gate 50 --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0 \
    --lr-const 0.01 --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 --out L_${nm}_pose > L_${nm}_pose.log 2>&1
  echo "$(date +%H:%M) hop $nm: $(score L_${nm}_pose) vs $(score $S)"
  tar czf /root/L_${nm}.tgz L_${nm}_pose L_${nm}_*.log auto_L.log
  if better $(score L_${nm}_pose) $(score $BEST); then BEST=L_${nm}_pose; fi
done
echo "$(date +%H:%M) best after size probes: $BEST $(score $BEST)"
python exp2_train.py --tokens $BEST/tokens_joint.u8 --state $BEST/joint_state.pt $P --steps 100 --window 8 --gate 100 \
  --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0.1 --lr-const 0.02 \
  --carrier-gn 22 --carrier-polish 4 --w-pose 1 --w-rate 1 --out L_poseA > L_poseA.log 2>&1
echo "$(date +%H:%M) pose hop: $(score L_poseA) vs $(score $BEST)"
tar czf /root/L_final.tgz L_poseA L_poseA.log auto_L.log
echo DONE > auto_L_done.flag
