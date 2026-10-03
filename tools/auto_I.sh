#!/bin/bash
# after the wide x4 search: pose stage on the pruned line (strengths gn + gray/amp + patterns), full score
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
while [ ! -f auto_H_done.flag ]; do sleep 30; done
score() { python -c "import json;print(json.load(open('$1/best.json'))['score'])"; }
python exp2_train.py --tokens H_wide4/tokens_joint.u8 --state H_wide4/joint_state.pt --prune-film-keep 1 --steps 600 \
  --window 8 --gate 50 --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0.02 \
  --lr-const 0.01 --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 --render-bytes wans --out I_pose > I_pose.log 2>&1
echo "$(date +%H:%M) pruned line pose stage: $(score I_pose)"
tar czf /root/auto_I.tgz auto_I.log I_pose I_pose.log
echo DONE > auto_I_done.flag
