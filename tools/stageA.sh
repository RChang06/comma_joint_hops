#!/bin/bash
# stage A pose hops: from a start state, try pattern / gray-amp moves of several sizes (strengths frozen during the
# move), each judged by one gate = full strengths gn + polish refit, accept only if pose term + rate improves.
# keep the best accepted candidate, repeat from it; stop when a round accepts nothing. usage: stageA.sh <state_dir> <rounds>
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
S=$1; R=${2:-4}
for r in $(seq 1 $R); do
  best=""; bests=$(python -c "import json;print(json.load(open('$S/best.json'))['score'])")
  echo "round $r start $S score $bests"
  for lb in 0.1 0.3 1.0; do
    O=sA_r${r}_b${lb}
    python exp2_train.py --tokens $S/tokens_joint.u8 --state $S/joint_state.pt --steps 100 --window 8 --gate 100 \
      --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis $lb --lr-const 0.02 \
      --carrier-gn 22 --carrier-polish 4 --w-pose 1 --w-rate 1 --render-bytes wans --out $O > $O.log 2>&1
    v=$(grep "^gate 100" $O.log | grep -o "ACCEPT\|ROLLBACK\|TIE"); sc=$(python -c "import json;print(json.load(open('$O/best.json'))['score'])")
    echo "  lr-basis $lb: $v score $sc"
    if [ "$v" = "ACCEPT" ] && python -c "import sys;sys.exit(0 if $sc < $bests else 1)"; then best=$O; bests=$sc; fi
  done
  [ -z "$best" ] && { echo "round $r: nothing accepted, stop"; break; }
  echo "round $r: keep $best ($bests)"; S=$best
done
echo "FINAL $S" ; echo DONE > stageA_done.flag
