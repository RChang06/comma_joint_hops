#!/bin/bash
# spend ceiling for the exp2 box ONLY. hardcoded to its own instance id so it can never
# touch any other box; was 52875597 first. no trainer logic: this box is driven interactively,
# so the only job is to destroy it at the pre-agreed minute ceiling.
# usage: bash guard_b.sh <ceiling_minutes>
CEILING_MIN=${1:-240}
INST=53664321
VAST="/c/Users/ruich/AppData/Roaming/Python/Python314/Scripts/vastai.exe"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
START=$(date +%s)
while true; do
  mins=$(( ($(date +%s) - START) / 60 ))
  st=$("$VAST" show instance "$INST" --raw 2>/dev/null | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("actual_status") or "unknown")
except Exception: print("api-unreachable")' 2>/dev/null)
  if [ "$st" != "running" ] && [ "$st" != "loading" ] && [ "$st" != "api-unreachable" ]; then
    echo "ALERT INSTANCE_GONE at ${mins}min: status=$st"; exit 0
  fi
  if [ "$mins" -ge "$CEILING_MIN" ]; then
    echo "ALERT CEILING ${CEILING_MIN}min reached - destroying $INST"
    "$VAST" destroy instance "$INST" -y 2>&1 | tail -1
    exit 0
  fi
  sleep 120
done
