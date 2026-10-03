#!/bin/bash
# Guard for the rented training box. Never spends money. Exits (which alerts) the moment
# anything needs a human decision, and destroys the box only when it would otherwise idle.
#
#   INSTANCE_GONE   host reclaimed it. billing already stopped. ALERT so the user can decide
#                   whether to re-rent; training resumes from the last checkpoint, nothing lost.
#   TRAINERS_DEAD   training stopped but the box is still billing -> pull results, DESTROY, alert.
#   FINISHED        same handling as above; the distinction is in the log line.
#   CEILING         hit the pre-authorised minute budget -> pull results, DESTROY, alert.
#   SSH_DOWN        instance still reports running -> transient, keep waiting (logged once).
#
# usage: bash guard.sh <ceiling_minutes>
CEILING_MIN=${1:-180}
IP=38.65.92.36
PORT=28805
INST=53624770
VAST="/c/Users/ruich/AppData/Roaming/Python/Python314/Scripts/vastai.exe"
RESULTS="/d/projects/comma_b/work/box_results"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
SSH="timeout 90 ssh -i $HOME/.ssh/id_ed25519_personal $SSHOPT -p $PORT root@$IP"
START=$(date +%s)
ssh_fails=0
# "no trainers" means two very different things: they have not STARTED yet (env still installing,
# targets precomputing) or they have DIED. Destroying on the first is exactly the bug that killed
# instance 52730537 seven minutes in, so only treat it as death once we have actually seen one alive.
seen_trainers=0
STARTUP_GRACE_MIN=30
# consecutive empty polls (about 2 min apart) before TRAINERS_DEAD fires
DEAD_TOLERANCE=6
dead_polls=0

vast_status() {
  # THREE outcomes, and conflating them is a bug: an empty reply means OUR network is down (or the API
  # is), which is NOT the same as the API saying the instance no longer exists. Treating the first as
  # "gone" made the guard quit during an operator internet outage and abandon the spend ceiling.
  local raw
  raw="$(timeout 60 "$VAST" show instance "$INST" --raw 2>/dev/null)"
  if [ -z "$raw" ]; then echo "api-unreachable"; return; fi
  printf '%s' "$raw" | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin)
    print(d.get("actual_status") or "unknown")
except Exception:
    print("api-unreachable")' 2>/dev/null
}

pull_results() {
  $SSH "cd /root/cb/work && tar czf /tmp/final.tgz *.log search_run transplant refit search600/*.json search600/*.npy search600/archive.zip search600/tokens_searched.u8 joint_run 2>/dev/null; du -h /tmp/final.tgz" 2>/dev/null
  mkdir -p "$RESULTS"
  if scp -q -i "$HOME/.ssh/id_ed25519_personal" $SSHOPT -P $PORT "root@$IP:/tmp/final.tgz" "$RESULTS/final_$(date +%H%M).tgz" 2>/dev/null; then
    echo "results pulled to $RESULTS/final_$(date +%H%M).tgz"
  else
    echo "WARNING could not pull results"
  fi
}

destroy() {
  timeout 120 "$VAST" destroy instance "$INST" -y 2>&1 | tail -1
  sleep 5
  echo "instances remaining: $(timeout 60 "$VAST" show instances --raw 2>/dev/null | head -c 5)"
  "$VAST" show user --raw 2>/dev/null | python3 -c 'import sys,json; print("credit left:", round(json.load(sys.stdin)["credit"],3))' 2>/dev/null
}

while true; do
  mins=$(( ($(date +%s) - START) / 60 ))

  alive=$($SSH "ps -eo args | grep -cE '^python (encode|score_map|exact_cost|search|refit|refit_gn|joint_train)135'" 2>/dev/null)

  if [ -z "$alive" ]; then
    st=$(vast_status)
    if [ "$st" = "api-unreachable" ]; then
      echo "both ssh and the vast API are unreachable at ${mins}min - assuming OUR network is down, holding the ceiling"
      sleep 120; continue
    fi
    if [ "$st" = "running" ]; then
      ssh_fails=$((ssh_fails + 1))
      [ "$ssh_fails" -eq 1 ] && echo "SSH_DOWN at ${mins}min (instance still running, waiting)"
      if [ "$ssh_fails" -ge 8 ]; then
        echo "ALERT SSH_DEAD: 8 consecutive failures over ~16 min while vast says running at ${mins}min"
        exit 0
      fi
      sleep 120; continue
    fi
    echo "ALERT INSTANCE_GONE at ${mins}min: vast status=$st. Billing has stopped."
    echo "Nothing to destroy. Training can resume from the last saved checkpoint once a new box is rented."
    exit 0
  fi

  ssh_fails=0

  if [ "$alive" != "0" ]; then
    seen_trainers=1
    dead_polls=0
    # heartbeat. the guard used to print NOTHING while healthy, which made it impossible to tell a
    # working guard from a dead one, and an unattended run needs a readable progress trail anyway.
    # newest log, whichever run is current, so the heartbeat survives switching runs
    last=$($SSH "cd /root/cb/work && tail -1 \$(ls -t *.log 2>/dev/null | head -1) 2>/dev/null" 2>/dev/null)
    best=$($SSH "cd /root/cb/work && for f in \$(ls -t runs/*/best.json 2>/dev/null); do echo \"\$f \$(cat \$f)\"; done | head -3" 2>/dev/null)
    echo "[${mins}min/${CEILING_MIN}] $last"
    [ -n "$best" ] && echo "[${mins}min] best: $best"
  elif [ "$seen_trainers" = "1" ]; then
    # tolerate a GAP rather than destroying on the first empty poll. between a finished probe and the
    # launch of the run it informs, there is legitimately no trainer alive, and destroying there throws
    # away the box mid-experiment. the spend ceiling still bounds the damage if nothing ever restarts.
    dead_polls=$((dead_polls + 1))
    if [ "$dead_polls" -lt "$DEAD_TOLERANCE" ]; then
      echo "no trainer at ${mins}min (${dead_polls}/${DEAD_TOLERANCE} before declaring death) - holding"
      sleep 120; continue
    fi
    echo "ALERT TRAINERS_DEAD at ${mins}min (box still billing, ${dead_polls} consecutive empty polls) - pulling results and destroying"
    pull_results; destroy; exit 0
  elif [ "$mins" -ge "$STARTUP_GRACE_MIN" ]; then
    echo "ALERT NEVER_STARTED: no trainer seen in ${mins} min (grace ${STARTUP_GRACE_MIN}) - launch likely failed"
    pull_results; destroy; exit 0
  else
    echo "waiting for training to start (${mins}min of ${STARTUP_GRACE_MIN} grace)"
  fi

  if [ "$mins" -ge "$CEILING_MIN" ]; then
    echo "ALERT CEILING reached (${CEILING_MIN}min budget) - pulling results and destroying"
    pull_results; destroy; exit 0
  fi

  sleep 120
done
