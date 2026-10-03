#!/bin/bash
# the search queue. each step writes its own log; a failure stops the queue.
set -e
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
A=${1:-0}; B=${2:-50}
# 1. transplant: #141's edited maps in a #135 archive
python encode135.py --tokens extracted/tokens_pr141.u8 --out transplant/archive_135coder_141maps.zip > q1_transplant_encode.log 2>&1
python score_map135.py --tokens extracted/tokens_pr141.u8 --archive transplant/archive_135coder_141maps.zip > q1_transplant_score.log 2>&1
# 2. price list for the search range (starting map = #135's own)
python exact_cost135.py --tokens pr135_cost/tokens_pr135.u8 --frames $((B+1)) --dump-costs $A:$B --out pr135_cost > q2_costs.log 2>&1
# 3. the search
python search135.py --frames $A:$B --rounds 40 --k 8 --costs pr135_cost/costs_${A}_${B}.npy --out search_run > q3_search.log 2>&1
# 4. exact rate of the searched map + real archive + score (carrier not yet re-solved)
python exact_cost135.py --tokens search_run/tokens_searched.u8 --out search_run > q4_cost.log 2>&1
python encode135.py --tokens search_run/tokens_searched.u8 --out search_run/archive_searched.zip > q4_encode.log 2>&1
python score_map135.py --tokens search_run/tokens_searched.u8 --archive search_run/archive_searched.zip > q4_score.log 2>&1
echo QUEUE_DONE > queue_done.flag
