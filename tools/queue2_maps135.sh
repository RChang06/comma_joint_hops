#!/bin/bash
# round 2: carrier re-solves for the two candidate maps, then the full 600-pair search from #141's maps.
set -e
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
mkdir -p refit
python refit135.py --tokens search_run/tokens_searched.u8 --out refit/coef_searched50.npy > r1_refit_searched.log 2>&1
python score_map135.py --tokens search_run/tokens_searched.u8 --archive search_run/archive_searched.zip --coefficients refit/coef_searched50.npy > r1_score_searched.log 2>&1
python refit135.py --tokens extracted/tokens_pr141.u8 --out refit/coef_transplant.npy > r2_refit_transplant.log 2>&1
python score_map135.py --tokens extracted/tokens_pr141.u8 --archive transplant/archive_135coder_141maps.zip --coefficients refit/coef_transplant.npy > r2_score_transplant.log 2>&1
python exact_cost135.py --tokens extracted/tokens_pr141.u8 --dump-costs 0:600 --out cost141 > r3_costs141.log 2>&1
python search135.py --frames 0:600 --rounds 100 --k 32 --tokens extracted/tokens_pr141.u8 --costs cost141/costs_0_600.npy --out search600 > r4_search600.log 2>&1
python exact_cost135.py --tokens search600/tokens_searched.u8 --out search600 > r5_cost.log 2>&1
python encode135.py --tokens search600/tokens_searched.u8 --out search600/archive.zip > r5_encode.log 2>&1
python refit135.py --tokens search600/tokens_searched.u8 --out search600/coef.npy > r5_refit.log 2>&1
python score_map135.py --tokens search600/tokens_searched.u8 --archive search600/archive.zip --coefficients search600/coef.npy > r5_score.log 2>&1
echo QUEUE2_DONE > queue2_done.flag
