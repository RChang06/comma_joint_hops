#!/bin/bash
# option 1 on our trained map: baseline (jA2 + gn carrier) then #140-style search from jA2, each exact-scored
set -e
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
T=jA2/tokens_joint.u8
mkdir -p s3
python encode135.py --tokens $T --out s3/archive_ja2.zip > s3_0_encode.log 2>&1
python refit_gn135.py --tokens $T --iters 22 --polish 4 --out s3/coef_ja2.npy > s3_1_refit_ja2.log 2>&1
python score_map135.py --tokens $T --archive s3/archive_ja2.zip --coefficients s3/coef_ja2.npy > s3_2_score_ja2.log 2>&1
python exact_cost135.py --tokens $T --dump-costs 0:600 --out s3 > s3_3_costs.log 2>&1
python search135.py --frames 0:600 --rounds 100 --k 32 --tokens $T --costs s3/costs_0_600.npy --out s3/search > s3_4_search.log 2>&1
python encode135.py --tokens s3/search/tokens_searched.u8 --out s3/search/archive.zip > s3_5_encode.log 2>&1
python refit_gn135.py --tokens s3/search/tokens_searched.u8 --iters 22 --polish 4 --out s3/search/coef.npy > s3_6_refit.log 2>&1
python score_map135.py --tokens s3/search/tokens_searched.u8 --archive s3/search/archive.zip --coefficients s3/search/coef.npy > s3_7_score.log 2>&1
echo DONE > s3_done.flag
