#!/bin/bash
# on the rented box: deps, unpack, compile rc64, then the queue. logs in /root/cb/work/*.log
set -x
exec >> /root/setup.log 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq build-essential >/dev/null
mkdir -p /root/cb && tar xzf /root/maps135_bundle.tgz -C /root/cb
python3 -m venv --system-site-packages /root/venv
source /root/venv/bin/activate
pip install -q --upgrade pip wheel
pip install -q torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -q numpy einops timm safetensors segmentation-models-pytorch tqdm pillow av brotli constriction pyppmd
cd /root/cb/work
cc -O3 -std=c11 -shared -fPIC pr135/submissions/semantic-pose-HPAC_CPR1_polished/runtime/entropy/rc64_backend.c -o /tmp/rc64_135.so
cc -O3 -std=c11 -shared -fPIC research/repos/CommaVideoCompressionChallenge_ExperimentBook/src/cpr1_sub4/entropy/rc64_backend.c -o /tmp/rc64_full.so
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
echo SETUP_DONE
