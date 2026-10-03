#!/bin/bash
# runs on the rented box as root. writes SETUP_DONE to /root/work/setup.log when finished.
# the video is uploaded separately (not in git); the judges come from git-lfs.
set -x
mkdir -p /root/work
cd /root/work
exec >> /root/work/setup.log 2>&1
echo "setup start $(date)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git git-lfs unzip ffmpeg pkg-config build-essential >/dev/null
git lfs install
export GIT_LFS_SKIP_SMUDGE=1
if [ ! -d challenge ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/commaai/comma_video_compression_challenge.git challenge
fi
cd challenge
git fetch origin pull/141/head:pr141
git checkout pr141
cp /root/work/challenge_upload/*.safetensors models/
ls -la models/
cd /root/work
python3 -m venv --system-site-packages /root/venv
source /root/venv/bin/activate
pip install -q --upgrade pip wheel
pip install -q torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -q numpy einops timm safetensors segmentation-models-pytorch tqdm pillow av charset-normalizer requests urllib3 "brotli==1.2.0" constriction pyppmd
pip install -q nvidia-dali-cuda120==1.52.0 --extra-index-url https://pypi.nvidia.com
python - <<'PY'
import torch, brotli, importlib.metadata as m
print("torch", torch.__version__, "cuda", torch.version.cuda, "brotli", m.version("Brotli"))
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(512, 512, device="cuda"); print("matmul ok", float((x @ x).sum()))
PY
echo caps=$NVIDIA_DRIVER_CAPABILITIES
echo "SETUP_DONE $(date)"
