#!/usr/bin/env bash
# T8 training session on the Spark (serving container must be stopped: training needs ~40-60 GB).
# usage: t8/trainrun.sh TAG [train.py args...]   data in /dev/shm/t8fp4 (capture.py), outputs lora_TAG.pt / log_TAG.txt there
set -euo pipefail
cd "$(dirname "$0")/.."; TAG=$1; shift
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"; D=/dev/shm/t8fp4
docker ps -q -f name=fp4local | grep -q . && { echo "stop the fp4local server first"; exit 1; }
run() { docker run --rm --name fp4-t8 --gpus all --ipc host --memory 70g --memory-swap 70g \
  -v "$HF_CACHE":/root/.cache/huggingface:ro -v $D:/t8 -v "$PWD":/w:ro --entrypoint python3 eugr/spark-vllm-b12x:latest "$@"; }
[ -f $D/dv.pt ] || run -c "import torch; ids=sorted({int(l) for l in open('/w/x/dv65k.txt') if l.strip()} | set(range(248320-320, 248320))); torch.save(torch.tensor(ids), '/t8/dv.pt'); print('dv', len(ids))"
run /w/t8/train.py --tag "$TAG" "$@"
