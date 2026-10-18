#!/usr/bin/env bash
# Merge the r32a drafter LoRA (adapters/r32a, from the companion repo's T8 self-distillation) into the checkpoint's BF16
# MTP tensors -> x/mtp_patch_r32a.safetensors (175 MB, 22 tensors). CPU-only, ~1 min. The FP4-Local MTP tensors are
# bit-identical to the pre-QAD checkpoint, so the result matches the companion repo's patch hash exactly.
set -euo pipefail
cd "$(dirname "$0")/.."
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
LORA_SHA=3e5a76538018b88648705fbda6420f3d4a4859f805d3b4b30dd5c9a77e4d9680
PATCH_SHA=bd3c1807244253342c6c83e7aaa2864856456e4ffcae3baa069cbde36076c302
[ "$(sha256sum adapters/r32a/lora_r32a.pt | cut -d' ' -f1)" = $LORA_SHA ] || { echo "LoRA hash mismatch"; exit 1; }
T=$(mktemp -d); trap 'docker run --rm -v "$T:/w" --entrypoint rm eugr/spark-vllm-b12x:latest -rf /w/out >/dev/null 2>&1; rm -rf "$T"' EXIT
mkdir -p "$T/out"; cp adapters/r32a/lora_r32a.pt "$T/out/"
docker run --rm --network none --cpus 4 --memory 8g -e CUDA_VISIBLE_DEVICES= -v "$HF_CACHE":/root/.cache/huggingface:ro \
  -v "$T/out":/t8 -v "$PWD/t8/merge.py":/merge.py:ro --entrypoint bash eugr/spark-vllm-b12x:latest \
  -c "python3 /merge.py r32a 32 64 | tail -1 && chmod a+r /t8/mtp_patch_r32a.safetensors"
[ "$(sha256sum "$T/out/mtp_patch_r32a.safetensors" | cut -d' ' -f1)" = $PATCH_SHA ] || { echo "patch hash mismatch"; exit 1; }
cp "$T/out/mtp_patch_r32a.safetensors" x/mtp_patch_r32a.safetensors; echo "built x/mtp_patch_r32a.safetensors (sha256 verified)"
