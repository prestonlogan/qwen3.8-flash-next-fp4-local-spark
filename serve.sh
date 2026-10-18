#!/usr/bin/env bash
# NVIDIA Qwen3.8-Flash-Next-FP4-Local on one DGX Spark (model-card Spark recipe, run detached).
# Default (FP4_OPT=1): fp4x decode plugin (x/fp4patch.py) + r32a drafter + 2 MTP drafts, ~55 tok/s greedy prose
# (card recipe ~36.7). FP4_OPT=0: card recipe + 3 MTP drafts + --async-scheduling (~39.9).
# fp4x patches (M<=16 decode path only; prefill untouched):
#   FP4X_MX8      exact MXFP8 GEMV for dense linears (b12x quantizes activations at small M)
#   FP4X_DENSE=w6 6-bit g32 copies of dense weights; FP4X_NV4_SHAPES: GDN in_proj_qkvz as NVFP4
#   FP4X_HC/_W8   hyper-connection mixers as int8 g64 GEMV (target only; MTP keeps BF16)
#   FP4X_HC_FUSED each mixer as two fused kernels (split-K down + silu epilogue, up + gate-mean epilogue; bit-exact)
#   FP4X_GATE=w8  MoE router gates as int8 g64 GEMV (decode PPL +0.04%)
#   FP4X_MTP_W8   MTP layer dense weights int8 g64 (draft only)
#   FP4X_DRAFT_VOCAB  MTP draft head over a 65k-token NVFP4 subset (draft only)
#   FP4X_DRAFT_INT2=64 draft head as INT2 coarse + exact NVFP4 refine of the top 64; draft q = those 64 (draft only)
#   FP4X_FAST_HEAD    INT2 coarse LM head + exact MXFP8 rescoring of the top 512 in one fused kernel (argmax exact)
#   FP4X_MTP_PATCH    r32a self-distilled drafter weights (t8/build_patch.sh; draft only, +4% prose)
# Decode-path PPL (12 texts): card 2.6501, optimized ~2.633 (exact MXFP8 more than offsets W6/NV4/W8).
# usage: serve.sh [port] [extra vllm args...]
set -euo pipefail
PORT="${1:-5810}"; shift || true
NAME="${FP4_NAME:-fp4local}"
ROOT="${FP4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
C=$ROOT/cache
X=$ROOT/x
SITE=/usr/local/lib/python3.12/dist-packages
if [ "${FP4_OPT:-1}" = 1 ]; then
  SPEC_DEFAULT='{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}'
  OPT_ARGS=(
    -v $X/fp4patch.py:$SITE/fp4patch.py:ro -v $X/fp4x_kernels.py:$SITE/fp4x_kernels.py:ro
    -v $X/fp4patch-0.1.dist-info:$SITE/fp4patch-0.1.dist-info:ro -v $X/dv65k.txt:/opt/fp4x/dv65k.txt:ro
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    -e FP4X_MX8=1 -e FP4X_DENSE=w6 -e FP4X_NV4_SHAPES=16384x2560
    -e FP4X_HC=1 -e FP4X_HC_W8=1 -e FP4X_HC_FUSED=1 -e FP4X_MTP_W8=1 -e FP4X_GATE=w8
    -e FP4X_DRAFT_VOCAB=/opt/fp4x/dv65k.txt -e FP4X_DRAFT_INT2=64 -e FP4X_FAST_HEAD=1
  )
  if [ -f $X/mtp_patch_r32a.safetensors ]; then
    OPT_ARGS+=(-v $X/mtp_patch_r32a.safetensors:/opt/fp4x/mtp_patch_r32a.safetensors:ro -e FP4X_MTP_PATCH=/opt/fp4x/mtp_patch_r32a.safetensors)
  else
    echo "serve.sh: x/mtp_patch_r32a.safetensors missing (run t8/build_patch.sh); serving the stock drafter" >&2
  fi
else
  SPEC_DEFAULT='{"method":"mtp","num_speculative_tokens":3}'
  OPT_ARGS=()
fi
SPEC="${FP4_SPEC:-$SPEC_DEFAULT}"
COMP_DEFAULT='{"pass_config":{"fuse_act_quant":true}}'
mkdir -p $C/vllm $C/flashinfer $C/triton $C/b12x
docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run -d --name "$NAME" \
  --gpus all --ipc host --network host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v $C/vllm:/root/.cache/vllm -v $C/flashinfer:/root/.cache/flashinfer \
  -v $C/triton:/root/.triton -v $C/b12x:/root/.cache/b12x \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_PLE_TABLE_MEMORY="${VLLM_PLE_TABLE_MEMORY:-disk}" \
  -e B12X_DISK_BACKEND=io_uring \
  -e CUTE_DSL_ARCH=sm_121a \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e B12X_POLICY_MODE=auto \
  -e VLLM_MXFP8_LM_HEAD=1 \
  -e VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 \
  -e VLLM_B12X_MOE_FP4_FORCE_A16=1 \
  -e B12X_W4A16_A4_PREFILL=1 \
  "${OPT_ARGS[@]}" \
  ${FP4_EXTRA_ENV:-} \
  --entrypoint vllm eugr/spark-vllm-b12x:latest \
  serve nvidia/Qwen3.8-Flash-Next-FP4-Local \
    --served-model-name "${FP4_SERVED:-qwen3.8-flash-next-fp4-local}" \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 \
    --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \
    --dtype bfloat16 --kv-cache-dtype fp8 --quantization modelopt_mixed \
    --block-size 16 --load-format b12x \
    --max-model-len "${FP4_MAXLEN:-262144}" --max-num-seqs "${FP4_SEQS:-8}" --max-num-batched-tokens "${FP4_MNBT:-4096}" \
    --speculative-config "$SPEC" \
    --gdn-decode-kernel b12x --linear-backend "${FP4_LINEAR:-b12x}" --moe-backend b12x \
    --no-enable-flashinfer-autotune --mm-encoder-tp-mode data \
    --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
    --compilation-config "${FP4_COMP:-$COMP_DEFAULT}" \
    --gpu-memory-utilization "${FP4_GMU:-0.8}" --async-scheduling "$@"
