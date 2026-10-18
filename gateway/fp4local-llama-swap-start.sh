#!/usr/bin/env bash
# llama-swap adapter: NVIDIA Qwen3.8-Flash-Next-FP4-Local (b12x vLLM) via serve.sh in $FP4_ROOT (default ~/fp4local)
set -euo pipefail
port=${1:?llama-swap backend port required}
cd "${FP4_ROOT:-$HOME/fp4local}"
FP4_NAME=fp4local FP4_SERVED=qwen3.8-flash-next-fp4-local ./serve.sh "$port" >/dev/null
nohup ./watchdog.sh >/dev/null 2>&1 < /dev/null &
exec docker wait fp4local
