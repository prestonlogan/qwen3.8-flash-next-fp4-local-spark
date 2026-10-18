#!/usr/bin/env bash
# Decode-path quality: serve with 16-token prefill chunks so prompt tokens run through the M<=16 kernels,
# then teacher-forced PPL. usage: exp_q.sh TAG "ENV_FLAGS" [extra vllm args...]
cd "$(dirname "$0")"; TAG=$1; ENVF=${2//@ROOT@/$PWD}; shift 2; export FP4_OPT=${FP4_OPT:-0}
docker stop -t 30 fp4local >/dev/null 2>&1; sleep 3; : > logs/mem.log
FP4_SEQS=1 FP4_MNBT=16 FP4_EXTRA_ENV="$ENVF" ./serve.sh 5810 --language-model-only --no-enable-prefix-caching "$@" >/dev/null
sleep 1; nohup ./watchdog.sh >/dev/null 2>&1 < /dev/null &
for i in $(seq 1 90); do curl -sf 127.0.0.1:5810/v1/models >/dev/null && break; docker ps -q -f name=fp4local | grep -q . || { echo "$TAG EXITED"; docker logs fp4local 2>&1 | grep -v rope_param | grep -iE "error|raise" | tail -6 | cut -c1-300; exit 1; }; sleep 10; done
PORT=5810 python3 ppl.py Q_$TAG | tee -a results/ppl_q.jsonl
