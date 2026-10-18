#!/usr/bin/env bash
# usage: exp.sh TAG "ENV_FLAGS (-e K=V ...)" [extra vllm args...]
# restarts fp4local standalone on 5810 with watchdog, then PPL + prose2 greedy + code/json greedy
cd "$(dirname "$0")"; TAG=$1; ENVF=${2//@ROOT@/$PWD}; shift 2; export FP4_OPT=${FP4_OPT:-0}
docker stop -t 30 fp4local >/dev/null 2>&1; sleep 3
: > logs/mem.log
FP4_EXTRA_ENV="$ENVF" ./serve.sh 5810 "$@" >/dev/null
sleep 1; nohup ./watchdog.sh >/dev/null 2>&1 < /dev/null &
t0=$(date +%s)
for i in $(seq 1 90); do curl -sf 127.0.0.1:5810/v1/models >/dev/null && break; docker ps -q -f name=fp4local | grep -q . || { echo "$TAG EXITED"; docker logs fp4local 2>&1 | grep -v rope_param | grep -iE "error|Traceback|raise" | tail -8 | cut -c1-300; docker logs fp4local > logs/$TAG.fail.log 2>&1; exit 1; }; sleep 10; done
echo "$TAG ready in $(( $(date +%s) - t0 )) s; min MemAvailable kB: $(sort -k2 -n logs/mem.log | head -1)"
docker logs fp4local > logs/$TAG.log 2>&1
[ -n "$NOBENCH" ] && exit 0
PORT=5810 python3 ppl.py $TAG | tee -a results/ppl.jsonl
python3 q38bench.py --port 5810 --tag ${TAG}_G --suite prose2 --max-tokens 400 --texts results/$TAG.texts.jsonl --out results/$TAG.jsonl >/dev/null
python3 q38bench.py --port 5810 --tag ${TAG}_Gcj --suite code json --max-tokens 400 --no-warmup --out results/$TAG.jsonl >/dev/null
[ -n "$SAMPLED" ] && python3 q38bench.py --port 5810 --tag ${TAG}_S --suite prose2 --max-tokens 400 --temperature 1.0 --top-p 0.95 --top-k 20 --no-warmup --out results/$TAG.jsonl >/dev/null
python3 summ.py results/$TAG.jsonl
