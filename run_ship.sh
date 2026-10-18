#!/usr/bin/env bash
# Validate the serve.sh defaults: bench (greedy/sampled/code/json), gates, 110k context. usage: run_ship.sh TAG
cd "$(dirname "$0")"; TAG=$1; export MODEL=${FP4_SERVED:-qwen3.8-flash-next-fp4-local}
docker stop -t 30 fp4local >/dev/null 2>&1; sleep 3; : > logs/mem.log
./serve.sh 5810 >/dev/null
sleep 1; nohup ./watchdog.sh >/dev/null 2>&1 < /dev/null &
t0=$(date +%s)
for i in $(seq 1 90); do curl -sf 127.0.0.1:5810/v1/models >/dev/null && break; sleep 10; done
echo "$TAG ready in $(( $(date +%s) - t0 )) s"
docker logs fp4local > logs/$TAG.log 2>&1
python3 q38bench.py --port 5810 --tag ${TAG}_G --suite prose2 --max-tokens 400 --out results/$TAG.jsonl >/dev/null
python3 q38bench.py --port 5810 --tag ${TAG}_Gcj --suite code json --max-tokens 400 --no-warmup --out results/$TAG.jsonl >/dev/null
python3 q38bench.py --port 5810 --tag ${TAG}_S --suite prose2 --max-tokens 400 --temperature 1.0 --top-p 0.95 --top-k 20 --no-warmup --out results/$TAG.jsonl >/dev/null
python3 summ.py results/$TAG.jsonl
python3 gates.py $TAG 2>&1 | tail -1
python3 t8_long.py 110000 0 ${TAG}_long110k | tee -a results/long.jsonl
echo "min MemAvailable kB: $(sort -k2 -n logs/mem.log | head -1)"
