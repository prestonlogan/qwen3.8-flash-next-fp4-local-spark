#!/usr/bin/env bash
# usage: quick.sh TAG  -> prose2 greedy, prose2 T1.0, code/json greedy at S=1
cd "$(dirname "$0")"; T=$1; S="--temperature 1.0 --top-p 0.95 --top-k 20"
python3 q38bench.py --port ${PORT:-5810} --tag ${T}_G --suite prose2 --max-tokens 400 --out results/${T}.jsonl >/dev/null &&
python3 q38bench.py --port ${PORT:-5810} --tag ${T}_S --suite prose2 --max-tokens 400 $S --no-warmup --out results/${T}.jsonl >/dev/null &&
python3 q38bench.py --port ${PORT:-5810} --tag ${T}_Gcj --suite code json --max-tokens 400 --no-warmup --out results/${T}.jsonl >/dev/null &&
python3 summ.py results/${T}.jsonl
