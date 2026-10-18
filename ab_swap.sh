#!/usr/bin/env bash
# In-process A/B of a hot-swapped MTP patch: ab_swap.sh TAG PATCH ROUNDS  (server on 5810 in dev mode)
cd "$(dirname "$0")"; TAG=$1; PATCH=$2; R=${3:-2}; OUT=results/$TAG.jsonl
swap() { echo "$1" > x/mtp_swap.arg; python3 x/rpc.py fx_exec /x/probes/mtp_swap.py > /dev/null; }
for i in $(seq 1 $R); do
  swap "apply $PATCH"; python3 q38bench.py --port 5810 --tag ${TAG}_P --suite prose2 --max-tokens 400 --out $OUT --note r$i >/dev/null
  swap restore;        python3 q38bench.py --port 5810 --tag ${TAG}_B --suite prose2 --max-tokens 400 --no-warmup --out $OUT --note r$i >/dev/null
done
swap "apply $PATCH"; python3 q38bench.py --port 5810 --tag ${TAG}_PS --suite prose2 --max-tokens 400 --temperature 1.0 --top-p 0.95 --top-k 20 --no-warmup --out $OUT >/dev/null
python3 q38bench.py --port 5810 --tag ${TAG}_Pcj --suite code json --max-tokens 400 --no-warmup --out $OUT >/dev/null
swap restore; python3 q38bench.py --port 5810 --tag ${TAG}_BS --suite prose2 --max-tokens 400 --temperature 1.0 --top-p 0.95 --top-k 20 --no-warmup --out $OUT >/dev/null
python3 q38bench.py --port 5810 --tag ${TAG}_Bcj --suite code json --max-tokens 400 --no-warmup --out $OUT >/dev/null
python3 summ.py $OUT
