#!/usr/bin/env bash
# CUDA-event span accounting of the decode step on the running dev server (5810). usage: spans.sh TAG
cd "$(dirname "$0")"; until curl -sf 127.0.0.1:5810/v1/models >/dev/null; do sleep 5; done
python3 x/rpc.py fx_exec /x/probes/probe16.py >/dev/null
python3 q38bench.py --port 5810 --tag SP_$1 --suite prose2 --max-tokens 400 --no-warmup --out /tmp/sp_$1.jsonl >/dev/null
python3 summ.py /tmp/sp_$1.jsonl
for p in probe16b probe16c probe16d; do python3 x/rpc.py fx_exec /x/probes/$p.py | python3 -c "import json,sys;print(json.load(sys.stdin)['results'][0])"; done
