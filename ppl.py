#!/usr/bin/env python3
"""Teacher-forced PPL over fixed texts via /v1/completions prompt_logprobs. usage: ppl.py TAG"""
import json, math, os, sys, urllib.request
P = os.environ.get("PORT", "5810"); M = os.environ.get("MODEL", "qwen3.8-flash-next-fp4-local")
T = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppl_texts.json")))
nll = n = 0
for t in T:
    b = {"model": M, "prompt": t, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{P}/v1/completions", json.dumps(b).encode(), {"content-type": "application/json"}), timeout=600))
    for d in r["choices"][0]["prompt_logprobs"][1:]:
        if d: lp = max(v["logprob"] for v in d.values()) if len(d) == 1 else next(iter(d.values()))["logprob"]; nll -= lp; n += 1
print(json.dumps({"tag": sys.argv[1] if len(sys.argv) > 1 else "", "tokens": n, "ppl": round(math.exp(nll / n), 4)}))
