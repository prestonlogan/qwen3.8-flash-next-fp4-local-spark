#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of qwen3.8-flash-next-fp4-local-spark.
"""Drafter self-distillation corpus: the target's own GREEDY prose with exact token ids.
Prompts come from t8/prompts.json (the companion repo's T8 sets, disjoint from the sealed PROSE2 benchmark).
usage: gen_tok.py SET OUT.jsonl [--mt 900] [--conc 4]   (resumable)"""
import argparse, json, os, threading, http.client, time
ap = argparse.ArgumentParser(); ap.add_argument("set"); ap.add_argument("out")
ap.add_argument("--mt", type=int, default=900); ap.add_argument("--conc", type=int, default=4)
ap.add_argument("--port", type=int, default=5810); ap.add_argument("--model", default="qwen3.8-flash-next-fp4-local")
a = ap.parse_args()
P = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")))[a.set]
if os.path.exists(a.out):
    done = set(json.loads(l)["prompt"] for l in open(a.out) if l.strip()); P = [p for p in P if p not in done]
def one(prompt, lock):
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": a.mt, "temperature": 0,
            "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": False}}
    c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=1800)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    d = json.loads(c.getresponse().read()); c.close()
    with lock, open(a.out, "a") as f:
        f.write(json.dumps({"prompt": prompt, "pids": d["prompt_token_ids"], "cids": d["choices"][0]["token_ids"]}) + "\n")
lock = threading.Lock(); t0 = time.time()
for k in range(0, len(P), a.conc):
    ths = [threading.Thread(target=one, args=(p, lock)) for p in P[k:k + a.conc]]
    [t.start() for t in ths]; [t.join() for t in ths]
    print(f"{k + len(ths)}/{len(P)} {time.time() - t0:.0f}s", flush=True)
