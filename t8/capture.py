#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of qwen3.8-flash-next-fp4-local-spark.
"""Replay greedy sequences as prefill (cache_salt => no prefix hit) and dump the target's MTP input hidden.
usage: capture.py SRC.jsonl TAG [MAXN]   -> /dev/shm/t8fp4/TAG_NNN.{pt,json}   (dev-mode server on 5810)"""
import json, sys, uuid, http.client, subprocess, os, time
src, tag = sys.argv[1], sys.argv[2]; maxn = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = "/dev/shm/t8fp4"; os.makedirs(OUT, exist_ok=True)
def rpc(mode):
    open(f"{ROOT}/x/t8cap.txt", "w").write(mode)
    o = subprocess.run(["python3", f"{ROOT}/x/rpc.py", "fx_exec", "/x/probes/t8_cap.py"], capture_output=True, text=True).stdout
    return json.loads(o)["results"][0]
print(rpc("on sh" if tag.startswith("ho") else "on"))
t0 = time.time()
for i, r in enumerate([json.loads(l) for l in open(src)][:maxn]):
    ids = (r["pids"] + r["cids"])[:2040]
    body = {"model": "qwen3.8-flash-next-fp4-local", "prompt": ids, "max_tokens": 1, "temperature": 0, "cache_salt": uuid.uuid4().hex}
    c = http.client.HTTPConnection("127.0.0.1", 5810, timeout=600)
    c.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"}); c.getresponse().read(); c.close()
    res = rpc(f"dump {OUT}/{tag}_{i:03d}.pt fp8")
    json.dump({"ids": ids, "np": len(r["pids"])}, open(f"{OUT}/{tag}_{i:03d}.json", "w"))
    print(i, len(ids), res, f"{time.time() - t0:.0f}s", flush=True)
print(rpc("off"))
