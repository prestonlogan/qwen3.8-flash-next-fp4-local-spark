# Prefill-only calibration pass: send the t8 train sequences (prompt + model completion ids) with max_tokens 1.
import json, os, sys, urllib.request
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rows = [json.loads(l) for l in open("/home/plogan/fp4local/t8/data/train.jsonl")][:N]
tok = 0
for r in rows:
    ids = r["pids"] + r["cids"]
    req = urllib.request.Request(f"http://127.0.0.1:{os.environ.get('PORT', '5810')}/v1/completions",
        data=json.dumps({"model": os.environ.get("MODEL", "qwen3.8-flash-next-fp4-local"), "prompt": ids, "max_tokens": 1}).encode(),
        headers={"Content-Type": "application/json"})
    tok += json.loads(urllib.request.urlopen(req, timeout=600).read())["usage"]["prompt_tokens"]
print("calib tokens", tok)
