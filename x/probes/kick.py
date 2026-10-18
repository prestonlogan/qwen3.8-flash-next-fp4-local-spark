# Send one uncached ~1.5k-token prompt (random prefix defeats the prefix cache) so probes see eager prefill calls.
import json, os, random, sys, urllib.request
sys.path.insert(0, "/home/plogan/fp4local")
from q38bench import PROSE2
txt = f"[{random.random():.12f}] " + " ".join(PROSE2) * 3
req = urllib.request.Request(f"http://127.0.0.1:{os.environ.get('PORT', '5810')}/v1/completions",
    data=json.dumps({"model": os.environ.get("MODEL", "qwen3.8-flash-next-fp4-local"), "prompt": txt, "max_tokens": 4}).encode(),
    headers={"Content-Type": "application/json"})
print(json.loads(urllib.request.urlopen(req, timeout=600).read())["usage"])
