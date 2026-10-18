# One greedy 300-token prose request (decode timeline probes).
import json, os, sys, urllib.request
sys.path.insert(0, "/home/plogan/fp4local")
from q38bench import PROSE2
req = urllib.request.Request(f"http://127.0.0.1:{os.environ.get('PORT', '5810')}/v1/chat/completions",
    data=json.dumps({"model": os.environ.get("MODEL", "qwen3.8-flash-next-fp4-local"), "messages": [{"role": "user", "content": PROSE2[0]}],
                     "max_tokens": 300, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode(),
    headers={"Content-Type": "application/json"})
print(json.loads(urllib.request.urlopen(req, timeout=600).read())["usage"])
