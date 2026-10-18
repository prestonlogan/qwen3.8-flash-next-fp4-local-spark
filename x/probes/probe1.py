# Inspect the worker/model runner structure and time real-weight kernels.
import torch, time, sys
sys.path.insert(0, "/x")
r = worker.model_runner
out = []
out.append("runner=" + type(r).__name__ + " attrs=" + ",".join(a for a in dir(r) if "graph" in a.lower() or "spec" in a.lower() or "capture" in a.lower()))
m = r.model
out.append("model=" + type(m).__name__)
names = [n for n, _ in m.named_modules()]
out.append("n_modules=%d" % len(names))
out.append("sample=" + " | ".join(n for n in names if n.endswith("layers.0.attn_hyper_connection") or n.endswith("layers.0.linear_attn.in_proj_qkvz") or n.endswith("layers.0.mlp") or "lm_head" in n)[:600])
RESULT = "\n".join(out)
