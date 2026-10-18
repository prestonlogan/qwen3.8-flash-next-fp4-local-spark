# Graph-time the 48 MoE blocks at verify M under b12x small-M env variants (read from /x/moeenv.arg: K=V lines).
import torch, os, traceback
r = worker.model_runner; m = r.model; mods = dict(m.named_modules())
out = []
def gtime(fn, it=20):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        with torch.cuda.graph(g): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(it): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / it
M = 3
torch.manual_seed(0)
h = torch.randn(M, 2560, device="cuda", dtype=torch.bfloat16)
mlps = [mods[f"language_model.model.layers.{i}.mlp"] for i in range(48)]
from vllm.forward_context import set_forward_context
variants = [l.strip() for l in open("/x/moeenv.arg") if l.strip()]
ref = None
for v in variants:
    saved = {}
    for kv in v.split():
        k, val = kv.split("=", 1); saved[k] = os.environ.get(k); os.environ[k] = val
    try:
        with set_forward_context(None, r.vllm_config):
            o = torch.stack([mm(h) for mm in mlps]).float()
            if ref is None: ref = o
            t = gtime(lambda: [mm(h) for mm in mlps])
        out.append(f"{v or 'default'}: {t:.2f} ms; maxdiff vs first {(o - ref).abs().max().item():.3g}")
    except Exception:
        out.append(f"{v} err " + traceback.format_exc()[-500:])
    for k, val in saved.items():
        if val is None: os.environ.pop(k, None)
        else: os.environ[k] = val
RESULT = "\n".join(out)
