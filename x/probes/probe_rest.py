# Graph-time small per-layer pieces at verify M: router gate, shared-expert gate, norms.
import torch, traceback
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
h = torch.randn(M, 2560, device="cuda", dtype=torch.bfloat16)
mlps = [mods[f"language_model.model.layers.{i}.mlp"] for i in range(48)]
g0 = mlps[0].gate
out.append(f"gate {type(g0).__name__} w {g0.weight.dtype} {tuple(g0.weight.shape)} qm {type(g0.quant_method).__name__}")
for name, fn in [("router gate x48", lambda: [mm.gate(h) for mm in mlps]),
                 ("shared_expert_gate x48", lambda: [mm.shared_expert_gate(h) for mm in mlps]),
                 ("shared_expert x48", lambda: [mm.shared_expert(h) for mm in mlps])]:
    try: out.append(f"{name}: {gtime(fn):.2f} ms")
    except Exception: out.append(name + " err " + traceback.format_exc()[-300:])
types = {}
for n, mod in mods.items():
    if "mtp" in n or "visual" in n: continue
    types[type(mod).__name__] = types.get(type(mod).__name__, 0) + 1
out.append(", ".join(f"{k}:{v}" for k, v in sorted(types.items(), key=lambda x: -x[1])[:30]))
RESULT = "\n".join(out)
