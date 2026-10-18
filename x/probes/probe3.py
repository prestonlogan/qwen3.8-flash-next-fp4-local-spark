import torch, traceback
r = worker.model_runner; m = r.model
out = []
cg = r.cudagraph_manager
out.append("cg type=" + type(cg).__name__ + " attrs=" + ",".join(a for a in vars(cg))[:800])
sp = r.speculator
out.append("spec type=" + type(sp).__name__ + " attrs=" + ",".join(a for a in vars(sp))[:800])
for nm in ("graphs",):
    g = getattr(cg, nm, None)
    if g is not None: out.append(f"cg.{nm} keys={list(g.keys())[:20]}")
# MoE layers
mods = dict(m.named_modules())
moes = [(n, mod) for n, mod in mods.items() if type(mod).__name__.endswith("MoE") and "mtp" not in n]
out.append(f"moe-like modules: {len(moes)} e.g. {[ (n, type(x).__name__) for n,x in moes[:3]]}")
RESULT = "\n".join(out)
