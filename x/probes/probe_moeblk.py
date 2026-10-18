# Structure of the decode MoE block (layer 0): module tree + forward source.
import inspect
r = worker.model_runner; mods = dict(r.model.named_modules())
mm = mods["language_model.model.layers.0.mlp"]
out = [type(mm).__module__ + "." + type(mm).__name__]
for n, c in mm.named_modules():
    if n.count(".") <= 1: out.append(f"  {n}: {type(c).__name__} " + ", ".join(f"{k}{tuple(p.shape)}:{p.dtype}" for k, p in list(c.named_parameters(recurse=False))[:4]))
f = getattr(mm.forward, "_fx_orig", None) or getattr(mm, "_fx_orig", None) or mm.forward
try: out.append(inspect.getsource(type(mm).forward))
except Exception as ex: out.append(repr(ex))
RESULT = "\n".join(out)[:9000]
