import torch
m = worker.model_runner.model
out = []
lay = None
for n, mod in m.named_modules():
    if hasattr(mod, "_b12x_prepared_experts") and "layers.0." in n:
        lay = mod; out.append(f"layer {n} {type(mod).__name__}"); break
p = lay._b12x_prepared_experts
def walk(o, pre, depth=0):
    if depth > 3: return
    if torch.is_tensor(o):
        out.append(f"{pre}: {tuple(o.shape)} {o.dtype} stride={o.stride()} ptr%256={o.data_ptr()%256}"); return
    d = getattr(o, "__dict__", None)
    if d is None and hasattr(o, "__slots__"): d = {k: getattr(o, k, None) for k in o.__slots__}
    if isinstance(o, (list, tuple)): d = {str(i): v for i, v in enumerate(o)}
    if isinstance(o, dict): d = o
    if not d: out.append(f"{pre}: {type(o).__name__} {str(o)[:80]}"); return
    out.append(f"{pre}: <{type(o).__name__}>")
    for k, v in list(d.items())[:40]:
        walk(v, f"{pre}.{k}", depth + 1)
walk(p, "prep")
RESULT = "\n".join(out)[:7000]
