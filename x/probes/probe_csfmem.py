# Bytes held by the NVFP4-CSF compressed scale planes (what pre-expanding could free), per layer and in total.
import torch
r = worker.model_runner
seen = set()
def tbytes(o, d=0):
    if id(o) in seen or d > 4: return 0
    seen.add(id(o))
    if torch.is_tensor(o): return o.untyped_storage().nbytes()
    if isinstance(o, (list, tuple)): return sum(tbytes(v, d + 1) for v in o)
    if isinstance(o, dict): return sum(tbytes(v, d + 1) for v in o.values())
    if hasattr(o, "__dict__"): return sum(tbytes(v, d + 1) for v in vars(o).values())
    if hasattr(o, "__slots__"): return sum(tbytes(getattr(o, k, None), d + 1) for k in o.__slots__)
    return 0
out = []; tot = 0; n = 0
for name, m in r.model.named_modules():
    p = getattr(m, "_b12x_prepared_experts", None)
    if p is None: continue
    dct = p._impl.nvfp4_csf
    b = tbytes(dct.first) + tbytes(dct.second) if dct is not None else 0
    tot += b; n += 1
    if n <= 2 or "mtp" in name: out.append(f"{name}: csf {b/1e6:.1f} MB, scratch ptr {p._impl.w1_blockscale.data_ptr() % 10**6}")
spec = getattr(r, "speculator", None)
if spec is not None:
    for name, m in spec.model.named_modules():
        p = getattr(m, "_b12x_prepared_experts", None)
        if p is not None: out.append(f"draft {name}: csf {'yes' if p._impl.nvfp4_csf is not None else 'no'}")
out.append(f"{n} expert layers, CSF planes total {tot/1e9:.2f} GB; expanded would be {n*157.3/1e3:.2f} GB")
RESULT = "\n".join(out)
