# Apply FP4X_MOE_EXPAND's transformation in-process (affects eager prefill only; captured decode graphs keep CSF).
import torch, gc
r = worker.model_runner; n = 0
m0 = torch.cuda.memory_allocated()
for name, mod in list(r.model.named_modules()):
    p = getattr(mod, "_b12x_prepared_experts", None)
    dec = getattr(getattr(p, "_impl", None), "nvfp4_csf", None)
    if dec is None: continue
    impl = p._impl
    s1 = torch.empty_like(impl.w1_blockscale); s2 = torch.empty_like(impl.w2_blockscale)
    dec.decode_all(s1, s2)
    object.__setattr__(impl, "w1_blockscale", s1); object.__setattr__(impl, "w2_blockscale", s2)
    object.__setattr__(impl, "nvfp4_csf", None)
    n += 1
torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
RESULT = f"expanded {n} layers, {(torch.cuda.memory_allocated() - m0)/1e9:+.2f} GB"
