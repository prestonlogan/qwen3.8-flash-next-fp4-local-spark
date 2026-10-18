import torch, builtins
r = worker.model_runner; sp = r.speculator; cg = r.cudagraph_manager
ev = builtins.__dict__.setdefault("_fx_ev", {})
ev.clear()
targets = [(r, n) for n in ("prepare_inputs", "prepare_attn", "sample", "postprocess_sampled", "sample_tokens")] + [(sp, "propose")]
targets += [(cg, n) for n in dir(type(cg)) if any(k in n for k in ("run", "replay", "forward", "execute")) and not n.startswith("__")]
targets += [(r.model, "compute_logits")]
done = []
for obj, n in targets:
    if not hasattr(obj, n) or getattr(obj, "_fxe_" + n, False): continue
    o = getattr(obj, n)
    if not callable(o): continue
    def w(*a, _o=o, _n=f"{type(obj).__name__}.{n}", **k):
        e0 = torch.cuda.Event(True); e0.record(); res = _o(*a, **k); e1 = torch.cuda.Event(True); e1.record()
        L = ev.setdefault(_n, [])
        if len(L) < 4000: L.append((e0, e1))
        return res
    setattr(obj, n, w); setattr(obj, "_fxe_" + n, True); done.append(f"{type(obj).__name__}.{n}")
RESULT = "wrapped " + ", ".join(done)
