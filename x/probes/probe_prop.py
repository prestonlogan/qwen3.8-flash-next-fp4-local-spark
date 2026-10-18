# CUDA-event spans inside MTPSpeculator.propose (draft prefill graph, draft decode graph, whole propose).
# First call installs wrappers; second call (after a bench) reports medians.
import torch, builtins, statistics as stt
r = worker.model_runner; sp = r.speculator
ev = builtins.__dict__.setdefault("_fx_pev", None)
if ev is None:
    ev = builtins._fx_pev = {}
    tg = [(sp, "propose"), (sp, "_multi_step_decode"), (sp, "_fused_multi_step_decode"), (sp, "_prefill"), (sp, "sample_draft")]
    for mgr in ("prefill_cudagraph_manager", "decode_cudagraph_manager"):
        m_ = getattr(sp, mgr, None)
        if m_ is not None: tg.append((m_, "run_fullgraph"))
    done = []
    for obj, n in tg:
        if not hasattr(obj, n): continue
        o = getattr(obj, n); name = f"{type(obj).__name__}@{id(obj) % 1000}.{n}"
        def w(*a, _o=o, _n=name, **k):
            e0 = torch.cuda.Event(True); e0.record(); res = _o(*a, **k); e1 = torch.cuda.Event(True); e1.record()
            L = ev.setdefault(_n, [])
            if len(L) < 3000: L.append((e0, e1))
            return res
        setattr(obj, n, w); done.append(name)
    RESULT = "wrapped " + ", ".join(done) + f" | fused_multi_step={getattr(sp, 'use_fused_multi_step_decode', None)}"
else:
    torch.cuda.synchronize()
    out = []
    for k, L in ev.items():
        if len(L) > 60:
            d = [a.elapsed_time(b) for a, b in L[50:]]
            out.append(f"{k}: n={len(L)} median {stt.median(d):.3f} ms")
    RESULT = "\n".join(out)
