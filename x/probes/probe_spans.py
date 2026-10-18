# GPU spans of the eager (non-graph) parts of a decode step: CUDA events at entry/exit of runner methods.
# Spans of nested methods overlap. Call 1 hooks; send kick2.py; call 2 reports per-call averages.
import torch, builtins, collections
r = worker.model_runner
targets = [(r, n) for n in ("sample", "postprocess_sampled", "postprocess_num_computed_tokens", "prepare_inputs",
                            "_prepare_inputs", "prepare_model_inputs", "execute_model", "sample_tokens")]
targets += [(r.model, "compute_logits"), (r.speculator, "propose"), (r.model_state, "prepare_inputs"),
            (r.model_state, "postprocess_state"), (r.speculator, "sample_draft")]
for n in ("rejection_sampler", "sampler"):
    o = getattr(r, n, None)
    if o is not None: targets.append((o, "__call__" if not hasattr(o, "forward") else "forward"))
if not hasattr(builtins, "_fx_sp"):
    st = builtins._fx_sp = {"ev": collections.defaultdict(list), "orig": []}
    hooked = []
    for obj, name in targets:
        o = getattr(obj, name, None)
        if o is None or not callable(o): continue
        key = f"{type(obj).__name__}.{name}"
        def f(*a, _o=o, _k=key, **k):
            e0 = torch.cuda.Event(enable_timing=True); e0.record()
            res = _o(*a, **k)
            e1 = torch.cuda.Event(enable_timing=True); e1.record()
            L = st["ev"][_k]
            if len(L) < 3000: L.append((e0, e1))
            return res
        try:
            setattr(obj, name, f); st["orig"].append((obj, name, o)); hooked.append(key)
        except Exception as ex:
            hooked.append(f"!{key}:{ex}")
    RESULT = "hooked " + ", ".join(hooked)
else:
    st = builtins._fx_sp
    for obj, name, o in st["orig"]: setattr(obj, name, o)
    torch.cuda.synchronize()
    out = []
    for k, L in st["ev"].items():
        L = L[10:]
        if not L: continue
        v = sorted(e0.elapsed_time(e1) for e0, e1 in L)
        out.append(f"{k:45s} n={len(L):4d} median {v[len(v)//2]:.3f} ms  mean {sum(v)/len(v):.3f}")
    del builtins._fx_sp
    RESULT = "\n".join(out)
