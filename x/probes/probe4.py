import torch
r = worker.model_runner
out = []
def timeit(fn, it=20):
    fn(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(it): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / it
def replayer(g):
    if hasattr(g, "replay"): return g.replay
    for a in ("graph", "cuda_graph", "cudagraph"):
        if hasattr(g, a): return getattr(g, a).replay
    if isinstance(g, (list, tuple)):
        fs = [replayer(x) for x in g]; return lambda: [f() for f in fs]
    raise TypeError(type(g).__name__ + str([a for a in dir(g) if not a.startswith("__")])[:300])
for k, g in r.cudagraph_manager.graphs.items():
    try: out.append(f"target tokens={k.num_tokens} reqs={k.num_reqs}: {timeit(replayer(g)):.2f} ms ({type(g).__name__})")
    except Exception as e: out.append(f"target {k.num_tokens}: {e}")
sp = r.speculator
for nm in ("decode_cudagraph_manager", "prefill_cudagraph_manager"):
    mg = getattr(sp, nm, None)
    if mg is None: continue
    for k, g in list(mg.graphs.items())[:6]:
        try: out.append(f"{nm} {getattr(k,'num_tokens',k)} reqs={getattr(k,'num_reqs','?')} uniform={getattr(k,'uniform_token_count','?')}: {timeit(replayer(g)):.2f} ms")
        except Exception as e: out.append(f"{nm}: {e}")
RESULT = "\n".join(out)
