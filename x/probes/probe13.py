import torch, time, types
r = worker.model_runner; m = r.model
mods = [mod for n, mod in m.named_modules() if hasattr(mod, "prepare_disk")]
st = {"n": 0, "cpu": 0.0, "gpu": 0.0}
for mod in mods:
    if getattr(mod, "_fx_wrapped", False): continue
    orig = mod.prepare_disk
    def wrapped(*a, _o=orig, **k):
        t0 = time.perf_counter(); e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True); e0.record()
        res = _o(*a, **k)
        e1.record(); torch.cuda.synchronize(); st["cpu"] += time.perf_counter() - t0; st["gpu"] += e0.elapsed_time(e1) / 1e3; st["n"] += 1
        return res
    mod.prepare_disk = wrapped; mod._fx_wrapped = True
import builtins; builtins._fx_ple_stats = st
RESULT = f"wrapped {len(mods)} modules: {[type(x).__name__ for x in mods]}"
