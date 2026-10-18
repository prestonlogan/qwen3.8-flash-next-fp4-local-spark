# Experiment: delay the verify-graph launch by builtins._fx_vdelay_us of CPU busy-wait (the GPU is idle then).
import builtins, time
r = worker.model_runner
m = r.cudagraph_manager
if not hasattr(m, "_fx_orig_run"):
    m._fx_orig_run = m.run_fullgraph
    def f(*a, _o=m._fx_orig_run, **k):
        d = builtins.__dict__.get("_fx_vdelay_us", 0)
        if d:
            t = time.perf_counter() + d * 1e-6
            while time.perf_counter() < t: pass
        return _o(*a, **k)
    m.run_fullgraph = f
builtins._fx_vdelay_us = int(open("/x/probes/vdelay.txt").read())
RESULT = f"V delay {builtins._fx_vdelay_us} us"
