import time, builtins, torch
r = worker.model_runner
names = [n for n in ("execute_model", "sample_tokens", "propose_draft", "_sample", "postprocess") if hasattr(r, n)]
st = builtins.__dict__.setdefault("_fx_rt", {})
for n in names:
    if getattr(r, "_fx_w_" + n, False): continue
    o = getattr(r, n)
    def w(*a, _o=o, _n=n, **k):
        t0 = time.perf_counter(); res = _o(*a, **k); dt = time.perf_counter() - t0
        s = st.setdefault(_n, [0, 0.0]); s[0] += 1; s[1] += dt
        return res
    setattr(r, n, w); setattr(r, "_fx_w_" + n, True)
st.clear()
RESULT = f"wrapped {names}"
