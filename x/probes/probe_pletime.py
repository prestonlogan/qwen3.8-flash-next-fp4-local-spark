# Wall time of B12xNGramEmbedding.prepare_disk per decode step (1st call wraps, 2nd call reports).
import time, builtins, statistics as stt
r = worker.model_runner
emb = [m for n, m in r.model.named_modules() if type(m).__name__ == "B12xNGramEmbedding"][0]
if not hasattr(builtins, "_fx_ple"):
    builtins._fx_ple = []
    orig = emb.prepare_disk
    def w(*a, _o=orig, **k):
        t0 = time.perf_counter(); res = _o(*a, **k); builtins._fx_ple.append(time.perf_counter() - t0); return res
    emb.prepare_disk = w
    RESULT = "wrapped"
else:
    L = builtins._fx_ple[20:]
    L.sort()
    RESULT = f"n={len(L)} prepare_disk wall ms: median {stt.median(L)*1e3:.3f} p10 {L[len(L)//10]*1e3:.3f} p90 {L[9*len(L)//10]*1e3:.3f}"
    builtins._fx_ple = []
