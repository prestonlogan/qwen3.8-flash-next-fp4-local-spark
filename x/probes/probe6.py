import torch, traceback
r = worker.model_runner; sp = r.speculator
out = []
def gtime(fn, it=50):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        with torch.cuda.graph(g): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(it): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / it
dm = getattr(sp, "model", None)
out.append("draft model: " + type(dm).__name__)
try:
    lm = dm.lm_head
    out.append(f"draft lm_head qm={type(lm.quant_method).__name__} rlq={getattr(lm,'runtime_lm_head_quantization',None)} params={[ (n, tuple(p.shape), p.dtype) for n,p in lm.named_parameters()][:6]}")
    out.append("lm attrs: " + ",".join(a for a in vars(lm) if not a.startswith("_"))[:600])
    for M in (1, 4):
        h = torch.randn(M, 2560, device="cuda", dtype=torch.bfloat16)
        out.append(f"draft compute_logits M={M}: {gtime(lambda: dm.compute_logits(h)):.3f} ms")
    tm = r.model
    for M in (4,):
        h = torch.randn(M, 2560, device="cuda", dtype=torch.bfloat16)
        out.append(f"target compute_logits M={M}: {gtime(lambda: tm.compute_logits(h)):.3f} ms")
    out.append("sampler-ish attrs: " + ",".join(a for a in dir(sp) if "sample" in a or "draft" in a)[:500])
except Exception: out.append(traceback.format_exc()[-800:])
RESULT = "\n".join(out)
