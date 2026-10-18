# Verify the reduced-vocab NVFP4 kernel against the real draft head and time it.
import torch, traceback, sys
sys.path.insert(0, "/x"); import importlib, fp4x_kernels as F; importlib.reload(F)
r = worker.model_runner; dm = r.speculator.model; lm = dm.lm_head
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
try:
    ids = sorted({int(l) for l in open("/x/dv65k.txt") if l.strip()})
    idx = torch.tensor(ids, device="cuda")
    w = lm.weight.data; sc = lm.weight_scale.data; n, kb = sc.shape
    scu = sc.view(torch.uint8).reshape(n // 128, kb // 4, 32, 4, 4).transpose(1, 3).reshape(n, kb)
    W = w.index_select(0, idx).contiguous(); S = scu.index_select(0, idx).contiguous().view(torch.float8_e4m3fn); gs = float(lm.weight_global_scale.item())
    h = torch.randn(1, 2560, device="cuda", dtype=torch.bfloat16)
    ref = dm.compute_logits(h).float()[:, idx]
    for cfg in ((32, 256, 4), (16, 256, 4), (64, 256, 4), (32, 512, 4), (16, 512, 2), (64, 128, 4), (32, 128, 2)):
        o = F.nv4_linear(h, W, S, gs, cfg).float()
        err = ((o - ref).abs().max() / ref.abs().max()).item()
        out.append(f"cfg {cfg}: relerr {err:.2e}  {gtime(lambda: F.nv4_linear(h, W, S, gs, cfg)):.3f} ms (ids {len(ids)}, {(W.numel()+S.numel())/1e6:.0f} MB)")
    full = torch.empty(1, n, device="cuda", dtype=torch.bfloat16)
    out.append(f"scatter -inf fill+index_copy: {gtime(lambda: full.fill_(float('-inf')).index_copy_(1, idx, ref.to(torch.bfloat16))):.3f} ms")
    out.append(f"orig head: {gtime(lambda: dm.compute_logits(h)):.3f} ms")
except Exception: out.append(traceback.format_exc()[-1200:])
RESULT = "\n".join(out)
