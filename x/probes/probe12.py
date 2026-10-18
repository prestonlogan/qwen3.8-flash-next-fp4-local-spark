import torch, sys
sys.path.insert(0, "/x"); import fp4x_kernels as F
r = worker.model_runner; lm = r.model.language_model
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
h = torch.randn(4, 2560, device="cuda", dtype=torch.bfloat16)
out = [f"compute_logits (patched) {gtime(lambda: lm.compute_logits(h)):.3f} ms"]
pk = lm.lm_head.b12x_mxfp8_packed_weight; w = pk.weight.values; s = pk.weight.scale_rows.view(torch.uint8).view(w.shape[0], -1)
out.append(f"mx8 full head triton {gtime(lambda: F.mx8_linear(h, w, s, (32, 512, 1, 4))):.3f} ms")
co = F.nv4_linear(h, *[t for t in (lambda q: q)(F.quantize_nv4(F.mx8_dequant(w[:4096], s[:4096])))], (32, 256, 4))
x = torch.randn(4, 248320, device="cuda", dtype=torch.bfloat16)
out.append(f"topk512 {gtime(lambda: torch.topk(x, 512, dim=-1)):.3f} ms")
RESULT = "\n".join(out)
