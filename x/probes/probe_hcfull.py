# Graph-timed breakdown of the hyper-connection mixers at verify M (no profiler).
import torch, traceback, sys
sys.path.insert(0, "/x")
import fp4x_kernels as F
r = worker.model_runner; m = r.model; mods = dict(m.named_modules())
out = []
def gtime(fn, it=20):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        with torch.cuda.graph(g): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(it): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / it
M = 3
hcs = [h for n, h in mods.items() if type(h).__name__ == "GatedResidual" and "mtp" not in n]
out.append(f"GatedResidual (non-mtp): {len(hcs)}; combine {sum(h.use_combine for h in hcs)}")
h0 = hcs[0]
wd = (h0.input_mix_weight_down_block_inject if h0.use_combine else h0.input_mix_weight_down).weight
wu = h0.input_mix_weight_up.weight
out.append(f"down {tuple(wd.shape)} up {tuple(wu.shape)} lora_rank {h0.lora_rank} hc_count {h0.hc_count}")
xs = torch.randn(M, wd.shape[1], device="cuda", dtype=torch.bfloat16)
bt = torch.randn(M, wu.shape[1], device="cuda", dtype=torch.bfloat16)
Q = [(F.quantize_w8((h.input_mix_weight_down_block_inject if h.use_combine else h.input_mix_weight_down).weight), F.quantize_w8(h.input_mix_weight_up.weight)) for h in hcs]
torch.cuda.synchronize()
nb = sum(q[0][0].numel() + q[1][0].numel() for q in Q)
out.append(f"W8 bytes total {nb/1e6:.0f} MB -> {nb/243e6:.2f} ms at 243 GB/s")
cd, cu = (32, 512, 4, 4), (32, 64, 1, 8)
try:
    t = gtime(lambda: [F.w8_linear(xs, *q[0], cd) for q in Q]); out.append(f"down W8 x{len(Q)}: {t:.2f} ms ({t/len(Q)*1000:.1f} us each)")
    t = gtime(lambda: [F.w8_linear(bt, *q[1], cu) for q in Q]); out.append(f"up W8 x{len(Q)}: {t:.2f} ms ({t/len(Q)*1000:.1f} us each)")
except Exception: out.append("w8 err " + traceback.format_exc()[-600:])
try:
    from vllm.forward_context import set_forward_context
    with set_forward_context(None, r.vllm_config):
        t = gtime(lambda: [h._mix_normalized(xs) for h in hcs]); out.append(f"full _mix_normalized x{len(hcs)}: {t:.2f} ms ({t/len(hcs)*1000:.1f} us each)")
except Exception: out.append("mix err " + traceback.format_exc()[-900:])
try:
    with set_forward_context(None, r.vllm_config):
        t = gtime(lambda: [h(xs) if False else None for h in hcs[:1]])
    fw = type(h0).forward.__code__
    out.append("GatedResidual.forward args: " + str(fw.co_varnames[:fw.co_argcount]))
except Exception: out.append("fw err " + traceback.format_exc()[-400:])
RESULT = "\n".join(out)
