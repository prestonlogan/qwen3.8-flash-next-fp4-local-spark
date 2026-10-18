# Fused shared-expert tail (fp4x_kernels.w6_sdown) vs the stock path on real 3-row inputs: exactness per layer and
# graph time for 48 layers. Call 1 hooks (kick.py), call 2 runs.
import torch, builtins, importlib.util
spec = importlib.util.spec_from_file_location("fp4x_k2", "/x/fp4x_kernels.py"); F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
r = worker.model_runner; mods = dict(r.model.named_modules())
mlps = [mods[f"language_model.model.layers.{i}.mlp"] for i in range(48)]
def gtime(fn, it=30):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        with torch.cuda.graph(g): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(it): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / it
if not hasattr(builtins, "_fx_sf"):
    st = builtins._fx_sf = {}
    for i, mm in enumerate(mlps):
        orig = mm.forward
        def f(*a, _o=orig, _i=i, **k):
            h = a[0] if a else k.get("hidden_states")
            if h.dim() == 2 and h.shape[0] >= 64 and not torch.cuda.is_current_stream_capturing() and _i not in st:
                st[_i] = h[16:19].detach().clone()
            return _o(*a, **k)
        mm.forward = f; mm._fx_orig = orig
    RESULT = "hooked"
else:
    st = builtins._fx_sf
    for mm in mlps: mm.forward = mm._fx_orig
    hs = [st[i] for i in range(48)]
    out = []
    qs = []
    for mm in mlps:
        pk = mm.shared_expert.down_proj.b12x_linear.packed
        w = pk.weight.values.view(torch.float8_e4m3fn); s = pk.weight.scale_rows.view(torch.uint8).view(w.shape[0], -1)
        qs.append(F.quantize_w6(F.mx8_dequant(w, s)))
    ses = [mm.shared_expert for mm in mlps]
    def fused(i, cfg=(32, 128, 2)):
        gu, _ = ses[i].gate_up_proj(hs[i])
        return F.w6_sdown(gu, *qs[i], hs[i], ses[i].expert_gate.weight, cfg)
    with torch.no_grad():
        nbad = 0; worst = 0.0
        for i in range(48):
            a = ses[i](hs[i]); b = fused(i)
            if not torch.equal(a, b):
                nbad += 1; worst = max(worst, ((a.float() - b.float()).abs().max() / a.float().abs().max()).item())
        out.append(f"layers not bit-equal: {nbad}/48, worst max-rel {worst:.3g}")
        out.append(f"stock shared expert x48: {gtime(lambda: [ses[i](hs[i]) for i in range(48)]):.3f} ms")
        for cfg in [(32, 128, 2), (16, 128, 2), (32, 64, 2), (64, 128, 4), (32, 128, 4), (16, 64, 1)]:
            out.append(f"fused {cfg}: {gtime(lambda: [fused(i, cfg) for i in range(48)]):.3f} ms")
        from vllm.forward_context import set_forward_context
        from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
        ofwd = Qwen2MoeMLP.forward; qmap = {id(ses[i]): qs[i] for i in range(48)}
        run = lambda: [mm(hs[i]) for i, mm in enumerate(mlps)]
        with set_forward_context(None, r.vllm_config):
            out.append(f"blocks stock {gtime(run):.3f} ms")
            for cfg in [(64, 128, 4), (32, 128, 2)]:
                def fwd(self, x, _c=cfg):
                    q = qmap.get(id(self))
                    if q is None or x.shape[0] > 16: return ofwd(self, x)
                    gu, _ = self.gate_up_proj(x)
                    return F.w6_sdown(gu, *q, x, self.expert_gate.weight, _c)
                Qwen2MoeMLP.forward = fwd
                try:
                    ys = [mm(hs[i]) for i, mm in enumerate(mlps)]
                    out.append(f"blocks fused {cfg} {gtime(run):.3f} ms")
                finally:
                    Qwen2MoeMLP.forward = ofwd
            out.append(f"blocks stock again {gtime(run):.3f} ms")
    del builtins._fx_sf
    RESULT = "\n".join(out)
