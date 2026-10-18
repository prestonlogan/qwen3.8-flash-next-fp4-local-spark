# Non-routed part of the decode MoE block (b12x routed GEMMs measured at the bandwidth floor): time each piece over 48
# layers at M=3 on real hidden states. Call 1 hooks to capture inputs (send kick.py), call 2 times.
import torch, builtins
import b12x.moe._shared.kernels.w4a16.kernel as K
r = worker.model_runner; mods = dict(r.model.named_modules())
mlps = [mods[f"language_model.model.layers.{i}.mlp"] for i in range(48)]
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
if not hasattr(builtins, "_fx_mp"):
    st = builtins._fx_mp = {}
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
    st = builtins._fx_mp
    for mm in mlps: mm.forward = mm._fx_orig
    hs = [st[i] for i in range(48)]
    from vllm.forward_context import set_forward_context
    orig = K.run_w4a16_moe
    def skip(a, p, w, ids, *args, output, **kw):
        output.zero_(); return output
    out = []
    with set_forward_context(None, r.vllm_config):
        out.append(f"block {gtime(lambda: [mm(hs[i]) for i, mm in enumerate(mlps)]):.2f}")
        K.run_w4a16_moe = skip
        try:
            out.append(f"block, routed GEMMs skipped {gtime(lambda: [mm(hs[i]) for i, mm in enumerate(mlps)]):.2f}")
            out.append(f"experts() only, routed skipped {gtime(lambda: [mm.experts(hidden_states=hs[i], router_logits=hs[i]) for i, mm in enumerate(mlps)]):.2f}")
        finally:
            K.run_w4a16_moe = orig
        out.append(f"gate {gtime(lambda: [mm.gate(hs[i]) for i, mm in enumerate(mlps)]):.2f}")
        out.append(f"shared_expert {gtime(lambda: [mm.shared_expert(hs[i]) for i, mm in enumerate(mlps)]):.2f}")
        out.append(f"shared_expert_gate {gtime(lambda: [mm.shared_expert_gate(hs[i]) for i, mm in enumerate(mlps)]):.2f}")
        se = mlps[0].shared_expert
        out.append(f"shared gate_up {gtime(lambda: [mm.shared_expert.gate_up_proj(hs[i]) for i, mm in enumerate(mlps)]):.2f}")
        gu = se.gate_up_proj(hs[0]); gu = gu[0] if isinstance(gu, tuple) else gu
        out.append(f"shared gate_up out {tuple(gu.shape)}; quant {type(se.gate_up_proj.quant_method).__name__}")
        e = mlps[0].experts
        out.append(f"MoERunner attrs: {[k for k in vars(e).keys()][:40]}")
    del builtins._fx_mp
    RESULT = "\n".join(out)
