# Upper bound for the shared-expert path inside the decode MoE block: graph-time 48 blocks at M=3 normally, with the
# shared expert MLP stubbed to zeros, and with its sigmoid gate stubbed. Call 1 hooks (kick.py), call 2 times.
import torch, builtins
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
if not hasattr(builtins, "_fx_sh"):
    st = builtins._fx_sh = {}
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
    st = builtins._fx_sh
    for mm in mlps: mm.forward = mm._fx_orig
    hs = [st[i] for i in range(48)]
    from vllm.forward_context import set_forward_context
    out = []
    z = torch.zeros_like(hs[0]); z1 = torch.zeros((3, 1), dtype=hs[0].dtype, device=z.device)
    run = lambda: [mm(hs[i]) for i, mm in enumerate(mlps)]
    with set_forward_context(None, r.vllm_config):
        out.append(f"normal {gtime(run):.2f} ms")
        for mm in mlps: mm.shared_expert._fx_f = mm.shared_expert.forward; mm.shared_expert.forward = lambda x, *a, **k: z
        try: out.append(f"shared MLP -> 0 {gtime(run):.2f} ms")
        finally:
            for mm in mlps: mm.shared_expert.forward = mm.shared_expert._fx_f
        g0 = mlps[0].shared_expert_gate
        for mm in mlps: mm.shared_expert_gate._fx_f = mm.shared_expert_gate.forward; mm.shared_expert_gate.forward = lambda x, *a, **k: (z1, None)
        try: out.append(f"shared gate -> 0 {gtime(run):.2f} ms")
        except Exception as ex: out.append(f"gate stub failed {ex!r}"[:200])
        finally:
            for mm in mlps: mm.shared_expert_gate.forward = mm.shared_expert_gate._fx_f
        out.append(f"normal again {gtime(run):.2f} ms")
        out.append(f"runner shared_expert_gate is block's: {mlps[0].experts.shared_expert_gate is mlps[0].shared_expert_gate}, fse_fuse_gate {mlps[0].experts._fse_fuse_gate}")
    del builtins._fx_sh
    RESULT = "\n".join(out)
