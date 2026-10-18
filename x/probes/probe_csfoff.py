# Feasibility of pre-expanded NVFP4 expert scales (no per-call CSF decode): on real 3-row inputs, compare a layer's MoE
# block with CSF vs. with nvfp4_csf=None after decode_all into its scratch, and graph-time 48 blocks both ways.
# Call 1 hooks (send kick.py), call 2 runs. Restores everything.
import torch, builtins
r = worker.model_runner; mods = dict(r.model.named_modules())
mlps = [mods[f"language_model.model.layers.{i}.mlp"] for i in range(48)]
exps = [mods[f"language_model.model.layers.{i}.mlp.experts.routed_experts"] for i in range(48)]
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
if not hasattr(builtins, "_fx_co"):
    st = builtins._fx_co = {}
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
    st = builtins._fx_co
    for mm in mlps: mm.forward = mm._fx_orig
    hs = [st[i] for i in range(48)]
    from vllm.forward_context import set_forward_context
    imps = [e._b12x_prepared_experts._impl for e in exps]
    dec = [p.nvfp4_csf for p in imps]
    out = []
    with set_forward_context(None, r.vllm_config):
        L = 5
        ref = [mlps[L](hs[L]).clone() for _ in range(2)]
        out.append(f"CSF run-to-run diff {(ref[0].float() - ref[1].float()).abs().max().item():.3g}")
        tb = gtime(lambda: [mm(hs[i]) for i, mm in enumerate(mlps)])
        try:
            for p in imps: object.__setattr__(p, "nvfp4_csf", None)
            dec[L].decode_all(imps[L].w1_blockscale, imps[L].w2_blockscale); torch.cuda.synchronize()
            y = mlps[L](hs[L])
            out.append(f"layer {L} no-CSF vs CSF: max abs {(y.float() - ref[0].float()).abs().max().item():.3g}, rel {((y.float() - ref[0].float()).norm() / ref[0].float().norm()).item():.3g}")
            ta = gtime(lambda: [mm(hs[i]) for i, mm in enumerate(mlps)])
            out.append(f"48 MoE blocks M=3: CSF {tb:.2f} ms, pre-expanded {ta:.2f} ms")
        except Exception as ex:
            import traceback; out.append(traceback.format_exc()[-1500:])
        finally:
            for p, d in zip(imps, dec): object.__setattr__(p, "nvfp4_csf", d)
    del builtins._fx_co
    RESULT = "\n".join(out)
