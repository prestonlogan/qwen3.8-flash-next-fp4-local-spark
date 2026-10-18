# Cost of the per-call NVFP4-CSF scale expansion at decode (M=3 rows x top-10): time decode() over 48 layers with real
# routes. Call 1 hooks run_w4a16_moe to keep 3-row routes per layer (send kick.py), call 2 times.
import torch, builtins
import b12x.moe._shared.kernels.w4a16.kernel as K
r = worker.model_runner; mods = dict(r.model.named_modules())
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
if not hasattr(builtins, "_fx_csf"):
    st = builtins._fx_csf = {"orig": K.run_w4a16_moe, "ids": []}
    def hook(a_input, prepared, topk_weights, topk_ids, *args, **kw):
        if a_input.shape[0] >= 64 and len(st["ids"]) < 48 and not torch.cuda.is_current_stream_capturing():
            st["ids"].append(topk_ids[16:19].clone())
        return st["orig"](a_input, prepared, topk_weights, topk_ids, *args, **kw)
    K.run_w4a16_moe = hook
    RESULT = "hooked"
else:
    st = builtins._fx_csf; K.run_w4a16_moe = st["orig"]
    ids = st["ids"]; out = [f"{len(ids)} routes captured, dtype {ids[0].dtype if ids else None}"]
    imps = [e._b12x_prepared_experts._impl for e in exps]
    if ids:
        idl = [ids[i % len(ids)] for i in range(48)]
        t = gtime(lambda: [p.nvfp4_csf.decode(idl[i], p.w1_blockscale, p.w2_blockscale) for i, p in enumerate(imps)])
        t1 = gtime(lambda: [p.nvfp4_csf.decode(idl[i][:1], p.w1_blockscale, p.w2_blockscale) for i, p in enumerate(imps)])
        f = imps[0].nvfp4_csf
        comp = sum(t.numel() * t.element_size() for b in (f.first, f.second) for t in vars(b).values() if torch.is_tensor(t))
        out.append(f"CSF decode 48 layers: M=3 {t:.3f} ms, M=1 {t1:.3f} ms; compressed planes/layer {comp/1e6:.1f} MB; "
                   f"expanded scratch/layer {(imps[0].w1_blockscale.numel() + imps[0].w2_blockscale.numel())/1e6:.1f} MB; "
                   f"same scratch across layers: {imps[0].w1_blockscale.data_ptr() == imps[1].w1_blockscale.data_ptr()}")
    del builtins._fx_csf
    RESULT = "\n".join(out)
