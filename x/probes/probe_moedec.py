# Custom decode MoE kernel (x/fp4x_moe.py) vs b12x run_w4a16_moe. Call 1: hook run_w4a16_moe; on eager prefill calls
# compare our kernel on 3 rows against b12x's output, and keep 3-row MoE-block inputs per layer.
# Call 2: report errors; graph-time the 48 MoE blocks at M=3 with b12x vs ours.
import torch, builtins, sys, importlib
sys.path.insert(0, "/x")
import fp4x_moe; importlib.reload(fp4x_moe)
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
def ours(a, p, w, ids, out):
    cfg = dict(builtins._fx_md["cfg"]); fn = getattr(fp4x_moe, cfg.pop("fn", "moe_decode2"))
    return fn(a, ids, w, p.w13, p.w13_scale, p.w13_global_scale, p.w2, p.w2_scale, p.w2_global_scale,
                               out_bf16=out, **cfg)
if not hasattr(builtins, "_fx_md"):
    st = builtins._fx_md = {"orig": K.run_w4a16_moe, "err": [], "cmp": [], "h": {}, "mode": "cmp", "cfg": {}, "attrs": None}
    def hook(a_input, prepared, topk_weights, topk_ids, *args, output, **kw):
        m = a_input.shape[0]
        if st["mode"] == "skip" and m <= 16:
            output.zero_(); return output
        if st["mode"] == "ours" and m <= 16:
            ours(a_input, prepared, topk_weights, topk_ids, output); return output
        res = st["orig"](a_input, prepared, topk_weights, topk_ids, *args, output=output, **kw)
        if st["mode"] == "cmp" and m >= 64 and len(st.setdefault("cap", [])) < 48 and not torch.cuda.is_current_stream_capturing():
            j = 16; st["cap"].append((a_input[j:j + 3].clone(), topk_ids[j:j + 3].clone(), topk_weights[j:j + 3].clone(), prepared))
        if st["mode"] == "cmp" and m >= 64 and len(st["cmp"]) < 96 and not torch.cuda.is_current_stream_capturing():
            if st["attrs"] is None:
                st["attrs"] = [f"{k}:{tuple(v.shape) if torch.is_tensor(v) else type(v).__name__}" for k, v in vars(prepared).items()] if hasattr(prepared, "__dict__") else dir(prepared)
            try:
                j = 16; o3 = torch.empty_like(output[j:j + 3])
                f = ours(a_input[j:j + 3].contiguous(), prepared, topk_weights[j:j + 3].contiguous(), topk_ids[j:j + 3].contiguous(), o3)
                ref = output[j:j + 3].float()
                st["cmp"].append(((f - ref).norm() / ref.norm()).item())
                st["cmp_r"] = (f.flatten()[:4].tolist(), ref.flatten()[:4].tolist(), (f.norm() / ref.norm()).item())
            except Exception as ex:
                st["err"].append(repr(ex)[:300])
        return res
    K.run_w4a16_moe = hook
    for i, mm in enumerate(mlps):
        orig = mm.forward
        def f(*a, _o=orig, _i=i, **k):
            h = a[0] if a else k.get("hidden_states")
            if h.dim() == 2 and h.shape[0] >= 64 and not torch.cuda.is_current_stream_capturing() and _i not in st["h"]:
                st["h"][_i] = h[16:19].detach().clone()
            return _o(*a, **k)
        mm.forward = f; mm._fx_orig = orig
    RESULT = "hooked"
else:
    st = builtins._fx_md
    for mm in mlps: mm.forward = mm._fx_orig
    out = [f"attrs {st['attrs']}", f"errors {st['err'][:3]}", f"n cmp {len(st['cmp'])}"]
    if st["cmp"]:
        c = st["cmp"]; out.append(f"rel err mean {sum(c)/len(c):.5f} max {max(c):.5f}; sample {st.get('cmp_r')}")
    from vllm.forward_context import set_forward_context
    if len(st["h"]) == 48 and not st["err"]:
        hs = [st["h"][i] for i in range(48)]
        with set_forward_context(None, r.vllm_config):
            st["mode"] = "off"; tb = gtime(lambda: [mm(hs[i]) for i, mm in enumerate(mlps)])
            yb = [mm(hs[i]).float() for i, mm in enumerate(mlps)]
            res = []
            st["mode"] = "skip"; res.append(f"routed skipped: {gtime(lambda: [mm(hs[i]) for i, mm in enumerate(mlps)]):.2f} ms")
            for cfg in builtins.__dict__.get("_fx_md_cfgs", [{}]):
                st["mode"] = "ours"; st["cfg"] = cfg
                to = gtime(lambda: [mm(hs[i]) for i, mm in enumerate(mlps)])
                yo = [mm(hs[i]).float() for i, mm in enumerate(mlps)]
                e = max(((a - b).norm() / b.norm()).item() for a, b in zip(yo, yb))
                res.append(f"cfg {cfg}: ours {to:.2f} ms  block rel err max {e:.4f}")
        out.append(f"48 MoE blocks M=3: b12x {tb:.2f} ms"); out += res
        cap = st["cap"]; U = sum(c[1].unique().numel() for c in cap)
        out.append(f"{len(cap)} calls, unique experts/layer {U/len(cap):.1f}; floor up {U*1.843/243:.2f} down {U*0.922/243:.2f} ms (243 GB/s)")
        F = fp4x_moe
        for cfg in builtins.__dict__.get("_fx_md_kcfgs", [dict(KT=4, nw=4, ns=3)]):
            bufs = []
            for (x, ids, w, p) in cap:
                m = x.shape[0]; ids_f = ids.reshape(-1).to(torch.int32).contiguous()
                act = torch.empty((m * 10, 16, 640), dtype=torch.bfloat16, device=x.device)
                o = torch.zeros((m, 2560), dtype=torch.float32, device=x.device)
                bufs.append((x, ids_f, w.reshape(-1).float().contiguous(), p, act, o))
            def up3():
                for (x, ids_f, w, p, act, o) in bufs:
                    F._moe_up3[(30, 20)](x, ids_f, p.w13, p.w13_scale.view(torch.uint8), p.w13_global_scale, gu, 3, x.stride(0),
                                         TOPK=10, I=640, K=2560, KT=cfg["KT"], num_warps=cfg["nw"], num_stages=cfg["ns"])
            def dn3():
                for (x, ids_f, w, p, act, o) in bufs:
                    F._moe_down3[(30, 40)](gu, ids_f, w, p.w2, p.w2_scale.view(torch.uint8), p.w2_global_scale, o, 3,
                                           TOPK=10, I=640, H=2560, KT=cfg["KT"], num_warps=cfg["nw"], num_stages=cfg["ns"])
            gu = torch.zeros((30, 16, 1280), dtype=torch.bfloat16, device="cuda")
            out.append(f"v3 {cfg}: up {gtime(up3):.3f} ms, down {gtime(dn3):.3f} ms")
            def up():
                for (x, ids_f, w, p, act, o) in bufs:
                    F._moe_up2[(30, 10)](x, ids_f, p.w13, p.w13_scale.view(torch.uint8), p.w13_global_scale, act, 3, x.stride(0),
                                         TOPK=10, I=640, K=2560, KT=cfg["KT"], num_warps=cfg["nw"], num_stages=cfg["ns"])
            def dn():
                for (x, ids_f, w, p, act, o) in bufs:
                    F._moe_down2[(30, 40)](act, ids_f, w, p.w2, p.w2_scale.view(torch.uint8), p.w2_global_scale, o, 3,
                                           TOPK=10, I=640, H=2560, KT=cfg["KT"], num_warps=cfg["nw"], num_stages=cfg["ns"])
            out.append(f"v2 {cfg}: up {gtime(up):.3f} ms, down {gtime(dn):.3f} ms")
    K.run_w4a16_moe = st["orig"]
    del builtins._fx_md
    RESULT = "\n".join(out)
