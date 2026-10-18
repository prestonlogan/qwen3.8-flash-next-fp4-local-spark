# Routed-MoE efficiency on real hidden states (3 consecutive prompt rows, a stand-in for a verify step). 1st call: hook each layer's MoE block to keep one M=3 input.
# 2nd call: graph-time the 48 MoE blocks on those inputs and compare with unique-expert bytes.
import torch, builtins
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
if not hasattr(builtins, "_fx_moe"):
    builtins._fx_moe = {}
    for i, mm in enumerate(mlps):
        orig = mm.forward
        def f(*a, _o=orig, _i=i, **k):
            h = a[0] if a else k.get("hidden_states")
            # decode runs inside FULL graphs (no Python), so sample 3 consecutive rows from eager prefill chunks
            if h.dim() == 2 and h.shape[0] >= 64 and not torch.cuda.is_current_stream_capturing():
                L = builtins._fx_moe.setdefault(_i, [])
                for j0 in range(16, h.shape[0] - 3, max(1, h.shape[0] // 4)):
                    if len(L) < 8: L.append(h[j0:j0 + 3].detach().clone())
            return _o(*a, **k)
        mm.forward = f; mm._fx_orig = orig
    RESULT = "hooked"
else:
    H = builtins._fx_moe
    for mm in mlps: mm.forward = mm._fx_orig
    out = [f"layers captured {len(H)}, samples {min(len(v) for v in H.values()) if H else 0}"]
    from vllm.forward_context import set_forward_context
    for j in range(min(4, min(len(v) for v in H.values()))):
        hs = [H[i][j] for i in range(48)]
        uniq = 0
        for i, mm in enumerate(mlps):
            logits = mm.gate(hs[i])
            logits = logits[0] if isinstance(logits, tuple) else logits
            uniq += torch.topk(logits.float(), 10, dim=-1).indices.unique().numel()
        with set_forward_context(None, r.vllm_config):
            t = gtime(lambda: [mm(hs[i]) for i, mm in enumerate(mlps)])
            ts = gtime(lambda: [mm.shared_expert(hs[i]) for i, mm in enumerate(mlps)])
        mb = uniq * 2.7648
        out.append(f"sample {j}: unique experts/layer {uniq/48:.1f}; MoE blocks {t:.2f} ms (shared alone {ts:.2f}); routed bytes {mb:.0f} MB -> floor {mb/243:.2f} ms")
    del builtins._fx_moe
    RESULT = "\n".join(out)
