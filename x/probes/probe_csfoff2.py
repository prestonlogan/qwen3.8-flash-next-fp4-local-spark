# Debug pre-expansion: on layer L, compare the eager 3-row MoE block (CSF) with (a) scales expanded into separate tensors
# and nvfp4_csf=None, (b) additionally with the compressed planes' parameters freed. Call 1 hooks (kick.py), call 2 runs.
import torch, builtins, gc
r = worker.model_runner; mods = dict(r.model.named_modules())
L = 5
mm = mods[f"language_model.model.layers.{L}.mlp"]; ex = mods[f"language_model.model.layers.{L}.mlp.experts.routed_experts"]
if not hasattr(builtins, "_fx_co2"):
    st = builtins._fx_co2 = {}
    orig = mm.forward
    def f(*a, _o=orig, **k):
        h = a[0] if a else k.get("hidden_states")
        if h.dim() == 2 and h.shape[0] >= 64 and not torch.cuda.is_current_stream_capturing() and "h" not in st:
            st["h"] = h[16:19].detach().clone()
        return _o(*a, **k)
    mm.forward = f; mm._fx_orig = orig
    RESULT = "hooked"
else:
    st = builtins._fx_co2; mm.forward = mm._fx_orig; hs = st["h"]
    from vllm.forward_context import set_forward_context
    impl = ex._b12x_prepared_experts._impl; dec = impl.nvfp4_csf
    out = []
    def tensors(o, acc, seen, d=0, path="", paths=None):
        if id(o) in seen or d > 5: return
        seen.add(id(o))
        if torch.is_tensor(o):
            acc.add(o.untyped_storage().data_ptr())
            if paths is not None: paths.append(f"{path}{tuple(o.shape)}{o.dtype}")
            return
        if isinstance(o, dict): it = list(o.items())
        elif isinstance(o, (list, tuple)): it = list(enumerate(o))
        elif hasattr(o, "__dict__"): it = list(vars(o).items())
        else: it = [(k, getattr(o, k, None)) for k in getattr(o, "__slots__", ())]
        for k, v in it: tensors(v, acc, seen, d + 1, f"{path}.{k}", paths)
    with set_forward_context(None, r.vllm_config):
        ref = mm(hs).float().clone()
        s1 = torch.empty_like(impl.w1_blockscale); s2 = torch.empty_like(impl.w2_blockscale)
        dec.decode_all(s1, s2); torch.cuda.synchronize()
        out.append(f"expanded == scratch-expanded? decode_all into scratch too:")
        dec.decode_all(impl.w1_blockscale, impl.w2_blockscale); torch.cuda.synchronize()
        out.append(f"  s1 eq {torch.equal(s1.view(torch.uint8), impl.w1_blockscale.view(torch.uint8))} s2 eq {torch.equal(s2.view(torch.uint8), impl.w2_blockscale.view(torch.uint8))}")
        old = (impl.w1_blockscale, impl.w2_blockscale)
        try:
            object.__setattr__(impl, "w1_blockscale", s1); object.__setattr__(impl, "w2_blockscale", s2)
            object.__setattr__(impl, "nvfp4_csf", None)
            y = mm(hs).float()
            out.append(f"(a) separate tensors: rel {((y - ref).norm() / ref.norm()).item():.3g}")
            ptrs = set(); paths = []
            tensors(dec.first, ptrs, set(), path="first", paths=paths); tensors(dec.second, ptrs, set(), path="second", paths=paths)
            out.append("plane tensors: " + "; ".join(paths[:30]))
            hits = []
            for mmod_n, mmod in mm.named_modules():
                for pn, t in list(mmod.named_parameters(recurse=False)) + list(mmod.named_buffers(recurse=False)):
                    if t.numel() and t.untyped_storage().data_ptr() in ptrs: hits.append(f"{mmod_n}.{pn}{tuple(t.shape)}")
            out.append("params sharing plane storage: " + ", ".join(hits))
        finally:
            object.__setattr__(impl, "w1_blockscale", old[0]); object.__setattr__(impl, "w2_blockscale", old[1])
            object.__setattr__(impl, "nvfp4_csf", dec)
    del builtins._fx_co2
    RESULT = "\n".join(out)
