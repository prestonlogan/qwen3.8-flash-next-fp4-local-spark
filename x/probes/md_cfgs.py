import builtins
builtins._fx_md_cfgs = [dict(fn="moe_decode2"), dict(fn="moe_decode3")]
builtins._fx_md_kcfgs = [dict(KT=k, nw=w, ns=n) for k, w, n in [(4, 4, 3), (2, 4, 3), (8, 4, 2), (2, 2, 4), (4, 4, 4)]]
RESULT = "ok"
