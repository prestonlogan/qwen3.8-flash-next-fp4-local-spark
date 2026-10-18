# T8 capture (run via fx_exec): record the target's multi-stream hidden [n,10240] (what the MTP drafter consumes)
# plus optionally the sampled hidden [n,2560] for eager prefill chunks. Control file /x/t8cap.txt:
#   on [sh] | dump <path> [fp8] | off
import torch, builtins
lm = worker.model_runner.model.language_model.model
D = builtins.__dict__.setdefault("_fx_t8", {"chunks": []})
arg = open("/x/t8cap.txt").read().split()
if arg[0] == "on":
    if "orig" not in D:
        orig = lm.forward; D["orig"] = orig
        def fwd(*a, **k):
            out = orig(*a, **k)
            pos = k.get("positions", a[1] if len(a) > 1 else None)
            p = pos[0] if pos.dim() == 2 else pos
            n = p.shape[0]
            if n > 64 and torch.is_tensor(out):
                D["chunks"].append((p[:n].to("cpu", torch.int32), lm._mtp_hidden_buffer[:n].to("cpu"),
                                    out[:n].to("cpu", torch.bfloat16) if D.get("sh") else None))
            return out
        lm.forward = fwd
    D["sh"] = len(arg) > 1 and arg[1] == "sh"; D["chunks"].clear(); RESULT = f"cap on sh={D['sh']}"
elif arg[0] == "dump":
    ch = D["chunks"]
    if not ch:
        RESULT = "EMPTY"
    else:
        pos = torch.cat([c[0] for c in ch]); mh = torch.cat([c[1] for c in ch])
        sh = torch.cat([c[2] for c in ch]) if ch[0][2] is not None else None
        if len(arg) > 2 and arg[2] == "fp8":
            x = mh.float().view(-1, 4, 2560); sc = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 448.0
            torch.save({"pos": pos, "mq": (x / sc).to(torch.float8_e4m3fn).view(-1, 10240), "ms": sc[..., 0].to(torch.bfloat16), "sh": sh}, arg[1])
        else:
            torch.save({"pos": pos, "mh": mh, "sh": sh}, arg[1])
        n = len(ch); ch.clear()
        RESULT = f"saved n={pos.shape[0]} chunks={n} pos0={int(pos[0])} last={int(pos[-1])} mono={bool((pos[1:] - pos[:-1] == 1).all())}"
else:
    if "orig" in D: lm.forward = D.pop("orig")
    D["chunks"].clear(); RESULT = "cap off"
