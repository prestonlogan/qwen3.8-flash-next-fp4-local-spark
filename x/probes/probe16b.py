import builtins, torch
torch.cuda.synchronize()
ev = builtins._fx_ev
out = []
for k, L in ev.items():
    ts = [a.elapsed_time(b) for a, b in L[50:]]
    if ts: out.append(f"{k}: n={len(L)} avg gpu-span {sum(ts)/len(ts):.2f} ms")
RESULT = "\n".join(out)
