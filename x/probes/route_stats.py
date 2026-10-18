# Routing statistics from eager prefill: per-token top-10 weights, distinct experts per 3-token window (verify width),
# and the effect of dropping low-weight experts. /x/route.arg: "on" | "stats"
import builtins, torch
from vllm.model_executor.layers.fused_moe import b12x as B
D = builtins.__dict__.setdefault("_fx_route", {"rec": []})
arg = open("/x/route.arg").read().split()
if arg[0] == "on":
    if "orig" not in D:
        orig = B.B12xExperts.apply; D["orig"] = orig
        def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, *a, **k):
            if hidden_states.shape[0] > 64 and len(D["rec"]) < 49 * 4:
                D["rec"].append((topk_ids.detach().clone(), topk_weights.detach().float().clone()))
            return orig(self, output, hidden_states, w1, w2, topk_weights, topk_ids, *a, **k)
        B.B12xExperts.apply = apply
    D["rec"].clear(); RESULT = "on"
else:
    out = []
    W = torch.cat([w for _, w in D["rec"]]); s = W.sort(-1, descending=True).values
    out.append(f"calls={len(D['rec'])} tokens={W.shape[0]} mean sorted weights={[round(x, 3) for x in s.mean(0).tolist()]}")
    for tau in (0.0, 0.02, 0.03, 0.05, 0.07, 0.1):
        d3 = []; mass = []; kept = []
        for ids, w in D["rec"]:
            keep = w >= tau
            keep[:, 0] = True if keep.dim() == 2 else keep[:, 0]
            top1 = ids.gather(1, w.argmax(1, keepdim=True))
            keep.scatter_(1, w.argmax(1, keepdim=True), True)
            ids2 = torch.where(keep, ids, top1.expand_as(ids))
            mass.append((w * ~keep).sum(1).mean().item()); kept.append(keep.float().sum(1).mean().item())
            n = ids2.shape[0] // 3 * 3
            tri = ids2[:n].view(-1, 30)
            d3.append(torch.tensor([len(set(r.tolist())) for r in tri[:200]]).float().mean().item())
        out.append(f"tau={tau}: experts/token={sum(kept)/len(kept):.2f} dropped mass={sum(mass)/len(mass):.4f} D(3)={sum(d3)/len(d3):.2f}")
    RESULT = "\n".join(out)
