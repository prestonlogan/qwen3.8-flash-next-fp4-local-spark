# Hot-swap MTP drafter tensors from a checkpoint-named patch (or restore originals).
# /x/mtp_swap.arg: "apply <patch.safetensors>" | "restore"
import builtins, glob, json, torch
from safetensors import safe_open
arg = open("/x/mtp_swap.arg").read().split()
m = worker.model_runner.speculator.model
P = dict(m.named_parameters())
SNAP = glob.glob("/root/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/*/")[0]
wm = json.load(open(SNAP + "model.safetensors.index.json"))["weight_map"]
def ck_tensor(k):
    with safe_open(SNAP + wm[k], "pt") as f: return f.get_tensor(k)
def target(ck):
    n = ck[len("mtp."):-len(".weight")]
    for a, b, lo, hi in (("self_attn.q_proj", "self_attn.qkv_proj", 0, 12288), ("self_attn.k_proj", "self_attn.qkv_proj", 12288, 12800),
                         ("self_attn.v_proj", "self_attn.qkv_proj", 12800, 13312),
                         ("shared_expert.gate_proj", "shared_expert.gate_up_proj", 0, 640), ("shared_expert.up_proj", "shared_expert.gate_up_proj", 640, 1280)):
        if n.endswith(a): return P["model." + n[:-len(a)] + b + ".weight"].data[lo:hi]
    if n.endswith("hyper_connection.input_mix_weight_down") and "layers" in n:
        return P["model." + n + "_block_inject.weight"].data[:320]
    return P["model." + n + ".weight"].data
S = builtins.__dict__.setdefault("_mtp_swap", {})
out = []
torch.cuda.synchronize()
if arg[0] == "apply":
    with safe_open(arg[1], "pt") as f:
        for ck in f.keys():
            dst = target(ck); new = f.get_tensor(ck).to(dst.device, dst.dtype)
            orig = ck_tensor(ck).to(dst.device)
            if ck not in S:
                if not torch.equal(dst, orig.to(dst.dtype)):
                    d = ((dst.float() - orig.float()).norm() / orig.float().norm()).item()
                    out.append(f"MISMATCH {ck} rel={d:.4g} (layout differs; skipped)"); continue
                S[ck] = dst.clone()
            dst.copy_(new); out.append(f"{ck} {tuple(dst.shape)}")
else:
    for ck, t in S.items():
        target(ck).copy_(t)
    out.append(f"restored {len(S)}")
torch.cuda.synchronize()
RESULT = "\n".join(out)
