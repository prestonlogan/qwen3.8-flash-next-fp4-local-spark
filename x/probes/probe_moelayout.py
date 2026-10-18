# Check my reading of b12x's packed W4A16 expert layout against the checkpoint nibbles (layer 0, a few experts).
import torch, glob, json
from safetensors import safe_open
r = worker.model_runner; mods = dict(r.model.named_modules())
out = []
prep = None
for n, m in mods.items():
    if hasattr(m, "_b12x_prepared_experts") and ".layers.0." in n:
        prep = m._b12x_prepared_experts; out.append(f"prep on {n}: {type(prep).__name__}"); break
impl = getattr(prep, "_impl", prep)
w1 = impl.w1_fp4; w2 = impl.w2_fp4
out.append(f"w1 {tuple(w1.shape)} w2 {tuple(w2.shape)}")
snap = glob.glob("/root/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/*/")[0]
wm = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
def ck(name):
    with safe_open(snap + wm[name], "pt", device="cpu") as f: return f.get_tensor(name)
def unpack(t, N):
    # t: [K/16, (N/64)*128] int32 -> nibbles [N, K]
    kt = t.shape[0]; nt = N // 64
    v = t.view(kt, nt, 8, 4, 4)                                   # (kt, nt, tc_col, r, warp)
    sh = torch.arange(8, device=t.device, dtype=torch.int32) * 4
    nib = (v[..., None] >> sh) & 0xF                               # (..., slot) slot = a*4 + b*2 + c
    nib = nib.view(kt, nt, 8, 4, 4, 2, 2, 2)                       # (kt, nt, tc_col, r, w, a, b, c)
    nib = nib.permute(1, 4, 6, 2, 0, 7, 3, 5)                      # (nt, w, b, tc_col, kt, c, r, a)
    return nib.reshape(N, kt * 16)
def nibbles(u8):  # [N, K/2] uint8, low nibble = even k
    return torch.stack([u8 & 15, u8 >> 4], -1).view(u8.shape[0], -1).to(torch.int32)
for e in (0, 1, 7):
    g = nibbles(ck(f"model.language_model.layers.0.mlp.experts.{e}.gate_proj.weight")).cuda()
    u = nibbles(ck(f"model.language_model.layers.0.mlp.experts.{e}.up_proj.weight")).cuda()
    d = nibbles(ck(f"model.language_model.layers.0.mlp.experts.{e}.down_proj.weight")).cuda()
    a = unpack(w1[e], 1280)
    b = unpack(w2[e], 2560)
    out.append(f"e{e}: w1[:640]==gate {(a[:640]==g).float().mean():.4f} ==up {(a[:640]==u).float().mean():.4f}; "
               f"w1[640:]==gate {(a[640:]==g).float().mean():.4f} ==up {(a[640:]==u).float().mean():.4f}; w2==down {(b==d).float().mean():.4f}")
RESULT = "\n".join(out)
