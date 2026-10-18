import torch, traceback
r = worker.model_runner; dm = r.speculator.model; lm = dm.lm_head
out = []
try:
    E = torch.eye(2560, device="cuda", dtype=torch.bfloat16)[:64]  # first 64 basis vectors
    L = dm.compute_logits(E)  # [64, V]: column v = W[v, :64]
    out.append(f"logits dtype {L.dtype} shape {tuple(L.shape)}")
    w = lm.weight.data; sc = lm.weight_scale.data; gs = lm.weight_global_scale.data
    out.append(f"global {gs.item()} alpha {lm.alpha.item() if hasattr(lm,'alpha') else None} igs {lm.input_global_scale_inv.item()}")
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
    rows = torch.tensor([0, 1, 1000, 50000, 248000], device="cuda")
    wb = w[rows]  # [5, 1280] bytes
    lo = lut[(wb & 0xF).long()]; hi = lut[(wb >> 4).long()]
    vals = torch.stack([lo, hi], -1).view(5, 2560)
    s = sc[rows].float().repeat_interleave(16, dim=1)
    for name, deq in (("x/gs", vals * s / gs), ("x*gs", vals * s * gs), ("plain", vals * s)):
        ref = L[:, rows].T.float()  # [5, 64]
        err = (deq[:, :64] - ref).abs().max().item(); mag = ref.abs().max().item()
        out.append(f"{name}: maxerr {err:.3e} vs mag {mag:.3e}")
except Exception: out.append(traceback.format_exc()[-1000:])
RESULT = "\n".join(out)
