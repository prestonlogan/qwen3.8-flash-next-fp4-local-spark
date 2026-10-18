import torch, traceback
r = worker.model_runner; dm = r.speculator.model; lm = dm.lm_head
out = []
try:
    E = torch.eye(2560, device="cuda", dtype=torch.bfloat16)[:64]
    L = dm.compute_logits(E)
    w = lm.weight.data; sc = lm.weight_scale.data; gs = lm.weight_global_scale.data.float()
    N, KB = sc.shape
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
    def unswz(flat, N, Kc):
        mt, kt = N // 128, Kc // 4
        return flat.view(torch.uint8).reshape(mt, kt, 32, 4, 4).transpose(1, 3).reshape(mt * 128, kt * 4)
    scu = unswz(sc, N, KB).view(torch.float8_e4m3fn)
    rows = torch.tensor([0, 1, 1000, 50000, 248000], device="cuda")
    ref = L[:, rows].T.float()
    for sname, S in (("raw", sc), ("unswz", scu)):
        for order in ("lohi", "hilo"):
            wb = w[rows]; lo = lut[(wb & 0xF).long()]; hi = lut[(wb >> 4).long()]
            vals = torch.stack([lo, hi] if order == "lohi" else [hi, lo], -1).view(5, 2560)
            deq = vals * S[rows].float().repeat_interleave(16, dim=1) * gs
            out.append(f"{sname}/{order}: maxerr {(deq[:, :64] - ref).abs().max().item():.3e} (mag {ref.abs().max().item():.3e})")
except Exception: out.append(traceback.format_exc()[-1000:])
RESULT = "\n".join(out)
