import torch, fp4x_kernels as F
m = worker.model_runner.model
out = []
for name, p in m.named_parameters():
    if ("input_mix_weight_down_block_inject" in name or "input_mix_weight_up" in name) and len(out) < 6:
        w = p.data
        if w.dim() != 2: continue
        x = torch.randn(4, w.shape[1], device=w.device, dtype=torch.bfloat16)
        ref = (x.float() @ w.float().t())
        o4 = F.nv4_linear(x, *F.quantize_nv4(w), (16, 64, 4, 1)).float()
        o8 = F.w8_linear(x, *F.quantize_w8(w), (32, 64, 1, 4)).float()
        rel = lambda o: ((o - ref).norm() / ref.norm()).item()
        a = w.float().abs()
        out.append(f"{name} {tuple(w.shape)} amax={a.max().item():.3g} med={a.median().item():.3g} nv4_rel={rel(o4):.4f} w8_rel={rel(o8):.4f}")
RESULT = "\n".join(out)
