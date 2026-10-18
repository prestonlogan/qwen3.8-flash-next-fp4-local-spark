import torch, sys, traceback
sys.path.insert(0, "/x")
import fp4x_kernels as F
r = worker.model_runner; m = r.model
out = []
def timeit(fn, it=10):
    fn(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(it): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / it
M = 4
mods = dict(m.named_modules())
# HC
hcs = [(n, mod) for n, mod in mods.items() if type(mod).__name__ == "GatedResidual" and "mtp" not in n]
xs = torch.randn(M, 10240, device="cuda", dtype=torch.bfloat16); bt = torch.randn(M, 320, device="cuda", dtype=torch.bfloat16)
def hc_torch():
    for n, h in hcs:
        d = h.input_mix_weight_down_block_inject if h.use_combine else h.input_mix_weight_down
        d(xs); h.input_mix_weight_up(bt)
def hc_tri():
    for n, h in hcs:
        d = h.input_mix_weight_down_block_inject if h.use_combine else h.input_mix_weight_down
        F.bf16_linear(xs, d.weight, (16, 512, 1, 4)); F.bf16_linear(bt, h.input_mix_weight_up.weight, (32, 64, 1, 4))
try:
    out.append(f"HC n={len(hcs)} torch-module {timeit(hc_torch):.2f} ms  triton {timeit(hc_tri):.2f} ms  bytes {sum(2*(h.input_mix_weight_down_block_inject.weight.numel() if h.use_combine else h.input_mix_weight_down.weight.numel()) + 2*h.input_mix_weight_up.weight.numel() for _, h in hcs)/1e9:.2f} GB")
except Exception: out.append("HC err " + traceback.format_exc()[-400:])
# MXFP8 dense via module forward
from vllm.model_executor.layers.linear import LinearBase
dense = [(n, mod) for n, mod in mods.items() if isinstance(mod, LinearBase) and hasattr(mod, "b12x_mxfp8_packed_weight") and "mtp" not in n and "visual" not in n]
xin = {}
def dense_run():
    for n, d in dense:
        k = d.b12x_mxfp8_packed_weight.in_features
        x = xin.get(k)
        if x is None: x = xin[k] = torch.randn(M, k, device="cuda", dtype=torch.bfloat16)
        d(x)
try:
    nb = sum(d.b12x_mxfp8_packed_weight.weight.values.numel() for _, d in dense)
    out.append(f"MXFP8 dense n={len(dense)} module-forward {timeit(dense_run):.2f} ms  bytes {nb/1e9:.2f} GB")
    def dense_tri():
        for n, d in dense:
            pk = d.b12x_mxfp8_packed_weight; w = pk.weight.values; s = pk.weight.scale_rows.view(torch.uint8).view(w.shape[0], -1)
            F.mx8_linear(xin[pk.in_features], w, s, (32, 512, 1, 4) if pk.in_features % 512 == 0 else (32, 128, 1, 4))
    out.append(f"MXFP8 dense triton {timeit(dense_tri):.2f} ms")
except Exception: out.append("dense err " + traceback.format_exc()[-600:])
# LM head
try:
    lm = mods["language_model.lm_head"]; h = torch.randn(M, 2560, device="cuda", dtype=torch.bfloat16)
    lp = m.language_model.logits_processor if hasattr(m.language_model, "logits_processor") else None
    out.append(f"lm_head type={type(lm).__name__} qm={type(lm.quant_method).__name__} attrs={[a for a in vars(lm) if 'b12x' in a or 'weight' in a][:8]}")
    if lp is not None: out.append(f"logits {timeit(lambda: lp(lm, h)):.2f} ms")
except Exception: out.append("lm err " + traceback.format_exc()[-600:])
RESULT = "\n".join(out)
