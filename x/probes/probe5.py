import torch, traceback, sys
sys.path.insert(0, "/x")
r = worker.model_runner; m = r.model; mods = dict(m.named_modules())
out = []
def gtime(fn, it=20):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        with torch.cuda.graph(g): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(it): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / it
M = 4
mlp0 = mods["language_model.model.layers.0.mlp"]
out.append("mlp type " + type(mlp0).__name__ + " children " + ",".join(f"{n}:{type(c).__name__}" for n, c in mlp0.named_children()))
from vllm.model_executor.layers.linear import LinearBase
dense = [d for n, d in mods.items() if isinstance(d, LinearBase) and hasattr(d, "b12x_mxfp8_packed_weight") and "mtp" not in n and "visual" not in n]
X = {}
for d in dense:
    k = d.b12x_mxfp8_packed_weight.in_features
    X.setdefault(k, torch.randn(M, k, device="cuda", dtype=torch.bfloat16))
try: out.append(f"dense graph (240 layers, b12x/current path): {gtime(lambda: [d(X[d.b12x_mxfp8_packed_weight.in_features]) for d in dense]):.2f} ms")
except Exception: out.append("dense graph err " + traceback.format_exc()[-500:])
hcs = [h for n, h in mods.items() if type(h).__name__ == "GatedResidual" and "mtp" not in n]
xs = torch.randn(M, 10240, device="cuda", dtype=torch.bfloat16); bt = torch.randn(M, 320, device="cuda", dtype=torch.bfloat16)
try: out.append(f"HC linears graph (cuBLAS): {gtime(lambda: [((h.input_mix_weight_down_block_inject if h.use_combine else h.input_mix_weight_down)(xs), h.input_mix_weight_up(bt)) for h in hcs]):.2f} ms")
except Exception: out.append("hc graph err " + traceback.format_exc()[-500:])
import fp4x_kernels as F
try: out.append(f"HC linears graph (triton): {gtime(lambda: [(F.bf16_linear(xs, (h.input_mix_weight_down_block_inject if h.use_combine else h.input_mix_weight_down).weight, (16,512,1,4)), F.bf16_linear(bt, h.input_mix_weight_up.weight, (32,64,1,4))) for h in hcs]):.2f} ms")
except Exception: out.append("hc tri err " + traceback.format_exc()[-500:])
# MoE experts: try calling the experts module directly
try:
    ex = [c for n, c in mlp0.named_children()]
    out.append("mlp0 forward sig: " + str(type(mlp0).forward.__code__.co_varnames[:6]))
    h = torch.randn(M, 2560, device="cuda", dtype=torch.bfloat16)
    mlps = [mods[f"language_model.model.layers.{i}.mlp"] for i in range(48)]
    from vllm.forward_context import set_forward_context
    with set_forward_context(None, r.vllm_config):
        out.append(f"MoE mlp x48 graph: {gtime(lambda: [mm(h) for mm in mlps]):.2f} ms")
except Exception: out.append("moe err " + traceback.format_exc()[-900:])
RESULT = "\n".join(out)
