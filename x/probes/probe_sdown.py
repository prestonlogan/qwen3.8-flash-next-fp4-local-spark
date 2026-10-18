# Where the B12xBlockscaledLinear object for the shared expert down_proj lives (for a fused shared-expert kernel).
import torch
from vllm.model_executor.kernels.linear import b12x_blockscaled as BB
r = worker.model_runner; mods = dict(r.model.named_modules())
dp = mods["language_model.model.layers.0.mlp.shared_expert.down_proj"]
out = [f"module vars: {[k for k in vars(dp) if not k.startswith('_')][:40]}", f"qm {type(dp.quant_method).__name__}: {list(vars(dp.quant_method))[:30]}"]
def find(o, path, d, seen):
    if d > 4 or id(o) in seen: return
    seen.add(id(o))
    if isinstance(o, BB.B12xBlockscaledLinear): out.append(f"FOUND at {path}"); return
    if isinstance(o, torch.Tensor): return
    items = o.items() if isinstance(o, dict) else enumerate(o) if isinstance(o, (list, tuple)) else vars(o).items() if hasattr(o, "__dict__") else []
    for k, v in items: find(v, f"{path}.{k}", d + 1, seen)
find(dp, "dp", 0, set())
eg = mods["language_model.model.layers.0.mlp.shared_expert"]
out.append(f"shared_expert expert_gate: {getattr(eg, 'expert_gate', None)}")
RESULT = "\n".join(out)
