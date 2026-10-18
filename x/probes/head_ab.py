# Runtime A/B of the target fast head's INT2 coarse pass (the head runs eagerly, outside the CUDA graphs):
# builtins._fx_head_mode = "g16" (C=512) or "g32" (C=1024). Builds the other variant once.
import torch, builtins, importlib.util
from vllm.models.qwen4_exp.nvidia import model as MD
import fp4x_kernels as F
fn = MD.Qwen4ExpForCausalLM.compute_logits
cells = dict(zip(fn.__code__.co_freevars, fn.__closure__))
fast = cells["fast"].cell_contents
fc = dict(zip(fast.__code__.co_freevars, fast.__closure__))
st = fc["st"].cell_contents
store = builtins.__dict__.setdefault("_fx_heads", {})
G = st["coarse"][1].shape[0] and 2560 // st["coarse"][1].shape[1]
store.setdefault(f"g{G}", (st["coarse"], fc["C"].cell_contents))
mode = builtins.__dict__.get("_fx_head_mode", "g16")
if mode not in store:
    g = int(mode[1:])
    wf = F.mx8_dequant(st["w"], st["s"])
    store[mode] = (F.quantize_int2(wf, group=g), 512 if g == 16 else 1024); del wf; torch.cuda.empty_cache()
st["coarse"], c = store[mode]; fc["C"].cell_contents = c
RESULT = f"head now {mode} C={c} (had g{G})"
