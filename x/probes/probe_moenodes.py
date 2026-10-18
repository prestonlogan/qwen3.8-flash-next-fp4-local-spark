# Kernel names in one decode MoE block (M=3) via a captured graph's nodes (no profiler). Inputs from probe_moeparts-style
# capture: call 1 hooks, send kick.py, call 2 lists.
import torch, builtins
from cuda.bindings import driver as cu
r = worker.model_runner; mods = dict(r.model.named_modules())
mlps = [mods[f"language_model.model.layers.{i}.mlp"] for i in range(48)]
def names(fn):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    raw = cu.CUgraph(int(g.raw_cuda_graph()))
    err, nodes, n = cu.cuGraphGetNodes(raw, 0); err, nodes, n = cu.cuGraphGetNodes(raw, n)
    res = []
    for nd in nodes:
        e, t = cu.cuGraphNodeGetType(nd)
        if t == cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            e, p = cu.cuGraphKernelNodeGetParams(nd)
            try:
                e2, nm = cu.cuFuncGetName(p.func); nm = nm.decode()[:90] if isinstance(nm, bytes) else str(nm)[:90]
            except Exception as ex:
                nm = f"? {ex}"[:60]
            res.append(f"K grid({p.gridDimX},{p.gridDimY},{p.gridDimZ}) blk {p.blockDimX} {nm}")
        else:
            res.append(t.name)
    return res
if not hasattr(builtins, "_fx_mn"):
    st = builtins._fx_mn = {}
    mm = mlps[5]; orig = mm.forward
    def f(*a, _o=orig, **k):
        h = a[0] if a else k.get("hidden_states")
        if h.dim() == 2 and h.shape[0] >= 64 and not torch.cuda.is_current_stream_capturing() and "h" not in st:
            st["h"] = h[16:19].detach().clone()
        return _o(*a, **k)
    mm.forward = f; mm._fx_orig = orig
    RESULT = "hooked"
else:
    st = builtins._fx_mn; mm = mlps[5]; mm.forward = mm._fx_orig
    from vllm.forward_context import set_forward_context
    with set_forward_context(None, r.vllm_config):
        res = names(lambda: mm(st["h"]))
    del builtins._fx_mn
    RESULT = "\n".join(res)
