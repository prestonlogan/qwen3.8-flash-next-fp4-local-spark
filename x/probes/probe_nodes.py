# Count kernel nodes in the captured decode CUDA graphs (no profiler): cudaGraphGetNodes + node types.
import torch
from cuda.bindings import runtime as rt
out = []
r = worker.model_runner
seen = set(); found = []
def walk(o, path, d):
    if d > 3 or id(o) in seen: return
    seen.add(id(o))
    if isinstance(o, torch.cuda.CUDAGraph): found.append((path, o)); return
    if isinstance(o, dict):
        for k, v in list(o.items())[:200]: walk(v, f"{path}[{str(k)[:60]}]", d + 1)
    elif hasattr(o, "__dict__") and not isinstance(o, (torch.nn.Module, torch.Tensor)):
        for k, v in vars(o).items(): walk(v, f"{path}.{k}", d + 1)
walk(r, "r", 0)
out.append(f"{len(found)} graphs")
for path, g in found:
    try:
        raw = g.raw_cuda_graph()
        err, nodes, n = rt.cudaGraphGetNodes(raw, 0)
        err, nodes, n = rt.cudaGraphGetNodes(raw, n)
        kinds = {}
        for nd in nodes:
            e, t = rt.cudaGraphNodeGetType(nd)
            kinds[t.name] = kinds.get(t.name, 0) + 1
        out.append(f"{path[:160]}: {n} nodes {kinds}")
    except Exception as e:
        out.append(f"{path[:160]}: err {type(e).__name__} {str(e)[:120]}")
RESULT = "\n".join(out)
