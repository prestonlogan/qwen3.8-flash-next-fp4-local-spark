# Profiler-free kernel-class costs inside vLLM's captured decode graphs: disable every kernel node of one name in the
# instantiated graph (cuGraphNodeSetEnabled) and time replays. Output is garbage while a class is disabled, and the
# replays overwrite KV/GDN state, so restart the server before any benchmark. builtins._fx_abl = {"graph": idx, "top": n}.
import torch, builtins, collections, re
from cuda.bindings import driver as cu
r = worker.model_runner
cfg = getattr(builtins, "_fx_abl", {})
seen = set(); found = []
def walk(o, path, d):
    if d > 4 or id(o) in seen: return
    seen.add(id(o))
    if isinstance(o, torch.cuda.CUDAGraph): found.append((path, o)); return
    if isinstance(o, dict):
        for k, v in list(o.items())[:300]:
            ks = str(k); m = re.search(r"num_tokens=(\d+)", ks); walk(v, f"{path}[{('T' + m.group(1)) if m else ks[:60]}]", d + 1)
    elif isinstance(o, (list, tuple)):
        for k, v in enumerate(o[:300]): walk(v, f"{path}[{k}]", d + 1)
    elif hasattr(o, "__dict__") and not isinstance(o, (torch.nn.Module, torch.Tensor)):
        for k, v in vars(o).items(): walk(v, f"{path}.{k}", d + 1)
walk(r, "r", 0)
def kernels(g):
    raw = cu.CUgraph(int(g.raw_cuda_graph()))
    _, nodes, n = cu.cuGraphGetNodes(raw, 0); _, nodes, n = cu.cuGraphGetNodes(raw, n)
    ks = []
    for nd in nodes:
        _, t = cu.cuGraphNodeGetType(nd)
        if t == cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            _, p = cu.cuGraphKernelNodeGetParams(nd)
            _, nm = cu.cuFuncGetName(p.func)
            ks.append((nd, nm.decode() if isinstance(nm, bytes) else str(nm)))
    return ks, n
out = []
for i, (path, g) in enumerate(found):
    try:
        ks, n = kernels(g); out.append(f"[{i}] {path[:120]}: {n} nodes, {len(ks)} kernels")
    except Exception as ex:
        out.append(f"[{i}] {path[:120]}: {type(ex).__name__} {str(ex)[:80]}")
if "graph" in cfg:
    path, g = found[cfg["graph"]]
    ks, n = kernels(g)
    ex = cu.CUgraphExec(int(g.raw_cuda_graph_exec()))
    def t(it=20):
        g.replay(); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
        for _ in range(it): g.replay()
        e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / it
    import sys
    groups = collections.defaultdict(list)
    for nd, nm in ks: groups[nm].append(nd)
    # Freeze routing: garbage hidden states would change the routed experts (and the MoE cost) otherwise.
    frozen = [nd for nm, nds in groups.items() if "softmax_topk" in nm for nd in nds]
    for _ in range(40): g.replay()
    torch.cuda.synchronize()
    full = t()
    for nd in frozen: cu.cuGraphNodeSetEnabled(ex, nd, 0)
    base = t()
    print(f"[abl] baseline {full:.3f}, routing frozen {base:.3f}", file=sys.stderr, flush=True)
    out.append(f"graph {cfg['graph']} replay {full:.3f} ms, routing frozen {base:.3f} ms ({len(frozen)} topk nodes off), {len(groups)} kernel names")
    deny = re.compile(r"topk|route|pack|sort|csf|index|gather|scatter|embed|argmax|sampl|reject|prefix|cumsum|copy|fill|slot|ple|W4A16|b12xmoe|metadata|reset|commit|ring|select|expand", re.I)
    def ok(nm, n):
        if n < 12 or deny.search(nm): return False
        if nm.startswith("_ZN2at6native"): return any(t_ in nm for t_ in ("BFloat16", "Half", "IfE", "IffE"))
        return True
    cand = [kv for kv in sorted(groups.items(), key=lambda kv: -len(kv[1])) if ok(kv[0], len(kv[1]))]
    out.append("skipped: " + ", ".join(f"{nm[:40]} x{len(nds)}" for nm, nds in groups.items() if not ok(nm, len(nds))))
    rows = []
    for j, (nm, nds) in enumerate(cand[:cfg.get("top", 40)]):
        if j and j % 6 == 0: base = t()
        print(f"[abl] disabling x{len(nds)} {nm[:90]}", file=sys.stderr, flush=True)
        for nd in nds: cu.cuGraphNodeSetEnabled(ex, nd, 0)
        try: tt = t()
        finally:
            for nd in nds: cu.cuGraphNodeSetEnabled(ex, nd, 1)
        rows.append((base - tt, len(nds), nm[:90]))
        print(f"[abl] {base - tt:7.3f} ms x{len(nds)} {nm[:90]}", file=sys.stderr, flush=True)
    for d, c, nm in sorted(rows, reverse=True):
        out.append(f"  {d:7.3f} ms  x{c:4d}  {nm}")
    for nd in frozen: cu.cuGraphNodeSetEnabled(ex, nd, 1)
    out.append(f"replay after {t():.3f} ms")
RESULT = "\n".join(out)
