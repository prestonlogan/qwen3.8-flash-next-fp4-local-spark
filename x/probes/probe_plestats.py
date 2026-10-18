# Find the disk PLE row cache under the B12xNGramEmbedding module and report its stats.
import torch
r = worker.model_runner
emb = [m for n, m in r.model.named_modules() if type(m).__name__ == "B12xNGramEmbedding"]
out = [f"{len(emb)} B12xNGramEmbedding"]
seen = set()
def walk(o, path, d):
    if d > 5 or id(o) in seen or isinstance(o, (torch.Tensor, str, int, float, bytes)): return
    seen.add(id(o))
    if hasattr(type(o), "stats") and callable(getattr(o, "stats")):
        try: out.append(f"{path} ({type(o).__name__}): {o.stats()}")
        except Exception as e: out.append(f"{path}: stats err {e}")
    items = []
    if isinstance(o, dict): items = list(o.items())[:50]
    elif isinstance(o, (list, tuple)): items = list(enumerate(o))[:50]
    else:
        try: items = list(vars(o).items())
        except TypeError:
            items = [(s, getattr(o, s, None)) for s in getattr(type(o), "__slots__", ())]
    for k, v in items:
        if isinstance(v, torch.nn.Module) and d > 0: continue
        walk(v, f"{path}.{k}", d + 1)
for e in emb: walk(e, "emb", 0)
RESULT = "\n".join(out)
