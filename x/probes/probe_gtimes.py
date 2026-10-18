# Replay time of each captured decode graph in isolation (replays overwrite KV/GDN state: restart before benchmarks).
import torch, re
r = worker.model_runner
out = []
def t(g, it=20):
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(it): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / it
for name, mgr in (("verify", r.cudagraph_manager), ("draft prefill", r.speculator.prefill_cudagraph_manager),
                  ("draft decode", r.speculator.decode_cudagraph_manager)):
    for k, g in mgr.graphs.items():
        m = re.search(r"num_tokens=(\d+)", str(k)); n = int(m.group(1)) if m else -1
        if n in (1, 2, 3, 4):
            out.append(f"{name} T{n}: {t(g):.3f} ms")
RESULT = "\n".join(out)
