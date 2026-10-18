# Per-step GPU timeline from CUDA events around each graph launch (verify, draft prefill, draft decode).
# Call 1 installs the hooks; send a decode request (kick2.py); call 2 reports busy spans and the idle gaps between them.
import torch, builtins, time
r = worker.model_runner
mgrs = (("V", r.cudagraph_manager), ("P", r.speculator.prefill_cudagraph_manager), ("D", r.speculator.decode_cudagraph_manager))
if not hasattr(builtins, "_fx_tl"):
    st = builtins._fx_tl = {"ev": [], "orig": {}}
    for tag, m in mgrs:
        o = m.run_fullgraph; st["orig"][tag] = o
        def f(*a, _o=o, _t=tag, **k):
            e0 = torch.cuda.Event(enable_timing=True); e0.record(); c0 = time.perf_counter()
            res = _o(*a, **k)
            e1 = torch.cuda.Event(enable_timing=True); e1.record()
            if len(st["ev"]) < 6000: st["ev"].append((_t, e0, e1, c0))
            return res
        m.run_fullgraph = f
    ms = r.model_state if hasattr(r, "model_state") else None
    embs = getattr(ms, "disk_embeddings", []) if ms is not None else []
    st["pd"] = []
    for emb in embs:
        o = emb.prepare_disk; st["orig"].setdefault("pd", []).append((emb, o))
        def pd(*a, _o=o, **k):
            e = torch.cuda.Event(enable_timing=True); e.record()
            c0 = time.perf_counter(); res = _o(*a, **k); c1 = time.perf_counter()
            if len(st["pd"]) < 6000: st["pd"].append((c0, c1, e))
            return res
        emb.prepare_disk = pd
    RESULT = f"hooked ({len(embs)} disk embeddings)"
else:
    st = builtins._fx_tl
    for tag, m in mgrs: m.run_fullgraph = st["orig"][tag]
    for emb, o in st["orig"].get("pd", []): emb.prepare_disk = o
    torch.cuda.synchronize()
    ev = st["ev"]; out = [f"{len(ev)} graph launches: " + " ".join(t for t, *_ in ev[:12])]
    # steps start at each V launch
    idx = [i for i, e in enumerate(ev) if e[0] == "V"]
    rows = []
    for a, b in zip(idx[5:-1], idx[6:]):
        seg = ev[a:b]; nxt = ev[b]
        dur = {t: 0.0 for t in "VPD"}; gaps = []
        for j, (t, e0, e1, c0) in enumerate(seg):
            dur[t] += e0.elapsed_time(e1)
            n2 = seg[j + 1] if j + 1 < len(seg) else nxt
            gaps.append(e1.elapsed_time(n2[1]))
        step = seg[0][1].elapsed_time(nxt[1])
        rows.append((step, dur["V"], dur["P"], dur["D"], gaps))
    if rows:
        n = len(rows); avg = lambda k: sum(x[k] for x in rows) / n
        ng = len(rows[0][4])
        g = [sum(x[4][j] for x in rows if len(x[4]) == ng) / n for j in range(ng)]
        out.append(f"{n} steps: step {avg(0):.3f} ms = V {avg(1):.3f} + P {avg(2):.3f} + D {avg(3):.3f} + between-graph time {[round(v, 3) for v in g]}")
        # CPU: D launch -> prepare_disk entry -> exit -> V launch
        pdl = st["pd"]; cv = [e[3] for e in ev]; dd = []
        for a, b in zip(idx[5:-1], idx[6:]):
            d_launch = ev[b - 1][3]; v_launch = ev[b][3]
            inside = [p for p in pdl if d_launch <= p[0] <= v_launch]
            if inside:
                p0, p1, pe = inside[-1]
                dd.append(((p0 - d_launch) * 1e3, (p1 - p0) * 1e3, (v_launch - p1) * 1e3,
                           ev[b - 1][2].elapsed_time(pe), pe.elapsed_time(ev[b][1])))
        if dd:
            out.append("CPU ms: D launch->prepare_disk %.3f, prepare_disk (sync wait + read) %.3f, ->V launch %.3f; GPU ms: D end->eager work done %.3f, ->V start %.3f" % tuple(sum(x[j] for x in dd) / len(dd) for j in range(5)))
        out.append("launch order per step: " + " ".join(t for t, *_ in ev[idx[6]:idx[7]]))
    del builtins._fx_tl
    RESULT = "\n".join(out)
