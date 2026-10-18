# GPU-time markers at the entry of each runner function between the draft decode graph and the next verify graph.
# Call 1 hooks; kick2.py; call 2 reports the mean GPU time from each marker to the next (per step).
import torch, builtins, collections
r = worker.model_runner
names = [(r, n) for n in ("execute_model", "update_pp_decode_requests", "finish_requests", "free_states", "add_requests",
                          "update_requests", "gather_batch_req_state", "prepare_inputs", "prepare_attn", "sample_tokens",
                          "postprocess_sampled", "postprocess_num_computed_tokens", "sample")]
names += [(r.block_tables, "apply_staged_writes"), (r.model_state, "preprocess_state"), (r.model_state, "prepare_inputs"),
          (r.model_state, "postprocess_state"), (r.speculator, "propose"), (r.model_state, "prepare_attn")]
for emb in getattr(r.model_state, "disk_embeddings", []): names.append((emb, "prepare_disk"))
mg = [("V", r.cudagraph_manager), ("P", r.speculator.prefill_cudagraph_manager), ("D", r.speculator.decode_cudagraph_manager)]
if not hasattr(builtins, "_fx_mk"):
    st = builtins._fx_mk = {"m": [], "orig": []}
    def mark(tag):
        if len(st["m"]) < 40000:
            e = torch.cuda.Event(enable_timing=True); e.record(); st["m"].append((tag, e))
    for obj, n in names:
        o = getattr(obj, n, None)
        if o is None: continue
        tag = f"{type(obj).__name__}.{n}"
        def f(*a, _o=o, _t=tag, **k):
            mark(_t); res = _o(*a, **k); mark(_t + ":end"); return res
        setattr(obj, n, f); st["orig"].append((obj, n, o))
    for t, m in mg:
        o = m.run_fullgraph
        def f(*a, _o=o, _t=t, **k):
            mark(_t); res = _o(*a, **k); mark(_t + ":end"); return res
        m.run_fullgraph = f; st["orig"].append((m, "run_fullgraph", o))
    RESULT = "hooked"
else:
    st = builtins._fx_mk
    for obj, n, o in st["orig"]: setattr(obj, n, o)
    torch.cuda.synchronize()
    m = st["m"]; acc = collections.defaultdict(float); cnt = collections.Counter()
    dstart = [i for i, (t, _) in enumerate(m) if t == "D:end"]
    nsteps = 0
    for a in dstart[5:-2]:
        j = a
        while j + 1 < len(m) and m[j + 1][0] != "V":
            acc[(m[j][0], m[j + 1][0])] += m[j][1].elapsed_time(m[j + 1][1]); j += 1
        if j + 1 < len(m): acc[(m[j][0], "V")] += m[j][1].elapsed_time(m[j + 1][1])
        nsteps += 1
    out = [f"{nsteps} steps; GPU ms between markers from draft-decode end to verify start:"]
    tot = 0
    for (x, y), v in acc.items():
        v /= nsteps; tot += v
        if v > 0.004: out.append(f"  {v:7.3f}  {x} -> {y}")
    out.append(f"  total {tot:.3f}")
    del builtins._fx_mk
    RESULT = "\n".join(out)
