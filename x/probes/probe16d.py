import builtins, torch
torch.cuda.synchronize()
ev = builtins._fx_ev
ref = ev["ModelCudaGraphManager.run_fullgraph"][100][0]
top = ["GPUModelRunner.prepare_inputs", "GPUModelRunner.prepare_attn", "ModelCudaGraphManager.run_fullgraph", "GPUModelRunner.sample_tokens", "GPUModelRunner.postprocess_sampled"]
iv = []
for k in top:
    for a, b in ev.get(k, []):
        t0 = ref.elapsed_time(a); t1 = ref.elapsed_time(b)
        iv.append((t0, t1, k))
iv = [x for x in iv if 0 <= x[0] <= 60000]
iv.sort()
lo, hi = iv[0][0], max(x[1] for x in iv)
cov = 0; end = lo
for a, b, k in iv:
    if b > end: cov += b - max(a, end); end = b
nsteps = sum(1 for x in iv if x[2] == "ModelCudaGraphManager.run_fullgraph")
# gaps by preceding phase
gaps = {}
end = iv[0][1]; prev = iv[0][2]
for a, b, k in iv[1:]:
    if a > end: gaps[(prev, k)] = gaps.get((prev, k), 0) + (a - end)
    if b > end: end = b; prev = k
RESULT = f"window {hi-lo:.0f} ms, {nsteps} steps, {(hi-lo)/nsteps:.2f} ms/step, covered {cov/nsteps:.2f} ms/step\n" + "\n".join(f"gap after {p} before {k}: {v/nsteps:.2f} ms/step" for (p, k), v in sorted(gaps.items(), key=lambda x: -x[1])[:6])
