import builtins, torch, statistics as stt
torch.cuda.synchronize()
ev = builtins._fx_ev
F = ev["ModelCudaGraphManager.run_fullgraph"]; S = ev["GPUModelRunner.sample_tokens"]; P = ev["GPUModelRunner.prepare_inputs"]
n = min(len(F), len(S), len(P)) - 1
g1 = [F[i][1].elapsed_time(S[i][0]) for i in range(50, n)]
g2 = [S[i][1].elapsed_time(P[i + 1][0]) for i in range(50, n)]
g3 = [P[i + 1][1].elapsed_time(F[i + 1][0]) for i in range(50, n)]
tot = [F[i][0].elapsed_time(F[i + 1][0]) for i in range(50, n)]
RESULT = f"step {stt.median(tot):.2f} | full->sample_tokens {stt.median(g1):.2f} | sample_tokens->next prepare_inputs {stt.median(g2):.2f} | prepare_inputs->fullgraph {stt.median(g3):.2f} ms (medians)"
