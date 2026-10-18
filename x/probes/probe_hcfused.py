# Validate + time the fused HC kernels against the served path (real weights, verify M).
import torch, sys, itertools, traceback
sys.path.insert(0, "/x")
import fp4x_kernels as F

r = worker.model_runner; m = r.model; mods = dict(m.named_modules())
out = []
def gtime(fn, it=20):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        with torch.cuda.graph(g): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(it): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / it
hcs = [h for n, h in mods.items() if type(h).__name__ == "GatedResidual" and "mtp" not in n]
h0 = hcs[0]
fx = F.HCFused("cuda")
from vllm.forward_context import set_forward_context
QD = [F.quantize_w8((h.input_mix_weight_down_block_inject if h.use_combine else h.input_mix_weight_down).weight) for h in hcs]
QU = [F.quantize_w8(h.input_mix_weight_up.weight) for h in hcs]
BJ = 8
QI = [F.HCFused.interleave_up(q, s, h0.hc_count, BJ) for q, s in QU]
lr, hc = h0.lora_rank, h0.hc_count
torch.manual_seed(0)
try:
  for M in (1, 3, 4, 8, 12, 16):
    xs = (torch.randn(M, 10240, device="cuda") * 0.5).to(torch.bfloat16)
    worst = [0, 0, 0, 0]
    with set_forward_context(None, r.vllm_config):
        for i in range(0, len(hcs), 12):
            h = hcs[i]
            ref_bi, ref_inj = h._mix_normalized(xs)   # served path (fused when FP4X_HC_FUSED=1)
            di, bot = fx.down(xs, *QD[i], lr, hc)
            api = __import__("vllm.models.qwen4_exp.nvidia.hyperconnection", fromlist=["x"])._hyperconnection_api()
            di_ref = F.w8_linear(xs, *QD[i], (32, 512, 4, 4))
            bot_ref = api.run_scaled_silu(di_ref[:, :lr], binding=h._binding(xs, "scaled_silu"))
            bi = F.HCFused.up(bot, *QI[i], xs, hc)
            bi_ref2 = api.run_gate_mean(xs, F.w8_linear(bot, *QU[i], (32, 64, 1, 8)), binding=h._binding(xs, "gate_mean"))
            worst[0] = max(worst[0], (di.float() - di_ref.float()).abs().max().item())
            worst[1] = max(worst[1], (bot.float() - bot_ref.float()).abs().max().item())
            worst[2] = max(worst[2], (bi.float() - bi_ref2.float()).abs().max().item() / bi_ref2.float().abs().max().item())
            worst[3] = max(worst[3], (bi.float() - ref_bi.float()).abs().max().item() / ref_bi.float().abs().max().item())
    out.append(f"M={M}: max|d| down {worst[0]:.3g} bot {worst[1]:.3g}; block_input rel vs same-input ref {worst[2]:.3g}, vs served {worst[3]:.3g}")
  M = 3
  xs = (torch.randn(M, 10240, device="cuda") * 0.5).to(torch.bfloat16)
  with set_forward_context(None, r.vllm_config):
      out.append(f"served _mix_normalized x97: {gtime(lambda: [h._mix_normalized(xs) for h in hcs]):.2f} ms")
  def fused_all(cd=(32, 512, 4, 4), cu=(8, 64, 8)):
      for i in range(len(hcs)):
          di, bot = fx.down(xs, *QD[i], lr, hc, cd)
          F.HCFused.up(bot, *QI[i], xs, hc, cu)
  out.append(f"fused x97: {gtime(fused_all):.2f} ms")
  res = []
  for cd in [(32, 512, 4, 4), (32, 512, 5, 4), (32, 256, 8, 4), (16, 512, 4, 4), (16, 256, 8, 2), (32, 512, 2, 4), (32, 1024, 2, 8), (16, 512, 5, 2)]:
      if 10240 % (cd[1] * cd[2]): continue
      try: res.append((gtime(lambda: [fx.down(xs, *QD[i], lr, hc, cd) for i in range(len(hcs))], it=10), cd))
      except Exception as e: out.append(f"cd {cd} err {str(e)[:100]}")
  res.sort(); out.append("down only: " + str([(round(a, 3), b) for a, b in res[:4]]))
  bots = [torch.randn(M, lr, device="cuda", dtype=torch.bfloat16) for _ in range(1)]
  res = []
  for bj, bk, nw in itertools.product([4, 8, 16], [64, 320 if False else 64], [2, 4, 8]):
      try:
          qi = [F.HCFused.interleave_up(q, s, hc, bj) for q, s in QU] if bj != BJ else QI
          res.append((gtime(lambda: [F.HCFused.up(bots[0], *qi[i], xs, hc, (bj, bk, nw)) for i in range(len(hcs))], it=10), (bj, bk, nw)))
      except Exception as e: out.append(f"cu {(bj,bk,nw)} err {str(e)[:100]}")
  res.sort(); out.append("up only: " + str([(round(a, 3), b) for a, b in res[:4]]))
except Exception:
  out.append(traceback.format_exc()[-1500:])
RESULT = "\n".join(out)
