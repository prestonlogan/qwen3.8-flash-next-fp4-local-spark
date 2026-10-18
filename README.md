# Qwen3.8-Flash-Next-FP4-Local on one DGX Spark: ~55 tok/s prose

This is a single-Spark serving recipe for [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), NVIDIA's QAD-trained FP4 checkpoint. It adds a small vLLM plugin (`x/fp4patch.py` plus Triton kernels in `x/fp4x_kernels.py`) that speeds up the per-token decode path on GB10.

It starts from the model card's Spark recipe and runs on the `eugr/spark-vllm-b12x` image (vLLM 0.1.dev21830, b12x backends). The target weights are unchanged. Every change happens at serve time, plus a self-distilled draft-model patch that can only affect speed.

Companion to [Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized](https://github.com/prestonlogan/Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized), which optimizes the pre-QAD NVFP4 checkpoint. Several kernels here are ports from that repo.

## Results

All numbers are tok/s, decode only, 400-token completions, one stream unless noted. They were measured on one DGX Spark (GB10, 128 GB).

| | Model card recipe | + 3 MTP drafts + async scheduling | **This repo** |
|---|---|---|---|
| Prose, greedy | 36.7 | 39.9 | **55.0 – 55.6** |
| Prose, sampled (T=1.0, top-p 0.95, top-k 20) | – | 36.7 | **53.4 – 54.4** |
| Code / JSON, greedy | 59 – 61 | 60 / 58 | 71 – 73 / 69 – 70 |
| Prose at 110k-token context | – | 40.8 | **51.5 – 54.3** |
| Aggregate at 2 / 4 streams | 51 / 68 | 58 / 75 | **74 / 98** (before r32a) |
| ms per decode step | 64 | 58 | **38.8** |

**Quality.** Decode-path perplexity is measured with prompts forced through the M≤16 kernels (`exp_q.sh`), over 12 texts and 9,588 tokens. It is **2.633** here (mean of two runs) vs **2.650** for the card recipe. The stock b12x path quantizes activations at small M, and the exact MXFP8 GEMV here wins back more than the 6-bit and NVFP4 weight copies lose.

Functional gates (`gates.py`) all pass: single and multi-turn tool calls, strict JSON, schema-guided JSON, 4/4 reasoning with thinking on, and a code-execution test.

Speculative decoding stays lossless. Greedy output matches argmax, and sampled output uses standard rejection sampling.

**Trade-off.** It uses 2 MTP drafts. With these cheaper steps, 2 drafts win on prose and sampled text. 3 drafts would give code/JSON about 75/72 tok/s but drop prose to about 51.7 greedy and 48.8 sampled. To get the 3-draft behaviour, set `FP4_SPEC='{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}'`.

## Quick start

Requirements:
- A DGX Spark with Docker and the NVIDIA runtime.
- The `eugr/spark-vllm-b12x:latest` image.
- The model in your Hugging Face cache (`huggingface-cli download nvidia/Qwen3.8-Flash-Next-FP4-Local`).

```bash
git clone https://github.com/prestonlogan/qwen3.8-flash-next-fp4-local-spark ~/fp4local
cd ~/fp4local
./t8/build_patch.sh               # one-time: merge the r32a drafter LoRA -> x/mtp_patch_r32a.safetensors (~1 min, CPU)
./serve.sh 5810                    # detached container "fp4local", OpenAI API on :5810
nohup ./watchdog.sh >/dev/null 2>&1 &   # optional: kills the container if MemAvailable < 1.5 GiB
```

- A warm boot takes about 4–4.5 minutes. The first boot ever runs a one-time b12x autotune of about 15 minutes.
- `FP4_OPT=0 ./serve.sh` runs the plain card recipe (3 drafts + async) for comparison.
- Other overrides: `FP4_SPEC`, `FP4_COMP`, `FP4_MAXLEN` (default 262144), `FP4_SEQS`, `FP4_MNBT`, `FP4_GMU` (0.8), `FP4_NAME`, `FP4_SERVED`, `HF_CACHE`, `FP4_ROOT`.

`gateway/fp4local-llama-swap-start.sh` is the llama-swap adapter used on the author's machine (`cmd: …start.sh ${PORT}`, `cmdStop: docker stop -t 30 fp4local`).

## What the plugin does

`x/fp4patch.py` is a vLLM `general_plugins` entry point. `serve.sh` bind-mounts it into site-packages with its dist-info. Each patch is gated by an env var, touches only decode-shaped calls (M ≤ 16 rows), and leaves prefill untouched.

| Flag | What | Exact? | Step time saved (3-draft stack) |
|---|---|---|---|
| `FP4X_MX8=1` | Dense MXFP8 linears read b12x's packed fp8 rows and e8m0 scales directly. They run a bandwidth-tuned Triton GEMV with BF16 activations. | exact weights; better than stock | ~0.5 ms |
| `FP4X_DENSE=w6` | 6-bit g32 copies of the dense weights (GDN/QSA projections, shared experts). | quality-gated (PPL 2.631) | ~2 ms |
| `FP4X_NV4_SHAPES=16384x2560` | GDN `in_proj_qkvz` as NVFP4. | quality-gated (PPL 2.634) | ~1 ms |
| `FP4X_HC=1 FP4X_HC_W8=1` | Hyper-connection mixer matmuls (97 target layers) as int8 g64. Was cuBLAS BF16. | quality-gated (PPL 2.635) | ~2.4 ms |
| `FP4X_HC_FUSED=1` | Each W8 mixer as two kernels instead of six: a split-K down projection whose last CTA per tile casts, applies the bottleneck SiLU and re-zeroes its accumulator, and an up projection over stream-interleaved rows whose epilogue does the sigmoid gate and stream mean. | exact (bit-identical to the W8 path) | ~0.7 ms |
| `FP4X_GATE=w8` | MoE router gates (48 × 512×2560 BF16) as int8 g64. | quality-gated (PPL +0.04%) | ~0.25 ms |
| `FP4X_MTP_W8=1` | MTP layer BF16 linears as int8 g64. | draft only | ~1 ms |
| `FP4X_DRAFT_VOCAB=dv65k.txt` | MTP draft head over a 65k-token NVFP4 subset instead of 248k. | draft only | ~2.5 ms |
| `FP4X_MTP_PATCH=…/mtp_patch_r32a.safetensors` | Self-distilled MTP drafter weights, applied right after load (see below). | draft only | tok/step +4.5% |
| `FP4X_FAST_HEAD=1` | Target LM head: INT2 coarse pass (BK 512, at DRAM bandwidth), then one fused kernel rescoring the top 512 exactly from the MXFP8 rows. A self-check runs at build time. | top-512 exact (argmax agree 1.000) | ~1.6 ms |
| `FP4X_DRAFT_INT2=64` | Draft head as INT2 coarse over the 65k draft vocab, then exact NVFP4 rescoring of the top 64; the draft distribution q is those 64 tokens. | draft only (lossless: the rejection sampler uses the same q) | ~0.2 ms; sampled tok/step +3% |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Removes a post-load dip in MemAvailable, to about 1.5 GiB, that could OOM the box. | – | stability |

Where the time goes now, per 38.8 ms step (CUDA-event spans and graph-timed components at M=3, `x/probes/probe_*.py`):

| Part | ms | Floor at 243 GB/s |
|---|---|---|
| Verify graph | ~34 | |
| ‣ routed MoE GEMMs (b12x W4A16, ~23 unique experts per 3-token step) | ~12.1 | ~12.4 |
| ‣ rest of the MoE block: router, CSF scale expansion (1.35), route packing, shared expert on the aux stream | ~4.2 | |
| ‣ dense linears (W6 / NVFP4 Triton GEMVs, 240 calls) | ~10.5 | 8.3 |
| ‣ hyper-connection mixers (97 × 2 fused kernels) | ~3.4 | 2.7 |
| ‣ router gates (W8) | ~0.45 | 0.26 |
| ‣ GDN / QSA attention, norms, combines, PLE add | ~4.5 | – |
| 2 draft steps (`propose`: draft prefill graph ~1.6, decode graph ~1.2) | ~2.9 | |
| Target LM head (fast head) | ~1.1 | 1.0 |
| Disk PLE read gap before the graph | ~0.7 | |

The big kernels already run at 75–90% of DRAM bandwidth; what's left is mostly per-kernel latency in long chains of small launches.

## Experiment log

Greedy prose2 tok/s, ms/step, and tokens per step. Raw JSONL is in `results/`.

| Run | Change | Prose | ms/step | tok/step | Code |
|---|---|---|---|---|---|
| A | card recipe + 3 drafts + async | 39.9 | 58.0 | 2.31 | 60.2 |
| B | same, 2 drafts | 41.3 | 50.4 | 2.08 | 55.0 |
| E1 | `FORCE_A16=0` | 39.4 | 58.5 | 2.31 | 58.7 |
| E2 | FULL_AND_PIECEWISE + custom ops | 40.4 | 57.4 | 2.32 | 59.7 |
| E4 | `VLLM_B12X_BF16_GEMV=1` | 40.3 | 56.8 | 2.29 | 59.9 |
| E5 | MX8 exact GEMV | 40.9 | 57.0 | 2.33 | 60.1 |
| E6 | HC BF16 Triton | 40.2 | 57.7 | 2.32 | 59.1 |
| E7 | + draft vocab 65k | 42.2 | 54.5 | 2.30 | 62.7 |
| E8b | + HC W8 | 43.9 | 52.1 | 2.29 | 65.6 |
| E9 | + W6 dense + MTP W8 | 47.0 | 48.9 | 2.30 | 70.8 |
| E10 | 4 drafts | 43.6 | 54.2 | 2.36 | 74.9 |
| E11 | NV4-coarse fast head | 46.8 | 48.7 | 2.28 | 69.5 |
| E13 | + qkvz and qkv NV4 | 49.2 | 47.0 | 2.31 | 70.9 |
| E14 | qkvz-only NV4 + probabilistic drafts | 47.9 | 47.5 | 2.28 | 71.4 |
| E15 | + INT2 fast head | 48.9 | 46.5 | 2.27 | 72.3 |
| E16b | HC as NVFP4 | 27.3 | 39.3 | **1.07** | broken |
| E17 | PLE io_uring QD 256 | 49.5 | 46.7 | 2.31 | 72.1 |
| E18 | E15 with 2 drafts | 50.7 | 40.4 | 2.05 | 67.7 |
| E19 | + PLE spin-wait | 50.6 | 40.5 | 2.05 | 67.4 |
| E20b | adaptive draft depth (≤3, start 2) | 50.2 | 42.5 | 2.14 | 72.2 |
| AB | r32a drafter, hot-swapped, same-process A/B, 2 rounds each | 52.8 vs 50.4 | 40.9 | 2.16 vs 2.06 | 68.3 vs 67.1 |
| **E21** | **E18 + r32a drafter (shipped)** | **52.7** | **40.5** | 2.14 | 68.9 |
| E22 | E21 with 3 drafts | 51.7 | 46.4 | 2.40 | 75.5 |
| SHIP | shipped `serve.sh` defaults | 53.1 | 40.4 | 2.15 | 69.2 |
| E23 | `torch.compile` (inductor) on the drafter path | 51.3 | 42.0 | 2.16 | 65.6 |
| E24 | + HC W8 on the MTP mixers | 53.5 | 40.3 | 2.16 | 69.3 |
| E25 | + PDL launches for every plugin GEMV | 54.0 | 40.6 | 2.19 | 69.3 |
| E26 | + router gates W8 | 52.9 | 40.1 | 2.12 | 70.7 |
| **E27** | **E26 + fused HC mixers (shipped)** | **54.4** | **39.4** | 2.14 | 71.1 |
| SHIP2 | shipped `serve.sh` defaults | 55.3 | 39.45 | 2.18 | 71.6 |
| E28 | E27 with the shared expert on the main stream (`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`) | 52.8 | 40.6 | 2.14 | 69.2 |
| E29 | E27 without the GDN input-projection overlap (`VLLM_QWEN3_8_FLASH_NEXT_OVERLAP=0`) | 53.3 | 40.3 | 2.15 | 69.4 |
| E30 | E27 + W8 + fused mixers on the MTP layer | 55.1 | 39.7 | 2.19 | 71.9 |
| **E31** | **E27 + fast head at BK 512 + fused refine (shipped)** | **55.0** | **39.0** | 2.14 | 72.9 |
| **E32** | **E31 + INT2 draft head, top-64 refine (shipped)** | **55.2** | **38.8** | 2.14 | 70.9 |
| SHIP3 | shipped `serve.sh` defaults | 55.6 | 38.8 | 2.16 | 73.3 |
| E33 | SHIP3 with 3 drafts | 53.9 | 44.5 | 2.40 | 78.5 |
| E35 | SHIP3 + NVFP4 expert scales pre-expanded at load (exact) | 55.1 | 39.3 | 2.17 | 71.4 |
| E36 | SHIP3 + shared-expert tail fused into one kernel (exact) | 54.8 | 39.0 | 2.13 | 71.7 |
| E37 | SHIP3 config, same-session baseline for E35/E36 | 55.6 | 38.8 | 2.16 | 72.0 |
| E38 | SHIP3 + 98k-token draft vocab (ids < 98304), 2 runs | 56.6, 55.0 | 38.9, 39.0 | 2.20, 2.15 | 73.1, 72.6 |
| E39 | SHIP3 + 164k-token draft vocab | 54.6 | 39.8 | 2.17 | 71.5 |
| E40 | same-server A/B, 4 prose runs each: 65k vs 98k draft vocab | 55.1 vs 55.3 | 39.1 vs 39.5 | 2.155 vs 2.183 | |
| E41 | SHIP3 + head INT2 with 32-wide scale groups, C=1024 (3 runs) | 56.3, 54.9, 55.0 | 38.5, 39.1, 38.9 | 2.17, 2.15, 2.14 | 72.7 |
| E42 | same-server alternating A/B, 4 rounds: head g16/C512 vs g32/C1024 | 55.2 vs 55.3 | 39.07 vs 39.08 | | |
| E43 | SHIP3 + PLE read overlapped with the attention-metadata kernels (3 runs) | 55.7, 54.4, 54.0 | 38.8, 39.2, 39.3 | 2.16, 2.13, 2.12 | 73.0 |
| E44 | same-server alternating A/B, 5 rounds: stock vs (head g32 + early PLE) | 54.6 vs 54.7 | 39.15 vs 39.12 | | |
| E34 | draft logits / T for sampled drafts (T 1.0 / 0.85 / 0.7, runtime-tuned, n=24 each) | sampled 53.9 / 54.5 / 53.7 | | | |

Sampled prose: E15 47.4, E18 50.8, E20b 48.6, **E21 50.9**, E22 48.8, E24 50.7, E25 50.5, E26 50.8, **E27 51.3**, SHIP2 51.3, E31 52.4, **E32 54.4** (sampled tok/step 2.07 → 2.14), SHIP3 53.4.

Greedy tok/step moves by ±0.03 between identical runs because the drafts are sampled, so ms/step is the cleaner signal for step-cost changes.

Decode-path PPL (`results/ppl_q.jsonl`):

Repeated runs of the same config differ by up to ±0.007 (about ±0.25%), because the split-K GEMVs accumulate with atomics and the summation order changes from run to run. Treat single readings near the +0.3% budget as inconclusive and repeat them.

| Config | PPL |
|---|---|
| base | 2.6501 |
| mx8 | 2.6362 |
| + HC W8 | 2.6350 |
| + W6 | 2.6312 |
| + qkvz NV4 (shipped) | 2.6339 |
| + fast head | 2.6322 |
| + router gates W8 | 2.6332, 2.6370 |
| + fused HC mixers | 2.6396, 2.6262 |
| + fused fast-head refine, INT2 draft head (shipped) | 2.6348 |
| + qkv NV4 | 2.6438 |
| out_proj NV4 | 2.7696 |
| all-dense NV4 | 2.7903 |
| drop routed experts with weight < 0.05 (renormalized) | 2.6623 |
| drop routed experts with weight < 0.07 (renormalized) | 2.8228 |

## Drafter self-distillation (r32a)

The FP4-Local MTP drafter, LM head and embeddings are **bit-identical** to the pre-QAD checkpoint (only the 48 target layers changed with QAD), so the companion repo's r32a adapter applies directly. That adapter is a rank-32 LoRA on the drafter's dense matrices and norms, trained by FastMTP-style self-distillation on the target's own greedy prose.

- `t8/build_patch.sh` merges `adapters/r32a/lora_r32a.pt` into 22 BF16 drafter tensors. The output's SHA-256 matches the companion repo's patch exactly.
- `FP4X_MTP_PATCH` copies the patch into the drafter's parameters right after `load_model`, before profiling, graph capture and the lazy draft-side W8 copies. Swapping tensors inside the loader's iterator is rejected by b12x's checkpoint routing.
- Served same-process A/B: prose greedy +4.7% (tok/step 2.06 → 2.16), sampled +2%, code +2%. Output is unchanged, because the target verifies every token.
- Retraining on FP4-Local's own data (`t8/`: `gen_tok.py` → `capture.py` → `trainrun.sh`) matched r32a offline (val tok/step 2.538 vs 2.544), and continuing from r32a added only +0.2%. This matches the companion repo's finding that more teacher-forced data doesn't transfer. `results/t8_offline.txt` has the logs.

## Dead ends, so you don't repeat them

- **NVFP4 hyper-connection mixers.** These carry about 9% relative error per matmul, vs 0.8% for int8, and 194 of them in series wreck the output. NVFP4 on all dense layers or on `out_proj` costs +5–6% PPL.
- **Adaptive draft depth** (`adaptive_speculative_tokens_window`).
  - Configs that use acceptance adaptation alone trip an assertion. They need `num_speculative_tokens_per_batch_size`, for example `[[1,8,3]]`.
  - They also need explicit `cudagraph_capture_sizes` that include 3/6/9. Otherwise b12x raises "kernel resolution is frozen" during capture.
  - Once it works, it is still slower than static 2 drafts on prose (step overhead without extra acceptance).
- **Faster disk PLE.** Each decode step issues about 128 random 4 KiB `O_DIRECT` reads for 64 rows of 90 B, taking about 0.85 ms.
  - Back to back the same batch takes about 0.25 ms. The difference is CPU idle wake after the ~40 ms GPU wait. NVMe power states and queue depth are not the cause.
  - Queue depth 256, spin-waiting and an NVMe keepalive did not move end-to-end speed.
  - Row reuse is only 18–39%, so a RAM row cache isn't worth it.
- **Retraining the drafter** on FP4-Local's own greedy outputs: no better than r32a (see above).
- **3 drafts with r32a**: 51.7 greedy / 48.8 sampled, vs 52.7 / 50.9 with 2 drafts. Code/JSON improve to 75/72.
- **RAM PLE** (`VLLM_PLE_TABLE_MEMORY=ram`). The 28.8 GB table counts against the GPU memory budget and doesn't fit next to the KV cache.
- **`torch.compile`** (E23). The FULL verify graph runs with compile mode NONE (b12x's breakable graphs force it), so only the drafter got compiled. Prose 51.3, slower than eager.
- **Skipping low-weight routed experts** at decode (`FP4X_MOE_TAU`, debug flag in `fp4patch.py`). The top-10 routing weights are flat (mean sorted weights 0.205 … 0.058), so dropping even the weakest costs PPL +1.1% (τ=0.05) or +7% (τ=0.07).
- **Programmatic dependent launch** for the plugin's GEMVs (E25). A chain of PDL GEMVs saves about 2 µs per kernel in isolation, but in the model most neighbours are b12x/torch kernels that never trigger early, so step time didn't move.
- **W6 hyper-connection mixers.** The mixer GEMVs (336×10240, 10240×320) are latency-bound at 17–21 µs; 6-bit weights were no faster than int8 and have 3× the error.
- **Semaphore split-K for the W6 dense GEMVs.** No gain for any dense shape (out_proj stays near 190 GB/s); split-K only pays off for the 11-tile mixer down projection.
- **b12x small-M MoE knobs** (`B12X_W4A16_SMALL_M_OCCUPANCY=2`, `B12X_W4A16_SMALL_M_SPLITK=1`): within ±2% on the 48-layer MoE graph.
- **A custom NVFP4 MoE decode kernel** (`x/fp4x_moe.py`, Triton, reads b12x's packed weights and scales in place). It matches b12x to 6e-4 relative, but on real 3-row routing it takes 18.3 ms for the 48 MoE blocks against b12x's 16.3. Skipping the routed GEMMs shows why: they cost 12.1 ms, which is the bandwidth floor for the ~23 unique experts per layer (12.4 ms at 243 GB/s). b12x's routed kernels have no headroom left. The rest of the block (4.2 ms) is the router, shared expert, scale expansion, and route packing.
- **Pre-expanding the NVFP4-CSF expert scales at load** (E35). b12x stores expert scales compressed (CSF, 3.97 GB) and expands the routed experts' scales into a shared scratch on every call; that kernel takes 1.35 ms/step at 3 rows. Expanding all 48 layers once at load (+7.55 GB) is bit-exact (PPL unchanged) but no faster: 55.1 tok/s at 39.3 ms/step. The expanded scales are 7 MB per layer per step read from DRAM, where CSF reads half that and its output is still in L2 when the GEMM reads it. Two traps if you retry it: the W4A16 kernels read `representation.value.w13_scale`, not `_impl.w1_blockscale`, and setting `nvfp4_csf=None` changes b12x's tuning plan and fails preparation.
- **Fusing the shared expert's tail** (E36). One kernel for silu-and-mul, the W6 down projection and the sigmoid gate, replacing five launches (act_and_mul, GEMV, cuBLAS dot + reduce, sigmoid, mul). It is bit-exact and saves 0.37 ms on the 48 graph-timed MoE blocks, but the full step was 39.0 ms against 38.8 for the same-session baseline. The shared expert runs on vLLM's aux stream, so it is off the critical path.
- **A bigger draft vocabulary** (E38–E40). 3.0–3.3% of the target's prose tokens fall outside the 65k draft vocab (common words such as " manageable", " classrooms"), so they can never be drafted. Ids below 98304 cut that to 0.5% and raised tok/step by 1.3%, but the larger INT2 draft head cost 0.4 ms/step, a wash (55.3 vs 55.1 over 4 runs each). A 164k vocab was slower. A frequency-chosen subset could do better, but there's no large target-output corpus to rank tokens by.
- **Removing idle gaps between graphs** (E43, E44, and the reason E35/E36 didn't pay either). CUDA-event timelines put ~2 ms of each step outside the three graphs: 1.3 ms between verify and drafting (mostly the 1.1 ms LM head) and 0.8 ms before the next verify (0.24 ms of attention-metadata kernels, then ~0.36 ms idle for the disk PLE read). `FP4X_PLE_EARLY=1` stages the PLE hash before the metadata kernels so the read overlaps them, and that gap shrinks by 0.25 ms. But the verify graph then runs 0.4–0.5 ms slower and the step doesn't move. The direct test was to busy-wait the CPU before the verify launch: 0.5 ms of extra GPU idle cost nothing (the graph ran correspondingly faster), and only 1 ms of idle started to cost time. With no throttle flags and clocks at 2.5 GHz, sub-0.5 ms idle on this box is free. Only less work (fewer bytes) shortens a step.
- **INT2 head with 32-wide scale groups** (`FP4X_HEAD_G=32 FP4X_HEAD_C=1024`): 16% fewer head bytes (0.99 → 0.81 ms in isolation, argmax exact, top-20 inside the top 1024 on all 869 rows tested, PPL 2.6355), but the alternating A/B showed no step change.
- **6-bit HC mixers** (`FP4X_HC_W6=1`, unfused, for a quality check): PPL 2.6615, +1.0%. **5-bit dense** (fake-quantized in the W6 layout, `FP4X_DENSE_QMAX=15`): PPL 2.6669, +1.25%. Both over budget. Per-group MSE clip search (`FP4X_DENSE_CLIP=15`) makes 5-bit worse (2.6837) and leaves 6-bit unchanged (2.6346): clipping the group outliers costs more than the finer step saves. An activation-aware variant (`FP4X_DENSE_STATS=1` calibration server + `x/probes/calib.py` / `dstats_dump.py`, then `FP4X_DENSE_AWQ=/x/dense_stats.pt`; power-of-2 per-channel scales, h-weighted error) is implemented but not yet measured: the first attempt matched stats to the wrong layers (the calibration server also loaded the vision tower; calibrate with `--language-model-only`). Upside if it ever fits the budget: ~180 MB/step, about 0.7 ms.
- **A 1-bit coarse head** for a two-stage head, re-checked on 857 rows: the true argmax ranks at most 412 (g64 scales), so C=2048 would be greedy-exact, but the top 20 needs C=8192. It would save ~0.3 ms of bandwidth; not built, given the step's insensitivity to sub-ms savings.
- **Draft temperature for sampled drafts** (E34, `FP4X_DRAFT_TSCALE`): 0.85 gave +0.6 tok/s sampled, inside the noise; not shipped.
- **3 drafts after the head kernels** (E33): prose 53.9 at 44.5 ms/step, but code/JSON reach 78.5/77.8. Use `FP4_SPEC` with 3 drafts for code-heavy work.
- **Hiding the PLE disk read behind layer 0.** The read takes 0.285 ms for 48 rows once the token ids are known. The CPU spends the rest of the wait blocked on the previous step, so hiding it is worth at most ~0.35 ms and needs an AOT-cache bust; deprioritized.
- **HC W8 on the MTP mixers** (E24), also with the fused kernels (E30): neutral.
- **Turning off vLLM's stream overlap.** Running the shared expert on the main stream (E28) costs 1.2 ms/step, and serializing GDN's two input projections (E29) costs 0.9 ms/step. The small decode kernels leave enough of the GPU idle that a second stream fills it, so the defaults stay.
- **e4m3 group scales for the INT2 head.** 16% fewer bytes, but no faster at M=3 (the dequant becomes the limit).
- **1-bit (sign × per-16 mean) coarse head.** On 1,200 real decode rows the true argmax ranked as low as 487 and the top-20 often fell outside the top 1024, against a worst rank of 17 for INT2.
- **Fast-head candidate recall, for the record.** On 1,200–1,500 real decode rows the true argmax was always within the INT2 top 30. The true top 20 was inside the top 512 for all but one row, where one top-20 logit stays a coarse estimate. Greedy output is unaffected; that row's top-k-20 sampling distribution is very slightly off.
- **Disabling kernel nodes in the captured graph to cost them** (`FP4X_KEEP_GRAPH=1` + `x/probes/probe_ablate.py`). Without a profiler this looked like a way to price each kernel class. It isn't reliable: garbage activations change the MoE routing and with it the MoE cost (one run dropped from 33.9 to 19 ms), and disabling index-producing kernels faults the GPU. Only the big GEMV classes gave usable numbers: W6 5.3 ms (204 calls), qkvz NV4 3.3 ms (36), HC mixers 2.7 ms (194).
- **Torch profiler.** One profiler run was followed by a hard reset of the Spark on the next reload. Use CUDA-event spans instead (`x/probes/probe16*.py`).

## Layout

```
serve.sh              launcher (FP4_OPT=1 optimized, 0 = card recipe)
watchdog.sh           MemAvailable guard
x/fp4patch.py         vLLM plugin (patches above; also unused FP4X_PLE_QD / FP4X_PLE_SPIN / FP4X_HC_NV4 and the
                      debug ablations FP4X_MOE_TAU / FP4X_MOE_ZERO / FP4X_HC_ZERO / FP4X_MX8_ZERO)
x/fp4x_kernels.py     Triton GEMVs: mx8, bf16, w8, w6, nv4, int2 logits + quantizers; HCFused mixer kernels
x/fp4patch-0.1.dist-info/   entry point registration
x/dv65k.txt           draft-vocab token ids (64,877 + tail)
x/fp4ext.py, x/rpc.py worker extension + /collective_rpc client for in-process probes (dev mode only)
x/stack2.txt, x/spec*.json  flag sets used by the experiments (@ROOT@ is substituted by exp.sh)
x/probes/             in-process timing/tuning scripts (run via rpc.py fx_exec /x/probes/<name>.py); mtp_swap.py hot-swaps drafter weights
adapters/r32a/        drafter LoRA (from the companion repo); t8/build_patch.sh merges it
t8/                   drafter self-distillation pipeline (corpus, capture, train, merge)
ab_swap.sh            same-process A/B of a hot-swapped drafter patch
exp.sh, exp_q.sh      restart-and-benchmark / decode-path-PPL harnesses
run_ship.sh           validate serve.sh defaults: bench, gates, 110k context
spans.sh, abl.sh      CUDA-event step accounting on the dev server; ablation sequences
q38bench.py, ppl.py, gates.py, t8_long.py, summ.py   benchmarks
results/              raw JSONL for every run above
gateway/              llama-swap adapter
```

Dev-mode runs (`x/stack2.txt`) set `VLLM_SERVER_DEV_MODE=1`, which exposes `/collective_rpc` and lets probes execute code inside the worker. Don't use dev mode on a network-reachable server. `serve.sh`'s default mode doesn't enable it.

## Credits

- NVIDIA, for the FP4-Local checkpoint and its Spark recipe.
- eugr, for the `spark-vllm-b12x` image.
- The b12x / vLLM contributors.
- The draft-vocab subset and several kernel designs come from the E46 work in the companion repo, which builds on MiaAI-Lab's recipe.

License: AGPL-3.0-or-later.
