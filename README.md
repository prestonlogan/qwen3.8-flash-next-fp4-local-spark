# Qwen3.8-Flash-Next-FP4-Local on one DGX Spark: ~51 tok/s prose

This is a single-Spark serving recipe for [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), NVIDIA's QAD-trained FP4 checkpoint. It adds a small vLLM plugin (`x/fp4patch.py` plus Triton kernels in `x/fp4x_kernels.py`) that speeds up the per-token decode path on GB10.

It starts from the model card's Spark recipe and runs on the `eugr/spark-vllm-b12x` image (vLLM 0.1.dev21830, b12x backends). The weights are unchanged. Every change happens at serve time.

Companion to [Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized](https://github.com/prestonlogan/Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized), which optimizes the pre-QAD NVFP4 checkpoint. Several kernels here are ports from that repo.

## Results

All numbers are tok/s, decode only, 400-token completions, one stream unless noted. They were measured on one DGX Spark (GB10, 128 GB).

| | Model card recipe | + 3 MTP drafts + async scheduling | **This repo** |
|---|---|---|---|
| Prose, greedy | 36.7 | 39.9 | **50.7 – 51.6** |
| Prose, sampled (T=1.0, top-p 0.95, top-k 20) | – | 36.7 | **49 – 51** |
| Code / JSON, greedy | 59 – 61 | 60 / 58 | 68 / 65.5 |
| Prose at 110k-token context | – | 40.8 | **47.2** |
| Aggregate at 2 / 4 streams | 51 / 68 | 58 / 75 | **74 / 98** |
| ms per decode step | 64 | 58 | **40.4** |

**Quality.** Decode-path perplexity is measured with prompts forced through the M≤16 kernels (`exp_q.sh`), over 12 texts and 9,588 tokens. It is **2.634** here vs **2.650** for the card recipe. The stock b12x path quantizes activations at small M, and the exact MXFP8 GEMV here wins back more than the 6-bit and NVFP4 weight copies lose.

Functional gates (`gates.py`) all pass: single and multi-turn tool calls, strict JSON, schema-guided JSON, 4/4 reasoning with thinking on, and a code-execution test.

Speculative decoding stays lossless. Greedy output matches argmax, and sampled output uses standard rejection sampling.

**Trade-off.** It uses 2 MTP drafts. With these cheaper steps, 2 drafts win on prose and sampled text. 3 drafts would give code/JSON about 72/71 tok/s but drop prose to about 49 greedy and 47 sampled. To get the 3-draft behaviour, set `FP4_SPEC='{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}'`.

## Quick start

Requirements:
- A DGX Spark with Docker and the NVIDIA runtime.
- The `eugr/spark-vllm-b12x:latest` image.
- The model in your Hugging Face cache (`huggingface-cli download nvidia/Qwen3.8-Flash-Next-FP4-Local`).

```bash
git clone https://github.com/prestonlogan/qwen3.8-flash-next-fp4-local-spark ~/fp4local
cd ~/fp4local
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
| `FP4X_MTP_W8=1` | MTP layer BF16 linears as int8 g64. | draft only | ~1 ms |
| `FP4X_DRAFT_VOCAB=dv65k.txt` | MTP draft head over a 65k-token NVFP4 subset instead of 248k. | draft only | ~2.5 ms |
| `FP4X_FAST_HEAD=1` | Target LM head: INT2 coarse pass, then exact MXFP8 rescoring of the top 512. A self-check runs at build time. | top-512 exact (argmax agree 1.000) | ~1 ms |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Removes a post-load dip in MemAvailable, to about 1.5 GiB, that could OOM the box. | – | stability |

Where the time goes now, per 40.4 ms step: about 33 ms in the verify CUDA graph (MoE experts are the largest part), about 5.5 ms for logits, sampling and the 2 draft steps, and about 1 ms of disk n-gram (PLE) reads.

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
| **E18** | **E15 with 2 drafts (shipped)** | **50.7** | **40.4** | 2.05 | 67.7 |
| E19 | + PLE spin-wait | 50.6 | 40.5 | 2.05 | 67.4 |
| E20b | adaptive draft depth (≤3, start 2) | 50.2 | 42.5 | 2.14 | 72.2 |

Sampled prose: E15 47.4, **E18 50.8**, E20b 48.6.

Decode-path PPL (`results/ppl_q.jsonl`):

| Config | PPL |
|---|---|
| base | 2.6501 |
| mx8 | 2.6362 |
| + HC W8 | 2.6350 |
| + W6 | 2.6312 |
| + qkvz NV4 (shipped) | 2.6339 |
| + qkv NV4 | 2.6438 |
| out_proj NV4 | 2.7696 |
| all-dense NV4 | 2.7903 |

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
- **RAM PLE** (`VLLM_PLE_TABLE_MEMORY=ram`). The 28.8 GB table counts against the GPU memory budget and doesn't fit next to the KV cache.
- **Torch profiler.** One profiler run was followed by a hard reset of the Spark on the next reload. Use CUDA-event spans instead (`x/probes/probe16*.py`).

## Layout

```
serve.sh              launcher (FP4_OPT=1 optimized, 0 = card recipe)
watchdog.sh           MemAvailable guard
x/fp4patch.py         vLLM plugin (patches above; also unused FP4X_PLE_QD / FP4X_PLE_SPIN / FP4X_HC_NV4)
x/fp4x_kernels.py     Triton GEMVs: mx8, bf16, w8, w6, nv4, int2 logits + quantizers
x/fp4patch-0.1.dist-info/   entry point registration
x/dv65k.txt           draft-vocab token ids (64,877 + tail)
x/fp4ext.py, x/rpc.py worker extension + /collective_rpc client for in-process probes (dev mode only)
x/stack2.txt, x/spec*.json  flag sets used by the experiments (@ROOT@ is substituted by exp.sh)
x/probes/             in-process timing/tuning scripts (run via rpc.py fx_exec /x/probes/<name>.py)
exp.sh, exp_q.sh      restart-and-benchmark / decode-path-PPL harnesses
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
