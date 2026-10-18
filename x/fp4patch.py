# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of qwen3.8-flash-next-fp4-local-spark.
# vLLM general plugin: decode-path Triton GEMVs for nvidia/Qwen3.8-Flash-Next-FP4-Local on GB10.
#   FP4X_MX8=1   MXFP8 dense linears (M <= FP4X_MX8_MAXM rows) read b12x's packed fp8 rows + e8m0 scale rows
#                exactly and run a bandwidth-tuned Triton GEMV instead of the b12x block-scaled GEMM.
#   FP4X_HC=1    hyper-connection down/inject and up BF16 projections through a Triton BF16 GEMV (M <= 16).
#   FP4X_HC_FUSED=1 (with FP4X_HC_W8=1) each mixer as two fused kernels (fp4x_kernels.HCFused), same rounding points.
# Both are exact-weight paths (fp32 accumulation; only the summation order differs).
import os, sys

CFG_MX8 = {  # (N, K): (BN, BK, KSPLIT, warps), tuned at M=4/8 on GB10 with L2-cold weights
    (16384, 2560): (16, 512, 1, 4), (96, 2560): (16, 128, 4, 2), (13312, 2560): (32, 512, 1, 4),
    (640, 2560): (16, 512, 1, 2), (1280, 2560): (16, 256, 1, 4), (2560, 6144): (32, 512, 1, 4),
    (2560, 640): (64, 128, 1, 4),
}
CFG_W6 = {(16384, 2560): (32, 256, 1, 4), (96, 2560): (16, 256, 2, 4), (13312, 2560): (16, 256, 1, 4),
          (640, 2560): (16, 256, 1, 2), (1280, 2560): (16, 256, 1, 2), (2560, 6144): (32, 256, 1, 4), (2560, 640): (32, 128, 1, 2)}
CFG_NV4 = {}
CFG_BG = {(336, 10240): (16, 512, 1, 4), (10240, 320): (32, 64, 1, 4), (512, 2560): (16, 256, 1, 4)}
CFG_HC_NV4 = {(336, 10240): (16, 128, 2, 20), (10240, 320): (32, 64, 4, 1)}
CFG_W8 = {(336, 10240): (32, 512, 4, 4), (10240, 320): (32, 64, 1, 8), (512, 2560): (16, 512, 1, 4)}
_stats = {"mx8": 0, "mx8_skip": 0}
_DRAFT_INV_T = {}
_LOWBIT_FOR = None  # set by _patch_mx8: B12xBlockscaledLinear -> its low-bit decode copy (or None)


def _log(msg):
    print(f"[fp4x] {msg}", file=sys.stderr, flush=True)


def _mx8_cfg(n, k):
    c = CFG_MX8.get((n, k))
    if c is None:
        bk = 512 if k % 512 == 0 else 256 if k % 256 == 0 else 128 if k % 128 == 0 else 64
        c = (32, bk, 1, 4)
    return c


def _patch_mx8():
    import torch
    from vllm.model_executor.kernels.linear import b12x_blockscaled as BB
    import fp4x_kernels as F
    maxm = int(os.environ.get("FP4X_MX8_MAXM", "16"))
    dense_mode = os.environ.get("FP4X_DENSE", "mx8")  # mx8 (exact) | w6 | nv4
    cache = {}
    orig = BB.B12xBlockscaledLinear.run

    nv4_shapes = {tuple(int(v) for v in t.split("x")) for t in os.environ.get("FP4X_NV4_SHAPES", "").split(",") if t}

    def mode_for(n, k):
        return "nv4" if (n, k) in nv4_shapes else dense_mode

    def lowbit(self, w, s):
        key = id(self)
        q = cache.get(key)
        if q is None and not torch.cuda.is_current_stream_capturing():
            wf = F.mx8_dequant(w, s)
            q = cache[key] = F.quantize_w6(wf) if mode_for(*w.shape) == "w6" else F.quantize_nv4(wf)
            del wf
            if len(cache) in (1, 200): _log(f"dense {dense_mode}: quantized {len(cache)} layers")
        return q

    def lowbit_for(lin):
        pk = lin.packed
        w = pk.weight.values
        if w.dtype == torch.uint8:
            w = w.view(torch.float8_e4m3fn)
        if lin.recipe != "mxfp8" or pk.padded_in_features != pk.in_features or w.dim() != 2 or mode_for(*w.shape) != "w6":
            return None
        return lowbit(lin, w, pk.weight.scale_rows.view(torch.uint8).view(w.shape[0], -1))

    global _LOWBIT_FOR
    if dense_mode == "w6":
        _LOWBIT_FOR = lowbit_for
    seen = set()

    def run(self, source, bias):
        key = (self.recipe, tuple(source.shape), bias is None, getattr(self.packed, "padded_in_features", None),
               getattr(self.packed, "in_features", None))
        if key not in seen and len(seen) < 60:
            seen.add(key)
            w_ = getattr(getattr(self.packed, "weight", None), "values", None)
            sr_ = getattr(getattr(self.packed, "weight", None), "scale_rows", None)
            _log(f"run {key} w={None if w_ is None else (w_.dtype, tuple(w_.shape), w_.stride())} s={None if sr_ is None else (sr_.dtype, tuple(sr_.shape))}")
        if self.recipe == "mxfp8" and source.dim() == 2 and source.shape[0] <= maxm and bias is None:
            pk = self.packed
            w = pk.weight.values
            if w.dtype == torch.uint8:
                w = w.view(torch.float8_e4m3fn)
            if pk.padded_in_features == pk.in_features and w.dtype == torch.float8_e4m3fn and w.dim() == 2:
                s = pk.weight.scale_rows.view(torch.uint8).view(w.shape[0], -1)
                n, k = w.shape
                _stats["mx8"] += 1
                if (_stats["mx8"] + _stats["mx8_skip"]) % 20000 == 0: _log(f"stats {_stats}")
                if dense_mode != "mx8" and n < 100000:
                    q = lowbit(self, w, s)
                    if q is not None:
                        if mode_for(n, k) == "w6":
                            return F.w6_linear(source, *q, CFG_W6.get((n, k)) or (32, 256, 1, 4))
                        return F.nv4_linear(source, *q, CFG_NV4.get((n, k)) or (32, 256 if k % 256 == 0 else 128 if k % 128 == 0 else 64, 4))
                if os.environ.get("FP4X_MX8_ZERO") == "1":
                    return torch.zeros((source.shape[0], n), dtype=source.dtype, device=source.device)
                return F.mx8_linear(source, w, s, _mx8_cfg(n, k))
        _stats["mx8_skip"] += 1
        if (_stats["mx8"] + _stats["mx8_skip"]) % 20000 == 0: _log(f"stats {_stats}")
        return orig(self, source, bias)

    BB.B12xBlockscaledLinear.run = run
    _log(f"MXFP8 decode GEMV installed (M <= {maxm})")


def _patch_hc():
    import torch
    from vllm.models.qwen4_exp.nvidia import hyperconnection as HC
    from vllm.utils.torch_utils import direct_register_custom_op
    import fp4x_kernels as F

    zero = os.environ.get("FP4X_HC_ZERO") == "1"
    w8 = os.environ.get("FP4X_HC_W8") == "1"
    hc_nv4 = os.environ.get("FP4X_HC_NV4") == "1"
    q4 = {}
    q8 = {}
    cnt = [0]

    def _bg(x: torch.Tensor, w: torch.Tensor, allow_w8: bool) -> torch.Tensor:
        cnt[0] += 1
        if cnt[0] in (1, 100, 10000): _log(f"hc op call #{cnt[0]} x={tuple(x.shape)} w={tuple(w.shape)}")
        if zero and x.shape[0] <= 16:
            return torch.zeros((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)
        if hc_nv4 and allow_w8 and x.shape[0] <= 16:
            key = (w.data_ptr(), tuple(w.shape))
            qs = q4.get(key)
            if qs is None and not torch.cuda.is_current_stream_capturing():
                qs = q4[key] = F.quantize_nv4(w)
            if qs is not None:
                return F.nv4_linear(x.contiguous(), *qs, CFG_HC_NV4.get(tuple(w.shape)) or (32, 64, 4, 1))
        if w8 and allow_w8 and x.shape[0] <= 16:
            key = (w.data_ptr(), tuple(w.shape))
            qs = q8.get(key)
            if qs is None and not torch.cuda.is_current_stream_capturing():
                qs = q8[key] = F.quantize_w8(w)
                if len(q8) in (1, 100): _log(f"HC W8 quantized {len(q8)} weights")
            if qs is not None:
                return F.w8_linear(x.contiguous(), *qs, CFG_W8.get(tuple(w.shape)) or (32, 256 if w.shape[1] % 256 == 0 else 64, 1, 4))
        if x.shape[0] <= 16:
            n, k = w.shape
            c = CFG_BG.get((n, k)) or (32, 256 if k % 256 == 0 else 64, 1, 4)
            return F.bf16_linear(x.contiguous(), w, c)
        return torch.nn.functional.linear(x, w)

    def _bg_fake(x: torch.Tensor, w: torch.Tensor, allow_w8: bool) -> torch.Tensor:
        return x.new_empty((x.shape[0], w.shape[0]))

    direct_register_custom_op(op_name="fp4x_bf16_linear", op_func=_bg, fake_impl=_bg_fake)
    op = torch.ops.vllm.fp4x_bf16_linear
    orig = HC.GatedResidual._mix_normalized
    fused = w8 and os.environ.get("FP4X_HC_FUSED") == "1"
    fx = []
    qu = {}

    def _mix_fused(self, normalized):
        # Two kernels per mixer (see fp4x_kernels.HCFused); bit-identical to the W8 GEMV + b12x silu / gate_mean path.
        wd = (self.input_mix_weight_down_block_inject if self.use_combine else self.input_mix_weight_down).weight
        wu = self.input_mix_weight_up.weight
        kd, ku = (wd.data_ptr(), tuple(wd.shape)), (wu.data_ptr(), tuple(wu.shape))
        if kd not in q8 or ku not in qu or not fx:
            if torch.cuda.is_current_stream_capturing():
                return None
            if not fx:
                fx.append(F.HCFused(normalized.device))
            if kd not in q8:
                q8[kd] = F.quantize_w8(wd)
            if ku not in qu:
                qu[ku] = F.HCFused.interleave_up(*F.quantize_w8(wu), self.hc_count, 8)
                if len(qu) in (1, 97): _log(f"HC fused: prepared {len(qu)} mixers")
        x = normalized.contiguous()
        di, bot = fx[0].down(x, *q8[kd], self.lora_rank, self.hc_count, (32, 256, 8, 4))
        block_input = F.HCFused.up(bot, *qu[ku], x, self.hc_count, (8, 64, 4))
        injection = di[:, self.lora_rank : self.lora_rank + self.hc_count] if self.use_combine else None
        return block_input, injection

    def _mix_normalized(self, normalized):
        if self.tp_size > 1:
            return orig(self, normalized)
        api = HC._hyperconnection_api() if self.workspace is not None else None
        aw = "mtp" not in (self._preparation_prefix or "") or os.environ.get("FP4X_HC_W8_MTP") == "1"
        if fused and aw and normalized.shape[0] <= 16:
            r = _mix_fused(self, normalized)
            if r is not None:
                return r
        if self.use_combine:
            down_and_injection = op(normalized, self.input_mix_weight_down_block_inject.weight, aw)
            projected_down = down_and_injection[:, : self.lora_rank]
            injection = down_and_injection[:, self.lora_rank : self.lora_rank + self.hc_count]
        else:
            projected_down = op(normalized, self.input_mix_weight_down.weight, aw)
            injection = None
        bottleneck = (
            api.run_scaled_silu(projected_down, binding=self._binding(normalized, "scaled_silu"))
            if api is not None else HC.hc_silu(projected_down, self.hc_count)
        )
        gate_logits = op(bottleneck, self.input_mix_weight_up.weight, aw)
        block_input = (
            api.run_gate_mean(normalized, gate_logits, binding=self._binding(normalized, "gate_mean"))
            if api is not None else HC.hc_gate_mix(normalized, gate_logits, self.hc_count)
        )
        return block_input, injection

    HC.GatedResidual._mix_normalized = _mix_normalized
    _log("HC BF16 decode GEMV installed")
    mtp_w8 = os.environ.get("FP4X_MTP_W8") == "1"
    gate_mode = os.environ.get("FP4X_GATE", "")  # target router gates: w8 (approx) | bf16 (exact weights)
    if mtp_w8 or gate_mode:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        orig_apply = UnquantizedLinearMethod.apply
        gate_calls = [0]

        def apply(self, layer, x, bias=None):
            if bias is None and layer.weight.dim() == 2:
                prefix = str(getattr(layer, "prefix", ""))
                if mtp_w8 and prefix.startswith("mtp."):
                    x2 = x.reshape(-1, x.shape[-1])
                    return op(x2, layer.weight, True).view(*x.shape[:-1], layer.weight.shape[0])
                if gate_mode and prefix.endswith(".mlp.gate") and x.reshape(-1, x.shape[-1]).shape[0] <= 16:
                    x2 = x.reshape(-1, x.shape[-1]).contiguous()
                    gate_calls[0] += 1
                    if gate_calls[0] in (1, 1000): _log(f"router gate {gate_mode} call #{gate_calls[0]} {prefix} x={tuple(x2.shape)}")
                    if gate_mode == "w8":
                        y = op(x2, layer.weight, True)
                    else:
                        y = F.bf16_linear(x2, layer.weight, CFG_BG.get(tuple(layer.weight.shape)) or (16, 256, 1, 4))
                    return y.view(*x.shape[:-1], layer.weight.shape[0])
            return orig_apply(self, layer, x, bias)

        UnquantizedLinearMethod.apply = apply
        _log(f"UnquantizedLinear decode hook: mtp_w8={mtp_w8} gate={gate_mode or 'off'}")


def _patch_draft_vocab():
    """MTP draft head over a reduced vocabulary (draft-only: verification stays exact)."""
    import torch
    from vllm.models.qwen4_exp.nvidia import mtp as MTP
    import fp4x_kernels as F
    path = os.environ["FP4X_DRAFT_VOCAB"]
    cfg = tuple(int(v) for v in os.environ.get("FP4X_DRAFT_CFG", "32,256,4").split(","))
    dc = int(os.environ.get("FP4X_DRAFT_INT2", "0"))  # >0: INT2 coarse + exact refine of the top dc rows
    dts = float(os.environ.get("FP4X_DRAFT_TSCALE", "1.0"))  # draft logits / dts (sampled drafts only differ)
    orig = MTP.Qwen4ExpMTP.compute_logits

    def build(self, dev):
        lm = self.lm_head
        ids = sorted({int(l) for l in open(path) if l.strip()} | set(range(lm.org_vocab_size - 320, lm.org_vocab_size)))
        idx = torch.tensor([i for i in ids if i < lm.weight.shape[0]], device=dev, dtype=torch.long)
        w = lm.weight.data; sc = lm.weight_scale.data
        n, kb = sc.shape
        scu = sc.view(torch.uint8).reshape(n // 128, kb // 4, 32, 4, 4).transpose(1, 3).reshape(n, kb)
        self._fx_idx = idx
        self._fx_w = w.index_select(0, idx).contiguous()
        self._fx_s = scu.index_select(0, idx).contiguous().view(torch.float8_e4m3fn)
        self._fx_gs = float(lm.weight_global_scale.item())
        self._fx_vocab = lm.weight.shape[0]
        _log(f"draft vocab head built: {idx.numel()} ids, {(self._fx_w.numel() + self._fx_s.numel()) / 1e6:.0f} MB")
        if dc:
            q = [F.quantize_int2(F.dequant_nv4(self._fx_w[a:a + 8192], self._fx_s[a:a + 8192], self._fx_gs)) for a in range(0, idx.numel(), 8192)]
            self._fx_c = (torch.cat([t[0] for t in q]), torch.cat([t[1] for t in q]))
            _log(f"draft head INT2 coarse + exact NVFP4 refine of the top {dc}: {sum(t.numel() * t.element_size() for t in self._fx_c) / 1e6:.0f} MB")

    def compute_logits(self, hidden_states, spec_step_idx=0):
        if getattr(self, "_fx_idx", None) is None:
            if torch.cuda.is_current_stream_capturing() or getattr(self.lm_head, "weight", None) is None:
                return orig(self, hidden_states, spec_step_idx)
            build(self, hidden_states.device)
        h = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
        if dc and h.shape[0] <= 16:
            # Draft q is restricted to the coarse top-dc rows, scored exactly (lossless: the rejection sampler uses this q).
            co = F.int2_logits(h, *self._fx_c, self._fx_idx.numel())
            cand = torch.topk(co, dc, dim=-1).indices
            vals = F.refine_nv4(h, cand, self._fx_w, self._fx_s, self._fx_gs)
            if dts != 1.0:
                # Sharper draft q at any temperature (argmax unchanged). A device scalar so probes can retune it
                # under CUDA graphs (fp4patch._DRAFT_INV_T).
                if _DRAFT_INV_T.get(h.device) is None:
                    _DRAFT_INV_T[h.device] = torch.full((1,), 1.0 / dts, dtype=torch.float32, device=h.device)
                vals = vals * _DRAFT_INV_T[h.device]
            full = torch.full((h.shape[0], self._fx_vocab), float("-inf"), dtype=torch.bfloat16, device=h.device)
            return full.scatter_(1, self._fx_idx[cand], vals.to(torch.bfloat16))
        sub = F.nv4_linear(h, self._fx_w, self._fx_s, self._fx_gs, cfg)
        full = torch.full((h.shape[0], self._fx_vocab), float("-inf"), dtype=sub.dtype, device=sub.device)
        full.index_copy_(1, self._fx_idx, sub)
        return full

    MTP.Qwen4ExpMTP.compute_logits = compute_logits
    _log(f"draft vocab head patch installed ({path})")


def _patch_fast_head():
    """Target LM head for decode rows: NVFP4 coarse pass over the vocabulary, then the top-C candidates per row are
    recomputed exactly from the MXFP8 head (fp32 accumulate) and scattered in. Argmax is exact whenever the true
    argmax is inside the coarse top-C; the top-C logits are exact (default chat top_k=20 sampling is unaffected)."""
    import torch
    from vllm.models.qwen4_exp.nvidia import model as MD
    import fp4x_kernels as F
    C = int(os.environ.get("FP4X_HEAD_C", "512"))
    orig = MD.Qwen4ExpForCausalLM.compute_logits
    st = {}

    def build(self, h):
        lp = self.logits_processor
        if lp.scale != 1.0 or lp.soft_cap is not None:
            st["off"] = True; _log("fast head disabled (scale/soft_cap)"); return
        pk = self.lm_head.b12x_mxfp8_packed_weight
        w = pk.weight.values; sc = pk.weight.scale_rows.view(torch.uint8).view(w.shape[0], -1)
        st["w"], st["s"] = w, sc
        P, S, gs = None, None, None
        wf = F.mx8_dequant(w, sc)
        st["int2"] = os.environ.get("FP4X_HEAD_COARSE", "int2") == "int2"
        st["coarse"] = F.quantize_int2(wf) if st["int2"] else F.quantize_nv4(wf); del wf
        ref = orig(self, h)
        st["V"], st["dtype"] = ref.shape[-1], ref.dtype
        out = fast(self, h)
        same = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
        top = torch.topk(ref.float(), 20, dim=-1).indices
        err = (out.float().gather(1, top) - ref.float().gather(1, top)).abs().max().item()
        _log(f"fast head built: C={C} coarse {sum(t.numel() for t in st['coarse'][:2]) / 1e6:.0f} MB; self-check argmax agree {same:.3f}, top20 max|d| {err:.2e}")
        if same < 1.0 or err > 1e-2:
            st["off"] = True; _log("fast head self-check failed -> disabled")

    def fast(self, h):
        if st["int2"]:
            P, S = st["coarse"]
            coarse = F.int2_logits(h, P, S, st["w"].shape[0]).to(torch.bfloat16)
        else:
            P, S, gs = st["coarse"]
            coarse = F.nv4_linear(h, P, S, gs, (32, 256, 4))
        idx = torch.topk(coarse, C, dim=-1).indices  # [M, C]
        return F.refine_mx8_(coarse, h, idx, st["w"], st["s"])  # exact MXFP8 rows scattered in place

    def compute_logits(self, hidden_states):
        h = hidden_states
        if st.get("off") or h.dim() != 2 or h.shape[0] > 16 or h.shape[0] == 0:
            return orig(self, hidden_states)
        if "coarse" not in st:
            if torch.cuda.is_current_stream_capturing():
                return orig(self, hidden_states)
            build(self, h.contiguous())
            if st.get("off"):
                return orig(self, hidden_states)
        return fast(self, h.contiguous())

    MD.Qwen4ExpForCausalLM.compute_logits = compute_logits
    _log("fast target head patch installed")


def _patch_ple_qd():
    # Deeper io_uring queue for the disk PLE table: a decode step issues ~128 4 KiB reads
    # (64 weight rows + 64 scale rows); the default depth of 64 needs two-plus waves.
    from b12x.sequence.ple_embedding import _disk as D
    qd = int(os.environ["FP4X_PLE_QD"])
    orig = D.DiskTable.__init__

    def init(self, layout, shard_rows, *, queue_depth=64):
        orig(self, layout, shard_rows, queue_depth=qd)

    D.DiskTable.__init__ = init
    _log(f"PLE disk queue depth {qd}")


def _patch_ple_spin():
    # The disk PLE read follows a ~40 ms blocking wait on the GPU. Waking a cold, down-clocked
    # core makes the io_uring batch take ~0.85 ms instead of ~0.37 ms; spin-poll the event
    # instead (GIL released every iteration).
    import time
    import torch
    from b12x.sequence._shared import disk_table as dt

    def _read_staged(self, count):
        ev = self._ids_ready
        with torch.cuda.device(self.device):
            while not ev.query():
                time.sleep(0)
        if self._gds is not None:
            self._gds.read(self._ids_buffer, count, self._transaction_stream)
            return
        self._native.ple_reader_run(self._reader, self._ids_buffer, self._weight_buffer, self._scale_buffer, count)

    dt.DiskRowCache._read_staged = _read_staged
    _log("PLE read spin-wait installed")


def _mtp_patch_target(params, ck):
    """Runtime parameter view for a checkpoint-named MTP tensor (q/k/v and shared gate/up are fused at load)."""
    n = ck[len("mtp."):-len(".weight")]
    for a, b, lo, hi in (("self_attn.q_proj", "self_attn.qkv_proj", 0, 12288), ("self_attn.k_proj", "self_attn.qkv_proj", 12288, 12800),
                         ("self_attn.v_proj", "self_attn.qkv_proj", 12800, 13312),
                         ("shared_expert.gate_proj", "shared_expert.gate_up_proj", 0, 640),
                         ("shared_expert.up_proj", "shared_expert.gate_up_proj", 640, 1280)):
        if n.endswith(a):
            return params["model." + n[:-len(a)] + b + ".weight"].data[lo:hi]
    if n.endswith("hyper_connection.input_mix_weight_down") and "layers" in n:
        return params["model." + n + "_block_inject.weight"].data[:320]
    return params["model." + n + ".weight"].data


def _patch_mtp_weights():
    # Overwrite MTP drafter tensors (checkpoint names, e.g. the merged r32a self-distillation LoRA) right after the
    # model runner loads, before profiling/capture and before any lazy draft-side W8 copies. Draft-only.
    import torch
    from safetensors.torch import load_file
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    path = os.environ["FP4X_MTP_PATCH"]
    orig = GPUModelRunner.load_model

    def load_model(self, *a, **k):
        out = orig(self, *a, **k)
        params = dict(self.speculator.model.named_parameters())
        patch = load_file(path)
        with torch.no_grad():
            for ck, t in patch.items():
                dst = _mtp_patch_target(params, ck)
                assert dst.shape == t.shape, (ck, dst.shape, t.shape)
                dst.copy_(t.to(dst.device, dst.dtype))
        torch.cuda.synchronize()
        _log(f"MTP patch {path}: applied {len(patch)} tensors")
        return out

    GPUModelRunner.load_model = load_model
    _log(f"MTP weight patch installed ({path})")


def _patch_moe_tau():
    # Quality-gated experiment: at decode shapes (M <= 16) drop routed experts whose normalized router weight is below
    # FP4X_MOE_TAU by pointing them at the token's top-1 expert with weight 0 (no extra expert weights are read).
    # FP4X_MOE_RENORM=1 renormalizes the kept weights to sum 1.
    import torch
    from vllm.model_executor.layers.fused_moe import b12x as B
    tau = float(os.environ["FP4X_MOE_TAU"]); renorm = os.environ.get("FP4X_MOE_RENORM") == "1"
    orig = B.B12xExperts.apply

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, *a, **k):
        if hidden_states.shape[0] <= 16:
            w = topk_weights.float()
            top = w.argmax(1, keepdim=True)
            keep = (w >= tau).scatter(1, top, True)
            topk_ids = torch.where(keep, topk_ids, topk_ids.gather(1, top)).contiguous()
            w = torch.where(keep, w, torch.zeros_like(w))
            if renorm:
                w = w / w.sum(1, keepdim=True)
            topk_weights = w.contiguous()
        return orig(self, output, hidden_states, w1, w2, topk_weights, topk_ids, *a, **k)

    B.B12xExperts.apply = apply
    _log(f"MoE decode expert drop installed (tau={tau}, renorm={renorm})")


def _patch_moe_zero():
    # Debug ablation: routed-expert output = 0 at decode shapes (measures the routed MoE share of a step; garbage output).
    from vllm.model_executor.layers.fused_moe import b12x as B
    orig = B.B12xExperts.apply

    def apply(self, output, hidden_states, *a, **k):
        if hidden_states.shape[0] <= 16:
            output.zero_()
            return None
        return orig(self, output, hidden_states, *a, **k)

    B.B12xExperts.apply = apply
    _log("MoE decode ZERO ablation installed")


def _patch_keep_graph():
    # Debug: keep every captured cudaGraph_t so probes can list and disable kernel nodes (x/probes/probe_ablate.py).
    import torch
    C = torch.cuda.CUDAGraph
    orig = C.__init__  # the pybind constructor reads keep_graph here, not in __new__
    C.__init__ = lambda self, keep_graph=False, *a, **k: orig(self, True, *a, **k)
    _log("CUDA graphs keep their cudaGraph_t (debug)")


def register():
    if os.environ.get("FP4X_KEEP_GRAPH") == "1":
        _patch_keep_graph()
    if os.environ.get("FP4X_MOE_ZERO") == "1":
        _patch_moe_zero()
    if os.environ.get("FP4X_MOE_TAU"):
        try:
            _patch_moe_tau()
        except Exception as e:
            _log(f"MoE tau patch failed: {type(e).__name__}: {e}")
    if os.environ.get("FP4X_MTP_PATCH"):
        try:
            _patch_mtp_weights()
        except Exception as e:
            _log(f"MTP weight patch failed: {type(e).__name__}: {e}")
    if os.environ.get("FP4X_PLE_SPIN") == "1":
        try:
            _patch_ple_spin()
        except Exception as e:
            _log(f"PLE spin patch failed: {type(e).__name__}: {e}")
    if os.environ.get("FP4X_PLE_QD"):
        try:
            _patch_ple_qd()
        except Exception as e:
            _log(f"PLE queue depth patch failed: {type(e).__name__}: {e}")
    if os.environ.get("FP4X_FAST_HEAD") == "1":
        try:
            _patch_fast_head()
        except Exception as e:
            _log(f"fast head patch failed: {type(e).__name__}: {e}")
    if os.environ.get("FP4X_DRAFT_VOCAB"):
        try:
            _patch_draft_vocab()
        except Exception as e:
            _log(f"draft vocab patch failed: {type(e).__name__}: {e}")
    for flag, fn in (("FP4X_MX8", _patch_mx8), ("FP4X_HC", _patch_hc)):
        if os.environ.get(flag) == "1":
            try:
                fn()
            except Exception as e:  # never break serving because of an optional patch
                _log(f"{flag} patch failed: {type(e).__name__}: {e}")
