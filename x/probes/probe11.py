import inspect
r = worker.model_runner; m = r.model
lm = m.language_model
src = inspect.getsource(type(m).compute_logits) if hasattr(type(m), "compute_logits") else "none"
src2 = inspect.getsource(type(lm).compute_logits) if hasattr(type(lm), "compute_logits") else "none"
RESULT = f"{type(m).__module__}.{type(m).__name__}\n{src}\n{type(lm).__module__}.{type(lm).__name__}\n{src2}\nhead={type(lm.lm_head).__name__} shape={tuple(lm.lm_head.b12x_mxfp8_packed_weight.weight.values.shape)}"
