sp = worker.model_runner.speculator
m = sp.model
RESULT = type(m).__name__ + "\n" + "\n".join(f"{n} {tuple(p.shape)} {p.dtype}" for n, p in m.named_parameters())[:8000]
