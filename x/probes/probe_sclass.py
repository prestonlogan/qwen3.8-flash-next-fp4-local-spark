import inspect
r = worker.model_runner; mods = dict(r.model.named_modules())
se = mods["language_model.model.layers.0.mlp.shared_expert"]
c = type(se)
RESULT = f"{c.__module__}.{c.__qualname__} file {inspect.getsourcefile(c)}\n" + inspect.getsource(c.forward) + f"\nact_fn {type(se.act_fn).__module__}.{type(se.act_fn).__name__}"
