r = worker.model_runner; sp = r.speculator
RESULT = "runner: " + ",".join(a for a in dir(type(r)) if not a.startswith("__") and callable(getattr(type(r), a, None)))[:2500] + "\nspec: " + ",".join(a for a in dir(type(sp)) if not a.startswith("__") and callable(getattr(type(sp), a, None)))[:1500]
