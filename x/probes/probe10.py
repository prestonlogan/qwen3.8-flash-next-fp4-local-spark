r = worker.model_runner
import torch
pre = [h._preparation_prefix for n, h in r.speculator.model.named_modules() if type(h).__name__ == "GatedResidual"]
pre2 = [h._preparation_prefix for n, h in r.model.named_modules() if type(h).__name__ == "GatedResidual"][:2]
RESULT = f"draft HC prefixes {pre}; target {pre2}"
