import torch
from safetensors.torch import load_file
r = worker.model_runner
f = type(r).__dict__["load_model"]
P = dict(r.speculator.model.named_parameters())
pt = load_file("/x/mtp_patch_r32a.safetensors")
w = P["model.fc_hidden.weight"].data
RESULT = f"{f.__module__}.{f.__qualname__} | fc_hidden equals patch: {torch.equal(w.cpu(), pt['mtp.fc_hidden.weight'])}"
