import builtins
for k in ("_fx_moe",):
    if hasattr(builtins, k): delattr(builtins, k)
RESULT = "cleared"
