import builtins, sys
builtins._fx_head_mode = open("/x/probes/head_mode.txt").read().strip()
RESULT = builtins._fx_head_mode
