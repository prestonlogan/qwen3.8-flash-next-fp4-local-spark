# Runtime A/B switch: /x/probes/ab_mode.txt = "<head g16|g32> <ple 0|1>" (A = "g16 0", B = "g32 1").
import builtins
m = open("/x/probes/ab_mode.txt").read().split()
if m == ["A"]: m = ["g16", "0"]
if m == ["B"]: m = ["g32", "1"]
builtins._fx_ple_early = m[1] == "1"
builtins._fx_head_mode = m[0]
exec(open("/x/probes/head_ab.py").read())
RESULT = f"mode {m}: ple_early={builtins._fx_ple_early}; " + RESULT
