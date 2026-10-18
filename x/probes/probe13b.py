import builtins
st = builtins._fx_ple_stats
RESULT = f"calls {st['n']} avg wall {1e3*st['cpu']/max(st['n'],1):.3f} ms (incl sync) avg gpu-span {1e3*st['gpu']/max(st['n'],1):.3f} ms"
