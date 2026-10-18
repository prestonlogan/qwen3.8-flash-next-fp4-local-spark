import builtins
st = builtins._fx_rt
RESULT = " | ".join(f"{k}: n={v[0]} avg {1e3*v[1]/max(v[0],1):.2f} ms" for k, v in st.items())
