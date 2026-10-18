import json,glob,sys,statistics as st,collections
for f in sys.argv[1:]:
    R=[json.loads(l) for l in open(f)]
    g=collections.defaultdict(list)
    for r in R: g[(r["tag"],r.get("suite"),r["S"])].append(r)
    for k,X in g.items():
        print("%-26s %-34s n=%2d tps=%6.2f agg=%6.2f ms=%6.2f tok/step=%.3f"%(f.split("/")[-1],k,len(X),st.mean(r["decode_tps_mean"] for r in X),st.mean(r.get("agg_tps",0) for r in X),st.mean(r["ms_per_step"] for r in X),st.mean(r["tok_per_step"] for r in X)))
