#!/usr/bin/env bash
# Run a sequence of span-accounting ablations: abl.sh "TAG|extra env" ...
cd "$(dirname "$0")"
BASE="$(cat x/stack2.txt) -e FP4X_FAST_HEAD=1 -e FP4X_MTP_PATCH=/x/mtp_patch_r32a.safetensors"
for spec in "$@"; do
  TAG=${spec%%|*}; EXTRA=${spec#*|}
  NOBENCH=1 FP4_SPEC="$(cat x/spec2.json)" ./exp.sh $TAG "$BASE $EXTRA" --worker-extension-cls fp4ext.Ext
  echo "== $TAG ($EXTRA)"; ./spans.sh $TAG 2>&1 | grep -E "tps=|run_fullgraph|propose|compute_logits|^step"
done
