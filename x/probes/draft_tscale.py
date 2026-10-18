# Set the draft logit sharpening at runtime: /x/draft_tscale.arg holds T (draft logits are divided by T).
import torch, fp4patch
t = float(open("/x/draft_tscale.arg").read())
with torch.inference_mode():
    for v in fp4patch._DRAFT_INV_T.values(): v.fill_(1.0 / t)
RESULT = f"draft tscale {t} on {len(fp4patch._DRAFT_INV_T)} device(s)"
