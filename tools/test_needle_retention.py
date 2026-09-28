# SPDX-License-Identifier: Apache-2.0
"""Direct measurement: is the needle token span retained after chunked online eviction?

Complements G6 generation-based inference. Tokenizes a synthetic needle doc,
prefills with ChunkedOnlinePress (r=0.5, replace), locates the needle's token
span in the context, and reports per-layer coverage of that span in the kept
index sets. Also reports coverage of a matched-length random span as control.
"""

import hashlib
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HOME", os.path.join(HERE, "cache", "hf"))

import torch  # noqa: E402
import transformers  # noqa: E402
import transformers.integrations.sdpa_attention as sdpa  # noqa: E402

sdpa.use_gqa_in_sdpa = lambda attention_mask, key: False

import presses.chunked_online_press as P  # noqa: E402

SRC = Path(P.__file__)
print(f"press source : {SRC}")
print(f"sha256       : {hashlib.sha256(SRC.read_bytes()).hexdigest()}")

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

MODEL = os.environ.get("KVPROVE_MODEL_PATH",
          r"D:\mxy\llm\kvpress_project\models\Qwen3-8B")  # configurable; override with env var
DEV = "cuda"
B = 256

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
).to(DEV)
model.eval()
NL = len(model.model.layers)

NEEDLE = "7c3a1e9f-4b2d-4e8a-9c1d-5f6a7b8c9d0e"
key_name = "zephyr"
words = " ".join(["memoranda"] * 300)
doc = (f"Welcome to the {key_name} archive. {words}. "
       f"\nOne of the special magic numbers for {key_name}-key is {NEEDLE}. "
       f"It is an important document.\n{words}")
ids = tok(doc, return_tensors="pt").input_ids.to(DEV)
L = ids.shape[1]

# locate needle span in token space
needle_ids = tok(NEEDLE, add_special_tokens=False).input_ids
n_len = len(needle_ids)
span_start = -1
for s in range(L - n_len):
    if ids[0, s: s + n_len].tolist() == needle_ids:
        span_start = s
        break
print(f"L={L}, needle span tokens [{span_start}, {span_start + n_len}) len={n_len}")
assert span_start > 0, "needle not found in tokenized doc"

# control span: same length, right before the needle
ctl_start = max(0, span_start - n_len)
spans = {"needle": list(range(span_start, span_start + n_len)),
         "control": list(range(ctl_start, ctl_start + n_len))}

for ratio in (0.25, 0.5, 0.75):
    press = P.ChunkedOnlinePress(compression_ratio=ratio, block_length=B, mode="replace")
    cache = DynamicCache()
    with torch.no_grad():
        with press(model):
            model.model(input_ids=ids, past_key_values=cache)
    K = int(cache.layers[0].keys.shape[2])
    covs = {k: [] for k in spans}
    for li in range(NL):
        kept = set(press._state[li]["idx"][0].tolist())  # head 0 (report per-head-0; sets vary by head)
        for name, sp in spans.items():
            covs[name].append(sum(1 for t in sp if t in kept) / len(sp))
    msg = " | ".join(
        f"{name}: mean {sum(v)/NL:.3f} min {min(v):.3f}" for name, v in covs.items()
    )
    print(f"r={ratio} K={K} ({K/L:.2f} keep) head0 coverage -> {msg}", flush=True)

# head spread at r=0.5
press = P.ChunkedOnlinePress(compression_ratio=0.5, block_length=B, mode="replace")
cache = DynamicCache()
with torch.no_grad():
    with press(model):
        model.model(input_ids=ids, past_key_values=cache)
Hkv = press._state[0]["idx"].shape[0]
per_head = []
for h in range(Hkv):
    kept = set(press._state[5]["idx"][h].tolist())
    per_head.append(sum(1 for t in spans["needle"] if t in kept) / len(spans["needle"]))
print(f"r=0.5 layer5 needle coverage per head (first 8): {[round(x,2) for x in per_head[:8]]}")
print("NEEDLE RETENTION MEASUREMENT DONE")
