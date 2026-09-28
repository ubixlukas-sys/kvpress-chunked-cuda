# SPDX-License-Identifier: Apache-2.0
"""Targeted profiling of the chunked-online prefill path (8K tokens, B=256, K=1024).

Profiler capture is SEPARATE from the timing runs in perf_bench.py (single pass,
no median). Outputs the top CUDA/CPU ops so the next optimization target is
chosen from evidence.
"""
import hashlib
import os
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", os.path.join(HERE, "cache", "hf"))

import torch  # noqa: E402
import transformers  # noqa: E402
import transformers.integrations.sdpa_attention as sdpa  # noqa: E402

sdpa.use_gqa_in_sdpa = lambda attention_mask, key: False

import presses.chunked_online_press as P  # noqa: E402

FUSED = os.environ.get("CHUNKED_FUSED", "0") == "1"


from transformers import AutoModelForCausalLM, DynamicCache  # noqa: E402

MODEL = os.environ.get("KVPROVE_MODEL_PATH",
          r"D:\mxy\llm\kvpress_project\models\Qwen3-8B")  # configurable; override with env var
L, B, K = 8192, 256, 1024

SRC = Path(P.__file__)
print(f"press sha256: {hashlib.sha256(SRC.read_bytes()).hexdigest()}")

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to("cuda")
model.eval()

press = P.ChunkedOnlinePress(budget=K, block_length=B, mode="replace", strict_checks=False,
                             fused_scoring=FUSED)
torch.manual_seed(0)
ids = torch.randint(1000, 20000, (1, L), device="cuda")

# one untimed warmup (cache/allocator steady state)
with torch.no_grad():
    with press(model):
        model.model(input_ids=ids, past_key_values=DynamicCache(), use_cache=True)
torch.cuda.synchronize()

from torch.profiler import ProfilerActivity, profile  # noqa: E402

with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        cache = DynamicCache()
        with press(model):
            model.model(input_ids=ids, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()

table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=45)
print(table)
out_dir = os.path.join(HERE, "evidence")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "profile_chunked_8k.txt"), "w", encoding="utf-8") as f:
    f.write(table)

# structured top list for the report
rows = []
for ev in prof.key_averages():
    if ev.self_device_time_total > 0:
        rows.append({
            "op": ev.key,
            "cuda_total_us": ev.device_time_total,
            "cuda_self_us": ev.self_device_time_total,
            "cpu_self_us": ev.self_cpu_time_total,
            "count": ev.count,
        })
rows.sort(key=lambda r: -r["cuda_self_us"])
with open(os.path.join(out_dir, "profile_chunked_8k_top.json"), "w", encoding="utf-8") as f:
    json.dump(rows[:40], f, indent=1)
print("PROFILE DONE")
