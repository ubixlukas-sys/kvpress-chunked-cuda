# SPDX-License-Identifier: Apache-2.0
"""Alternating-order paired perf test: ChunkedOnlinePress frozen vs fused scoring.

Differences vs perf_fused_e2e.py (supersedes it for reporting):
- config order ALTERNATES within each repetition (ref, fused, ref, fused, ...)
  to cancel drift/thermal effects
- EVERY raw timing is saved (no median-only); report median + min + max
- peak allocated VRAM and decode latency recorded per run
"""
import hashlib
import json
import os
import statistics
import sys
import time
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

from transformers import AutoModelForCausalLM, DynamicCache  # noqa: E402

MODEL = os.environ.get("KVPROVE_MODEL_PATH",
          r"D:\mxy\llm\kvpress_project\models\Qwen3-8B")  # configurable; override with env var
DEV = "cuda"
B, K = 256, 1024
DECODE_STEPS = 32
REPS = 4

SRC = Path(P.__file__)
print(f"press sha256: {hashlib.sha256(SRC.read_bytes()).hexdigest()}")
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV)
model.eval()
print(f"model load (untimed): {time.time() - t0:.1f}s")

LENGTHS = [8192, 16384, 32768]
torch.manual_seed(0)
INPUTS = {L: torch.randint(1000, 20000, (1, L), device=DEV) for L in LENGTHS}
eos = torch.tensor([[model.config.eos_token_id]], device=DEV)


def run_once(fused, L):
    cache = DynamicCache()
    press = P.ChunkedOnlinePress(budget=K, block_length=B, mode="replace",
                                 strict_checks=False, fused_scoring=fused)
    ids = INPUTS[L]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    with press(model):
        ev0.record()
        with torch.no_grad():
            out = model.model(input_ids=ids, past_key_values=cache, use_cache=True)
        ev1.record()
    pos = L
    ev2, ev3 = torch.cuda.Event(True), torch.cuda.Event(True)
    dec_pairs = []
    with torch.no_grad():
        ev2.record()
        nxt = out.logits[0, -1].argmax() if hasattr(out, "logits") and out.logits is not None else eos[0, 0]
        first = model(input_ids=nxt.view(1, 1), past_key_values=cache,
                      position_ids=torch.tensor([[pos]], device=DEV))
        ev3.record()
        cur = first.logits[0, -1].argmax()
        for i in range(1, DECODE_STEPS):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            o = model(input_ids=cur.view(1, 1), past_key_values=cache,
                      position_ids=torch.tensor([[pos + i]], device=DEV))
            e1.record()
            dec_pairs.append((e0, e1))
            cur = o.logits[0, -1].argmax()
    torch.cuda.synchronize()
    res = {
        "prefill_ms": ev0.elapsed_time(ev1),
        "ttft_tail_ms": ev2.elapsed_time(ev3),
        "decode_ms_per_token": statistics.median(a.elapsed_time(b) for a, b in dec_pairs),
        "peak_alloc_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "kv_tokens": cache.get_seq_length(),
    }
    del cache, out, first
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return res


out = {"meta": {
    "press_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
    "B": B, "K": K, "strict_checks": False, "decode_steps": DECODE_STEPS,
    "reps_per_arm": REPS, "order": "alternating (ref, fused) x reps within each length",
    "aggregate": "median/min/max over raw runs", "peak_stat": "max over runs (allocated GiB)",
    "ttft_boundary": "prefill(incl. compression) + first decode forward",
}, "runs": []}

for L in LENGTHS:
    # warmup both arms once (not recorded)
    run_once(False, L)
    run_once(True, L)
    for rep in range(REPS):
        for fused in (False, True):
            r = run_once(fused, L)
            rec = {"length": L, "fused": fused, "rep": rep, **r}
            out["runs"].append(rec)
            print(f"L={L:>6d} fused={int(fused)} rep{rep} prefill={r['prefill_ms']:>9.1f}ms "
                  f"ttft_tail={r['ttft_tail_ms']:>6.1f}ms dec/tok={r['decode_ms_per_token']:>6.1f} "
                  f"peak={r['peak_alloc_GiB']}GiB", flush=True)
            with open(os.path.join(HERE, "evidence_cudaop", "perf_fused_alt.json"), "w") as f:
                json.dump(out, f, indent=1)

# summary
print()
print("%7s %6s %10s %10s %10s   %8s %8s" % ("L", "arm", "med_ms", "min_ms", "max_ms", "peakGiB", "dec/tok"))
summary = []
for L in LENGTHS:
    for fused in (False, True):
        rs = [r for r in out["runs"] if r["length"] == L and r["fused"] == fused]
        pre = [r["prefill_ms"] for r in rs]
        rec = {"length": L, "fused": fused,
               "prefill_median_ms": round(statistics.median(pre), 1),
               "prefill_min_ms": round(min(pre), 1), "prefill_max_ms": round(max(pre), 1),
               "prefill_stdev_ms": round(statistics.stdev(pre), 1) if len(pre) > 1 else 0.0,
               "peak_alloc_GiB": max(r["peak_alloc_GiB"] for r in rs),
               "decode_ms_per_token": round(statistics.median(r["decode_ms_per_token"] for r in rs), 1),
               "ttft_tail_median_ms": round(statistics.median(r["ttft_tail_ms"] for r in rs), 1)}
        summary.append(rec)
        print("%7d %6s %10.1f %10.1f %10.1f   %8.2f %8.1f" % (
            L, "fused" if fused else "ref", rec["prefill_median_ms"], rec["prefill_min_ms"],
            rec["prefill_max_ms"], rec["peak_alloc_GiB"], rec["decode_ms_per_token"]))

out["summary"] = summary
with open(os.path.join(HERE, "evidence_cudaop", "perf_fused_alt.json"), "w") as f:
    json.dump(out, f, indent=1)
print("ALT PERF DONE")
