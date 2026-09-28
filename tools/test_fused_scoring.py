# SPDX-License-Identifier: Apache-2.0
"""Correctness + paired microbenchmark for the fused scoring kernel.

Shapes cover the real scoring regime (B=256 fixed; T = candidate length in
[K-ish .. K+B] range and beyond), plus adversarial and edge inputs.
All logs bind the fused source SHA256 and library versions.
"""
import hashlib
import os
import statistics
import sys
from pathlib import Path

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from presses import fused_scoring as FS  # noqa: E402
from presses.nvrtc_runner import Module  # noqa: E402

H, G, B = 8, 4, 256
FAILURES = []


def check(name, ok, detail=""):
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"{tag} {name} {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


src_sha = hashlib.sha256(Path(FS.__file__).read_bytes()).hexdigest()
print(f"fused source : {FS.__file__}")
print(f"sha256       : {src_sha}")
print(f"torch        : {torch.__version__} | device: {torch.cuda.get_device_name(0)}")
print(f"kernel column cap: T <= {FS.MAX_T}")

torch.manual_seed(0)
dev = "cuda"


def make_logits(T, kind="randn", scale=1.0):
    if kind == "randn":
        x = torch.randn(1, H, G, B, T, device=dev) * scale
    elif kind == "uniform":
        x = torch.rand(1, H, G, B, T, device=dev) * scale
    elif kind == "const":  # pathological: identical values everywhere
        x = torch.full((1, H, G, B, T), 0.7, device=dev)
    elif kind == "extreme":  # large magnitudes pre-softmax
        x = (torch.randn(1, H, G, B, T, device=dev) * 30.0)
    return x


def maxdiff(a, b):
    return (a - b).abs().max().item()


# ---------------- correctness ----------------
# Contract: candidates = survivors + block, so T == kept + B in the press.
# Generalization tested: T <= kept + B (columns beyond kept+B never visible and
# excluded from the contract; counts are only defined for j < kept+B).
for T in (256, 512, 768, 1024, 1280, 2048):
    kept = T - B  # the press-invariant configuration
    if kept < 0:
        continue
    x = make_logits(T)
    ref = FS.reference_chain(x, kept, B)
    got = FS.fused_scoring_mass(x, kept, B)
    d = maxdiff(ref, got)
    check(f"C.T{T}.kept{T-B}", d <= 1e-4, f"max_abs={d:.3e}")

# contract violation must raise
try:
    FS.fused_scoring_mass(make_logits(512), 0, B)
    check("C.contract_raise", False, "no exception for T > kept+B")
except AssertionError:
    check("C.contract_raise", True, "T > kept+B rejected")

for kind in ("const", "extreme", "uniform"):
    x = make_logits(1280, kind=kind)
    ref = FS.reference_chain(x, 1024, B)
    got = FS.fused_scoring_mass(x, 1024, B)
    d = maxdiff(ref, got)
    check(f"C.adv_{kind}", d <= 1e-4, f"max_abs={d:.3e}")

# determinism: bitwise identical across two runs
x = make_logits(1280)
a = FS.fused_scoring_mass(x, 1024, B)
b2 = FS.fused_scoring_mass(x, 1024, B)
check("C.deterministic", torch.equal(a, b2))

# fallback path: T beyond the column cap -> reference, flagged (contract kept = T - B)
x = make_logits(FS.MAX_T + 16)
_ = FS.fused_scoring_mass(x, FS.MAX_T + 16 - B, B)
check("C.fallback_flag", FS.last_fallback is True)
_ = FS.fused_scoring_mass(make_logits(1280), 1024, B)
check("C.nofallback_normal", FS.last_fallback is False)

# ---------------- paired microbenchmark ----------------
# The replaced segment: everything AFTER the einsum. Reference side includes
# mask build + masked_fill + softmax + sum + mean + counts build + division.
# Fused side: kernel + partial.sum + division by (G * counts) (counts cached).
def bench_once(T, kept, side):
    x = make_logits(T)
    if side == "ref":
        def fn():
            return FS.reference_chain(x, kept, B)
    else:
        cnt = FS._counts(kept, B, T, dev)
        mod = FS._module()
        logits = x[0].reshape(H, G * B, T).contiguous()
        rows = H * G * B
        rmax = torch.empty(rows, device=dev, dtype=torch.float32)
        invsum = torch.empty(rows, device=dev, dtype=torch.float32)
        partial = torch.empty(H, G, T, device=dev, dtype=torch.float32)
        nb = (rows + FS.ROWS_PER_BLOCK - 1) // FS.ROWS_PER_BLOCK
        gy = (T + FS.THREADS - 1) // FS.THREADS
        def fn():
            mod.launch("row_max_sum", grid=(nb,), block=(FS.THREADS,),
                       args=[logits, rmax, invsum, rows, T, kept, B])
            mod.launch("col_accum", grid=(H * G, gy), block=(FS.THREADS,),
                       args=[logits, rmax, invsum, partial, H, G, B, T, kept])
            return partial.sum(dim=1) / (G * cnt.unsqueeze(0))
    # warmup
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(50):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


print()
print("paired microbenchmark (median of 50, us) — the replaced segment only:")
print("(press-invariant configuration: kept = T - B, i.e. candidates = survivors + block)")
print("%6s %6s %10s %10s %8s" % ("T", "kept", "ref", "fused", "speedup"))
bench_rows = []
for T in (512, 768, 1024, 1280, 2048):
    kept = T - B
    t_ref = bench_once(T, kept, "ref") * 1000
    t_fus = bench_once(T, kept, "fused") * 1000
    print("%6d %6d %10.1f %10.1f %7.2fx" % (T, kept, t_ref, t_fus, t_ref / t_fus))
    bench_rows.append({"T": T, "kept": kept, "ref_us": t_ref, "fused_us": t_fus})

# 8K prefill steady-state shapes (candidates capped at K + B = 1280 after first eviction)
for T in (1280, 1056):
    kept = T - B
    t_ref = bench_once(T, kept, "ref") * 1000
    t_fus = bench_once(T, kept, "fused") * 1000
    print("%6d %6d %10.1f %10.1f %7.2fx" % (T, kept, t_ref, t_fus, t_ref / t_fus))
    bench_rows.append({"T": T, "kept": kept, "ref_us": t_ref, "fused_us": t_fus})

# with the einsum included (context: share of the whole scoring step)
for T in (1280,):
    kept = 1024
    x = make_logits(T)
    q = torch.randn(1, H, G, B, 128, device=dev)
    k = torch.randn(1, H, T, 128, device=dev)

    def full_ref():
        lg = torch.einsum("bhgqd,bhkd->bhgqk", q, k) / (128 ** 0.5)
        return FS.reference_chain(lg, kept, B)

    def full_fused():
        lg = torch.einsum("bhgqd,bhkd->bhgqk", q, k) / (128 ** 0.5)
        cnt = FS._counts(kept, B, T, dev)
        logits = lg[0].reshape(H, G * B, T).contiguous()
        rows = H * G * B
        rmax = torch.empty(rows, device=dev, dtype=torch.float32)
        invsum = torch.empty(rows, device=dev, dtype=torch.float32)
        partial = torch.empty(H, G, T, device=dev, dtype=torch.float32)
        mod = FS._module()
        nb = (rows + FS.ROWS_PER_BLOCK - 1) // FS.ROWS_PER_BLOCK
        gy = (T + FS.THREADS - 1) // FS.THREADS
        mod.launch("row_max_sum", grid=(nb,), block=(FS.THREADS,),
                   args=[logits, rmax, invsum, rows, T, kept, B])
        mod.launch("col_accum", grid=(H * G, gy), block=(FS.THREADS,),
                   args=[logits, rmax, invsum, partial, H, G, B, T, kept])
        return partial.sum(dim=1) / (G * cnt.unsqueeze(0))

    for _ in range(5):
        full_ref(); full_fused()
    torch.cuda.synchronize()
    ts = []
    for _ in range(30):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); full_fused(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    t_ff = statistics.median(ts) * 1000
    ts = []
    for _ in range(30):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); full_ref(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    t_fr = statistics.median(ts) * 1000
    print("with einsum (d=128, T=1280): ref %.1fus vs fused %.1fus -> %.2fx" % (t_fr, t_ff, t_fr / t_ff))
    bench_rows.append({"T": 1280, "with_einsum": True, "ref_us": t_fr, "fused_us": t_ff})

out = {"src_sha256": src_sha, "torch": torch.__version__,
       "device": torch.cuda.get_device_name(0), "warmup": 5, "reps": 50,
       "aggregate": "median", "bench": bench_rows}
import json  # noqa: E402

os.makedirs(os.path.join(HERE, "evidence_cudaop"), exist_ok=True)
with open(os.path.join(HERE, "evidence_cudaop", "fused_scoring_microbench.json"), "w") as f:
    json.dump(out, f, indent=1)

print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): {FAILURES}")
    sys.exit(1)
print("ALL FUSED-SCORING CHECKS PASSED")
