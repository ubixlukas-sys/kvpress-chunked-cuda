# SPDX-License-Identifier: Apache-2.0
"""Fused scoring sub-chain for ChunkedOnlinePress (CUDA via NVRTC, no MSVC needed).

Replaces the per-(chunk, layer) PyTorch chain
    allowed-mask build -> masked_fill -> softmax(query dim) -> sum(queries) -> mean(G)
    -> counts build -> division
with two synchronization-free kernels + a tiny combine:

  k1 row_max_sum   one warp per logits row (rows = H*G*B); computes the row max
                   over ALLOWED columns (causal mask generated inline) and
                   1/rowsum of the numerically-stable exp. Warp shuffle
                   reductions, no shared memory, no barriers.
  k2 col_accum     one thread per column per (h,g) group tile; accumulates
                   exp(v - max[r]) * invsum[r] over the B query rows into a
                   REGISTER (fully coalesced reads, no atomics, no barriers),
                   writes partial (H, G, T).
  combine (torch)  out = partial.sum(G) / (G * counts)   [counts tensor cached]

The einsum (logits = q @ k^T / sqrt(d)) is NOT part of the kernels: it is a
real GEMM kept in PyTorch on purpose (we are not rewriting attention).

Contract (documented)
---------------------
logits   : (1, H, G, B, T) float32 CUDA, contiguous; row = g*B + q after
           reshape; values are pre-softmax scores from the kept GEMM.
kept     : survivors before the current block; columns j < kept are visible to
           all B queries of the block.
mask rule: column j visible to query q  <=>  j <= kept + q.
counts   : counts[j] = B if j < kept else B - (j - kept); defined for
           j < kept + B. Therefore the contract requires **T <= kept + B**
           (in the press, T == kept + B exactly: candidates = survivors + block).
out      : (H, T) float32 = mean over G of the per-group column sums of the
           row softmax, divided by counts.
numerics : vs the PyTorch reference chain, max_abs <= 1e-4 on random and
           adversarial inputs (scores only feed a top-k selection).
edge     : kept = 0 and T = B (first chunk) supported; non-finite logits are
           not expected from the GEMM and are not special-cased.
determin : fixed reduction order -> bitwise identical across runs.
fallback : T beyond the per-block column cap (COLS_PER_THREAD * threads) falls
           back to the reference chain and sets module-level `last_fallback`.
"""
import hashlib

import torch

from presses.nvrtc_runner import Module

# thread owns up to COLS_PER_THREAD columns in k2 -> T cap per launch
THREADS = 256
COLS_PER_THREAD = 8
MAX_T = THREADS * COLS_PER_THREAD  # 2048
ROWS_PER_BLOCK = 8                 # warps per block in k1

_SRC = r"""
extern "C" __global__ void row_max_sum(
    const float* __restrict__ logits,  // (H, G*B, T)
    float* __restrict__ rmax,          // (H*G*B)
    float* __restrict__ invsum,        // (H*G*B)
    int rows, int T, int kept, int B)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    int q = row % B;                    // query position within its group
    const float* rowp = logits + (size_t)row * T;
    float m = __int_as_float(0xff800000);            // -inf
    for (int j = lane; j < T; j += 32) {
        bool vis = (j < kept) || (j - kept <= q);
        if (vis) {
            float v = rowp[j];
            if (v > m) m = v;
        }
    }
    // butterfly reduction: EVERY lane ends with the full max (shfl_down would
    // leave lanes 1..31 with stale partial values and corrupt the row sum)
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        float o = __shfl_xor_sync(0xffffffff, m, s);
        m = fmaxf(m, o);
    }
    float s = 0.f;
    for (int j = lane; j < T; j += 32) {
        bool vis = (j < kept) || (j - kept <= q);
        if (vis) s += __expf(rowp[j] - m);
    }
    #pragma unroll
    for (int s2 = 16; s2 > 0; s2 >>= 1) {
        s += __shfl_xor_sync(0xffffffff, s, s2);
    }
    if (lane == 0) {
        rmax[row] = m;
        invsum[row] = 1.f / s;
    }
}

extern "C" __global__ void col_accum(
    const float* __restrict__ logits,  // (H, G*B, T)
    const float* __restrict__ rmax,    // (H*G*B)
    const float* __restrict__ invsum,  // (H*G*B)
    float* __restrict__ partial,       // (H, G, T)
    int H, int G, int B, int T, int kept)
{
    int hg = blockIdx.x;               // h*G + g
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= T) return;
    int g = hg % G;
    int h = hg / G;
    const float* base = logits + (size_t)hg * B * T;
    const float* rmaxb = rmax + (size_t)hg * B;
    const float* invb = invsum + (size_t)hg * B;
    float acc = 0.f;
    float comp = 0.f;                    // Kahan compensation: 256 sequential adds
    for (int r = 0; r < B; ++r) {        // would otherwise lose ~1e-3 to rounding
        bool vis = (c < kept) || (c - kept <= r);
        if (vis) {
            float p = expf(base[(size_t)r * T + c] - rmaxb[r]) * invb[r];
            float y = p - comp;
            float t = acc + y;
            comp = (t - acc) - y;
            acc = t;
        }
    }
    partial[((size_t)h * G + g) * T + c] = acc;
}
"""

_mod = None
_key = None


def _module():
    global _mod, _key
    key = hashlib.sha256(_SRC.encode()).hexdigest()[:16]
    if _mod is None or _key != key:
        _mod = Module(_SRC)
        _key = key
    return _mod


_counts_cache = {}


def _counts(kept: int, b: int, T: int, device):
    """counts[j] = b if j < kept else b - (j - kept); the G-mean factor folded in."""
    key = (kept, b, T, str(device))
    c = _counts_cache.get(key)
    if c is None:
        j = torch.arange(T, device=device, dtype=torch.float32)
        c = torch.where(j < kept, torch.full_like(j, float(b)), b - (j - kept))
        _counts_cache[key] = c
    return c


def reference_chain(logits5d: torch.Tensor, kept_for_mask: int, b: int) -> torch.Tensor:
    """Frozen PyTorch reference: logits5d (1,H,G,b,T) fp32 -> (H,T) fp32.

    Exactly the ops of the frozen implementation (mask build, masked_fill,
    softmax, sum over queries, mean over G, counts build+division).
    """
    dev = logits5d.device
    T = logits5d.shape[-1]
    jpos = torch.arange(b, device=dev).view(b, 1)
    mpos = torch.arange(T, device=dev).view(1, T)
    allowed = mpos <= kept_for_mask + jpos                       # (b, T)
    probs = torch.softmax(logits5d.masked_fill(~allowed.view(1, 1, 1, b, T), float("-inf")), dim=-1)
    mass = probs.sum(dim=-2).mean(dim=2)[0]                      # (H, T)
    counts = torch.cat(
        [torch.full((T - b,), float(b), device=dev),
         torch.arange(b, 0, -1, device=dev, dtype=torch.float32)]
    ).view(1, T)
    return mass / counts


last_fallback = False


def fused_scoring_mass(logits5d: torch.Tensor, kept_for_mask: int, b: int) -> torch.Tensor:
    """CUDA-fused equivalent of reference_chain. Returns (H, T) float32.

    Contract: T <= kept_for_mask + b (in the press T == kept_for_mask + b).
    Falls back to reference_chain when CUDA is unavailable or T exceeds the
    per-launch column cap; sets module-level `last_fallback` accordingly.
    """
    global last_fallback
    assert logits5d.dim() == 5 and logits5d.shape[0] == 1
    H, G, B, T = logits5d.shape[1], logits5d.shape[2], logits5d.shape[3], logits5d.shape[4]
    assert B == b
    assert T <= kept_for_mask + b, (
        f"invalid scoring contract: T={T} > kept+B={kept_for_mask + b} "
        "(counts undefined beyond kept+B)")
    if (not torch.cuda.is_available()) or T > MAX_T or logits5d.dtype != torch.float32 \
            or not logits5d.is_cuda:
        last_fallback = True
        return reference_chain(logits5d, kept_for_mask, b)
    last_fallback = False
    logits = logits5d[0].reshape(H, G * B, T).contiguous()
    rows = H * G * B
    rmax = torch.empty(rows, device=logits.device, dtype=torch.float32)
    invsum = torch.empty(rows, device=logits.device, dtype=torch.float32)
    partial = torch.empty(H, G, T, device=logits.device, dtype=torch.float32)
    mod = _module()
    nblocks1 = (rows + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
    mod.launch("row_max_sum", grid=(nblocks1,), block=(THREADS,),
               args=[logits, rmax, invsum, rows, T, kept_for_mask, B])
    grid2_y = (T + THREADS - 1) // THREADS
    mod.launch("col_accum", grid=(H * G, grid2_y), block=(THREADS,),
               args=[logits, rmax, invsum, partial, H, G, B, T, kept_for_mask])
    counts = _counts(kept_for_mask, b, T, logits.device)
    out = partial.sum(dim=1)
    out = out / (G * counts.unsqueeze(0))
    return out
