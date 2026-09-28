# Chunked KV prefill and CUDA scoring experiments

A reproducible inference-optimization case study built on [NVIDIA KVPress](https://github.com/NVIDIA/kvpress), fixed at commit `a13a1da`.

`ChunkedOnlinePress` executes context prefill in real blocks, physically evicts KV entries after each block, and keeps a fixed per-layer token budget. An optional pair of CUDA kernels accelerates the scoring chain. The measured CUDA change reduces median prefill time by **4.5%–6.0%** on the tested RTX 3090 configuration while preserving all 650 saved per-sample benchmark scores. **17 prediction strings differ**, and the underlying compression method has a substantial accuracy cost relative to the uncompressed model.

This is an experimental implementation and evidence repository. It is not an upstream KVPress release or an accepted upstream contribution. Development and submission preparation used AI coding agents.

## Results

Qwen3-8B, bf16, batch 1; chunk size B=256, KV budget K=1024, four sink tokens, replace scoring; Windows, RTX 3090 24 GB, PyTorch 2.9.1+cu126, Transformers 5.2.0. Both arms use the same forced repeat-KV SDPA path.

| Synthetic input length | Reference prefill median | Fused prefill median | Reduction |
| --- | ---: | ---: | ---: |
| 8,192 | 5,235.9 ms | 4,988.9 ms | 4.72% |
| 16,384 | 10,575.7 ms | 10,097.8 ms | 4.52% |
| 32,768 | 21,702.0 ms | 20,407.6 ms | 5.96% |

Each length has four measurements per arm after warmup, in repeated reference-then-fused order. This is not a randomized or counterbalanced experiment. Fused is faster in 3/4, 3/4, and 4/4 pairs, respectively. All 24 measurements are included in [the timing JSON](data/performance/perf_fused_alt.json). Peak allocated memory is essentially unchanged (15.517 → 15.512 GiB).

The replaced scoring subchain is **2.04–2.46× faster** in the saved microbenchmark. Including the unchanged scoring einsum, one measured shape improves from 749.1 to 556.0 µs (**1.35×**). These are distinct from end-to-end prefill improvements. See [the original kernel log](data/performance/test_fused_scoring.log).

| Fixed RULER-4096 dev subset (first 50 examples per task, 13 tasks) | Result |
| --- | --- |
| Identical prediction strings | 633/650; 17 differ |
| Identical per-sample official scores | 650/650 |
| Task metrics unchanged | 13/13 |
| Compressed reference → fused macro score | 36.39 → 36.39 |
| Historical uncompressed macro score on the same subset | 77.25 |

The CUDA change preserves the **compressed baseline's measured scores**; compression is not lossless. The 650 examples are a fixed development subset, not the full 6,500-example benchmark. We do not attribute the 17 string differences to a particular numerical or selection mechanism without additional evidence.

## Implementation

- [chunked_online_press.py](presses/chunked_online_press.py): chunk execution, per-layer scoring, sink protection, fixed-budget selection, and physical cache compaction. Original positions drive RoPE; physical cache positions drive the causal mask and append positions.
- [fused_scoring.py](presses/fused_scoring.py): after the PyTorch scoring einsum, `row_max_sum` computes row normalization, and `col_accum` accumulates causal attention mass with Kahan compensation. PyTorch performs the final group reduction and count normalization.
- [nvrtc_runner.py](presses/nvrtc_runner.py): Windows NVRTC compilation and CUDA driver launches, avoiding a local MSVC dependency.

The source defaults to `fused_scoring=False`; the reported experiments explicitly use `mode="replace"`. The source class itself defaults to EMA, which is not the reported configuration. The CUDA kernels do not replace model attention or implement FlashAttention.

## Reproduce or audit

No GPU, model, or third-party Python package is needed to audit the saved predictions and timing data:

```shell
python tools/audit_existing_predictions.py --reference data/reference_dev --fused data/fused_dev --perf data/performance/perf_fused_alt.json --out verified/local
```

The script checks subset identities and task metrics, independently scores each prediction, and recomputes timing medians. Expected output: 633 identical strings, 17 changed strings, 650 unchanged sample scores, and macro score 36.39 for both arms. [The checked-in audit](verified/verified_summary.json) records the source CSV hashes.

For model execution, environment setup, and exact validation commands, see [REPRODUCE.md](REPRODUCE.md). The GPU results are supplied experimental evidence; publication preparation reran the offline audit and Python syntax checks, not the GPU experiments.

## Scope and limitations

- Validated with one model and one Windows GPU configuration. No mobile-device, Linux, multi-GPU, batching, quantization, or speculative-decoding claim is made.
- The press targets a fresh, empty-cache context prefill. Question processing and decode occur outside its context. It does not support a supplied attention mask. Other call patterns are not covered by these results.
- The supported scoring shape for this evidence is `(1, H, G, B, T)` with `T = kept + B`. Although the source assertion permits smaller T, this repository does not claim those inputs are supported.
- The fused route handles float32 scoring logits and T≤2048. Some unsupported inputs use the reference chain. A missing Windows NVRTC DLL is **not** an automatic fallback: disable fused scoring if the runtime is unavailable.
- NVRTC is configured for CUDA Toolkit 11.6 and the tested sm_86 device. The runtime is not portable as shipped.
- The timing script's `ttft_tail_ms` measures the first decode forward, not complete TTFT. Its `kv_tokens=1056` is sampled after 32 decode steps, not immediately after prefill. Decode speed was not optimized.

## Evidence and provenance

- [Technical account in Chinese](docs/SUMMARY.zh-CN.md)
- [Original test logs](data/correctness/) and [profiler tables](data/performance/)
- [Historical compression accuracy comparison](data/historical_method/dev_accuracy_summary.json)
- [Release notes and evidence interpretation](docs/RELEASE_NOTES.md)
- [Source manifest](provenance/source_manifest.json): code and selected raw evidence copied byte-for-byte from the final handoff, with SHA-256 hashes
- [Attribution](NOTICE.md) and [Apache-2.0 license](LICENSE)

Model weights, dataset contexts, caches, private credentials, historical archive bundles, and temporary debugging programs are not part of this repository.
