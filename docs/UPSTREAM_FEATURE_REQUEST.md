## Feature request: true chunked prefill with bounded online KV compression

I would like to discuss adding an execution path that processes context prefill in real chunks and physically compresses each layer's KV cache between chunks. This follows the distinction documented in #186 and #187: BlockPress performs block-wise compression after a full forward pass. This proposal concerns incremental execution, not a request to revert that documentation fix.

A working prototype, source, raw results, and offline verification script are available in the [companion experiment repository](https://github.com/ubixlukas-sys/kvpress-chunked-cuda).

### Proposed behavior

- Forward the context in B-token chunks, starting from an empty cache.
- Score the candidates (previous survivors plus the current chunk) with the current chunk's queries, using causal attention mass normalized by visibility counts.
- Retain sink tokens and physically compact to a fixed per-layer budget K after each chunk; subsequent chunks use the compacted history.
- Keep original token positions for RoPE while using physical positions for cache appends and causal masks.
- Leave question processing and decode outside this context-prefill wrapper.

The prototype currently wraps the model forward and installs per-layer hooks. I would appreciate guidance on whether an upstream version should instead expose a pipeline-level chunk-execution interface so that scoring policies can remain separate.

### Evidence and limits

The fixed experimental base is KVPress `a13a1da`; model Qwen3-8B bf16; batch 1; B=256, K=1024, four sinks, replace scoring; Windows/RTX 3090; PyTorch 2.9.1+cu126 and Transformers 5.2.0. The current upstream API has not yet been integrated or validated.

The repository also contains an optional CUDA scoring optimization, disabled by default. Relative to the same compressed PyTorch baseline, the replaced subchain is 2.04–2.46× faster in saved microbenchmarks; median prefill time drops 4.72%, 4.52%, and 5.96% at 8K/16K/32K synthetic lengths. Each length has four runs per arm in repeated reference→fused order, with no statistical-significance claim; peak allocated memory is essentially unchanged.

On a fixed RULER-4096 development subset of 650 examples (50 per task), all sample scores and all 13 task metrics match between reference and fused paths, with macro score 36.39→36.39. Prediction strings match for 633 examples and differ for 17. This is not lossless compression: the historical uncompressed baseline on the same subset scores 77.25.

Saved validation includes chunk/eviction gates, position/mask probes, and 13 CUDA unit checks (maximum absolute subchain error ≤2.8e-9). The current scope is one model/device, batch 1, fresh context prefill without an explicit attention mask. The fused loader is Windows-specific and tested with CUDA Toolkit 11.6; it is not proposed as a mandatory upstream dependency.

### Suggested contribution boundary

If this direction is useful, I propose discussing the chunk-execution interface first, then submitting a focused implementation with budget/position/cache tests against the supported upstream API. CUDA acceleration can remain a separate follow-up using the backend preferred by maintainers. The scoring-policy quality gap is an explicit limitation of the prototype.

Is there interest in this execution capability, and which integration point and validation matrix would you prefer? If related work is already planned, I would be happy to align with it.

This proposal and its supporting experiment materials were prepared with AI coding-agent assistance. 🤖🤖🤖
