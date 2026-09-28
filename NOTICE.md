# Attribution and data provenance

This project extends and depends on NVIDIA KVPress at commit `a13a1da`:
https://github.com/NVIDIA/kvpress

KVPress is licensed under Apache-2.0. The accompanying LICENSE is copied from the fixed upstream base and retains its attribution, including Copyright 2024 NVIDIA Corporation. The additional press and benchmark programs are experimental extensions; this is not an NVIDIA-endorsed release. Existing SPDX identifiers are preserved.

The saved predictions, reference-answer strings, and subset IDs derive from the `4096` test configuration of `simonjegou/ruler`:
https://huggingface.co/datasets/simonjegou/ruler

The RULER benchmark is described at https://github.com/NVIDIA/RULER. The saved results are included for reproducible scoring; full context documents, model weights, and the complete dataset are not redistributed here. Dataset and model source materials retain their respective terms; the code license does not relicense them.

Scoring semantics were checked against the fixed KVPress evaluator:
https://github.com/NVIDIA/kvpress/blob/a13a1da/evaluation/benchmarks/ruler/calculate_metrics.py

Development, review, and publication preparation used AI coding agents. Historical experiment logs are retained as evidence and are not represented as newly executed GPU tests.
