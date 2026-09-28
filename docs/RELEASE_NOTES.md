# Publication preparation

The public snapshot was organized from `KVPress_Final_Submission_Package.zip` on 2026-09-28. Python implementation and selected evidence files retain their submitted bytes; the provenance manifest records their hashes. No new GPU execution or numerical-kernel change was made in this step.

The public tree omits debugging scripts, driver/queue scripts, nested milestone archives, old performance aggregates, and redundant comparison scripts. It includes the independent standard-library audit, which understands NumPy-style answer strings containing adjacent quoted entries and applies the fixed-base RULER scorer's cleanup and matching rules.

The English README and reproduction guide distinguish observed scores from text equivalence, kernel microbenchmarks from end-to-end timing, and the CUDA optimization from the quality of the compression method. The Chinese technical account also corrects the following reporting details:

- The prototype is compared with the documented BlockPress behavior; it does not claim every current KVPress method runs after a full prefill.
- Warp shuffle intrinsics synchronize lanes; the two kernels are described as avoiding block-wide barriers rather than being synchronization-free.
- The tested counts contract is T=kept+B; the implementation's weaker assertion is disclosed rather than claimed to enforce equality.
- The masked-fill profiler time of about 119.6 ms is an aggregate across 1,152 calls, not a per-call duration.

Profiler totals use the complete table footers: 4.094 s reference and 3.894 s fused Self CUDA time. Summing overlapping operator/kernel rows would double-count work. The older 7,954→7,422 ms totals and the claim that all 650 prediction strings match are retracted.

Some preserved comments mention the original machine, an earlier design document, or an unrelated `submission.py` runtime workflow. Those comments do not describe additional shipped components or support promises. Use README.md and REPRODUCE.md for the publication contract.

The independent offline audit, source hash checks, and Python syntax checks are rerun before publishing. Their results do not replace the historical GPU validation logs.
