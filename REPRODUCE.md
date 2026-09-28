# Reproduction

## 1. Audit the saved results (CPU only)

From the repository root, using Python 3.12 or a compatible Python 3:

```powershell
python tools/audit_existing_predictions.py --reference data/reference_dev --fused data/fused_dev --perf data/performance/perf_fused_alt.json --out verified/local
```

This uses only Python's standard library. It verifies 650 matched rows and subset IDs, scores the original answer-array strings correctly, checks the shipped per-task metrics, and recomputes performance medians. It does not download data or load a model. Use this script as the canonical audit; the earlier comparison utility is intentionally omitted because it did not robustly parse all NumPy answer-array representations.

## 2. GPU environment

The recorded environment is Windows, Python 3.12, RTX 3090 (sm_86), PyTorch 2.9.1+cu126, Transformers 5.2.0, Qwen3-8B bf16, and CUDA Toolkit 11.6 NVRTC. A fresh environment installation has not been independently tested during publication preparation. The version list is a record of the tested environment, not a complete dependency lockfile.

The expected layout is:

```text
kvpress-chunked-cuda/
  presses/
  tools/
  data/
  kvpress/       # separate upstream checkout, ignored by this repository
```

From the project root, with your Python environment activated:

```powershell
git clone https://github.com/NVIDIA/kvpress.git kvpress
git -C kvpress checkout a13a1da
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install transformers==5.2.0
python -m pip install -e './kvpress[eval]'
```

Inspect dependency-resolution output and retain the recorded PyTorch/Transformers versions; the complete dependency set was not frozen in the experiment. Do not substitute current upstream `main` for the fixed base without treating that as a new validation run. Model weights and benchmark downloads remain separate dependencies.

The Windows runner currently loads `nvrtc64_112_0.dll` from `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin`, and loads `nvcuda.dll` from the driver installation. It compiles for sm_86. Install/use the matching toolkit to reproduce the fused path, or leave `fused_scoring=False`. Changing the loader for another toolkit/platform requires new validation.

Set the model path before running any GPU script:

```powershell
$env:KVPROVE_MODEL_PATH = 'D:\models\Qwen3-8B'
```

All published GPU tools read this variable. Historical defaults and comments still contain the original machine's paths because these files are preserved byte-for-byte. `run_one_eval.py` first downloads `simonjegou/ruler`, config `4096`; its optional local-Parquet fallback points to the original machine and is not distributed. Use a working dataset connection or explicitly adapt that fallback for a new environment.

## 3. Correctness and performance

Allow at least about 17 GiB free GPU memory for the model experiments. Run from the repository root.

```powershell
python tools/validate_chunked.py
$env:CHUNKED_FUSED = '1'
python tools/validate_chunked.py
python tools/test_scoring_reference.py
python tools/test_mask_probe.py
Remove-Item Env:CHUNKED_FUSED
python tools/test_fused_scoring.py
python tools/perf_fused_alt.py
```

Expected saved-log results: G1–G6 pass; 144 mask/position probes pass; 13 fused scoring checks pass with maximum absolute error ≤2.8e-9. The separate eager-attention scoring reference has a worst error around 5.2e-3 and measures a different comparison.

`perf_fused_alt.py` warms up each arm and runs `(reference, fused)` four times at each of 8K/16K/32K. Despite an old source comment, this ordering does not cancel all thermal/drift effects and is not AB/BA counterbalancing. Outputs go to `evidence_cudaop/`.

For profiler evidence:

```powershell
python tools/profile_hotspots.py
$env:CHUNKED_FUSED = '1'
python tools/profile_hotspots.py
Remove-Item Env:CHUNKED_FUSED
```

## 4. Fixed development subset

```powershell
python tools/run_one_eval.py --press_name chunked_replace_k1024 --compression_ratio 0.0 --fraction 0.05 --dev_first_k 50 --output_dir results_reference
python tools/run_one_eval.py --press_name chunked_replace_k1024_fused --compression_ratio 0.0 --fraction 0.05 --dev_first_k 50 --output_dir results_fused
```

`--dev_first_k 50` selects the first 50 examples of each of 13 tasks in original dataset order; it overrides the fraction sampler. This is 650 rows, not 5% of the 6,500-row benchmark. Each run writes its exact IDs and metadata. Supply the actual generated run directories to the CPU audit, alongside a new timing JSON if measuring performance again.

The wrapper forces `use_gqa_in_sdpa=False`, bf16, a project-local Hugging Face cache, and a Windows-safe result directory tag. These modifications are part of the measured configuration, not general recommendations for all models or PyTorch versions.
