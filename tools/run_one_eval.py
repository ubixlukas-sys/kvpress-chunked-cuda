# SPDX-License-Identifier: Apache-2.0
"""llm4 / stage-1 single-run wrapper around the official kvpress evaluate.py.

Differences vs the official CLI invocation (all disclosed in run_meta.json):
  1. HF caches redirected to D:/mxy/llm4/cache/hf (avoids bulk cleanup in the
     default user cache; measured 351 deletion events there on this machine).
  2. transformers' use_gqa_in_sdpa forced to False so GQA is materialised with
     repeat_kv and PyTorch picks the memory-efficient SDPA kernel instead of the
     MATH fallback (measured on this box: 19.97 GiB / 2.76 s vs 16.16 GiB / 1.58 s
     prefill for 3745 tokens; greedy answers verified byte-identical on 8/8
     RULER samples by the llm/kvpress_project study).
  3. dtype forced to bfloat16 (task brief; Qwen3-8B config.json is bfloat16, so
     "auto" would resolve to the same).
  4. Dataset source: official simonjegou/ruler via load_dataset(data_dir="4096").
     If the network fetch fails, falls back to the local full-split parquet copy
     (D:/mxy/llm/kvpress_project/data/raw/ruler4096_test.parquet, 6500 official
     rows) and records fallback=true.

Usage:
  python tools/run_one_eval.py --press_name knorm --compression_ratio 0.25 \
      [--fraction 0.003077] [--query_aware] [--output_dir results] [--tag smoketest]
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # D:/mxy/llm4
REPO = os.path.join(HERE, "kvpress")
CACHE = os.path.join(HERE, "cache", "hf")
LOCAL_RULER = r"D:\mxy\llm\kvpress_project\data\raw\ruler4096_test.parquet"
MODEL = os.environ.get("KVPROVE_MODEL_PATH",
          r"D:\mxy\llm\kvpress_project\models\Qwen3-8B")  # configurable; override with env var

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for sub in ("", "datasets", "hub", "modules", "xet"):
    os.makedirs(os.path.join(CACHE, sub), exist_ok=True)
for key, sub in (
    ("HF_HOME", ""),
    ("HF_DATASETS_CACHE", "datasets"),
    ("HF_HUB_CACHE", "hub"),
    ("HF_MODULES_CACHE", "modules"),
    ("HF_XET_CACHE", "xet"),
):
    os.environ.setdefault(key, os.path.join(CACHE, sub))

for p in (HERE, os.path.join(REPO, "evaluation"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402


def force_repeat_kv():
    import transformers.integrations.sdpa_attention as sdpa

    sdpa.use_gqa_in_sdpa = lambda attention_mask, key: False
    print("[llm4] use_gqa_in_sdpa forced to False -> memory-efficient SDPA kernel", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--press_name", required=True)
    ap.add_argument("--compression_ratio", type=float, default=0.0)
    ap.add_argument("--query_aware", action="store_true")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--output_dir", default="results")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default=None, help="label written into run_meta.json")
    ap.add_argument("--dev_first_k", type=int, default=None,
                    help="override fraction sampling: first K rows per task in ORIGINAL dataset order; "
                         "writes dev_ids.json into the run dir")
    args = ap.parse_args()

    force_repeat_kv()

    import evaluate as ev  # official script, imported from REPO/evaluation

    if args.press_name.startswith("chunked_"):
        from presses.chunked_online_press import register_chunked_presses

        print("[llm4] registered candidate press:", register_chunked_presses(ev.PRESS_REGISTRY, args.press_name), flush=True)

    used_fallback = {"value": False}

    def load_ruler(path, data_dir=None, split="test", **kwargs):
        try:
            ds = ev._hf_load_dataset(path, data_dir=data_dir, split=split, **kwargs)
            print(f"[llm4] dataset loaded from HF hub ({len(ds)} rows)", flush=True)
            return ds
        except Exception as e:  # noqa: BLE001
            used_fallback["value"] = True
            print(f"[llm4] HF load failed ({type(e).__name__}: {e}); using local parquet copy", flush=True)
            return Dataset.from_pandas(pd.read_parquet(LOCAL_RULER), preserve_index=False)

    ev._hf_load_dataset = ev.load_dataset
    ev.load_dataset = load_ruler

    dev_info = None
    if args.dev_first_k:
        dev_info = {"dev_first_k": args.dev_first_k, "rows": []}

        def _dev_sample(self, *a, **kw):  # replaces pd.DataFrame.sample under this flag
            parts = []
            for task, grp in self.groupby("task", sort=False):
                take = grp.head(args.dev_first_k)
                for idx in take.index:
                    dev_info["rows"].append({"pos": int(idx), "task": task})
                parts.append(take)
            out = pd.concat(parts).sort_index()
            print(f"[llm4] dev subset: first {args.dev_first_k} per task -> {len(out)} rows "
                  f"(original order)", flush=True)
            return out

        pd.DataFrame.sample = _dev_sample

    # Official get_results_dir embeds the raw model string in the output dir name;
    # a Windows local path contains ':' which is illegal in a directory name.
    # Sanitize to the model basename (disclosed in run_meta.json "model_dir_tag").
    import re as _re

    _orig_get_results_dir = ev.EvaluationConfig.get_results_dir

    def _patched_get_results_dir(self, output_dir):
        real_model = self.model
        self.model = _re.sub(r'[<>:"/\\|?*]+', "--", os.path.basename(real_model.rstrip("\\/")))
        try:
            return _orig_get_results_dir(self, output_dir)
        finally:
            self.model = real_model

    ev.EvaluationConfig.get_results_dir = _patched_get_results_dir

    cfg = ev.EvaluationConfig(
        dataset="ruler",
        data_dir="4096",
        model=args.model,
        device=args.device,
        press_name=args.press_name,
        compression_ratio=args.compression_ratio,
        fraction=args.fraction,
        query_aware=args.query_aware,
        output_dir=args.output_dir,
        seed=args.seed,
        model_kwargs={"dtype": "bfloat16"},
    )

    out_root = os.path.join(HERE, args.output_dir)
    os.makedirs(out_root, exist_ok=True)
    before = {d for d in os.listdir(out_root) if os.path.isdir(os.path.join(out_root, d))}

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    runner = ev.EvaluationRunner(cfg)
    runner.run_evaluation()
    wall = time.time() - t0

    new = [d for d in os.listdir(out_root) if os.path.isdir(os.path.join(out_root, d)) and d not in before]
    run_dir = os.path.join(out_root, new[0]) if len(new) == 1 else None

    peak_alloc = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None

    metrics = None
    if run_dir and os.path.isfile(os.path.join(run_dir, "metrics.json")):
        with open(os.path.join(run_dir, "metrics.json"), encoding="utf-8") as f:
            raw = json.load(f)
        vals = [v["string_match"] for v in raw.values() if isinstance(v, dict) and "string_match" in v]
        metrics = round(sum(vals) / len(vals), 2) if vals else None

    meta = {
        "tag": args.tag,
        "press_name": args.press_name,
        "compression_ratio": args.compression_ratio,
        "query_aware": args.query_aware,
        "fraction": args.fraction,
        "model": args.model,
        "seed": args.seed,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa (use_gqa_in_sdpa forced False -> repeat_kv path)",
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "dataset": "simonjegou/ruler data_dir=4096" + (" (LOCAL PARQUET FALLBACK)" if used_fallback["value"] else ""),
        "wall_seconds_incl_model_load": round(wall, 1),
        "run_peak_vram_alloc_GiB": round(peak_alloc, 3) if peak_alloc else None,
        "run_peak_vram_reserved_GiB": round(peak_reserved, 3) if peak_reserved else None,
        "mean_string_match": metrics,
        "run_dir": run_dir,
    }
    if run_dir:
        with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    if run_dir and dev_info is not None:
        dev_info["n_rows"] = len(dev_info["rows"])
        with open(os.path.join(run_dir, "dev_ids.json"), "w", encoding="utf-8") as f:
            json.dump(dev_info, f, indent=1)

    print("RUN_SUMMARY:" + json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
