"""Audit saved predictions only; no model loading or GPU execution.

Scoring semantics checked against NVIDIA/kvpress at a13a1da:
https://raw.githubusercontent.com/NVIDIA/kvpress/a13a1da/evaluation/benchmarks/ruler/calculate_metrics.py
"""
import argparse
import ast
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_answers(value):
    # CSV contains numpy array repr, including adjacent literals without commas.
    literal = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"", re.S)
    tokens = literal.findall(value)
    remainder = literal.sub("", value)
    assert tokens and not re.sub(r"[\s\[\],]", "", remainder), value
    return [ast.literal_eval(token) for token in tokens]


def sample_score(row):
    prediction = re.sub(r"[\x00-\x1f]", "", row["predicted_answer"].strip()).strip().lower()
    answers = parse_answers(row["answer"])
    matches = [answer.lower() in prediction for answer in answers]
    return float(any(matches)) if row["task"].split("_")[0] == "qa" else sum(matches) / len(matches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--perf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    a = read_rows(args.reference / "predictions.csv")
    b = read_rows(args.fused / "predictions.csv")
    assert len(a) == len(b) == 650
    assert json.loads((args.reference / "dev_ids.json").read_text(encoding="utf-8")) == json.loads((args.fused / "dev_ids.json").read_text(encoding="utf-8"))
    tasks = defaultdict(list)
    diffs = []
    scores = []
    for i, (old, new) in enumerate(zip(a, b)):
        for field in ("task", "question", "answer", "answer_prefix", "max_new_tokens"):
            assert old[field] == new[field], (i, field)
        sa, sb = sample_score(old), sample_score(new)
        changed = old["predicted_answer"] != new["predicted_answer"]
        row = {"row_zero_based": i, "task": old["task"], "text_changed": changed,
               "reference_score": sa, "fused_score": sb, "score_delta": sb - sa}
        scores.append(row)
        tasks[old["task"]].append(row)
        if changed:
            diffs.append({**row, "answer": old["answer"], "question": old["question"],
                          "reference_prediction": old["predicted_answer"], "fused_prediction": new["predicted_answer"]})
    per_task = {}
    shipped_a = json.loads((args.reference / "metrics.json").read_text(encoding="utf-8"))
    shipped_b = json.loads((args.fused / "metrics.json").read_text(encoding="utf-8"))
    for task, group in sorted(tasks.items()):
        ma = round(mean(r["reference_score"] for r in group) * 100, 2)
        mb = round(mean(r["fused_score"] for r in group) * 100, 2)
        assert ma == shipped_a[task]["string_match"], (task, ma, shipped_a[task])
        assert mb == shipped_b[task]["string_match"], (task, mb, shipped_b[task])
        per_task[task] = {"n": len(group), "text_changes": sum(r["text_changed"] for r in group),
                          "score_changes": sum(r["score_delta"] != 0 for r in group),
                          "reference_metric": ma, "fused_metric": mb}
    perf = json.loads(args.perf.read_text(encoding="utf-8"))
    timing = []
    for length in sorted({r["length"] for r in perf["runs"]}):
        ref = [r for r in perf["runs"] if r["length"] == length and not r["fused"]]
        fused = [r for r in perf["runs"] if r["length"] == length and r["fused"]]
        x = median(r["prefill_ms"] for r in ref)
        y = median(r["prefill_ms"] for r in fused)
        order = ["fused" if r["fused"] else "ref" for r in perf["runs"] if r["length"] == length]
        timing.append({"length": length, "reps_per_arm": len(ref), "reference_median_ms": round(x, 1),
                       "fused_median_ms": round(y, 1), "reduction_percent": round(100 * (x-y) / x, 2),
                       "fused_faster_pairs": sum(rb["prefill_ms"] < ra["prefill_ms"] for ra, rb in zip(ref, fused)),
                       "recorded_order": order})
    summary = {"n_rows": len(a), "identical_prediction_texts": len(a)-len(diffs), "changed_prediction_texts": len(diffs),
               "identical_per_sample_scores": sum(r["score_delta"] == 0 for r in scores),
               "worse_sample_scores": sum(r["score_delta"] < 0 for r in scores),
               "better_sample_scores": sum(r["score_delta"] > 0 for r in scores),
               "mean_reference": round(mean(x["reference_metric"] for x in per_task.values()), 2),
               "mean_fused": round(mean(x["fused_metric"] for x in per_task.values()), 2),
               "per_task": per_task, "timing": timing,
               "scorer_source": "https://raw.githubusercontent.com/NVIDIA/kvpress/a13a1da/evaluation/benchmarks/ruler/calculate_metrics.py",
               "source_csv_sha256": {"reference": hashlib.sha256((args.reference / "predictions.csv").read_bytes()).hexdigest(),
                                     "fused": hashlib.sha256((args.fused / "predictions.csv").read_bytes()).hexdigest()}}
    (args.out / "verified_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, rows in (("prediction_text_changes.csv", diffs), ("sample_scores.csv", scores)):
        with (args.out / name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_task", "source_csv_sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
