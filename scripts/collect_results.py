"""Aggregate results/evaluation/*.json into results/experiments.csv.

Idempotent: rebuilds the CSV from every eval JSON each run. One row per scored
run (tag, benchmark, corpus size, NER model, LLM, EM, F1, latency, git hash,
run date). `source` names the evaluation file and is the unambiguous run key:
replicate runs (`_rep2`, `_strat`) share benchmark, tag and model with the
canonical run.

`date` is the `run_date` recorded in the run's meta sidecar, not a file mtime,
so the CSV is identical in a fresh clone.

Latency columns: `wall_time_s` is raw wall-clock and absorbs
any time the machine spent asleep, so it is NOT safe to quote. We therefore also
derive median/p95/mean per-query latency from the source jsonl's in-loop timers,
plus `wall_clock_suspect` — true when mean > p95, which is physically impossible
for a well-behaved latency distribution and so a reliable sleep detector.
HotpotQA `base` trips it (69.5 s mean vs 39.0 s p95). See
results/analysis/latency-notes.md.

Usage:
    python scripts/collect_results.py
"""
import sys, json, csv, glob, statistics as st
from pathlib import Path

FIELDS = ["date", "benchmark", "tag", "model", "ner_model", "corpus_size",
          "use_hyperedges", "compression_enabled", "num_queries",
          "exact_match", "f1",
          "lat_median_s", "lat_p95_s", "lat_mean_s", "wall_time_s",
          "wall_clock_suspect", "refusal_pct",
          "git_hash", "source"]


def _pct(xs, q: float) -> float:
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def latency_from_jsonl(path: str) -> dict:
    """Per-query latency + refusal rate from the in-loop timers.

    Returns empty dict when the jsonl is missing or carries no timings, so a
    partially-populated results/ tree still aggregates instead of crashing.
    """
    p = Path(path)
    if not p.exists():
        return {}
    tot, refus, n = [], 0, 0
    with open(p) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            t = (r.get("timings") or {}).get("total_time")
            if t is not None:
                tot.append(float(t))
            if "cannot answer" in (r.get("answer") or "").lower():
                refus += 1
    if not tot:
        return {}
    mean, p95 = st.mean(tot), _pct(tot, 0.95)
    return {
        "lat_median_s": round(st.median(tot), 2),
        "lat_p95_s": round(p95, 2),
        "lat_mean_s": round(mean, 2),
        # mean above p95 => a handful of huge outliers => the machine slept.
        "wall_clock_suspect": mean > p95,
        "refusal_pct": round(100.0 * refus / n, 1) if n else None,
    }


def row_from(path: str) -> dict:
    d = json.load(open(path))
    em_f1 = d.get("em_f1") or {}
    row = {
        "date": d.get("run_date"),
        "benchmark": d.get("benchmark"),
        "tag": d.get("tag"),
        "model": d.get("model"),
        "ner_model": d.get("ner_model"),
        "corpus_size": d.get("corpus_size"),
        "use_hyperedges": d.get("use_hyperedges"),
        "compression_enabled": d.get("compression_enabled"),
        "num_queries": d.get("num_queries"),
        "exact_match": round(em_f1["exact_match"], 2) if "exact_match" in em_f1 else None,
        "f1": round(em_f1["f1"], 2) if "f1" in em_f1 else None,
        "lat_median_s": None, "lat_p95_s": None, "lat_mean_s": None,
        "wall_clock_suspect": None, "refusal_pct": None,
        "wall_time_s": d.get("wall_time_s"),
        "git_hash": d.get("git_hash"),
        "source": Path(path).name,
    }
    if d.get("source_jsonl"):
        row.update(latency_from_jsonl(d["source_jsonl"]))
    return row


def main():
    paths = sorted(glob.glob("./results/evaluation/*.json"))
    rows = []
    for p in paths:
        try:
            rows.append(row_from(p))
        except Exception as e:
            print(f"  skip {p}: {e}", file=sys.stderr)
    out = Path("./results/experiments.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out}")
    # Echo a compact table so you see it without opening the file.
    for r in rows:
        flag = "  <-- WALL-CLOCK SUSPECT, do not quote" if r["wall_clock_suspect"] else ""
        print(f"  {r['benchmark']:<13} {str(r['tag']):<11} "
              f"{str(r['model']).split('/')[-1]:<34} n={r['num_queries']} "
              f"EM={r['exact_match']} F1={r['f1']} "
              f"med={r['lat_median_s']}s p95={r['lat_p95_s']}s corpus={r['corpus_size']}{flag}")
    susp = [r["source"] for r in rows if r["wall_clock_suspect"]]
    if susp:
        print(f"\n{len(susp)} run(s) have sleep-contaminated wall-clock/mean: "
              f"{', '.join(susp)}\n  -> quote lat_median_s / lat_p95_s instead. "
              f"See results/analysis/latency-notes.md")


if __name__ == "__main__":
    main()
