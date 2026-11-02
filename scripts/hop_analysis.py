#!/usr/bin/env python3
"""Does the graph's MuSiQue advantage grow with hop count? (dose-response test)

MuSiQue encodes hop structure in its question ids -- `2hop__...`, `3hop1__...`,
`4hop3__...` -- but `scripts/download_data.py` writes only question/answer/
ground_truth to `data/raw/musique_val.json`, so the labels are not in the local
copy. This script re-reads the source dataset (cached by download_data.py, so
normally no network is needed), joins hop counts onto the 500-question run by
question text, and runs the paired comparison per hop count.

WHY IT MATTERS: `scripts/subgroup_analysis.py` shows the graph beats dense on
compositional questions and loses on comparison questions. If the advantage
also rises monotonically with the number of hops required, that is a
dose-response relationship -- substantially more persuasive than a subgroup
difference, and it costs no inference time.

Needs Hugging Face reachable, or the dataset cached by download_data.py.

Usage:
    python scripts/hop_analysis.py
"""
from __future__ import annotations

import collections
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.evaluation.em_f1 import (  # noqa: E402
    _metric_max_over_ground_truths, exact_match_score)

QDIR = REPO / "results" / "queries"
DENSE = "musique_vector_llama-3.1-8b"
GRAPH = "musique_base_llama-3.1-8b"
OUT = REPO / "results" / "analysis" / "musique_by_hop.json"


def hop_map() -> dict[str, int]:
    """question text -> hop count, from the source dataset's ids."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets  (inside the project venv)")
    try:
        from download_data import REVISIONS
        ds = load_dataset("dgslibisey/MuSiQue", revision=REVISIONS["dgslibisey/MuSiQue"])
    except Exception as e:
        raise SystemExit(
            f"could not load dgslibisey/MuSiQue ({type(e).__name__}: {e}).\n"
            "This needs Hugging Face reachable, or the dataset already in\n"
            "~/.cache/huggingface/datasets from a previous download_data.py run.")
    out = {}
    for split in ds:
        for rec in ds[split]:
            m = re.match(r"(\d+)hop", str(rec.get("id", "")))
            if m and rec.get("question"):
                out[rec["question"].strip()] = int(m.group(1))
    return out


def load(stem: str) -> list[dict]:
    return [json.loads(l) for l in (QDIR / f"{stem}.jsonl").open() if l.strip()]


def em(rec) -> float:
    return _metric_max_over_ground_truths(exact_match_score, rec["answer"], rec["ground_truth"])


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def main() -> None:
    hops = hop_map()
    print(f"hop labels recovered for {len(hops):,} source questions")

    D, G = load(DENSE), load(GRAPH)
    gmap = {r["question"]: r for r in G}
    pairs = [(d, gmap[d["question"]]) for d in D if d["question"] in gmap]
    print(f"paired {len(pairs)}/{len(D)} runs on question text")

    buckets = collections.defaultdict(list)
    missing = 0
    for d, g in pairs:
        h = hops.get(d["question"].strip())
        if h is None:
            missing += 1
            continue
        buckets[h].append((d, g))
    print(f"hop label matched for {len(pairs) - missing}/{len(pairs)}"
          + (f"  ({missing} unmatched -- check question normalisation)" if missing else ""))
    if not buckets:
        raise SystemExit("no questions matched; the join key is question text and it failed")

    rows, hdr = {}, f"\n{'hops':>5} {'n':>5} {'dense EM':>9} {'graph EM':>9} {'delta':>7} {'g-only':>7} {'d-only':>7} {'McNemar':>8}"
    print(hdr)
    print("-" * len(hdr.strip()))
    for h in sorted(buckets):
        grp = buckets[h]
        ed = [em(d) for d, _ in grp]
        eg = [em(g) for _, g in grp]
        n = len(grp)
        go = sum(1 for a, b in zip(ed, eg) if b > a)
        do = sum(1 for a, b in zip(ed, eg) if a > b)
        rows[h] = {"n": n,
                   "dense_em": round(100 * sum(ed) / n, 1),
                   "graph_em": round(100 * sum(eg) / n, 1),
                   "delta_em": round(100 * (sum(eg) - sum(ed)) / n, 1),
                   "graph_only": go, "dense_only": do,
                   "mcnemar_p": round(mcnemar_exact(go, do), 4)}
        r = rows[h]
        print(f"{h:>5} {n:>5} {r['dense_em']:>8.1f}% {r['graph_em']:>8.1f}% "
              f"{r['delta_em']:>+7.1f} {go:>7} {do:>7} {r['mcnemar_p']:>8.4f}")

    deltas = [rows[h]["delta_em"] for h in sorted(rows)]
    print(f"\ndeltas by hop count: {deltas}")

    # A single bucket makes monotonicity vacuously true. That is not a
    # dose-response result, it is evidence the SAMPLE is degenerate: the
    # runners take `json.load(f)[:max_samples]` (run_queries_batch.py:56,
    # run_vector_baseline.py:72) and the MuSiQue validation split is ORDERED BY
    # HOP COUNT, so the first 500 questions are all 2-hop. Say so loudly
    # instead of reporting a trend that does not exist.
    if len(rows) < 2:
        only = next(iter(rows))
        monotone = None
        print(f"\n*** DEGENERATE SAMPLE: all {rows[only]['n']} questions are "
              f"{only}-hop. ***")
        print("No dose-response can be computed from this run. The 500-question")
        print("sample is the sequential head of a hop-sorted file, so every")
        print("MuSiQue number in the paper is measured on 2-hop questions only.")
        print("Disclose that in Limitations. Testing dose-response needs a NEW")
        print("run with hop-stratified or shuffled sampling.")
    else:
        monotone = all(b >= a for a, b in zip(deltas, deltas[1:]))
        print("MONOTONE INCREASING -- dose-response supported." if monotone
              else "NOT monotone -- report the per-hop table without a dose-response claim.")
        print("Subgroups here are small; treat per-hop p-values as descriptive and")
        print("lean on the trend across hops rather than any single cell.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"by_hop": rows, "monotone_increasing": monotone,
               "degenerate_single_bucket": len(rows) < 2,
               "note": ("sample is the sequential head of a hop-sorted file; "
                        "all questions share one hop count")
               if len(rows) < 2 else None}, OUT.open("w"), indent=2)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
