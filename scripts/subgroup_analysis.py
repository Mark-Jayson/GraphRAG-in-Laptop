#!/usr/bin/env python3
"""Graph vs dense retrieval, broken out by the benchmarks' own question types.

WHY THIS EXISTS
---------------
The aggregate graph-vs-dense comparisons are nulls: 0.0 EM on HotpotQA, +1.4 on
2Wiki. This script shows those nulls are not "no effect" but two real opposite
effects cancelling inside each benchmark's question mix:

  * the graph WINS on questions that require composing facts across documents
    (HotpotQA `bridge`, 2Wiki `compositional`);
  * dense WINS on questions that name their own entities (HotpotQA
    `comparison`), where question-embedding similarity lands both passages
    directly and PPR expansion only dilutes.

MuSiQue carries no type labels but is composition end-to-end by construction,
which is why it is the one benchmark where the graph wins outright (+4.2 EM).

It also reports gold-in-context and conversion PER TYPE, which is what defuses
the obvious objection ("your contexts are just bigger"): on HotpotQA `bridge`
the graph retrieves gold LESS often than dense and still wins, because
conversion is ~8 points better. A larger net cannot explain winning while
catching less gold.

CAVEAT TO CARRY INTO THE PAPER: these subgroups are post hoc. The defence is
that the partition is the benchmarks' own published type labels rather than one
searched for, and the direction was predicted by the mechanism. State it.

Multiple comparisons: six subgroup tests here, so a Bonferroni threshold is
.05/6 = .0083. Only 2Wiki `compositional` (p = .0023) clears that. The stronger
evidence is the consistency of the DIRECTION across three independently
constructed benchmarks, which decode variance cannot manufacture.

Usage:
    python scripts/subgroup_analysis.py
    python scripts/subgroup_analysis.py --json results/analysis/subgroup_by_type.json
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.evaluation.em_f1 import (  # noqa: E402
    _metric_max_over_ground_truths, exact_match_score, f1_score, normalize_answer)

# reuse the conversion-analysis helpers verbatim so the subset definition here
# cannot drift from the one the paper already quotes
_spec = importlib.util.spec_from_file_location("ca", REPO / "scripts" / "conversion_analysis.py")
ca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ca)

QDIR = REPO / "results" / "queries"

# (benchmark label, dense jsonl stem, graph jsonl stem)
PAIRS = [
    ("hotpotqa", "hotpotqa_vector_llama-3.1-8b", "hotpotqa_base_llama-3.1-8b"),
    ("2wiki", "2wikimultihop_vector_llama-3.1-8b", "2wikimultihop_base_llama-3.1-8b"),
    ("musique", "musique_vector_llama-3.1-8b", "musique_base_llama-3.1-8b"),
]


def load(stem: str) -> list[dict]:
    return [json.loads(l) for l in (QDIR / f"{stem}.jsonl").open() if l.strip()]


def em(rec) -> float:
    return _metric_max_over_ground_truths(exact_match_score, rec["answer"], rec["ground_truth"])


def f1(rec) -> float:
    return _metric_max_over_ground_truths(f1_score, rec["answer"], rec["ground_truth"])


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def strict_subset(recs: list[dict]) -> list[dict]:
    """The paper's `strict` subset: gold exists, is not yes/no, and is not
    quoted in the question itself."""
    out = []
    for r in recs:
        g = ca.golds(r)
        if not g or set(g) & ca.YESNO:
            continue
        q = normalize_answer(r.get("question", ""))
        if any(ca.present(x, q) for x in g):
            continue
        out.append(r)
    return out


def gold_reached(rec) -> bool:
    ctx = normalize_answer(ca.context_of(rec))
    return any(ca.present(g, ctx) for g in ca.golds(rec))


def analyse(bench: str, dense_stem: str, graph_stem: str) -> dict:
    D, G = load(dense_stem), load(graph_stem)
    gmap = {r["question"]: r for r in G}
    pairs = [(d, gmap[d["question"]]) for d in D if d["question"] in gmap]
    if len(pairs) != len(D):
        print(f"  ! {bench}: paired {len(pairs)}/{len(D)} on question text", file=sys.stderr)

    buckets = collections.defaultdict(list)
    for d, g in pairs:
        buckets[d.get("type") or "(unlabelled)"].append((d, g))
    buckets["ALL"] = pairs

    # conversion needs the strict subset, indexed by question for the join
    strict_q = {
        "dense": {r["question"] for r in strict_subset(D)},
        "graph": {r["question"] for r in strict_subset(G)},
    }

    rows = {}
    for t, grp in buckets.items():
        ed = [em(d) for d, _ in grp]
        eg = [em(g) for _, g in grp]
        n = len(grp)
        g_only = sum(1 for a, b in zip(ed, eg) if b > a)
        d_only = sum(1 for a, b in zip(ed, eg) if a > b)
        row = {
            "n": n,
            "dense_em": round(100 * sum(ed) / n, 1),
            "graph_em": round(100 * sum(eg) / n, 1),
            "delta_em": round(100 * (sum(eg) - sum(ed)) / n, 1),
            "dense_f1": round(100 * sum(f1(d) for d, _ in grp) / n, 2),
            "graph_f1": round(100 * sum(f1(g) for _, g in grp) / n, 2),
            "graph_only": g_only,
            "dense_only": d_only,
            "discordant_pct": round(100 * (g_only + d_only) / n, 1),
            "mcnemar_p": round(mcnemar_exact(g_only, d_only), 4),
        }
        for arm, idx in (("dense", 0), ("graph", 1)):
            sub = [p[idx] for p in grp if p[idx]["question"] in strict_q[arm]]
            if not sub:
                continue
            hit = [r for r in sub if gold_reached(r)]
            row[f"{arm}_strict_n"] = len(sub)
            row[f"{arm}_gold_pct"] = round(100 * len(hit) / len(sub), 1)
            row[f"{arm}_conversion_pct"] = (
                round(100 * sum(em(r) for r in hit) / len(hit), 1) if hit else None)
            row[f"{arm}_ctx_tokens_est"] = int(
                st.median([len(ca.context_of(r)) for r in sub]) / 4)
        rows[t] = row
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/analysis/subgroup_by_type.json")
    args = ap.parse_args()

    out = {}
    for bench, ds, gs in PAIRS:
        out[bench] = analyse(bench, ds, gs)

    hdr = (f"{'benchmark / question type':32} {'n':>4} {'dense':>7} {'graph':>7} {'delta':>7} "
           f"{'g-only':>7} {'d-only':>7} {'McNemar':>8} | "
           f"{'gold d':>7} {'gold g':>7} {'conv d':>7} {'conv g':>7} {'ctx d':>6} {'ctx g':>6}")
    print(hdr)
    print("-" * len(hdr))
    for bench, rows in out.items():
        ordered = sorted((k for k in rows if k != "ALL"), key=lambda k: -rows[k]["n"]) + ["ALL"]
        for t in ordered:
            r = rows[t]
            star = " *" if r["mcnemar_p"] < 0.05 and t != "ALL" else "  "
            def g(k, fmt="{:>7.1f}"):
                v = r.get(k)
                return " " * 7 if v is None else fmt.format(v)
            print(f"{bench + ' / ' + t:32} {r['n']:>4} "
                  f"{r['dense_em']:>7.1f} {r['graph_em']:>7.1f} {r['delta_em']:>+7.1f} "
                  f"{r['graph_only']:>7} {r['dense_only']:>7} {r['mcnemar_p']:>7.4f}{star}| "
                  f"{g('dense_gold_pct')} {g('graph_gold_pct')} "
                  f"{g('dense_conversion_pct')} {g('graph_conversion_pct')} "
                  f"{g('dense_ctx_tokens_est', '{:>6,}')} {g('graph_ctx_tokens_est', '{:>6,}')}")
        print()

    print("delta/gold/conv are graph minus dense where signed. * = McNemar p < .05")
    print("uncorrected; Bonferroni over the six typed subgroups is .05/6 = .0083.")
    print("gold/conv/ctx are on the paper's `strict` subset; EM columns are on all")
    print("paired questions, so the two halves of each row have different n.")

    p = REPO / args.json
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, p.open("w"), indent=2)
    print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
