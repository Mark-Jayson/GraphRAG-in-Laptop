#!/usr/bin/env python3
"""Where the graph arms' errors come from: retrieval, generation, or refusal.

On the `strict` extractive subset defined in conversion_analysis.py (gold is a
string, not yes/no, and not quoted in the question), every question falls in
exactly one bucket:

  correct      gold reached the context and the answer is an exact match
  refuse_gold  gold reached the context and the model declined ("cannot answer")
  wrong_gold   gold reached the context and the answer is wrong
  no_gold      gold never reached the context (a retrieval failure; the few
               lucky exact matches without gold are counted here too)

Section 6 quotes these as shares of `n` for the `base` arms.

Usage:
    python scripts/error_decomposition.py
"""
import sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.em_f1 import (normalize_answer, exact_match_score,
                                  _metric_max_over_ground_truths)
from conversion_analysis import YESNO, golds, present, context_of

RUNS = [(b, t) for b in ("hotpotqa", "2wikimultihop", "musique") for t in ("base", "full")]


def decompose(recs) -> dict:
    out = {"n": 0, "correct": 0, "refuse_gold": 0, "wrong_gold": 0, "no_gold": 0}
    for r in recs:
        g = golds(r)
        if not g or set(g) & YESNO:
            continue
        q = normalize_answer(r.get("question", ""))
        if any(present(x, q) for x in g):
            continue
        out["n"] += 1
        ctx = normalize_answer(context_of(r))
        if not any(present(x, ctx) for x in g):
            out["no_gold"] += 1
        elif _metric_max_over_ground_truths(exact_match_score, r["answer"], r["ground_truth"]):
            out["correct"] += 1
        elif "cannot answer" in r["answer"].lower():
            out["refuse_gold"] += 1
        else:
            out["wrong_gold"] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-slug", default="llama-3.1-8b")
    ap.add_argument("--json", default="results/analysis/error_decomposition.json")
    args = ap.parse_args()

    out = {}
    for bench, tag in RUNS:
        p = Path(f"results/queries/{bench}_{tag}_{args.model_slug}.jsonl")
        recs = [json.loads(l) for l in p.open() if l.strip()]
        out[f"{bench}_{tag}"] = d = decompose(recs)
        n = d["n"]
        print(f"{bench + '_' + tag:<20} n={n:<4} " + "  ".join(
            f"{k}={100 * d[k] / n:5.1f}%" for k in ("correct", "no_gold", "wrong_gold", "refuse_gold")))

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {args.json}")


if __name__ == "__main__":
    main()
