"""How much does the cascade depend on how a refusal is detected?

Section 6.3 rests the whole routing result on a free runtime signal: the generator
declining to answer. That makes the DETECTOR a load-bearing choice, and a reviewer
will ask what happens if it is drawn differently. This script answers it from the
artifacts and persists the answer, so the claim in sections 6.3 and 7 is checkable
rather than asserted.

Two detectors:
  exact  the literal instructed phrase, "cannot answer"
  wide   that phrase plus eight further refusal formulations (nine patterns total)

Reports, per benchmark: refusal rate under each detector, escalation rate, and the
resulting dense-first cascade exact match. If the two detectors give the same EM, the
signal is robust to the detector and the paper can say so.

Usage: python scripts/refusal_sensitivity.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.em_f1 import exact_match_score, _metric_max_over_ground_truths

EXACT = re.compile(r"cannot answer", re.I)
WIDE = re.compile(
    r"cannot answer|not mentioned|not in the (provided )?context|"
    r"(does not|doesn't|do not) (mention|contain|provide|specify|state)|"
    r"no information|not specified|i (don't|do not) know|"
    r"cannot be determined|insufficient (information|context)", re.I)

BENCH = {"HotpotQA": "hotpotqa", "2Wiki": "2wikimultihop", "MuSiQue": "musique"}
Q = Path("results/queries")


def load(stem):
    return [json.loads(l) for l in (Q / stem).open() if l.strip()]


def em(recs):
    return round(100.0 * sum(
        _metric_max_over_ground_truths(exact_match_score, r["answer"], r["ground_truth"])
        for r in recs) / len(recs), 1)


def main():
    out = {}
    for label, slug in BENCH.items():
        dense = load(f"{slug}_vector_llama-3.1-8b.jsonl")
        graph = load(f"{slug}_base_llama-3.1-8b.jsonl")
        assert len(dense) == len(graph), f"{label}: arms differ in length"
        row = {"n": len(dense), "dense_em": em(dense), "graph_em": em(graph)}
        for name, det in (("exact", EXACT), ("wide", WIDE)):
            # dense-first cascade: keep the dense answer unless dense declined
            casc = [g if det.search(d["answer"]) else d
                    for d, g in zip(dense, graph)]
            esc = sum(1 for d in dense if det.search(d["answer"]))
            row[name] = {
                "dense_refusal_pct": round(100.0 * esc / len(dense), 1),
                "escalations": esc,
                "cascade_em": em(casc),
            }
        row["cascade_em_delta_wide_minus_exact"] = round(
            row["wide"]["cascade_em"] - row["exact"]["cascade_em"], 1)
        out[label] = row

    p = Path("results/analysis/refusal_sensitivity.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))

    print(f"{'benchmark':<10} {'dense':>6} {'graph':>6} | "
          f"{'exact: ref%':>11} {'esc':>4} {'EM':>5} | "
          f"{'wide: ref%':>10} {'esc':>4} {'EM':>5} | {'dEM':>5}")
    print("-" * 84)
    for b, r in out.items():
        e, w = r["exact"], r["wide"]
        print(f"{b:<10} {r['dense_em']:>6.1f} {r['graph_em']:>6.1f} | "
              f"{e['dense_refusal_pct']:>10.1f}% {e['escalations']:>4} {e['cascade_em']:>5.1f} | "
              f"{w['dense_refusal_pct']:>9.1f}% {w['escalations']:>4} {w['cascade_em']:>5.1f} | "
              f"{r['cascade_em_delta_wide_minus_exact']:>+5.1f}")
    tot = sum(1 for b in BENCH for tag in ("vector", "base")
              for _ in load(f"{BENCH[b]}_{tag}_llama-3.1-8b.jsonl"))
    hits = sum(1 for b in BENCH for tag in ("vector", "base")
               for r in load(f"{BENCH[b]}_{tag}_llama-3.1-8b.jsonl")
               if EXACT.search(r["answer"]))
    print(f"\nexact instructed phrase across all six routing arms: {hits:,} of {tot:,} answers")
    print(f"Saved -> {p}")


if __name__ == "__main__":
    main()
