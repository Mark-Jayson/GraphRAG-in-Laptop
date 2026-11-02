"""Retrieval ceiling: of the questions whose answer is a string (not yes/no), for
how many did the retriever actually put that answer into the context the model saw?

This is the ceiling that matters for interpreting EM/F1:
  - 100% - retrieval_ceiling   = answers lost in RETRIEVAL (right passage never
                                  reached the model — a retriever/NER problem).
  - retrieval_ceiling - actual EM = answers lost in GENERATION (answer was in
                                  context but the model didn't produce it).
So it tells you where to spend effort. HotpotQA/2Wiki corpora are built from each
question's own context, so "is the answer anywhere in the corpus" is ~100% by
construction and uninformative — hence we measure the retrieved context, not the
whole corpus.

Runs on a jsonl produced by run_queries_batch.py (has per-question `contexts` and
`ground_truth`). Substring match after normalization, so it's an OPTIMISTIC upper
bound (coincidental matches possible). yes/no answers are reported separately —
their ceiling is two-passage reasoning, not string presence.

Usage:
    python scripts/check_answerability.py --input results/queries/hotpotqa_full_llama-3.1-8b.jsonl
"""
import sys, json, re, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.em_f1 import normalize_answer

YESNO = {"yes", "no"}


def _golds(rec):
    """Normalized gold answer(s). MuSiQue carries a list (answer + aliases)."""
    gt = rec.get("ground_truth", "")
    if isinstance(gt, str):
        gt = [gt]
    return [g for g in (normalize_answer(x) for x in gt) if g]


def _present(gold: str, ctx: str) -> bool:
    """Word-boundary containment, not raw substring.

    Raw `in` produced false positives that inflated the ceiling: gold "Nice"
    (a place of death) matched "Ve-nice Film Festival". Short gold answers —
    places, years, single names — are exactly the common case, so bound it.
    """
    return re.search(rf"(?<!\w){re.escape(gold)}(?!\w)", ctx) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jsonl from run_queries_batch.py")
    args = ap.parse_args()

    records = [json.loads(line) for line in open(args.input) if line.strip()]
    yesno, extractive, retrieved = [], [], []
    # Questions whose gold answer is quoted in the question itself (2Wiki
    # comparison / bridge_comparison: "Which film has the director who died
    # earlier, A or B?" — the answer is A or B). For these, "gold string is in
    # the retrieved context" is nearly free and says nothing about whether the
    # facts needed to *decide* between A and B were retrieved. 36% of 2Wiki vs
    # 9% of HotpotQA, so mixing them makes the ceiling incomparable across
    # benchmarks. Reported as a separate bucket, never folded into the headline.
    self_ref, self_ref_hit = [], []
    for r in records:
        golds = _golds(r)
        if not golds:
            continue
        if any(g in YESNO for g in golds):
            yesno.append(r)
            continue
        # Prefer the exact post-compression context the model saw (newer jsonls
        # log it as `llm_context`); fall back to the full retrieved passages.
        ctx = normalize_answer(r["llm_context"] if r.get("llm_context")
                               else " ".join(r.get("contexts", [])))
        hit = any(_present(g, ctx) for g in golds)
        q = normalize_answer(r.get("question", ""))
        if any(_present(g, q) for g in golds):
            self_ref.append(r)
            if hit:
                self_ref_hit.append(r)
            continue
        extractive.append(r)
        if hit:
            retrieved.append(r)

    n = len(extractive)
    ceil = 100.0 * len(retrieved) / n if n else 0.0
    print(f"Input: {args.input}")
    print(f"  questions: {len(records)} | yes/no: {len(yesno)} | "
          f"gold-quoted-in-question: {len(self_ref)} | scored extractive: {n}")
    print(f"  RETRIEVAL CEILING (gold answer present in retrieved context): "
          f"{len(retrieved)}/{n} = {ceil:.1f}%  (optimistic upper bound)")
    print(f"  -> extractive EM/F1 cannot exceed ~{ceil:.1f}%. Gap below that is a "
          f"generation loss; the {100.0 - ceil:.1f}% missing here is a retrieval loss.")
    if self_ref:
        sr = 100.0 * len(self_ref_hit) / len(self_ref)
        print(f"  [separate bucket] gold quoted in the question: "
              f"{len(self_ref_hit)}/{len(self_ref)} = {sr:.1f}% present in context "
              f"— uninformative as a retrieval ceiling (the answer string comes "
              f"free with the question); excluded from the headline above.")
    if yesno:
        print(f"  ({len(yesno)} yes/no questions excluded — ceiling for those is "
              f"reasoning over two passages, not string presence.)")


if __name__ == "__main__":
    main()
