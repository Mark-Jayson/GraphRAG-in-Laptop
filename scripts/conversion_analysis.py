"""Retrieval->answer conversion analysis: where accuracy is actually lost.

The question this answers: when the gold answer IS in the context the model saw,
how often does the model produce it? That conversion rate is the tax every
retrieval improvement pays before it reaches EM, and at 8B-4bit it is low enough
to explain why base/ehrag/compression/full/vector all land within a point of each
other.

Reports, per run:
  gold_in_ctx   fraction of extractive questions whose gold string reaches the context
  em_given_gold CONVERSION RATE - EM on just those questions
  em_given_no   EM when gold is absent (sanity check; should be near zero)
  ctx_chars     median context length, because "more gold" and "more text" are
                entangled: at matched top_k=20 the graph arm's 2Wiki context is
                2.5x the dense arm's, so gold_in_ctx is partly a net-size effect

TWO SUBSET DEFINITIONS are reported side by side, because the choice moves 2Wiki's
conversion rate by ~10 points and an undocumented filter is a reviewer trap:

  extractive        gold exists and is not yes/no
  strict            extractive AND the gold string is not quoted in the question

`strict` matters on 2Wiki: its comparison / bridge_comparison questions name both
candidates ("Which film has the director who died earlier, A or B?"), so the gold
string is in the retrieved context for free while the facts needed to *decide*
may be absent. 36.2% of 2Wiki vs 8.8% of HotpotQA. Quote `strict` in the paper and
say so.

Usage:
    python scripts/conversion_analysis.py
    python scripts/conversion_analysis.py --inputs "results/queries/hotpotqa_*_llama-3.2-*.jsonl"
"""
import sys, json, re, glob, argparse, statistics as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.em_f1 import (normalize_answer, exact_match_score, f1_score,
                                  _metric_max_over_ground_truths)

YESNO = {"yes", "no"}


def golds(rec):
    gt = rec.get("ground_truth", "")
    if isinstance(gt, str):
        gt = [gt]
    return [g for g in (normalize_answer(x) for x in gt) if g]


def present(gold: str, haystack: str) -> bool:
    """Word-boundary containment. Raw substring inflates: gold "Nice" matches
    "Ve-nice Film Festival", and short golds (places, years, names) are the
    common case."""
    return re.search(rf"(?<!\w){re.escape(gold)}(?!\w)", haystack) is not None


def context_of(rec) -> str:
    """What the model actually saw. llm_context is post-compression; fall back to
    the joined retrieved passages for runs that predate that field."""
    return rec.get("llm_context") or "\n\n".join(rec.get("contexts", []))


def analyse(recs, strict: bool) -> dict:
    subset = []
    for r in recs:
        g = golds(r)
        if not g or set(g) & YESNO:
            continue
        if strict:
            q = normalize_answer(r.get("question", ""))
            if any(present(x, q) for x in g):
                continue
        subset.append(r)
    if not subset:
        return {}
    hit, miss = [], []
    for r in subset:
        ctx = normalize_answer(context_of(r))
        (hit if any(present(g, ctx) for g in golds(r)) else miss).append(r)

    def score(rs, fn):
        if not rs:
            return None
        return round(100.0 * sum(_metric_max_over_ground_truths(fn, r["answer"], r["ground_truth"])
                                 for r in rs) / len(rs), 1)

    return {
        "n_subset": len(subset),
        "gold_in_ctx_pct": round(100.0 * len(hit) / len(subset), 1),
        "em_given_gold": score(hit, exact_match_score),      # THE CONVERSION RATE
        "f1_given_gold": score(hit, f1_score),
        "em_given_no_gold": score(miss, exact_match_score),
        "n_gold_present": len(hit),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="results/queries/*_llama-*.jsonl")
    ap.add_argument("--min-n", type=int, default=500,
                    help="skip sanity runs; set lower to include them")
    ap.add_argument("--json", default="results/analysis/conversion_rates.json")
    args = ap.parse_args()

    out = {}
    for p in sorted(glob.glob(args.inputs)):
        if "_n20" in p or "_n5" in p:
            continue
        recs = [json.loads(l) for l in open(p) if l.strip()]
        if len(recs) < args.min_n:
            continue
        meta_p = Path(p + ".meta.json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        ctx = [len(context_of(r)) for r in recs]
        row = {
            "benchmark": meta.get("benchmark"), "tag": meta.get("tag"),
            "model": meta.get("model"), "n": len(recs),
            "em_overall": round(100.0 * sum(
                _metric_max_over_ground_truths(exact_match_score, r["answer"], r["ground_truth"])
                for r in recs) / len(recs), 1),
            "ctx_chars_median": int(st.median(ctx)),
            "ctx_tokens_est": int(st.median(ctx) / 4),   # ~4 chars/token, same as the compressor
            "n_passages_median": int(st.median([len(r.get("retrieved_ids", [])) for r in recs])),
            "refusal_pct": round(100.0 * sum(
                1 for r in recs if "cannot answer" in r["answer"].lower()) / len(recs), 1),
            "extractive": analyse(recs, strict=False),
            "strict": analyse(recs, strict=True),
        }
        out[Path(p).stem] = row

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.json, "w"), indent=2)

    hdr = (f"{'run':<38} {'EM':>5} | {'gold%':>6} {'conv%':>6} {'noGold%':>7} "
           f"| {'gold%':>6} {'conv%':>6} {'noGold%':>7} | {'ctx tok':>8} {'refuse':>7}")
    print(hdr)
    print(f"{'':<38} {'':>5} | {'--- extractive ---':^21} | {'------ strict ------':^21} |")
    print("-" * len(hdr))
    for k, r in out.items():
        e, s = r["extractive"], r["strict"]
        print(f"{k:<38} {r['em_overall']:>5.1f} | "
              f"{e['gold_in_ctx_pct']:>5.1f}% {e['em_given_gold']:>5.1f}% {e['em_given_no_gold']:>6.1f}% | "
              f"{s['gold_in_ctx_pct']:>5.1f}% {s['em_given_gold']:>5.1f}% {s['em_given_no_gold']:>6.1f}% | "
              f"{r['ctx_tokens_est']:>8,} {r['refusal_pct']:>6.1f}%")
    print(f"\nconv% = EM given the gold string reached the context = the conversion rate.")
    print(f"noGold% near zero confirms EM is tracking retrieval, not memorisation.")
    print(f"Compare ctx tok across arms before reading gold% as a ranking result.")
    print(f"\nSaved -> {args.json}")


if __name__ == "__main__":
    main()
