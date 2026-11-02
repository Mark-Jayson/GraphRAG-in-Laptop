"""Paired significance + power for two tag runs on the SAME question set.

Why this exists: `base` vs `full` differences in this project are small (0.2 pts EM
on HotpotQA), so the honest claim is a *bounded* null — "no difference within
±X pts" — not "no difference". That needs a paired test and a CI, because the two
tags answer identical questions and agree on ~92% of them; unpaired marginal CIs
badly understate the precision available.

Reports, for EM and F1:
  - the 2x2 contingency table (both right / a-only / b-only / both wrong)
  - McNemar exact test on the discordant pairs (EM), two-sided
  - paired bootstrap 95% CI on the b - a difference
  - the min detectable effect at 80% power for n = 100/300/500, so the sample
    size for the next benchmark is a calculation and not a guess

Usage:
    python scripts/paired_significance.py \
        --a results/queries/hotpotqa_base_llama-3.1-8b.jsonl \
        --b results/queries/hotpotqa_full_llama-3.1-8b.jsonl
    # add --json results/analysis/paired_hotpotqa_base_full.json to persist
"""
import sys, json, math, argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.em_f1 import _metric_max_over_ground_truths, exact_match_score, f1_score


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def score(rec, fn):
    """Score one record, honouring list-valued gold (MuSiQue answer_aliases)."""
    return _metric_max_over_ground_truths(fn, rec["answer"], rec["ground_truth"])


def mcnemar_exact_p(b_only: int, f_only: int) -> float:
    """Two-sided exact McNemar: binomial(n_disc, 0.5) on the discordant split."""
    n = b_only + f_only
    if n == 0:
        return 1.0
    k = min(b_only, f_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


# Same resample count, seed and percentile method as router_significance.py, so
# every interval in the paper is the one Sect. 4 describes.
SEED, BOOT = 20260814, 100_000


def bootstrap_ci(a_scores, b_scores, iters=BOOT, seed=SEED):
    """Paired bootstrap on the per-question difference (resample questions)."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(b_scores, float) - np.asarray(a_scores, float)
    n = len(diffs)
    means = np.empty(iters)
    for s in range(0, iters, 10_000):        # chunked: a full (iters, n) index is ~400 MB
        idx = rng.integers(0, n, (min(10_000, iters - s), n))
        means[s:s + len(idx)] = 100.0 * diffs[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline jsonl (e.g. the base tag)")
    ap.add_argument("--b", required=True, help="comparison jsonl (e.g. the full tag)")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--json", default=None, help="also write the report here")
    args = ap.parse_args()

    la = args.label_a or Path(args.a).stem
    lb = args.label_b or Path(args.b).stem

    ra, rb = load(args.a), load(args.b)
    by_q = {r["question"]: r for r in rb}
    pairs = [(x, by_q[x["question"]]) for x in ra if x["question"] in by_q]
    n = len(pairs)
    if n == 0:
        sys.exit("No overlapping questions — are these the same benchmark?")
    if n != len(ra) or n != len(rb):
        print(f"WARNING: aligned {n} of {len(ra)}/{len(rb)} — comparing the overlap only")

    em_a = [score(x, exact_match_score) for x, _ in pairs]
    em_b = [score(y, exact_match_score) for _, y in pairs]
    f1_a = [score(x, f1_score) for x, _ in pairs]
    f1_b = [score(y, f1_score) for _, y in pairs]

    both = sum(1 for i in range(n) if em_a[i] and em_b[i])
    a_only = sum(1 for i in range(n) if em_a[i] and not em_b[i])
    b_only = sum(1 for i in range(n) if em_b[i] and not em_a[i])
    neither = n - both - a_only - b_only
    disc = a_only + b_only

    p = mcnemar_exact_p(a_only, b_only)
    em_lo, em_hi = bootstrap_ci(em_a, em_b)
    f1_lo, f1_hi = bootstrap_ci(f1_a, f1_b)

    em_a_pct, em_b_pct = 100 * sum(em_a) / n, 100 * sum(em_b) / n
    f1_a_pct, f1_b_pct = 100 * sum(f1_a) / n, 100 * sum(f1_b) / n

    print(f"n = {n} paired questions   [{la}] vs [{lb}]")
    print(f"  EM  {em_a_pct:.2f} -> {em_b_pct:.2f}  (diff {em_b_pct - em_a_pct:+.2f}, "
          f"95% CI [{em_lo:+.2f}, {em_hi:+.2f}])")
    print(f"  F1  {f1_a_pct:.2f} -> {f1_b_pct:.2f}  (diff {f1_b_pct - f1_a_pct:+.2f}, "
          f"95% CI [{f1_lo:+.2f}, {f1_hi:+.2f}])")
    print(f"  EM contingency: both={both} {la}_only={a_only} {lb}_only={b_only} "
          f"neither={neither}")
    print(f"  discordant = {disc} ({100 * disc / n:.1f}% of questions flip)")
    print(f"  McNemar exact (two-sided) p = {p:.4f}"
          f"  -> {'significant' if p < 0.05 else 'NOT significant'} at .05")

    # Power: SE of the paired EM difference scales as sqrt(disc_rate/n).
    print("  power (extrapolating this discordance rate):")
    power_tbl = {}
    for nn in (100, 300, 500):
        se = math.sqrt(disc * nn / n) / nn * 100
        power_tbl[nn] = {"se_paired_em_diff": round(se, 2), "mde_80pct": round(2.8 * se, 1)}
        print(f"    n={nn}: SE={se:.2f}pts  min detectable effect(80%,.05)={2.8 * se:.1f}pts")

    ci_width = max(abs(em_lo), abs(em_hi))
    print(f"\n  reportable claim: EM difference is {em_b_pct - em_a_pct:+.2f} pts, "
          f"bounded within ±{ci_width:.1f} pts at 95% confidence.")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({
            "a": args.a, "b": args.b, "label_a": la, "label_b": lb, "n": n,
            "em": {"a": round(em_a_pct, 2), "b": round(em_b_pct, 2),
                   "diff": round(em_b_pct - em_a_pct, 2),
                   "ci95": [round(em_lo, 2), round(em_hi, 2)]},
            "f1": {"a": round(f1_a_pct, 2), "b": round(f1_b_pct, 2),
                   "diff": round(f1_b_pct - f1_a_pct, 2),
                   "ci95": [round(f1_lo, 2), round(f1_hi, 2)]},
            "em_contingency": {"both": both, "a_only": a_only, "b_only": b_only,
                               "neither": neither, "discordant": disc,
                               "discordant_rate": round(disc / n, 4)},
            "mcnemar_exact_p": round(p, 6),
            "bootstrap": {"resamples": BOOT, "seed": SEED},
            "power": power_tbl,
        }, open(out, "w"), indent=2)
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
