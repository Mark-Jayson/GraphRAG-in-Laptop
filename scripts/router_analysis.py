#!/usr/bin/env python3
"""Retriever routing: eight policies, three benchmarks, no GPU time.

Section 6.2 measures an oracle bound of +5.8 to +11.0 EM from choosing between
graph and dense retrieval per question. This script asks how much of that a
real policy can capture, which turns a finding about an opportunity into a
result about a method.

It costs no inference: both arms already answered the SAME questions, so a
routing policy only selects among answers already on disk.

POLICIES
--------
Selection policies decide before retrieving, and pay one arm's cost:
  always-dense / always-graph  the two baselines
  random                       the null floor (50 seeds, averaged)
  heuristic                    a hand-written question-form rule
  learned                      logistic regression on question unigrams,
                               trained on HotpotQA, applied out-of-domain to
                               2Wiki and MuSiQue (HotpotQA itself gets 5-fold
                               cross-validation so it is never self-tested)
  type-oracle                  gold question type + the best arm per type. An
                               UPPER BOUND on type-based routing, not a
                               deployable policy: it reads labels the deployed
                               system would not have, and picks the per-type
                               winner from the same split it is scored on.
Sequential policies run the cheap arm and escalate, paying dense always plus
graph sometimes:
  cascade                      run dense; if it refuses, fall back to graph
  answer-oracle                per-question maximum; the absolute ceiling

DEPLOYABILITY CONSTRAINT: every non-oracle policy uses the QUESTION STRING
only. A router that reads retrieved contexts is not deployable, because you
would have to run both retrievers to decide which retriever to run.

Usage:  python scripts/router_analysis.py
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.evaluation.em_f1 import (  # noqa: E402
    _metric_max_over_ground_truths, exact_match_score, f1_score)

QDIR = REPO / "results" / "queries"
OUT = REPO / "results" / "analysis" / "router_ladder.json"
SEED = 20260814

BENCH = [
    ("HotpotQA", "hotpotqa_vector_llama-3.1-8b", "hotpotqa_base_llama-3.1-8b"),
    ("2Wiki", "2wikimultihop_vector_llama-3.1-8b", "2wikimultihop_base_llama-3.1-8b"),
    ("MuSiQue", "musique_vector_llama-3.1-8b", "musique_base_llama-3.1-8b"),
]

COMPARATIVE = re.compile(
    r"\b(earlier|later|first|last|older|younger|longer|shorter|larger|smaller|"
    r"more|less|greater|higher|lower|same|both|either|which one)\b", re.I)


def load(stem):
    return [json.loads(l) for l in (QDIR / f"{stem}.jsonl").open() if l.strip()]


def em(r):  return _metric_max_over_ground_truths(exact_match_score, r["answer"], r["ground_truth"])
def f1(r):  return _metric_max_over_ground_truths(f1_score, r["answer"], r["ground_truth"])
# The grounding prompt templates the refusal, so 1,161 of 3,000 answers are the
# verbatim string "I cannot answer based on the provided context." Validated
# 14 Aug against a nine-pattern detector (not mentioned / no information /
# does not specify / I do not know / cannot be determined / ...): it catches only
# 3 more answers in 3,000 (0.10%) and moves cascade EM by 0.0 on all three
# benchmarks. Two answers contain both a real answer and the refusal string and
# so escalate needlessly (0.07%). The narrow test is therefore sound.
REFUSAL = re.compile(r"cannot answer", re.I)
REFUSAL_WIDE = re.compile(
    r"cannot answer|not mentioned|not in the (provided )?context|"
    r"(does not|doesn't|do not) (mention|contain|provide|specify|state)|"
    r"no information|not specified|i (don't|do not) know|"
    r"cannot be determined|insufficient (information|context)", re.I)


def refused(r, wide=False):
    return bool((REFUSAL_WIDE if wide else REFUSAL).search(r["answer"]))
def lat(r): return r["timings"]["total_time"]


def pairs(vs, bs):
    V, B = load(vs), load(bs)
    bm = {r["question"]: r for r in B}
    return [(v, bm[v["question"]]) for v in V if v["question"] in bm]


# ----------------------------------------------------------------- routers
def looks_comparison(q: str) -> bool:
    """Comparison questions name both candidates, so similarity finds them and
    graph expansion only dilutes (section 6.2). Question string only."""
    has_or = re.search(r",\s*\w[\w\s.'-]*\s+or\s+\w", q) is not None
    caps = len(re.findall(r"\b[A-Z][a-z]{2,}", q))
    return (has_or and caps >= 2) or (COMPARATIVE.search(q) is not None and caps >= 2)


def featurise(questions, vocab=None, min_df=2):
    toks = [re.findall(r"[a-z']+", q.lower()) for q in questions]
    if vocab is None:
        df = {}
        for t in toks:
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        vocab = {w: i for i, w in enumerate(sorted(w for w, c in df.items() if c >= min_df))}
    X = np.zeros((len(questions), len(vocab) + 4))
    for i, (q, t) in enumerate(zip(questions, toks)):
        for w in t:
            j = vocab.get(w)
            if j is not None:
                X[i, j] = 1.0
        X[i, -4] = 1.0                                    # bias
        X[i, -3] = min(len(t), 40) / 40.0                 # length
        X[i, -2] = 1.0 if looks_comparison(q) else 0.0    # the heuristic as a feature
        X[i, -1] = 1.0 if " or " in q.lower() else 0.0
    return X, vocab


def fit_logreg(X, y, l2=1.0, iters=600, lr=0.5):
    w = np.zeros(X.shape[1])
    n = len(y)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (p - y) / n + l2 * np.r_[w[:-4], np.zeros(4)] / n
        w -= lr * g
    return w


def predict(X, w):
    return (1.0 / (1.0 + np.exp(-X @ w))) >= 0.5          # True -> route to graph


# ----------------------------------------------------------------- scoring
def score(pr, choose, sequential=False):
    """choose: list of bools, True = use the graph arm.
    sequential: cascade cost model -- always pay dense, add graph on escalation."""
    e, f, la, ng = [], [], [], 0
    for (v, b), g in zip(pr, choose):
        r = b if g else v
        e.append(em(r)); f.append(f1(r)); ng += bool(g)
        la.append(lat(v) + lat(b) if (sequential and g) else lat(r))
    n = len(pr)
    # MEDIAN, not mean: hotpotqa_base contains one 21,301 s query (the host
    # slept mid-run) whose mean, 69.5 s, exceeds that run's own p95. The paper
    # quotes medians everywhere for exactly this reason and a routing table
    # that quietly used means would contradict Table 3.
    return dict(em=100 * sum(e) / n, f1=100 * sum(f) / n,
                pct_graph=100 * ng / n, med_lat=float(np.median(la)), per_q=e)


def boot_ci(a, b, iters=2000, rng=None):
    rng = rng or np.random.default_rng(SEED)
    a, b = np.array(a), np.array(b)
    idx = rng.integers(0, len(a), (iters, len(a)))
    d = 100 * (b[idx].mean(1) - a[idx].mean(1))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    rng = np.random.default_rng(SEED)

    # --- train the learned router on HotpotQA discordant questions only ---
    hp = pairs(BENCH[0][1], BENCH[0][2])
    tr_q, tr_y = [], []
    for v, b in hp:
        ev, eb = em(v) > .5, em(b) > .5
        if ev != eb:                       # ties carry no routing signal
            tr_q.append(v["question"]); tr_y.append(1.0 if eb else 0.0)
    Xtr, vocab = featurise(tr_q)
    w_full = fit_logreg(Xtr, np.array(tr_y))
    print(f"learned router: trained on {len(tr_q)} discordant HotpotQA questions, "
          f"{len(vocab)} vocab terms\n")

    results = {}
    for name, vs, bs in BENCH:
        pr = pairs(vs, bs)
        n = len(pr)
        qs = [v["question"] for v, _ in pr]
        rows = {}

        rows["always-dense"] = score(pr, [False] * n)
        rows["always-graph"] = score(pr, [True] * n)

        rnd = [score(pr, list(rng.random(n) < 0.5)) for _ in range(50)]
        rows["random"] = {k: float(np.mean([r[k] for r in rnd]))
                          for k in ("em", "f1", "pct_graph", "med_lat")}
        rows["random"]["per_q"] = rnd[0]["per_q"]

        rows["heuristic"] = score(pr, [not looks_comparison(q) for q in qs])

        if name == "HotpotQA":            # 5-fold CV so it never self-tests
            pred = np.zeros(n, dtype=bool)
            folds = rng.permutation(n) % 5
            for k in range(5):
                tr = [i for i in range(n) if folds[i] != k]
                trq = [qs[i] for i in tr]
                try_y = []
                keep = []
                for i in tr:
                    v, b = pr[i]
                    ev, eb = em(v) > .5, em(b) > .5
                    if ev != eb:
                        keep.append(qs[i]); try_y.append(1.0 if eb else 0.0)
                Xk, vk = featurise(keep)
                wk = fit_logreg(Xk, np.array(try_y))
                te = [i for i in range(n) if folds[i] == k]
                Xte, _ = featurise([qs[i] for i in te], vk)
                pred[te] = predict(Xte, wk)
            rows["learned (5-fold CV)"] = score(pr, list(pred))
        else:
            Xte, _ = featurise(qs, vocab)
            rows["learned (transfer)"] = score(pr, list(predict(Xte, w_full)))

        types = [v.get("type") for v, _ in pr]
        if any(t for t in types):
            best = {}
            for t in set(types):
                idx = [i for i, x in enumerate(types) if x == t]
                dv = sum(em(pr[i][0]) for i in idx)
                gv = sum(em(pr[i][1]) for i in idx)
                best[t] = gv > dv
            rows["type-oracle*"] = score(pr, [best[t] for t in types])

        rows["cascade (dense->graph)"] = score(
            pr, [refused(v) for v, _ in pr], sequential=True)
        # the mirror image, for completeness: run the graph first and fall back
        # to dense when it abstains. Cheap arm first is the usual cascade
        # instinct, but on a benchmark where the graph is the stronger arm the
        # ordering is not obvious a priori.
        rows["cascade (graph->dense)"] = score(
            pr, [not refused(b) for _, b in pr], sequential=True)
        rows["answer-oracle*"] = score(pr, [em(b) > em(v) for v, b in pr])

        base = rows["always-dense"]["per_q"]
        for k, r in rows.items():
            lo, hi = boot_ci(base, r["per_q"]) if "per_q" in r else (float("nan"),) * 2
            r["ci_vs_dense"] = [round(lo, 1), round(hi, 1)]
            r.pop("per_q", None)
        results[name] = rows

    # --------------------------------------------------------------- report
    hdr = f"{'policy':24} {'EM':>6} {'F1':>6} {'%graph':>7} {'med lat':>8} {'dEM vs dense':>16}"
    for name in results:
        print(f"=== {name} ===")
        print(hdr); print("-" * len(hdr))
        d = results[name]["always-dense"]["em"]
        for k, r in results[name].items():
            ci = r["ci_vs_dense"]
            print(f"{k:24} {r['em']:>6.1f} {r['f1']:>6.2f} {r['pct_graph']:>6.1f}% "
                  f"{r['med_lat']:>8.1f} {r['em']-d:>+7.1f} [{ci[0]:+.1f},{ci[1]:+.1f}]")
        print()
    print("* oracles are upper bounds, not deployable policies.")
    print("cascade latency = dense always + graph on escalation. All latencies are medians.")
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
