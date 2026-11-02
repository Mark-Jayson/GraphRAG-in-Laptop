#!/usr/bin/env python3
"""Exact paired tests for the routing ladder, with multiplicity corrections.

Every policy is compared against always-dense on the SAME questions, so the
right test is an exact two-sided McNemar on the discordant pairs, and the right
interval is a paired bootstrap on the EM difference.

Because the cascade was chosen after looking at nine policies, the p-value needs
a multiplicity correction. Three family definitions are reported rather than one,
because the choice is a judgement and the reader should see it:

  m_all      every policy-vs-dense test actually run (oracles included)
  m_deploy   deployable policies only -- the oracles are upper bounds, not
             hypotheses about a system anyone could ship
  m_cascade  the cascade claim alone (both orderings x three benchmarks)

Holm is computed within each family, so it depends on the other p-values in that
family; Bonferroni does not. Bonferroni-adjusted intervals use confidence
1 - alpha/m, which is why they are wider than the nominal 95%.

Usage: python scripts/router_significance.py
"""
from __future__ import annotations

import json, math, re, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.evaluation.em_f1 import (  # noqa: E402
    _metric_max_over_ground_truths, exact_match_score)

QDIR = REPO / "results" / "queries"
OUT = REPO / "results" / "analysis" / "router_significance.json"
SEED, BOOT = 20260814, 100_000
REFUSAL = re.compile(r"cannot answer", re.I)

BENCH = [("HotpotQA", "hotpotqa_vector_llama-3.1-8b", "hotpotqa_base_llama-3.1-8b"),
         ("2Wiki", "2wikimultihop_vector_llama-3.1-8b", "2wikimultihop_base_llama-3.1-8b"),
         ("MuSiQue", "musique_vector_llama-3.1-8b", "musique_base_llama-3.1-8b")]
COMPARATIVE = re.compile(r"\b(earlier|later|first|last|older|younger|longer|shorter|larger|"
                         r"smaller|more|less|greater|higher|lower|same|both|either|which one)\b", re.I)


def load(s): return [json.loads(l) for l in (QDIR / f"{s}.jsonl").open() if l.strip()]
def em(r): return _metric_max_over_ground_truths(exact_match_score, r["answer"], r["ground_truth"]) > .5
def ref(r): return bool(REFUSAL.search(r["answer"]))


def looks_comparison(q):
    has_or = re.search(r",\s*\w[\w\s.'-]*\s+or\s+\w", q) is not None
    caps = len(re.findall(r"\b[A-Z][a-z]{2,}", q))
    return (has_or and caps >= 2) or (COMPARATIVE.search(q) is not None and caps >= 2)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    m = len(pvals)
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    per_bench, policies = {}, {}

    for name, vs, bs in BENCH:
        V, B = load(vs), load(bs)
        bm = {r["question"]: r for r in B}
        pr = [(v, bm[v["question"]]) for v in V if v["question"] in bm]
        n = len(pr)
        qs = [v["question"] for v, _ in pr]
        types = [v.get("type") for v, _ in pr]
        dense = np.array([em(v) for v, _ in pr])
        graph = np.array([em(b) for _, b in pr])

        pol = {"always-graph": graph,
               "random": np.where(rng.random(n) < .5, graph, dense),
               "heuristic": np.where([not looks_comparison(q) for q in qs], graph, dense),
               "cascade (dense->graph)": np.where([ref(v) for v, _ in pr], graph, dense),
               "cascade (graph->dense)": np.where([ref(b) for _, b in pr], dense, graph),
               "answer-oracle*": np.maximum(dense, graph)}
        if any(t for t in types):
            best = {}
            for t in set(types):
                idx = [i for i, x in enumerate(types) if x == t]
                best[t] = graph[idx].sum() > dense[idx].sum()
            pol["type-oracle*"] = np.where([best[t] for t in types], graph, dense)

        rows = {}
        idx = rng.integers(0, n, (BOOT, n))
        for k, arr in pol.items():
            b = int(((arr == 1) & (dense == 0)).sum())
            c = int(((arr == 0) & (dense == 1)).sum())
            d = 100 * (arr[idx].mean(1) - dense[idx].mean(1))
            rows[k] = {"em": round(100 * arr.mean(), 1),
                       "delta": round(100 * (arr.mean() - dense.mean()), 2),
                       "b_policy_only": b, "c_dense_only": c, "discordant": b + c,
                       "p_raw": mcnemar_exact(b, c), "boot": d}
            policies.setdefault((name, k), rows[k])
        per_bench[name] = rows

    # ---------------- families ----------------
    all_keys = list(policies)
    deploy = [k for k in all_keys if not k[1].endswith("*")]
    casc = [k for k in all_keys if k[1].startswith("cascade")]
    fams = {"m_all": all_keys, "m_deploy": deploy, "m_cascade": casc}
    print("family sizes: " + ", ".join(f"{k}={len(v)}" for k, v in fams.items()) + "\n")

    adj = {}
    for fam, keys in fams.items():
        praw = {k: policies[k]["p_raw"] for k in keys}
        h = holm(praw)
        m = len(keys)
        for k in keys:
            adj.setdefault(k, {})[fam] = {
                "m": m,
                "bonferroni": min(1.0, m * praw[k]),
                "holm": h[k],
                "ci_bonf": [round(float(np.percentile(policies[k]["boot"], 100 * (0.05 / m) / 2)), 2),
                            round(float(np.percentile(policies[k]["boot"], 100 * (1 - (0.05 / m) / 2))), 2)],
            }

    out = {}
    for name in per_bench:
        print(f"=== {name} ===")
        print(f"{'policy':24} {'dEM':>7} {'95% CI':>16} {'b/c':>9} {'p raw':>9} "
              f"{'Bonf(all)':>10} {'Holm(all)':>10} {'Holm(casc)':>11}")
        for k, r in per_bench[name].items():
            key = (name, k)
            lo95 = np.percentile(r["boot"], 2.5); hi95 = np.percentile(r["boot"], 97.5)
            a = adj[key]
            hc = a.get("m_cascade", {}).get("holm")
            print(f"{k:24} {r['delta']:>+7.2f} [{lo95:>+6.2f},{hi95:>+6.2f}] "
                  f"{r['b_policy_only']:>4}/{r['c_dense_only']:<4} {r['p_raw']:>9.2e} "
                  f"{a['m_all']['bonferroni']:>10.2e} {a['m_all']['holm']:>10.2e} "
                  f"{('%.2e' % hc) if hc is not None else '--':>11}")
            out.setdefault(name, {})[k] = {
                "em": r["em"], "delta_em": r["delta"],
                "ci95": [round(float(lo95), 2), round(float(hi95), 2)],
                "policy_only": r["b_policy_only"], "dense_only": r["c_dense_only"],
                "p_raw": r["p_raw"], "families": a}
        print()
    print("b = policy correct & dense wrong; c = dense correct & policy wrong.")
    print(f"Bootstrap: {BOOT:,} resamples, seed {SEED}. * = oracle upper bound.")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
