#!/usr/bin/env python3
"""Generate the paper's results tables from artifacts, in md / csv / tex.

The paper's tables are *generated*, not transcribed. This script reads
`results/experiments.csv`, `results/analysis/*.json`,
`results/indexing/indexing_stats_*.json` and the query jsonls, and writes
`tab-main-results`, `tab-ablation`, `tab-systems`, `tab-router` and
`tab-router-stats` into `paper/tables/` (or `../manuscript/tables/` when the
manuscript sits beside this repository, as in the authors' checkout).

TWO DEFECTS IN THE HAND-BUILT tab-systems THAT THIS FIXES
---------------------------------------------------------
1. Its stage columns were **means** while its total was a **median**, so the
   columns did not sum to the total, and the HotpotQA `base` generation cell
   carried 67.8 s -- the sleep-contaminated mean that the same table's footnote
   says is excluded. Everything here is a median.
2. It had no `vector` rows at all, which is the arm the paper's retrieval
   argument now rests on.

Derived claims (generation's share of latency, the cost of the graph apparatus,
the compression trade) are COMPUTED and written into the prose lines, because
those three were each wrong in the hand-built version.

Usage:  python scripts/make_tables.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_MS = REPO.parent / "manuscript"
TABDIR = (_MS if _MS.is_dir() else REPO / "paper") / "tables"
QDIR = REPO / "results" / "queries"
ANA = REPO / "results" / "analysis"

STAGES = ["entity_extraction_time", "vector_search_time", "ppr_time",
          "compression_time", "generation_time"]
GRAPH_APPARATUS = ["entity_extraction_time", "vector_search_time", "ppr_time"]

BENCH = [("hotpotqa", "HotpotQA", 66581),
         ("2wikimultihop", "2WikiMultiHopQA", 54957),
         ("musique", "MuSiQue", 21100)]
ARMS = ["vector", "base", "full"]
SLUG8B = "llama-3.1-8b"
MUS_MARK = "‡"   # MuSiQue is its 2-hop tier only


def pct(v, p):
    v = sorted(v)
    k = (len(v) - 1) * p
    f = int(k); c = min(f + 1, len(v) - 1)
    return v[f] if f == c else v[f] * (c - k) + v[c] * (k - f)


def load_stem(stem: str) -> dict | None:
    f = QDIR / f"{stem}.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in f.open() if l.strip()]
    t = [r["timings"] for r in rows if "timings" in r]
    tot = [x["total_time"] for x in t]
    med = {k: st.median([x[k] for x in t]) for k in STAGES}
    return {
        "n": len(rows),
        "median": med,
        "total_median": st.median(tot),
        "total_p95": pct(tot, 0.95),
        "apparatus": sum(med[k] for k in GRAPH_APPARATUS),
        "gen_share": 100 * med["generation_time"] / st.median(tot),
        "refusal": 100 * sum(1 for r in rows
                             if "cannot answer" in r["answer"].lower()) / len(rows),
    }


def load_run(bench: str, arm: str) -> dict | None:
    return load_stem(f"{bench}_{arm}_{SLUG8B}")


def conv() -> dict:
    return json.loads((ANA / "conversion_rates.json").read_text())


def em_f1() -> dict:
    """(benchmark, tag) -> (EM, F1) for the canonical 8B runs.

    Keyed on the source file, not on (benchmark, tag): the `_rep2` and `_strat`
    replicates share both with the canonical run, and whichever row came last
    would silently replace it in Table 1. The 1B/3B ladder belongs to
    tab-ablation 2b.
    """
    out = {}
    with (REPO / "results" / "experiments.csv").open() as f:
        for r in csv.DictReader(f):
            if r["source"] == f"{r['benchmark']}_{r['tag']}_{SLUG8B}.json":
                out[(r["benchmark"], r["tag"])] = (float(r["exact_match"]), float(r["f1"]))
    return out


ROUTER_NAME = {"hotpotqa": "HotpotQA", "2wikimultihop": "2Wiki", "musique": "MuSiQue"}


def paired(bench: str, a: str, b: str) -> dict | None:
    f = ANA / f"paired_{bench}_{a}_{b}.json"
    return json.loads(f.read_text()) if f.exists() else None


def router_ci(bench: str) -> list | None:
    """graph − dense CI from the always-graph row of router_significance.json.

    Table 1b and Table 6 both report this comparison; reading it from one file
    keeps them identical instead of agreeing only up to bootstrap noise.
    """
    f = ANA / "router_significance.json"
    if not f.exists():
        return None
    row = json.loads(f.read_text()).get(ROUTER_NAME[bench], {}).get("always-graph")
    return row["ci95"] if row else None


def fmt(v, nd=2, dash="--"):
    return dash if v is None else f"{v:.{nd}f}"


# ----------------------------------------------------------------- T1
def build_main(scores, runs):
    rows_a, rows_b = [], []
    for key, label, corpus in BENCH:
        mark = MUS_MARK if key == "musique" else ""
        cells = []
        for arm in ARMS:
            s = scores.get((key, arm))
            cells.append(f"{s[0]:.1f} / {s[1]:.2f}" if s else "--")
        rows_a.append([label + mark, f"{corpus:,}"] + cells)

        # short labels: the verbose forms overflowed the LNCS text block
        for a, b, name in [("vector", "base", "graph − dense"),
                           ("base", "full", "full − base")]:
            p = paired(key, a, b)
            if not p:
                continue
            lo, hi = (a == "vector" and router_ci(key)) or p["em"]["ci95"]
            rows_b.append([label + mark, name, str(p["n"]),
                           f"{p['em']['diff']:+.1f}",
                           f"[{lo:+.1f}, {hi:+.1f}]",
                           f"{p['mcnemar_exact_p']:.3f}",
                           f"{p['f1']['diff']:+.2f}"])
    return rows_a, rows_b


# ------------------------------------------------------------ tab-systems
def build_systems(runs):
    idx = []
    for key, label, _ in BENCH:
        f = REPO / "results" / "indexing" / f"indexing_stats_{key}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text()); tg = d["trigraph_stats"]
        idx.append([label + (MUS_MARK if key == "musique" else ""),
                    f"{tg['num_passages']:,}", f"{tg['num_sentences']:,}",
                    f"{tg['num_entities']:,}", f"{tg['num_es_edges']:,}",
                    f"{d['num_hyperedges']:,}",
                    f"{d['total_time']/60:.1f} min",
                    "13.82 GB" if key == "hotpotqa" else "not measured"])
    lat = []
    for key, label, _ in BENCH:
        for arm in ARMS:
            r = runs.get((key, arm))
            if not r:
                continue
            m = r["median"]
            lat.append([label + (MUS_MARK if key == "musique" else ""), f"`{arm}`",
                        f"{r['total_median']:.1f} ({r['total_p95']:.1f})",
                        fmt(m["entity_extraction_time"]), fmt(m["vector_search_time"]),
                        fmt(m["ppr_time"]), fmt(m["compression_time"]),
                        fmt(m["generation_time"], 1), f"{r['refusal']:.1f}%",
                        f"{r['gen_share']:.1f}%"])
    return idx, lat


# Tag semantics. AUTHORITATIVE definition lives in run_queries_batch.py's
# TAG_FLAGS and docs/tag-configurations.md; mirrored here because importing the
# runner would pull in mlx and duckdb just to read two booleans.
TAG_FLAGS = {"base": (False, False), "ehrag": (True, False),
             "compression": (False, True), "full": (True, True)}
LADDER = [("hotpotqa_base_llama-3.2-1b", "Llama-3.2-1B-Instruct-4bit"),
          ("hotpotqa_base_llama-3.2-3b", "Llama-3.2-3B-Instruct-4bit"),
          ("hotpotqa_base_llama-3.1-8b", "Llama-3.1-8B-Instruct-4bit")]
YN = {True: "yes", False: "no"}


def build_ablation(scores, cr):
    rows_a = []
    for tag, (hy, cp) in TAG_FLAGS.items():
        s_ = scores.get(("hotpotqa", tag))
        r = load_stem(f"hotpotqa_{tag}_{SLUG8B}")
        c = cr.get(f"hotpotqa_{tag}_{SLUG8B}", {}).get("strict", {})
        if not (s_ and r and c):
            continue
        rows_a.append([f"`{tag}`", YN[hy], YN[cp], f"{s_[0]:.1f}", f"{s_[1]:.2f}",
                       f"{c['em_given_gold']:.1f}%", f"{r['total_median']:.1f} s"])
    rows_b = []
    for stem, label in LADDER:
        r, c = load_stem(stem), cr.get(stem, {})
        if not (r and c):
            continue
        st_ = c["strict"]
        rows_b.append([label, f"{st_['gold_in_ctx_pct']:.1f}%",
                       f"{st_['em_given_gold']:.1f}%", f"{c['em_overall']:.1f}",
                       f"{c.get('f1_overall', float('nan')):.2f}"
                       if c.get("f1_overall") else "--",
                       f"{c['refusal_pct']:.1f}%", f"{r['total_median']:.1f} s"])
    return rows_a, rows_b



def energy_section() -> str:
    """Energy measured 14 Aug. Reported as a ratio first: the measurement window
    ran throttled (2.6x the generation latency of the latency table above), so
    the absolute figures are order-of-magnitude estimates while the ratio, which
    divides out a slowdown that hit both arms equally, is sound."""
    f = REPO / "results" / "energy" / "energy.json"
    if not f.exists():
        return ("## Energy -- NOT MEASURED\n\nNo `powermetrics` artifacts in the repo.\n")
    d = json.loads(f.read_text())
    a = d.get("arms", {})
    if not a:
        return "## Energy -- no arms parsed\n"
    hdr = ["Arm", "net draw", "Wh / query (measured)", "Wh / query (at Table med.)", "Wh / 100 q"]
    med = {"base": 26.04, "vector": 18.30}
    rows = []
    for arm in ("vector", "base"):
        r = a.get(arm)
        if not r:
            continue
        at_med = r["net_mw"] * med.get(arm, 0) / 3_600_000
        rows.append([f"`{arm}`", f"{r['net_mw']:.0f} mW",
                     f"{r['wh_per_query']:.4f}", f"{at_med:.4f}", f"{100*at_med:.2f}"])
    ratio = ""
    if "base" in a and "vector" in a:
        rb = a["base"]["net_mw"] * med["base"]
        rv = a["vector"]["net_mw"] * med["vector"]
        ratio = (f"\n\nThe graph arm costs **{rb/rv:.2f}x** the dense arm's energy per query "
                 f"at the medians above ({a['base']['wh_per_query']/a['vector']['wh_per_query']:.2f}x "
                 f"as measured) -- the 1.42x latency gap plus a "
                 f"{100*(a['base']['net_mw']/a['vector']['net_mw']-1):.0f}% power premium.")
    return (f"## Energy (powermetrics, 1 Hz, 100 questions per arm)\n\n"
            f"Idle draw {d['idle_mw']:.0f} mW; all figures net of idle.\n\n"
            f"{md_table(hdr, rows)}{ratio}\n\n"
            f"**The ratio is the quotable figure, not the absolute.** The measurement window "
            f"ran at 2.6x the generation latency of the table above (the machine had been under "
            f"sustained load for nine hours and the GPU was thermally throttled), so the measured "
            f"per-query energy corresponds to a slower operating point. Both arms slowed by the "
            f"same factor, so the ratio survives; the 'at Table med.' column applies the measured "
            f"draw to the reported medians and is an estimate.\n")


# Multiplicity families for the routing tests. These are COUNTS OF TESTS ACTUALLY
# RUN, not policies x benchmarks: MuSiQue has no published question-type label so
# it contributes no type-oracle row, and always-dense is the reference arm rather
# than a comparison. 7 + 7 + 6 = 20.
N_ALL, N_DEPLOY, N_CASCADE = 20, 15, 6

ROUTER_ORDER = ["always-dense", "always-graph", "random", "heuristic",
                "learned (5-fold CV)", "learned (transfer)", "type-oracle*",
                "cascade (dense->graph)", "cascade (graph->dense)", "answer-oracle*"]


def router_table() -> tuple[str, list, str]:
    """tab-router from results/analysis/router_ladder.json.

    Each cell is EM (delta vs always-dense) / median latency, so accuracy and
    cost are never separated -- a policy that reaches oracle accuracy by always
    choosing the expensive arm has achieved nothing.
    """
    f = ANA / "router_ladder.json"
    if not f.exists():
        return "", [], ""
    d = json.loads(f.read_text())
    benches = list(d)
    hdr = ["Policy"] + benches
    rows = []
    for pol in ROUTER_ORDER:
        if not any(pol in d[b] for b in benches):
            continue
        cells = []
        for b in benches:
            r = d[b].get(pol)
            # An oracle is charged one arm's latency with no selection
            # mechanism to pay for, so its cost column is not comparable with a
            # deployable policy's; print n/c rather than a misleading number.
            if r is None:
                cells.append("--")
            else:
                lat_cell = "n/c" if pol.endswith("*") else f"{r['med_lat']:.0f}s"
                cells.append(f"{r['em']:.1f} "
                             f"({r['em']-d[b]['always-dense']['em']:+.1f}) / {lat_cell}")
        rows.append([pol.replace("->", "→")] + cells)
    md = (f"# Table: retriever routing (tab-router)\n\n"
            f"Generated by `scripts/make_tables.py` from "
            f"`results/analysis/router_ladder.json`. Each cell is **EM "
            f"(difference vs always-dense) / median latency**. 500 questions per "
            f"benchmark; both arms answered the identical questions, so every "
            f"policy is evaluated offline with no additional inference.\n\n"
            f"{md_table(hdr, rows)}\n\n"
            f"Policies marked * are upper bounds, not deployable: the type-oracle "
            f"reads gold question-type labels and picks the winning arm per type "
            f"from the same split it is scored on; the answer-oracle has "
            f"per-question hindsight. Every other policy uses only the question "
            f"string, or -- for the cascades -- the first arm's own abstention.\n\n"
            f"`random` is the mean over 50 coin-flip draws; the paired tests in "
            f"Table 6 necessarily use a single seeded draw, which differs by up to "
            f"1.8 EM. Every policy-by-benchmark comparison against always-dense "
            f"forms a family of {N_ALL} tests; corrected p-values are in Table 6.\n")
    return md, [hdr] + rows, tex_table(hdr, rows, "lccc")


def sig_table() -> tuple[str, list, str, str]:
    """tab-router-stats from results/analysis/router_significance.json.

    Reports the paired evidence the EM deltas in Table 5 rest on: a 95% bootstrap
    interval on the paired difference, the exact two-sided McNemar p-value from the
    discordant counts, and that p-value corrected for multiplicity.

    LAYOUT IS LOAD-BEARING. The nine-column version of this table (both corrections
    x three families, benchmark as its own column) overflows the svproc A4 text block
    by 34.9pt. This grouped seven-column form compiles clean with all 20 rows intact.
    Do not solve the width problem by dropping rows: the point of the table is that
    14 of the 20 tests do NOT survive correction, so a subset showing only the
    cascade would delete the evidence that the surviving result was not fished.

    Bonferroni is dropped from the body because Holm controls the same familywise
    error rate with uniformly more power, making it strictly redundant; the full
    grid (three families x both corrections) goes to the supplement.

    THREE FAMILIES exist because there is no single defensible choice and hiding the
    choice is the reviewer trap:
      m=20  every policy-by-benchmark test run (most conservative)
      m=15  the deployable subset, dropping the two oracle upper bounds
      m=6   the cascade tests alone, the family if the cascade is read as the
            pre-specified hypothesis and the ladder as description
    """
    f = ANA / "router_significance.json"
    if not f.exists():
        return "", [], "", ""
    d = json.loads(f.read_text())

    # U+2212 MINUS, not ASCII hyphen: tex_table maps it to $-$, so signs in a
    # statistics table typeset as minus signs rather than hyphens.
    def fmt(x):
        if x is None:
            return "--"
        if x >= 1e-3:
            return f"{x:.3f}"
        return f"{x:.1e}".replace("e-0", "e-").replace("e-", "e\u2212")

    def num(x):
        return f"{x:+.1f}".replace("-", "\u2212")

    hdr = ["Policy", "ΔEM", "95% CI", "b/c", "p raw", "Holm₂₀", "Holm₆"]
    NCOL = len(hdr)
    tex_rows, md_rows, full_rows = [], [], []
    for b, pols in d.items():
        # benchmark as a spanning row header rather than a repeated column: that is
        # what buys the width this table needs (see the docstring).
        tex_rows.append([f"\\multicolumn{{{NCOL}}}{{l}}{{\\emph{{{b}}}}}"])
        md_rows.append([f"**{b}**"] + [""] * (NCOL - 1))
        for pol, r in pols.items():
            fam = r["families"]
            cas = fam.get("m_cascade", {})
            cells = [num(r["delta_em"]),
                     f"[{num(r['ci95'][0])}, {num(r['ci95'][1])}]",
                     f"{r['policy_only']}/{r['dense_only']}",
                     fmt(r["p_raw"]), fmt(fam["m_all"]["holm"]),
                     fmt(cas.get("holm"))]
            arrowed = pol.replace("->", "\u2192")
            # The paper's caption omits the random router (a single seeded draw,
            # not Table 5's 50-draw mean); the .md and -full.csv keep it.
            if pol != "random":
                tex_rows.append(["~~" + arrowed] + cells)
            md_rows.append(["  " + arrowed] + cells)
            full_rows.append([b, pol, r["delta_em"], r["ci95"][0], r["ci95"][1],
                              r["policy_only"], r["dense_only"], r["p_raw"]]
                             + [fam.get(k, {}).get(c)
                                for k in ("m_all", "m_deploy", "m_cascade")
                                for c in ("holm", "bonferroni")])

    caption = (
        "Generated by `scripts/make_tables.py` from "
        "`results/analysis/router_significance.json`. Every policy is compared against "
        "always-dense on the same 500 questions. ΔEM is the paired difference in exact "
        "match; the interval is a 95% percentile bootstrap over 100,000 resamples of the "
        "per-question difference (seed 20260814); b/c are the discordant counts "
        "(questions the policy alone answers correctly / always-dense alone answers "
        "correctly) and feed an exact two-sided McNemar test, which is the p-value "
        "corrected in the last two columns. Subscripts are the multiplicity family "
        "size: Holm₂₀ corrects across every test in this table, Holm₆ across the six "
        "cascade tests alone; the deployable family (m=15, dropping the two oracles) "
        "falls between them. Holm is reported "
        "rather than Bonferroni because it controls the same familywise error rate with "
        "uniformly more power; the full grid of both corrections across all three "
        "families is in `tab-router-stats-full.csv`. Policies marked * are upper bounds, "
        "not deployable. `random` is a single seeded draw here against Table 5's "
        "50-draw mean, a difference of up to 1.8 EM.")

    verdict = (
        "**Both cascade directions survive correction on 2Wiki and MuSiQue, and "
        "dense-first survives on all three benchmarks under every family definition "
        "reported here.** Graph-first does not clear correction on HotpotQA at m=20 "
        "(Holm 0.106, Bonferroni 0.177), where it clears only within the cascade family "
        "(0.009). **No other deployable policy survives any correction on any "
        "benchmark**: the heuristic's +2.8 on 2Wiki, the type-oracle's +3.8 there and "
        "always-graph's +4.2 on MuSiQue are nominally significant at p < .05 and all "
        "fail once the family is accounted for. Note that the dense-first tests are "
        "one-sided by construction — b/c of 19/0, 38/0 and 26/0 — because the policy "
        "returns the dense answer unchanged unless dense refused, and a refusal is "
        "scored wrong; the effect size and its interval carry that result, not the "
        "p-value (§6.3).")

    md = (f"# Table: routing significance (tab-router-stats)\n\n{caption}\n\n"
          f"{md_table(hdr, md_rows)}\n\n{verdict}\n")
    full_hdr = ["benchmark", "policy", "delta_em", "ci95_lo", "ci95_hi",
                "b_policy_only", "c_dense_only", "p_raw"] + \
               [f"{k}_{c}" for k in ("m20", "m15", "m6")
                for c in ("holm", "bonferroni")]
    return md, [full_hdr] + full_rows, tex_table(hdr, tex_rows, "lrcrrrr"), caption


def cols(hdr, rows, keep):
    """Project a table onto a subset of columns for the LaTeX view.

    The .md and .csv views keep every column: they are the repository record.
    The paper's Table 3 carries only the columns its argument rests on -- corpus
    size, indexing cost, peak memory, total latency and generation's share --
    because the per-stage breakdown and the graph-size counts are reproducibility
    detail that the main text never cites.
    """
    return [hdr[i] for i in keep], [[r[i] for i in keep] for r in rows]


def md_table(hdr, rows):
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def tex_table(hdr, rows, align):
    # pdflatex cannot render raw Unicode under OT1/T1, so every non-ASCII
    # glyph the tables can contain is mapped to a LaTeX command here. A stray
    # one is a fatal build error, not a cosmetic issue -- U+0394 cost a build.
    SUBS = [("`", ""), ("_", r"\_"),
            ("−", "$-$"), ("–", "--"), ("—", "---"),
            ("Δ", r"$\Delta$"), ("×", r"$\times$"),
            ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("±", r"$\pm$"),
            ("‡", r"$\ddagger$"), ("†", r"$\dagger$"),
            ("→", r"$\rightarrow$"), ("’", "'"),
            ("₂₀", "$_{20}$"), ("₁₅", "$_{15}$"), ("₆", "$_{6}$"),
            ("%", r"\%"), ("&", r"\&"),
            ("[", "{[}"), ("]", "{]}")]
    def esc(s):
        for a, b in SUBS:
            s = s.replace(a, b)
        return s
    out = [r"\begin{tabular}{" + align + "}", r"\toprule",
           " & ".join(esc(h) for h in hdr) + r" \\", r"\midrule"]
    out += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def write(stem, md, csv_rows, tex):
    TABDIR.mkdir(parents=True, exist_ok=True)
    (TABDIR / f"{stem}.md").write_text(md)
    (TABDIR / f"{stem}.tex").write_text(tex)
    with (TABDIR / f"{stem}.csv").open("w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"  wrote {stem}.md / .csv / .tex")


def main():
    scores = em_f1()
    runs = {(b, a): load_run(b, a) for b, _, _ in BENCH for a in ARMS}
    runs = {k: v for k, v in runs.items() if v}
    missing = [f"{b}_{a}" for b, _, _ in BENCH for a in ARMS if (b, a) not in runs]
    if missing:
        print(f"  note: no jsonl for {', '.join(missing)}", file=sys.stderr)

    # ---------------- T1 ----------------
    hdr_a = ["Benchmark", "corpus", "`vector` EM / F1", "`base` EM / F1", "`full` EM / F1"]
    hdr_b = ["Benchmark", "comparison", "n", "ΔEM", "95% CI", "McNemar p", "ΔF1"]
    rows_a, rows_b = build_main(scores, runs)
    foot = (f"<sup>{MUS_MARK}</sup> MuSiQue is evaluated on its 2-hop tier only: the "
            "runners take the first *n* questions and that split is ordered by hop "
            "count. HotpotQA and 2Wiki head samples match their full-file question-type "
            "mixes within three points. See §7.")
    md = (f"# Table 1: main results (tab-main-results)\n\n"
          f"Generated by `scripts/make_tables.py` from `results/experiments.csv` and "
          f"`results/analysis/paired_*.json`. 500 questions per cell, "
          f"`Meta-Llama-3.1-8B-Instruct-4bit`, `en_core_web_trf` index on all graph arms.\n\n"
          f"## 1a — exact match / F1\n\n{md_table(hdr_a, rows_a)}\n\n{foot}\n\n"
          f"## 1b — paired differences (same questions, so paired CIs and exact McNemar)\n\n"
          f"{md_table(hdr_b, rows_b)}\n\n"
          f"Minimum detectable effect at n=500, 80% power: about ±3.6 EM. Every `full`−`base` "
          f"row is therefore a **bounded null**, not merely an undetected effect. The one "
          f"interval excluding zero is MuSiQue{MUS_MARK} graph−dense; note that its F1 "
          f"difference does not follow it, and that a Bonferroni threshold across the three "
          f"graph−dense comparisons (.0167) would not clear it.\n")
    write("tab-main-results", md,
          [hdr_a] + rows_a + [[]] + [hdr_b] + rows_b,
          # Panel 1b stays in the .md/.csv only: the paper quotes its intervals in
          # Sect. 5.1, and printing it too pushed the paper over the page limit.
          tex_table(hdr_a, rows_a, "llrrr") + "\n")

    # ------------- tab-systems -------------
    idx, lat = build_systems(runs)
    hdr_i = ["Benchmark", "passages", "sentences", "entities", "es-edges",
             "hyperedges", "index time", "peak footprint"]
    hdr_l = ["Benchmark", "tag", "total med (p95)", "NER", "vector", "PPR",
             "compr", "generation", "refusal", "gen share"]
    shares = [r["gen_share"] for r in runs.values()]
    gshare = [r["gen_share"] for (b, a), r in runs.items() if a != "vector"]
    app = [r["apparatus"] for (b, a), r in runs.items() if a != "vector" and b != "musique"]
    app_mus = [r["apparatus"] for (b, a), r in runs.items() if a != "vector" and b == "musique"]
    w2b, w2f = runs.get(("2wikimultihop", "base")), runs.get(("2wikimultihop", "full"))
    comp = ""
    if w2b and w2f:
        comp = (f" Compression pays for itself: it adds "
                f"{w2f['median']['compression_time']:.2f} s and removes "
                f"{w2b['median']['generation_time']-w2f['median']['generation_time']:.1f} s of "
                f"generation on 2Wiki ({w2b['total_median']:.1f} s -> {w2f['total_median']:.1f} s "
                f"median, **{100*(w2f['total_median']-w2b['total_median'])/w2b['total_median']:+.1f}%**).")
    md = (f"# Table: systems measurements (tab-systems)\n\n"
          f"Generated by `scripts/make_tables.py`. **Every latency figure is a median; "
          f"p95 in parentheses on the total.** Stage columns are medians too, so they sum "
          f"to the total (within the small non-additivity of medians). No mean and no "
          f"aggregate wall-clock appears anywhere: HotpotQA `base` slept mid-run and its "
          f"mean (69.5 s) exceeds its own p95. See `results/analysis/latency-notes.md`.\n\n"
          f"## Indexing (one-time, offline, zero LLM calls, `en_core_web_trf`)\n\n"
          f"{md_table(hdr_i, idx)}\n\n"
          f"Peak footprint uses `phys_footprint` via `proc_pid_rusage`, **not** RSS, which "
          f"is blind to Metal allocations and under-reports by ~7 GB at this scale. Only "
          f"HotpotQA was instrumented.\n\n"
          f"## Query latency (500 q, Llama-3.1-8B-Instruct-4bit, seconds)\n\n"
          f"{md_table(hdr_l, lat)}\n\n{foot}\n\n"
          f"**Reading:** generation is {min(shares):.1f}-{max(shares):.1f}% of every cell "
          f"({min(gshare):.1f}-{max(gshare):.1f}% across the graph arms, "
          f"{min(s for (b,a),r in runs.items() if a=='vector' for s in [r['gen_share']]):.1f}% "
          f"and above in the dense arms). The entire graph apparatus -- query NER, vector "
          f"seeding and PPR -- costs {min(app):.2f}-{max(app):.2f} s on HotpotQA and 2Wiki "
          f"and about {st.fmean(app_mus):.2f} s on MuSiQue.{comp}\n\n"
          f"{energy_section()}")
    # LaTeX view drops the graph-size counts (sentences/entities/es-edges/
    # hyperedges) and the per-stage timings; both stay in the .md and .csv.
    tex_i = tex_table(*cols(hdr_i, idx, [0, 1, 6, 7]), "lrrr")
    tex_l = tex_table(*cols(hdr_l, lat, [0, 1, 2, 7, 8, 9]), "llrrrr")
    write("tab-systems", md,
          [hdr_i] + idx + [[]] + [hdr_l] + lat,
          tex_i + "\n\n% query latency\n\n" + tex_l)

    # ------------- tab-ablation (T2) -------------
    cr = conv()
    ab_a, ab_b = build_ablation(scores, cr)
    f1map = {(r["benchmark"], r["tag"], r["model"]): None for r in []}
    # F1 for the ladder comes from experiments.csv, keyed by model
    # match on the model slug, not the display name: experiments.csv carries
    # "Meta-Llama-3.1-8B-Instruct-4bit" while the ladder label omits "Meta-",
    # and a silent .get() miss printed "nan" into the table.
    with (REPO / "results" / "experiments.csv").open() as fh:
        lad_f1 = {}
        for r in csv.DictReader(fh):
            if r["benchmark"] != "hotpotqa" or r["tag"] != "base":
                continue
            norm = r["model"].split("/")[-1].lower().replace("meta-", "")
            lad_f1[norm] = float(r["f1"])
    for row, (stem, label) in zip(ab_b, LADDER):
        slug = stem.replace("hotpotqa_base_", "")          # e.g. llama-3.2-1b
        hit = [v for k, v in lad_f1.items() if k.startswith(slug)]
        assert len(hit) == 1, f"F1 lookup for {slug!r} matched {len(hit)} rows in experiments.csv"
        row[4] = f"{hit[0]:.2f}"

    mde = json.loads((ANA / "paired_hotpotqa_base_full.json").read_text())
    mde_500 = mde["power"]["500"]["mde_80pct"]
    hdr_2a = ["Tag", "hyperedges", "compression", "EM", "F1", "conversion", "median latency"]
    hdr_2b = ["Generator", "gold in context", "conversion", "EM", "F1", "refusal", "median latency"]
    ems_a = [float(r[3]) for r in ab_a]
    cvs_a = [float(r[5].rstrip("%")) for r in ab_a]
    ems_b = [float(r[3]) for r in ab_b]
    lat_b = [float(r[6].split()[0]) for r in ab_b]
    md = (f"# Table 2: ablation (tab-ablation)\n\n"
          f"Generated by `scripts/make_tables.py`. HotpotQA, 500 questions, "
          f"`en_core_web_trf` index. Conversion is EM given the gold string reached the "
          f"context, on the paper's `strict` subset. Minimum detectable effect at n=500, "
          f"80% power: **+/-{mde_500:.1f} EM** -- the nulls in 2a are bounded, not merely "
          f"undetected.\n\n"
          f"## 2a -- pipeline features (generator fixed: Llama-3.1-8B-Instruct-4bit)\n\n"
          f"{md_table(hdr_2a, ab_a)}\n\n"
          f"Span: **{max(ems_a)-min(ems_a):.1f} EM**, inside the detectable effect. "
          f"Conversion varies by only {max(cvs_a)-min(cvs_a):.1f} points across all four "
          f"configurations.\n\n"
          f"## 2b -- generator scale (retrieval held constant: tag `base`)\n\n"
          f"Retrieval is identical across these rows -- same index, same questions, same "
          f"configuration -- so gold-in-context is {ab_b[0][1]} in every row. "
          f"**Only the generator changes.**\n\n"
          f"{md_table(hdr_2b, ab_b)}\n\n"
          f"**Generator scale moves EM by {max(ems_b)-min(ems_b):.1f} points "
          f"({min(ems_b):.1f} -> {max(ems_b):.1f}); no retrieval configuration in 2a moves "
          f"it by more than {max(ems_a)-min(ems_a):.1f}.** The gain is strongly concave: "
          f"1B->3B is +{ems_b[1]-ems_b[0]:.1f} EM, 3B->8B only +{ems_b[2]-ems_b[1]:.1f}. "
          f"The 3B generator reaches {100*ems_b[1]/ems_b[2]:.0f}% of the 8B's exact match "
          f"at {lat_b[1]/lat_b[2]:.2f}x its median latency.\n")
    write("tab-ablation", md,
          [hdr_2a] + ab_a + [[]] + [hdr_2b] + ab_b,
          tex_table(hdr_2a, ab_a, "lccrrrr") + "\n\n% panel 2b\n\n"
          + tex_table(hdr_2b, ab_b, "lrrrrrr"))

    rmd, rrows, rtex = router_table()
    if rmd:
        write("tab-router", rmd, rrows, rtex)

    smd, sfull, stex, _ = sig_table()
    if smd:
        (TABDIR / "tab-router-stats.md").write_text(smd)
        (TABDIR / "tab-router-stats.tex").write_text(stex)
        with (TABDIR / "tab-router-stats-full.csv").open("w", newline="") as fh:
            csv.writer(fh).writerows(sfull)
        print("  wrote tab-router-stats.md / .tex + -full.csv")

    print(f"\n  generation share : {min(shares):.1f}-{max(shares):.1f}% overall; "
          f"graph arms {min(gshare):.1f}-{max(gshare):.1f}%")
    print(f"  graph apparatus  : {min(app):.2f}-{max(app):.2f} s (HotpotQA/2Wiki), "
          f"{st.fmean(app_mus):.2f} s (MuSiQue)")


if __name__ == "__main__":
    main()
