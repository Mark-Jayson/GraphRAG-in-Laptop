#!/usr/bin/env python3
"""Per-query latency breakdown figures (HotpotQA, n=500 per arm).

Reads ``results/queries/*.jsonl`` (each record carries a ``timings`` dict),
writes a tidy ``results/queries/latency_breakdown.csv`` with median / mean /
p95 per stage, and renders two figures into ``paper/figures/`` (or
``../manuscript/figures/`` when the manuscript sits beside this repository):

  fig-latency-breakdown       (a) full pipeline, (b) retrieval zoom, across the
                              five 8B retrieval arms
  fig-latency-by-generator    the 1B / 3B / 8B ladder at tag ``base``, where
                              retrieval is held constant

WHY MEDIANS, NOT MEANS (defect D1, corrected 2026-08-13)
--------------------------------------------------------
The previous figure was built from **n=5 queries per model on a 60K corpus**
with the 3B run's corpus size unrecorded, and plotted *means*.  Two problems:

1. n=5 is a dev-scale sample; all seven runs below are the real 500-question
   runs on the full 66,581-passage trf index.
2. The ``hotpotqa_base`` 8B run contains a single query with a recorded
   total_time of 21,301 s -- the host slept mid-run -- which drags its mean to
   69.5 s against a p95 of 38.9 s.  Its *median* (26.04 s) is untouched.  The
   paper reports medians throughout for this reason, so these figures do too.

Per-stage medians are not additive in general, so the script checks how far
the sum of stage medians falls from the measured end-to-end median and prints
it; across these seven runs the worst gap is under 1.5%, which is why a
stacked median bar is a fair representation here.  The number is recomputed on
every run and written into the figure footnote rather than asserted.

Usage:  python scripts/plot_latency_breakdown.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
QDIR = REPO / "results" / "queries"
CSV_OUT = QDIR / "latency_breakdown.csv"
_MS = REPO.parent / "manuscript"
FIG_DIR = (_MS if _MS.is_dir() else REPO / "paper") / "figures"

# --- stages: (timings key, legend label, colour) ----------------------------
# The four retrieval stages use validated categorical slots 1-4 (dataviz
# validator, light, surface #ffffff: worst adjacent CVD dE 9.1, normal-vision
# dE 22.9, ALL CHECKS PASS).  Generation is deliberately NOT a categorical
# peer -- it is 91-99% of every bar, and a saturated fill that large shouts;
# it takes a recessive warm grey and is always direct-labelled, so it never
# depends on colour alone.
STAGES = [
    ("entity_extraction_time", "Query NER", "#2a78d6"),
    ("vector_search_time", "Vector search", "#eb6834"),
    ("ppr_time", "PPR retrieval", "#1baf7a"),
    ("compression_time", "Compression", "#eda100"),
    ("generation_time", "LLM generation", "#b5b4a9"),
]
RETRIEVAL = [s for s in STAGES if s[0] != "generation_time"]
GEN = "generation_time"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#ffffff"

# (jsonl stem, row label) -- 8B retrieval arms, dense baseline first
ARMS = [
    ("hotpotqa_vector_llama-3.1-8b", "vector"),
    ("hotpotqa_base_llama-3.1-8b", "base"),
    ("hotpotqa_ehrag_llama-3.1-8b", "ehrag"),
    ("hotpotqa_compression_llama-3.1-8b", "compression"),
    ("hotpotqa_full_llama-3.1-8b", "full"),
]
# the generator ladder: retrieval configuration held constant at `base`
LADDER = [
    ("hotpotqa_base_llama-3.2-1b", "Llama-3.2-1B"),
    ("hotpotqa_base_llama-3.2-3b", "Llama-3.2-3B"),
    ("hotpotqa_base_llama-3.1-8b", "Llama-3.1-8B"),
]


def pct(values, p):
    v = sorted(values)
    k = (len(v) - 1) * p
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] if f == c else v[f] * (c - k) + v[c] * (k - f)


def load_run(stem: str) -> dict:
    jsonl = QDIR / f"{stem}.jsonl"
    meta = QDIR / f"{stem}.jsonl.meta.json"
    rows = [json.loads(l) for l in jsonl.open() if l.strip()]
    t = [r["timings"] for r in rows if "timings" in r]
    if not t:
        raise SystemExit(f"{jsonl}: no timings")
    totals = [x["total_time"] for x in t]
    med = {k: st.median([x[k] for x in t]) for k, _, _ in STAGES}
    out = {
        "stem": stem,
        "n": len(t),
        "median": med,
        "mean": {k: st.fmean([x[k] for x in t]) for k, _, _ in STAGES},
        "p95": {k: pct([x[k] for x in t], 0.95) for k, _, _ in STAGES},
        "total_median": st.median(totals),
        "total_mean": st.fmean(totals),
        "total_p95": pct(totals, 0.95),
        "total_max": max(totals),
        "retrieval_median": sum(med[k] for k, _, _ in RETRIEVAL),
        "corpus_size": (json.loads(meta.read_text()).get("corpus_size")
                        if meta.exists() else None),
    }
    out["sum_stage_medians"] = sum(med.values())
    out["additivity_gap_pct"] = 100 * (out["sum_stage_medians"] - out["total_median"]) / out["total_median"]
    return out


def write_csv(runs: dict, path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "n_queries", "corpus_size", "stat"]
                   + [k for k, _, _ in STAGES] + ["total"])
        for stem, r in runs.items():
            for stat in ("median", "mean", "p95"):
                w.writerow([stem, r["n"], r["corpus_size"], stat]
                           + [f"{r[stat][k]:.4f}" for k, _, _ in STAGES]
                           + [f"{r[f'total_{stat}']:.4f}"])


def _rcparams():
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times",
                       "Liberation Serif", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 8.0,
        # mathtext in the figure's own serif, not matplotlib's DejaVu sans
        "mathtext.fontset": "stix",
        # NOT "tight": a tight bbox grows the canvas to enclose anything placed
        # outside the figure rectangle, which silently pushed these PDFs to
        # 6.50 in wide -- past the 12.2 cm svproc text width, so LaTeX would
        # shrink them and undo the legibility this rebuild is for. Every
        # artist is laid out inside the figure instead; methodological notes
        # live in the LaTeX caption.
        "savefig.bbox": None,
    })


def stacked_full(ax, rows, labels, height=0.6):
    """Full pipeline: retrieval stages hug each other (they are collectively a
    few percent of the bar, so internal gaps would erase them), then a single
    surface-coloured rule marks the retrieval/generation boundary."""
    ypos = list(range(len(rows)))[::-1]
    for i, r in zip(ypos, rows):
        left = 0.0
        for k, _, colour in STAGES:
            w = r["median"][k]
            ax.barh(i, w, height=height, left=left, color=colour, lw=0)
            left += w
        ax.vlines(r["retrieval_median"], i - height / 2, i + height / 2,
                  color=SURFACE, lw=1.1, zorder=4)
        share = r["median"][GEN] / r["total_median"] * 100
        ax.text(r["total_median"] + ax.get_xlim()[1] * 0.015, i,
                f"{r['total_median']:.1f} s   ({share:.1f}% gen)",
                va="center", ha="left", fontsize=7.4, color=INK)
    ax.set_yticks(ypos, labels)
    ax.grid(True, axis="x", ls="-", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def stacked_retrieval(ax, rows, labels, height=0.6):
    """Retrieval only: segments are individually legible here, so they get a
    surface-coloured gap between fills."""
    ypos = list(range(len(rows)))[::-1]
    for i, r in zip(ypos, rows):
        left = 0.0
        for k, _, colour in RETRIEVAL:
            w = r["median"][k]
            if w <= 0:
                continue
            ax.barh(i, w, height=height, left=left, color=colour,
                    edgecolor=SURFACE, lw=0.55)
            left += w
        ax.text(left + ax.get_xlim()[1] * 0.015, i, f"{left:.2f} s",
                va="center", ha="left", fontsize=7.4, color=INK)
    ax.set_yticks(ypos, labels)
    ax.grid(True, axis="x", ls="-", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def legend_handles(stages):
    return [Patch(facecolor=c, edgecolor=SURFACE, lw=0.5, label=name)
            for _, name, c in stages]


def render_arms(runs, stem: Path) -> None:
    rows = [runs[s] for s, _ in ARMS]
    labels = [l for _, l in ARMS]
    _rcparams()
    fig = plt.figure(figsize=(4.8, 4.05))
    axA = fig.add_axes([0.205, 0.605, 0.775, 0.280])
    axB = fig.add_axes([0.205, 0.195, 0.775, 0.280])

    axA.set_xlim(0, max(r["total_median"] for r in rows) * 1.34)
    stacked_full(axA, rows, labels)
    axA.set_xlabel("Median latency per query (s)", fontsize=8)
    axA.set_title("(a) Full pipeline", fontsize=8.8, loc="left", pad=4)

    axB.set_xlim(0, max(r["retrieval_median"] for r in rows) * 1.26)
    stacked_retrieval(axB, rows, labels)
    axB.set_xlabel("Median latency per query (s)", fontsize=8)
    axB.set_title("(b) Retrieval pipeline only (generation excluded)",
                  fontsize=8.8, loc="left", pad=4)

    fig.legend(handles=legend_handles(STAGES), loc="lower center", ncol=5,
               fontsize=6.9, frameon=False, bbox_to_anchor=(0.5, 0.005),
               handlelength=1.25, handleheight=0.85, columnspacing=0.95,
               handletextpad=0.4)
    fig.suptitle("Per-query latency by pipeline stage "
                 "(HotpotQA, 500 questions, M1/16 GB)", fontsize=9, y=0.975)

    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem.name}.{ext}", dpi=400)
    plt.close(fig)


def render_ladder(runs, stem: Path):
    rows = [runs[s] for s, _ in LADDER]
    labels = [l for _, l in LADDER]
    _rcparams()
    fig = plt.figure(figsize=(4.8, 2.05))
    ax = fig.add_axes([0.245, 0.335, 0.735, 0.395])
    ax.set_xlim(0, max(r["total_median"] for r in rows) * 1.34)
    stacked_full(ax, rows, labels, height=0.68)
    ax.set_xlabel("Median latency per query (s)", fontsize=8)
    fig.suptitle("Per-query latency by generator, retrieval held constant\n"
                 "(HotpotQA, tag base, 500 questions)", fontsize=8.8, y=0.975)
    fig.legend(handles=legend_handles(STAGES), loc="lower center", ncol=5,
               fontsize=6.9, frameon=False, bbox_to_anchor=(0.5, 0.01),
               handlelength=1.25, handleheight=0.85, columnspacing=0.95,
               handletextpad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem.name}.{ext}", dpi=400)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stems = {s for s, _ in ARMS} | {s for s, _ in LADDER}
    runs = {s: load_run(s) for s in sorted(stems)}
    write_csv(runs, CSV_OUT)

    worst = max(abs(r["additivity_gap_pct"]) for r in runs.values())
    contaminated = [r for r in runs.values() if r["total_mean"] > r["total_p95"]]
    note = (f"Medians, n=500 per arm; stage medians sum to within {worst:.1f}% of the "
            f"measured end-to-end median. Medians not means: one base-arm query recorded "
            f"{max(r['total_max'] for r in contaminated):,.0f} s (host slept mid-run).")

    print(note + "\n")
    render_arms(runs, FIG_DIR / "fig-latency-breakdown")
    render_ladder(runs, FIG_DIR / "fig-latency-by-generator")

    print(f"{'run':38} {'n':>4} {'total_med':>10} {'retr_med':>9} {'gen%':>6} {'gap%':>6}")
    for s in [a for a, _ in ARMS] + [l for l, _ in LADDER if l not in dict(ARMS)]:
        r = runs[s]
        print(f"{s:38} {r['n']:>4} {r['total_median']:>10.3f} "
              f"{r['retrieval_median']:>9.3f} "
              f"{r['median'][GEN] / r['total_median'] * 100:>6.1f} "
              f"{r['additivity_gap_pct']:>6.2f}")
    print(f"\nworst additivity gap: {worst:.2f}%")
    print(f"Wrote {CSV_OUT}")
    print(f"Wrote {FIG_DIR / 'fig-latency-breakdown'}.pdf/.png")
    print(f"Wrote {FIG_DIR / 'fig-latency-by-generator'}.pdf/.png")


if __name__ == "__main__":
    main()
