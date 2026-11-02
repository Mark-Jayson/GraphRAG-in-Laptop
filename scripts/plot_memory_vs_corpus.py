#!/usr/bin/env python3
"""Indexing-time peak memory vs. corpus size (HotpotQA).

Parses the scale-up log at ``results/indexing/scaleup-log.md`` -- both the ``## N = ...``
sweep blocks (run with ``en_core_web_sm``) and the ``# Final Quality Index
(trf)`` section -- into a tidy table, writes
``results/indexing/memory_vs_corpus.csv``, and renders
``paper/figures/fig-memory-vs-corpus.{pdf,png}`` (or ``../manuscript/figures/``
when the manuscript sits beside this repository).

WHAT THIS FIGURE MEASURES (corrected 2026-08-13, defect D2)
-----------------------------------------------------------
Every ``Peak footprint`` line in the scale-up log was recorded during *indexing*
(the accompanying JSON reports trigraph / storage / embedding / hyperedge
times).  Indexing is LLM-free, so the ~7.6 GB floor is the spaCy pipeline,
the sentence embedder, DuckDB and the Python runtime -- **not** the 8B-4bit
LLM, which is not resident during indexing.

The previous version of this figure was titled "query-time peak memory",
annotated its floor as "8B-4bit LLM weights + KV cache", and drew a
"~12 GB query-time working ceiling" that the real final-quality index
(13.82 GB with ``en_core_web_trf``) exceeds.  All four were wrong.  There is
no query-time peak-memory measurement in the artifacts, so this figure no
longer claims one.

THE ACTUAL FINDING: corpus size is nearly free (+0.82 GB across 13.3x corpus
growth); NER model choice is not (+5.39 GB at identical corpus, 6.6x the
whole corpus sweep).  The memory wall is a model decision, not a data-scale
decision, and the index we actually report sits at 86% of RAM.

Usage:  python scripts/plot_memory_vs_corpus.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# --- paths ------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "indexing" / "scaleup-log.md"
CSV_OUT = REPO / "results" / "indexing" / "memory_vs_corpus.csv"
_MS = REPO.parent / "manuscript"
FIG_STEM = (_MS if _MS.is_dir() else REPO / "paper") / "figures" / "fig-memory-vs-corpus"

PHYS_RAM_GB = 16.0  # MacBook Pro 13" M1 unified memory

# validated categorical slots (dataviz validator, light, surface #ffffff:
# adjacent CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 contrast)
C_SM = "#2a78d6"     # slot 1 blue   - the swept curve (en_core_web_sm)
C_TRF = "#eb6834"    # slot 2 orange - the trf operating point
C_RAM = "#52514e"    # secondary ink - physical-RAM threshold
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BAND = "#e1e0d9"

FOOTPRINT_RE = re.compile(r"Peak footprint:\s*([\d.]+)\s*GB")


def first_json_object(text: str) -> dict | None:
    """Return the first brace-balanced JSON object in ``text``.

    A greedy ``\\{.*\\}`` match breaks once the log grows further sections
    after the sweep (it spans several concatenated objects and raises), which
    is why this scans for a balanced object instead.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _row(section: str, ner_model: str) -> dict | None:
    fp = FOOTPRINT_RE.search(section)
    stats = first_json_object(section)
    if not (fp and stats):
        return None
    tg = stats.get("trigraph_stats", {})
    return {
        # the actual indexed passage count -- the last sweep block is headed
        # "N = 70000" but caps at the full corpus of 66,581
        "corpus": int(tg.get("num_passages")),
        "ner_model": ner_model,
        "footprint_gb": float(fp.group(1)),
        "entities": int(tg.get("num_entities", 0)),
        "es_edges": int(tg.get("num_es_edges", 0)),
        "hyperedges": int(stats.get("num_hyperedges", 0)),
        "total_time_s": float(stats.get("total_time", 0.0)),
    }


def parse_phase3(md_path: str | Path) -> tuple[list[dict], dict]:
    text = Path(md_path).read_text()

    # isolate the sweep from everything that follows it
    split = re.split(r"^#\s+Final Quality Index.*$", text, flags=re.MULTILINE)
    sweep_text = split[0]
    trf_text = split[1] if len(split) > 1 else ""
    # the trf section ends where the next top-level heading begins
    trf_text = re.split(r"^#\s+\S", trf_text, flags=re.MULTILINE)[0]

    sweep: list[dict] = []
    for block in re.split(r"^##\s*N\s*=.*$", sweep_text, flags=re.MULTILINE)[1:]:
        row = _row(block, "en_core_web_sm")
        if row:
            sweep.append(row)
    sweep.sort(key=lambda r: r["corpus"])

    trf = _row(trf_text, "en_core_web_trf")
    if trf is None:
        raise SystemExit("could not parse the trf final-quality index block")
    return sweep, trf


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def render(sweep: list[dict], trf: dict, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    xs = [r["corpus"] for r in sweep]
    ys = [r["footprint_gb"] for r in sweep]
    floor = min(ys)
    x_max = max(max(xs), trf["corpus"]) * 1.10
    jump = trf["footprint_gb"] - ys[-1]

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
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "mathtext.fontset": "stix",
        "savefig.bbox": None,
    })
    fig = plt.figure(figsize=(4.8, 3.0))
    ax = fig.add_axes([0.145, 0.155, 0.835, 0.735])

    # fixed-cost band: what is resident regardless of corpus size
    ax.axhspan(0, floor, color=BAND, alpha=0.55, zorder=0, lw=0)
    ax.text(x_max * 0.985, floor * 0.70,
            "fixed cost, LLM-free: spaCy pipeline + sentence\n"
            "embedder + DuckDB + Python runtime",
            ha="right", va="center", fontsize=6.8, color=INK2, style="italic")

    # physical-RAM threshold (a real threshold, so dashed is meaningful)
    ax.axhline(PHYS_RAM_GB, ls=(0, (5, 3)), lw=1.0, color=C_RAM, zorder=1)
    ax.text(x_max * 0.015, PHYS_RAM_GB + 0.16, "16 GB physical RAM",
            ha="left", va="bottom", fontsize=7.4, color=C_RAM)

    # the swept curve (en_core_web_sm)
    ax.plot(xs, ys, "-o", color=C_SM, lw=1.5, ms=3.8, mfc="white",
            mec=C_SM, mew=1.1, zorder=3)

    # the trf operating point
    ax.plot([trf["corpus"]], [trf["footprint_gb"]], "D", color=C_TRF,
            ms=6.4, mec="white", mew=1.0, zorder=5)
    ax.annotate(f"{trf['footprint_gb']:.2f} GB  "
                f"({trf['footprint_gb'] / PHYS_RAM_GB * 100:.0f}% of RAM)",
                xy=(trf["corpus"], trf["footprint_gb"]),
                xytext=(trf["corpus"] * 0.975, trf["footprint_gb"] + 0.75),
                ha="right", va="bottom", fontsize=7.4, color=C_TRF, zorder=6)

    # the jump that matters: same corpus, different NER model
    ax.annotate("", xy=(trf["corpus"], trf["footprint_gb"]),
                xytext=(trf["corpus"], ys[-1]),
                arrowprops=dict(arrowstyle="<->", color=C_TRF, lw=1.0,
                                shrinkA=1.5, shrinkB=1.5), zorder=4)
    ax.annotate(f"sm $\\rightarrow$ trf, identical corpus: +{jump:.2f} GB\n"
                f"({jump / (ys[-1] - ys[0]):.1f}$\\times$ the entire corpus sweep)",
                xy=(trf["corpus"], (trf["footprint_gb"] + ys[-1]) / 2),
                xytext=(trf["corpus"] * 0.955, (trf["footprint_gb"] + ys[-1]) / 2),
                ha="right", va="center", fontsize=7.2, color=C_TRF, zorder=6)

    # how flat the corpus dimension is
    ax.annotate(f"+{ys[-1] - ys[0]:.2f} GB across "
                f"{xs[-1] / xs[0]:.1f}$\\times$ corpus growth",
                xy=(xs[len(xs) // 2], ys[len(ys) // 2]),
                xytext=(xs[1], ys[-1] + 1.45),
                fontsize=7.2, color=C_SM, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=C_SM, lw=0.8,
                                shrinkB=3))

    ax.set_xlabel("Corpus size (indexed passages)")
    ax.set_ylabel("Peak memory during indexing (GB)")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, PHYS_RAM_GB + 1.3)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0"))
    ax.grid(True, axis="y", ls="-", lw=0.5, color=GRID, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        Line2D([], [], color=C_SM, lw=1.5, marker="o", ms=3.8, mfc="white",
               mec=C_SM, mew=1.1, label="\\texttt{en\\_core\\_web\\_sm} sweep"
               if False else "en_core_web_sm sweep"),
        Line2D([], [], color=C_TRF, lw=0, marker="D", ms=6.0, mec="white",
               mew=1.0, label="en_core_web_trf (final-quality index)"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.885),
              fontsize=7.2, frameon=False, handletextpad=0.6, labelspacing=0.35)
    ax.set_title("Indexing-time peak memory vs. corpus size (HotpotQA, M1/16 GB)",
                 fontsize=9, pad=7)

    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=400)
    plt.close(fig)


def main() -> None:
    sweep, trf = parse_phase3(SRC)
    if not sweep:
        raise SystemExit(f"No sweep data parsed from {SRC}")
    write_csv(sweep + [trf], CSV_OUT)
    render(sweep, trf, FIG_STEM)
    print(f"sweep: {len(sweep)} sizes, {sweep[0]['corpus']:,}..{sweep[-1]['corpus']:,} "
          f"passages, {sweep[0]['footprint_gb']:.2f}-{sweep[-1]['footprint_gb']:.2f} GB "
          f"({sweep[0]['ner_model']})")
    print(f"trf   : {trf['corpus']:,} passages, {trf['footprint_gb']:.2f} GB "
          f"= {trf['footprint_gb'] / PHYS_RAM_GB * 100:.1f}% of {PHYS_RAM_GB:.0f} GB")
    print(f"corpus dimension : +{sweep[-1]['footprint_gb'] - sweep[0]['footprint_gb']:.2f} GB "
          f"across {sweep[-1]['corpus'] / sweep[0]['corpus']:.1f}x growth")
    print(f"NER-model jump   : +{trf['footprint_gb'] - sweep[-1]['footprint_gb']:.2f} GB "
          f"at identical corpus")
    print(f"Wrote {CSV_OUT}")
    print(f"Wrote {FIG_STEM}.pdf and .png")


if __name__ == "__main__":
    main()
