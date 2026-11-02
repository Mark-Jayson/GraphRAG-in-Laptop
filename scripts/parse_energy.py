#!/usr/bin/env python3
"""Per-query energy from powermetrics logs, net of idle draw.

TWO CORRECTIONS over the first version (14 Aug):

1. USE `Combined Power (CPU + GPU + ANE)`, not CPU+GPU paired by hand.
   powermetrics emits TWO `GPU Power:` lines per sample (one in the power
   block, one in the GPU-usage block), so pairing each CPU reading with the
   next GPU reading mis-associates half of them. The Combined line is the
   package total and needs no pairing.

2. DURATION COMES FROM THE RUN'S OWN TIMERS, not the sample count.
   `powermetrics -n 5400` stopped 31 minutes before the `base` batch finished,
   so the sampled window covers only part of it. Because draw is near-constant
   under load (the three thirds of each window agree to within 4%), mean power
   x true batch duration is the sound estimator; sample-count x mean is not.

Energy is reported net of idle so it is the marginal cost of answering a
question rather than the cost of the laptop being switched on.
"""
from __future__ import annotations
import json, re, statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EDIR, QDIR = REPO / "results" / "energy", REPO / "results" / "queries"
COMBINED = re.compile(r"^Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW", re.M)


def draw(p: Path) -> list[float]:
    return [float(m.group(1)) for m in COMBINED.finditer(p.read_text())]


def batch_seconds(arm: str, n: int) -> float | None:
    f = QDIR / f"hotpotqa_{arm}_n{n}_llama-3.1-8b.jsonl"
    if not f.exists():
        return None
    return sum(json.loads(l)["timings"]["total_time"] for l in f.open() if l.strip())


def main() -> None:
    idle = draw(EDIR / "idle.txt")
    idle_mw = st.fmean(idle)
    print(f"idle draw: {idle_mw:.0f} mW over {len(idle)} samples\n")

    out = {"idle_mw": round(idle_mw, 1), "arms": {},
           "method": "Combined Power (CPU+GPU+ANE), net of idle; "
                     "duration from the run's in-loop timers, not the sample count"}
    rows = []
    for pm in sorted(EDIR.glob("*_n*.txt")):
        arm, n = pm.stem.split("_n")[0], int(pm.stem.split("_n")[1])
        s = draw(pm)
        if not s:
            print(f"  {arm}: no Combined Power lines parsed"); continue
        jf = QDIR / f"hotpotqa_{arm}_n{n}_llama-3.1-8b.jsonl"
        nq = sum(1 for _ in jf.open()) if jf.exists() else n
        dur = batch_seconds(arm, n) or len(s)
        covered = 100 * len(s) / dur
        net_mw = st.fmean(s) - idle_mw
        wh_q = net_mw * dur / 3_600_000 / nq
        per_q_s = dur / nq
        out["arms"][arm] = {
            "n_queries": nq, "batch_seconds": round(dur, 1),
            "samples": len(s), "window_coverage_pct": round(covered, 1),
            "mean_mw": round(st.fmean(s), 1), "net_mw": round(net_mw, 1),
            "wh_per_query": round(wh_q, 5), "wh_per_100_queries": round(100 * wh_q, 3),
            "s_per_query_during_measurement": round(per_q_s, 1),
        }
        rows.append((arm, nq, st.fmean(s), covered, wh_q, per_q_s))

    print(f"{'arm':8} {'n':>4} {'mean draw':>11} {'sampled':>8} {'Wh/query':>10} {'Wh/100q':>9} {'s/query':>9}")
    for arm, nq, mw, cov, whq, sq in rows:
        print(f"{arm:8} {nq:>4} {mw:>9.0f} mW {cov:>7.0f}% {whq:>10.5f} {100*whq:>9.3f} {sq:>9.1f}")
    if len(rows) == 2:
        a, b = rows
        print(f"\n{a[0]} / {b[0]} energy ratio: {a[4]/b[4]:.2f}x")
    (EDIR / "energy.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {EDIR/'energy.json'}")


if __name__ == "__main__":
    main()
