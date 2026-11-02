# Energy measurement — provenance, and why the claim was cut from the paper

Written 2026-09-20, after re-parsing the raw traces in this directory.

**Short version: `energy.json`'s "6% power premium" is a sampling artifact. Corrected,
the graph arm draws about 1% *less* power than the dense arm, and the energy ratio
collapses onto the latency ratio. The energy claim was removed from the WCTP2026
paper on this basis. Do not re-quote these numbers without reading the rest of this
file.**

## What was measured

Produced by `scripts/measure_energy.sh` on 2026-08-14 (`powermetrics --samplers
cpu_power,gpu_power -i 1000 -n 5400`), parsed by `scripts/parse_energy.py`:

| file | samples | window |
|---|---|---|
| `idle.txt` | 60 | 08:58:27–08:59:27, idle baseline, mean 136.8 mW |
| `base_n100.txt` | 5400 | 08:59:29–10:43:05, HotpotQA `base`, 100 questions |
| `vector_n100.txt` | 5400 | 11:22:11–12:53:19, HotpotQA `vector`, 100 questions |
| `energy.json` | — | the derived summary; the only numeric basis for the paper's prose |

One benchmark, two arms, 100 questions each — against 500 everywhere else in the
paper. The run followed roughly nine hours of sustained inference (an overnight
batch, 21:54→06:52) with the GPU thermally throttled: median generation in this run is
63.4 s (`base`) and 47.3 s (`vector`) against 24.3 s and 18.2 s in `tab-systems`, a
factor of ~2.6. The absolute Wh/query figures therefore describe a slower machine
than the rest of the paper, which is why only the ratio was ever quoted.

## The defect

**The two sampling windows are misaligned with their batches in opposite
directions.** `base` sampling stopped ~18 min *before* its batch ended, so every
`base` sample is under load. `vector` sampling ran ~8 min *past* its batch end, so
its tail is post-run cooldown.

Power by thirds of each window, and the last 300 samples:

```
base    3334 / 3387 / 3392 mW   (spread  1.7%)    last 300:  3387 mW
vector  3399 / 3406 / 2762 mW   (spread 18.9%)    last 300:   965 mW
```

Truncating `vector` at the point its load ends (sample 4955 of 5400 — 445 trailing
cooldown samples):

```
base   net draw                       3234.2 mW
vector net draw, full window          3052.0 mW   ->  graph +6.0%   <- energy.json
vector net draw, cooldown truncated   3263.1 mW   ->  graph -0.9%   <- corrected
```

So the "6% power premium" is entirely the cooldown tail. Corrected, the per-query
energy ratio falls from **1.55× to ≈1.45×**, which is indistinguishable from this
run's own latency ratio of **1.46×** (72.64 s vs 49.80 s per query). The energy
measurement carries no information beyond the latency measurement it was meant to
corroborate.

Two related notes:

- `parse_energy.py`'s docstring justifies the means with "the three thirds of each
  window agree to within 4%". That holds for `base` (1.7%) but not for `vector`
  (18.9%) — the check was never applied per-arm.
- The coverage figures in `energy.json` (74.3% / 108.4%) divide 5400 *samples* by
  batch *seconds*. Real spacing is ~1.15 s (`base`) and ~1.01 s (`vector`), so true
  wall-clock coverage is ~85.6% / ~109.8%.

## What was done about it

- The energy claim was cut from the paper (Sections 5 and 7, the abstract and
  the contributions list).
- `energy.json`, the generated `tab-systems` table and `parse_energy.py`
  were **left as they are**. This note records the defect rather than rewriting the
  artifacts; anything reusing them should apply the truncation above first.

## If the measurement is ever redone

Stop sampling when the batch stops, not on a fixed sample count — or record batch
start/end timestamps alongside the trace and truncate at parse time. Cover more than
one benchmark and more than two arms, and take it on a machine that is not already
nine hours into a thermal-throttled run.
