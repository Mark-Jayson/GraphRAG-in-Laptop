#!/usr/bin/env python3
"""Build a hop-stratified MuSiQue sample, because the default sample is 2-hop only.

THE BUG THIS EXISTS FOR
-----------------------
`run_queries_batch.py` and `run_vector_baseline.py` both select questions with
`json.load(f)[: args.max_samples]` -- the sequential head of the file, no
shuffle. That is harmless for HotpotQA and 2Wiki, whose dev files are not
ordered by anything we measure (their first-500 question-type mixes match the
full files within ~3 points). It is NOT harmless for MuSiQue: the
`dgslibisey/MuSiQue` validation split is ordered by hop count, so rows 0-499
are entirely 2-hop. Every MuSiQue number reported from a 500-question run is
therefore measured on the benchmark's easiest tier, and no dose-response over
hop count can be computed from it at all.

This script writes a NEW question file, stratified across hop counts, leaving
`data/raw/musique_val.json` untouched. Nothing in this repo mutates a raw data
file in place: a silent reorder would make every existing artifact
irreproducible without any visible sign, which is the same class of mistake as
the 2026-07-19 filename overwrite.

  --scheme proportional  (default) mirrors the true hop mix, so the aggregate
                         score means "MuSiQue" rather than "MuSiQue, 2-hop".
                         Use this if the sample is replacing a T1 cell.
  --scheme equal         equal n per hop tier, which maximises power for the
                         dose-response test at the cost of a non-representative
                         aggregate. Use this if the question is "does the
                         graph's advantage grow with hops?" and NOT for T1.

Requires Hugging Face reachable (or the dataset cached from download_data.py).

Usage:
    python scripts/make_stratified_sample.py --n 500 --scheme proportional
    python scripts/make_stratified_sample.py --n 501 --scheme equal --seed 20260813

Then, with the --questions-file / --out-suffix patch applied to the runners:
    python scripts/run_queries_batch.py --config config_musique.yaml \
        --benchmark musique --tag base --max-samples 500 \
        --questions-file data/raw/musique_val_strat500.json --out-suffix _strat
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "raw" / "musique_val.json"


def hop_of_source() -> dict[str, int]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets  (inside the project venv)")
    try:
        from download_data import REVISIONS
        ds = load_dataset("dgslibisey/MuSiQue", revision=REVISIONS["dgslibisey/MuSiQue"])
    except Exception as e:
        raise SystemExit(f"cannot load dgslibisey/MuSiQue ({type(e).__name__}: {e}). "
                         "Needs Hugging Face reachable or the dataset cached.")
    out = {}
    for split in ds:
        for rec in ds[split]:
            m = re.match(r"(\d+)hop", str(rec.get("id", "")))
            if m and rec.get("question"):
                out[rec["question"].strip()] = int(m.group(1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--scheme", choices=["proportional", "equal"], default="proportional")
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    local = json.loads(SRC.read_text())
    if isinstance(local, dict):
        local = local.get("data") or list(local.values())[0]
    hops = hop_of_source()

    tiers: dict[int, list] = collections.defaultdict(list)
    unlabelled = 0
    for rec in local:
        h = hops.get(rec["question"].strip())
        if h is None:
            unlabelled += 1
            continue
        tiers[h].append(rec)

    print(f"local dev file: {len(local):,} questions, {unlabelled} without a hop label")
    total = sum(len(v) for v in tiers.values())
    for h in sorted(tiers):
        print(f"  {h}-hop: {len(tiers[h]):>5}  ({100*len(tiers[h])/total:.1f}%)")

    # confirm the ordering problem rather than asserting it
    head = [hops.get(r["question"].strip()) for r in local[:args.n]]
    seen = sorted({h for h in head if h is not None})
    print(f"\nfirst {args.n} rows of the file span hop counts {seen} "
          f"-- this is what the runners currently select")

    rng = random.Random(args.seed)
    picked = []
    if args.scheme == "equal":
        per = args.n // len(tiers)
        for h in sorted(tiers):
            picked += rng.sample(tiers[h], min(per, len(tiers[h])))
    else:
        for h in sorted(tiers):
            k = round(args.n * len(tiers[h]) / total)
            picked += rng.sample(tiers[h], min(k, len(tiers[h])))
        # correct any rounding drift against the largest tier
        while len(picked) != args.n:
            big = max(tiers, key=lambda h: len(tiers[h]))
            pool = [r for r in tiers[big] if r not in picked]
            if len(picked) < args.n and pool:
                picked.append(rng.choice(pool))
            elif len(picked) > args.n:
                picked.pop()
            else:
                break

    rng.shuffle(picked)
    got = collections.Counter(hops[r["question"].strip()] for r in picked)
    print(f"\nsampled {len(picked)} questions ({args.scheme}, seed {args.seed}):")
    for h in sorted(got):
        print(f"  {h}-hop: {got[h]:>4}  ({100*got[h]/len(picked):.1f}%)")

    out = Path(args.out) if args.out else (
        REPO / "data" / "raw" / f"musique_val_strat{len(picked)}.json")
    if out.resolve() == SRC.resolve():
        raise SystemExit("refusing to overwrite the canonical musique_val.json")
    out.write_text(json.dumps(picked, indent=1))
    print(f"\nWrote {out}")
    print("Pass it with --questions-file and a MANDATORY --out-suffix so the run")
    print("cannot resolve to the same filename as the canonical 2-hop artifact.")


if __name__ == "__main__":
    main()
