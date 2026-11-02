#!/bin/bash
# Per-query energy via powermetrics, for the graph and dense arms.
#
#   idle baseline (60 s)  ->  hotpotqa base 100q  ->  hotpotqa vector 100q
#
# TWO arms rather than one, deliberately: a single number needs no figure and
# says little, whereas base-vs-vector measures the energy cost of the graph
# apparatus against a dense arm that matches it on accuracy. That is the same
# argument 6.5 already makes in latency, in a second currency.
#
# Runs at --max-samples 100, so the runner's own suffix rule writes to
# *_n100_*.jsonl and cannot touch a 500-question artifact.
#
# NOTE: powermetrics is macOS-only and needs sudo. Check the first parse
# against the raw log before trusting the number.
#
# Usage:  bash scripts/measure_energy.sh          (~75 min for both arms)
#         ARMS=base bash scripts/measure_energy.sh   (~45 min, one arm)
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO" || exit 1
PY="$REPO/venv/bin/python"; [ -x "$PY" ] || PY="$(command -v python)"
EDIR="$REPO/results/energy"; mkdir -p "$EDIR"
MODEL="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
ARMS="${ARMS:-base vector}"
N="${N:-100}"
log() { echo "$(date '+%H:%M:%S')  $*"; }

command -v powermetrics >/dev/null || { echo "powermetrics not found (macOS only)"; exit 1; }
ls -d "$REPO"/results/.run_*.lock >/dev/null 2>&1 && { echo "a batch is running; refusing"; exit 1; }

log "authenticating sudo once up front so nothing blocks mid-run"
sudo -v || exit 1

# ---- idle baseline: 60 samples at 1 s -------------------------------------
if [ ! -s "$EDIR/idle.txt" ]; then
  log "idle baseline, 60 s -- do not touch the machine"
  sudo powermetrics --samplers cpu_power,gpu_power -i 1000 -n 60 > "$EDIR/idle.txt" 2>/dev/null
  log "  idle done"
else
  log "idle baseline already present, skipping"
fi

# ---- one measured run per arm ---------------------------------------------
for arm in $ARMS; do
  out="$REPO/results/queries/hotpotqa_${arm}_n${N}_llama-3.1-8b.jsonl"
  pmlog="$EDIR/${arm}_n${N}.txt"
  if [ -s "$pmlog" ]; then log "$arm already measured, skipping"; continue; fi

  log "=== $arm : $N queries with powermetrics sampling ==="
  sudo powermetrics --samplers cpu_power,gpu_power -i 1000 -n 5400 > "$pmlog" 2>/dev/null &
  PM=$!
  sleep 2
  if [ "$arm" = vector ]; then
    "$PY" scripts/run_vector_baseline.py --config config.yaml --benchmark hotpotqa \
        --model "$MODEL" --max-samples "$N"
  else
    "$PY" scripts/run_queries_batch.py --config config.yaml --benchmark hotpotqa \
        --tag "$arm" --model "$MODEL" --max-samples "$N"
  fi
  rc=$?
  sudo kill "$PM" 2>/dev/null; wait "$PM" 2>/dev/null
  log "  $arm finished (exit $rc), $(wc -l < "$out" 2>/dev/null || echo 0) records"
done

log "parsing"
"$PY" scripts/parse_energy.py
