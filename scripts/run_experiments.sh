#!/usr/bin/env bash
# Every inference run behind the paper, in the order it is easiest to replay.
#
# Prints the commands by default. Set RUN=1 to execute them, and STAGE to run
# one stage only:
#
#   bash scripts/run_experiments.sh                  # print the full plan
#   RUN=1 STAGE=data   bash scripts/run_experiments.sh
#   RUN=1 STAGE=index  bash scripts/run_experiments.sh
#   RUN=1 STAGE=main   bash scripts/run_experiments.sh
#
# Stages and approximate wall-clock on a 16 GB Apple M1 (from results/):
#   data      download + checksum the three benchmarks             minutes
#   index     LLM-free indexing, en_core_web_trf                  ~3.4 h
#   main      graph arms (base/full x 3, ehrag + compression)     ~29 h
#   vector    dense baseline x 3                                  ~7 h
#   ladder    1B / 3B generators on HotpotQA base                 ~2.5 h
#   replicate 2Wiki base re-run (noise floor) + stratified MuSiQue ~9 h
#   energy    powermetrics, 100 q x {base, vector}; needs sudo    ~1.3 h
#   score     EM/F1 for every run; seconds
#
# Each run writes results/queries/<benchmark>_<tag>[suffix]_<model>.jsonl plus a
# .meta.json sidecar. Runs never overwrite one another: replicates use
# --out-suffix and non-500 samples get an _nN suffix.
#
# Keep the machine awake (caffeinate -i is used below): a run that sleeps keeps
# valid per-query timers but its wall-clock and mean latency become unusable
# (results/analysis/latency-notes.md).
set -u
cd "$(dirname "$0")/.." || exit 1

PY="${PY:-python}"
M8="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
M3="mlx-community/Llama-3.2-3B-Instruct-4bit"
M1="mlx-community/Llama-3.2-1B-Instruct-4bit"
STAGE="${STAGE:-all}"
AWAKE=""; command -v caffeinate >/dev/null && AWAKE="caffeinate -i"

cfg() { case "$1" in hotpotqa) echo config.yaml ;; *) echo "config_$1.yaml" ;; esac; }

step() {
  if [ "${RUN:-0}" = 1 ]; then
    echo "+ $*"; "$@" || { echo "FAILED: $*" >&2; exit 1; }
  else
    echo "$*"
  fi
}

want() { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }

if want data; then
  echo "# --- data"
  step $PY scripts/download_data.py
  step $PY scripts/make_stratified_sample.py --n 500 --scheme proportional
  step shasum -a 256 -c results/provenance/data-sha256.txt
fi

if want index; then
  echo "# --- index (one DuckDB graph + FAISS index per benchmark)"
  for b in hotpotqa 2wikimultihop musique; do
    step $AWAKE $PY scripts/run_indexing.py --config "$(cfg $b)" --passages "data/raw/${b}_corpus.json"
  done
fi

if want main; then
  echo "# --- graph arms, 500 questions, Llama-3.1-8B"
  for t in base ehrag compression full; do
    step $AWAKE $PY scripts/run_queries_batch.py --config config.yaml --benchmark hotpotqa \
      --tag $t --model "$M8" --max-samples 500
  done
  for b in 2wikimultihop musique; do
    for t in base full; do
      step $AWAKE $PY scripts/run_queries_batch.py --config "$(cfg $b)" --benchmark $b \
        --tag $t --model "$M8" --max-samples 500
    done
  done
fi

if want vector; then
  echo "# --- dense baseline (same encoder, top-20, generator and prompt)"
  for b in hotpotqa 2wikimultihop musique; do
    step $AWAKE $PY scripts/run_vector_baseline.py --config "$(cfg $b)" --benchmark $b \
      --model "$M8" --max-samples 500
  done
fi

if want ladder; then
  echo "# --- generator ablation: retrieval held at base"
  for m in "$M1" "$M3"; do
    step $AWAKE $PY scripts/run_queries_batch.py --config config.yaml --benchmark hotpotqa \
      --tag base --model "$m" --max-samples 500
  done
fi

if want replicate; then
  echo "# --- decode-noise floor and hop-stratified MuSiQue (Sect. 6-7)"
  step $AWAKE $PY scripts/run_queries_batch.py --config config_2wikimultihop.yaml \
    --benchmark 2wikimultihop --tag base --model "$M8" --max-samples 500 --out-suffix _rep2
  step $AWAKE $PY scripts/run_queries_batch.py --config config_musique.yaml \
    --benchmark musique --tag base --model "$M8" --max-samples 500 \
    --questions-file data/raw/musique_val_strat500.json --out-suffix _strat
fi

if want energy; then
  echo "# --- energy (macOS powermetrics, sudo); writes results/energy/"
  step bash scripts/measure_energy.sh
  step $PY scripts/parse_energy.py
fi

if want score; then
  echo "# --- EM/F1 for every scored run"
  for f in results/evaluation/*.json; do
    step $PY scripts/score_answers.py --input "results/queries/$(basename "$f" .json).jsonl"
  done
fi
