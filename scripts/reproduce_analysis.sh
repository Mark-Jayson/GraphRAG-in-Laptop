#!/usr/bin/env bash
# Regenerate every number, table and figure in the paper from the committed
# per-question outputs in results/queries/. No GPU, no index, no model download.
# hop_analysis.py is the one step that reads a dataset: the cached MuSiQue from
# download_data.py, or Hugging Face if it is reachable.
#
#   bash scripts/reproduce_analysis.sh            (~5 min on an M1)
#
# Then `git status` should report nothing: every regenerated file matches the
# release. Figures go to paper/figures/, which is not tracked.
set -eu
cd "$(dirname "$0")/.."
PY="${PY:-python}"
Q=results/queries
A=results/analysis
M=llama-3.1-8b

echo "== EM/F1 per run"
for f in results/evaluation/*.json; do
  $PY scripts/score_answers.py --input "$Q/$(basename "$f" .json).jsonl" >/dev/null
done
$PY scripts/collect_results.py >/dev/null

echo "== paired tests: features (base vs full) and retrievers (vector vs base)"
for b in hotpotqa 2wikimultihop musique; do
  $PY scripts/paired_significance.py --a $Q/${b}_base_$M.jsonl --b $Q/${b}_full_$M.jsonl \
    --label-a base --label-b full --json $A/paired_${b}_base_full.json >/dev/null
  $PY scripts/paired_significance.py --a $Q/${b}_vector_$M.jsonl --b $Q/${b}_base_$M.jsonl \
    --label-a vector --label-b base --json $A/paired_${b}_vector_base.json >/dev/null
done

echo "== decode-noise floor (2Wiki base, two runs)"
$PY scripts/paired_significance.py --a $Q/2wikimultihop_base_$M.jsonl \
  --b $Q/2wikimultihop_base_rep2_$M.jsonl --label-a run1 --label-b run2 \
  --json $A/noise_floor_2wiki_base.json >/dev/null

echo "== conversion, error decomposition, question types, hops"
$PY scripts/conversion_analysis.py >/dev/null
$PY scripts/conversion_analysis.py --inputs "$Q/musique_*_llama-*.jsonl" \
  --json $A/conversion_rates_musique.json >/dev/null
$PY scripts/conversion_analysis.py --inputs "$Q/hotpotqa_base_llama-*.jsonl" \
  --json $A/conversion_by_scale.json >/dev/null
$PY scripts/error_decomposition.py >/dev/null
$PY scripts/subgroup_analysis.py >/dev/null
$PY scripts/hop_analysis.py >/dev/null

echo "== routing ladder, its significance, refusal-detector sensitivity"
$PY scripts/router_analysis.py >/dev/null
$PY scripts/router_significance.py >/dev/null
$PY scripts/refusal_sensitivity.py >/dev/null

echo "== energy (parsed from the committed powermetrics logs)"
$PY scripts/parse_energy.py >/dev/null

echo "== tables and figures"
$PY scripts/make_tables.py >/dev/null
$PY scripts/plot_latency_breakdown.py >/dev/null
$PY scripts/plot_memory_vs_corpus.py >/dev/null

echo
if git rev-parse --git-dir >/dev/null 2>&1; then
  git status --short -- results paper
fi
echo "done. Any file listed above differs from the release."
