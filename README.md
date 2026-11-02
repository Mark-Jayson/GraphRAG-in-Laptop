# GraphRAG on a Laptop

Code, configurations and per-question results for

> M. J. Agao. *GraphRAG on a Laptop: Evaluating Fully Local Multi-Hop Retrieval
> on Consumer Hardware.* WCTP 2026.

The paper evaluates a fully local GraphRAG pipeline on a 16 GB Apple M1 MacBook
Pro. Indexing uses no LLM, and nothing leaves the machine at query time. The
pipeline is compared with a dense-retrieval baseline that answers the same 500
questions from each of HotpotQA, 2WikiMultiHopQA and MuSiQue.

```
Corpus  -> spaCy NER (en_core_web_trf) -> Tri-Graph (entity-sentence-passage)
        -> DuckDB graph store + FAISS IVF-PQ entity index (MiniLM, 384-d)
        -> [optional] BIRCH semantic hyperedges
Query   -> query NER + vector seeds -> personalised PageRank (alpha 0.15, top-20)
        -> [optional] evidence compression (3,072 tokens)
        -> Llama-3.1-8B-Instruct-4bit via MLX
```

There are two ways to check the paper:

| | Needs | Time |
|---|---|---|
| **A. Re-derive every number from the committed outputs** | any machine with Python 3.13 | ~5 min |
| **B. Re-run the experiments end to end** | Apple Silicon Mac, ≥16 GB, ~15 GB disk | ~55 h of compute |

## A. Re-derive the results (no GPU)

`results/queries/` holds each run's per-question output: question, gold answer,
retrieved passage ids, the exact context the generator saw, the answer, and
per-stage timings. All statistics, tables and the routing analysis are computed
from those files.

```bash
python3.13 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash scripts/reproduce_analysis.sh
git status            # expect a clean tree
```

The script re-scores every run, recomputes the paired bootstrap intervals
(100,000 resamples, seed 20260814) and exact McNemar tests, and reruns the
conversion, error, question-type, hop and routing analyses. It then regenerates
the paper's tables into `paper/tables/`. If every tracked file is unchanged,
each number below has been reproduced. `hop_analysis.py` reads MuSiQue's hop
labels from Hugging Face, so that step needs network access the first time.

`pip install -r requirements.txt` pulls MLX, which only installs on macOS.
To re-derive the results elsewhere, install everything in `requirements.txt`
except `mlx` and `mlx-lm`, which the analysis does not import.

### Where each result comes from

| Paper | Artifact | Produced by |
|---|---|---|
| Table 1, main results | `paper/tables/tab-main-results.*` | `make_tables.py` ← `experiments.csv`, `paired_*_vector_base.json`, `router_significance.json` |
| Sect. 5.1, graph vs dense and feature intervals | `results/analysis/paired_*.json` | `paired_significance.py` |
| Table 2, ablation and generator ladder | `paper/tables/tab-ablation.*` | `make_tables.py` ← `experiments.csv`, `conversion_rates.json` |
| Table 3, systems (indexing, latency) | `paper/tables/tab-systems.*` | `make_tables.py` ← `results/indexing/`, per-query timers |
| Table 4, conversion rates | `results/analysis/conversion_rates*.json`, `conversion_by_scale.json` | `conversion_analysis.py` |
| Sect. 6.1, error breakdown | `results/analysis/error_decomposition.json` | `error_decomposition.py` |
| Table 5, graph-dense agreement | `results/analysis/paired_*_vector_base.json` | `paired_significance.py` |
| Tables 6–7, question types | `results/analysis/subgroup_by_type.json` | `subgroup_analysis.py` |
| Sect. 6.2, decode-noise floor (1.6%) | `results/analysis/noise_floor_2wiki_base.json` | `paired_significance.py` |
| Tables 8–9, routing ladder and tests | `paper/tables/tab-router*.*`, `router_ladder.json`, `router_significance.json` | `router_analysis.py`, `router_significance.py`, `make_tables.py` |
| Sect. 6.3, refusal-detector robustness | `results/analysis/refusal_sensitivity.json` | `refusal_sensitivity.py` |
| Sect. 7, MuSiQue hop tiers | `musique_by_hop.json`, `conversion_rates_musique.json` | `hop_analysis.py`, `conversion_analysis.py` |
| Sect. 5.3, indexing memory | `results/indexing/memory_profile.json`, `memory_vs_corpus.csv` | `profile_memory.py`, `plot_memory_vs_corpus.py` |

The supplementary figures from `plot_latency_breakdown.py` and
`plot_memory_vs_corpus.py` are written to `paper/figures/`, which is not
tracked. Energy measurements (`results/energy/`) were collected but are not
claimed in the paper; `results/energy/README.md` explains why.

## B. Re-run the experiments

Hardware used: MacBook Pro, Apple M1, 16 GB unified memory, Python 3.13.13.
MLX requires Apple Silicon.

```bash
python3.13 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # exact pins; requirements.lock.txt is the full freeze
pytest -q tests                          # smoke test, no LLM
bash scripts/run_experiments.sh          # prints every command, stage by stage
RUN=1 STAGE=data  bash scripts/run_experiments.sh
RUN=1 STAGE=index bash scripts/run_experiments.sh
RUN=1 STAGE=main  bash scripts/run_experiments.sh   # then vector, ladder, replicate, score
bash scripts/reproduce_analysis.sh
```

`STAGE=data` downloads the benchmarks at pinned revisions and verifies them
against `results/provenance/data-sha256.txt`. Indexing is deterministic given
those files. Generation runs at temperature 0.1, so a re-run reproduces the
results within decode noise, not byte for byte. Running 2Wiki `base` twice
changed 1.6% of answers (Sect. 6.2).

### Configurations

`config.yaml` (HotpotQA), `config_2wikimultihop.yaml` and `config_musique.yaml`
differ only in their index paths. Each run picks one of four tags, which the
runner applies over the config file:

| tag | semantic hyperedges | evidence compression |
|---|---|---|
| `base` | off | off |
| `ehrag` | on | off |
| `compression` | off | on |
| `full` | on | on |
| `vector` | dense passage retrieval, no graph (`scripts/run_vector_baseline.py`) | |

`docs/tag-configurations.md` explains what each feature does in the code.

### Pinned versions

| | Hugging Face id | revision |
|---|---|---|
| HotpotQA (distractor, validation) | `hotpotqa/hotpot_qa` | `1908d6afbbead072334abe2965f91bd2709910ab` |
| 2WikiMultiHopQA (dev) | `voidful/2WikiMultihopQA` | `16852fde9d85cba158cf7e6517e7a3f9415a28c0` |
| MuSiQue-Ans (validation) | `dgslibisey/MuSiQue` | `c8f4f8c9465fb69d31a8eae894c3fd509c4ca321` |
| Generator | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | `241a666dad6cb93c8ff213d39a7f34a36bf26db4` |
| Ablation generators | `mlx-community/Llama-3.2-3B-Instruct-4bit` | `7f0dc925e0d0afb0322d96f9255cfddf2ba5636e` |
| | `mlx-community/Llama-3.2-1B-Instruct-4bit` | `08231374eeacb049a0eade7922910865b8fce912` |
| Encoder | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| NER | `en_core_web_trf` 3.8.0 | wheel hash in `requirements.txt` |

The dataset revisions are enforced in `scripts/download_data.py`. The model
revisions above are the ones that were cached during the runs. To pin them too,
download each one first with
`huggingface-cli download <id> --revision <revision>`.

## Things a careful reader should know

- **MuSiQue covers the 2-hop tier only.** The runners take the first 500
  questions, and that split is sorted by hop count. A hop-stratified re-run
  (`musique_base_strat_*`) quantifies the effect (Sect. 7).
- **One latency run slept.** HotpotQA `base` has valid per-query timers, but
  its wall-clock and mean latency include sleep, so the paper reports medians
  throughout. See `results/analysis/latency-notes.md`.
- **The `git_hash` values in the result metadata do not resolve here.**
  `results/provenance/README.md` explains why and what is verifiable instead.
- **No LLM judge.** Scoring is deterministic SQuAD-style EM/F1. A local RAGAS
  judge was tried and dropped because it ran out of Metal memory (Sect. 4).

## Repository layout

```
src/indexer/      NER, sentence splitting, Tri-Graph, BIRCH hyperedges
src/storage/      DuckDB graph store
src/retrieval/    FAISS IVF-PQ, personalised PageRank, evidence compression, dense baseline
src/generation/   MLX generator wrapper
src/pipeline/     indexing and query pipelines
src/evaluation/   EM/F1 (SQuAD normalisation)
scripts/          download, indexing, runs, scoring, analyses, tables
results/queries/      per-question outputs + run metadata (the primary artifacts)
results/evaluation/   EM/F1 per run;  results/experiments.csv aggregates them
results/analysis/     statistical analyses (JSON)
results/indexing/     index build statistics and memory measurements
results/provenance/   input checksums and provenance notes
paper/tables/         the paper's generated tables (md / csv / tex)
```

## Licence and data

The code is released under the MIT licence (`LICENSE`). That licence covers this
repository's code only. The files in `results/queries/` contain question,
answer and passage text from the three benchmarks, which remain under their own
licences: HotpotQA (CC BY-SA 4.0), 2WikiMultiHopQA (Apache 2.0) and MuSiQue
(CC BY 4.0). Model weights are governed by their providers' licences (Llama 3.1
and 3.2 Community Licences; MiniLM, Apache 2.0).

## Citation

See `CITATION.cff`, or GitHub's "Cite this repository" button.
