# Provenance

What this directory records, and the one known gap in the provenance chain.

## Inputs

- `data-sha256.txt`: SHA-256 of every file in `data/raw/` that a run read.
  `scripts/download_data.py` pins the Hugging Face dataset revisions and
  reproduces these bytes exactly; check with
  `shasum -a 256 -c results/provenance/data-sha256.txt` from the repository root.
- `phase4.2-input-hashes.txt`: the same hashes for the four 2Wiki/MuSiQue files,
  recorded when those runs were launched (4 August 2026). They match
  `data-sha256.txt`.
- `config_2wikimultihop.yaml`, `config_musique.yaml`: the configuration files
  exactly as launched. The copies at the repository root are identical apart
  from an unused `evaluation.ragas_metrics` block that has since been removed.

### MuSiQue gold answers include aliases

MuSiQue's official evaluation accepts `answer` plus `answer_aliases`. The
released `download_data.py` writes both into a `ground_truth` list, and the
MuSiQue runs were scored against that file (`musique_val.json`, SHA-256
`9fed9858…`). An earlier version of the script kept only `answer`.
`phase4.2-scripts.patch` is the diff that introduced the aliases (139 of the
first 500 MuSiQue questions carry at least one). It also touched
`check_answerability.py`, which is diagnostic only.

## Index state

The indexes are not in git (about 2.2 GB of DuckDB plus FAISS files). They are
rebuilt deterministically from `data/raw/` by `scripts/run_indexing.py`;
build statistics are in `results/indexing/indexing_stats_*.json`.

| benchmark | passages | entities | hyperedges | index wall-clock |
| --- | ---: | ---: | ---: | ---: |
| HotpotQA | 66,581 | 352,255 | 60,747 | 5,860 s |
| 2Wiki | 54,957 | 276,713 | 46,511 | 4,404 s |
| MuSiQue | 21,100 | 116,985 | 20,171 | 1,783 s |

All three were built with `en_core_web_trf`. The 2Wiki and MuSiQue builds ran
under `caffeinate -i`, so their wall-clocks contain no sleep.

## Known gap: `git_hash` values do not resolve

Each run's `.meta.json` stamps the commit of the authors' development
repository it ran from. The eight graph runs carry `4cac37d`. That repository
was re-initialised on 11 August 2026, so the commit no longer exists anywhere.
Later runs carry `ed88706`, `a5d23b3`, `d1c94cb` and `3cc74d4`, which do exist
in the development repository. This public repository was published with a
fresh history, so none of these hashes resolve here.

They are kept as historical labels rather than re-stamped, because re-stamping
would claim that this release contains the exact code that ran, and that cannot
be verified. What can be verified is the following:

- every per-question output (`results/queries/*.jsonl`) is committed together
  with its configuration, run date, corpus size and NER model;
- every number in the paper regenerates from those outputs with
  `scripts/reproduce_analysis.sh`;
- since the last run (14 August 2026), the pipeline code under `src/` and the
  two runners have changed only to remove the unused RAGAS judge and to record
  `run_date`. Analysis scripts, documentation and output paths were tidied for
  release.
