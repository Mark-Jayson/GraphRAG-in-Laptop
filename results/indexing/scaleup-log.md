> Development log of the indexing scale-up, kept verbatim because
> `scripts/plot_memory_vs_corpus.py` parses it. It records decisions as they
> were made; a few passages refer to internal notes that are not part of this
> repository.

# Finding the Corpus Ceiling (Memory Bound)
## N = 5000
Peak footprint: 7.61 GB (RSS alone: 3.40 GB) across 188 samples
{
  "trigraph_time": 59.02542805671692,
  "trigraph_stats": {
    "num_passages": 5000,
    "num_sentences": 20967,
    "num_entities": 35762,
    "num_es_edges": 72008,
    "num_sp_edges": 20967
  },
  "storage_time": 71.40475606918335,
  "embedding_time": 34.683207988739014,
  "index_time": 3.4433751106262207,
  "hyperedge_time": 4.050206184387207,
  "num_hyperedges": 6541,
  "total_time": 172.61664986610413
}

## N = 10000
Peak footprint: 7.68 GB (RSS alone: 2.28 GB) across 188 samples
{
  "trigraph_time": 114.47788405418396,
  "trigraph_stats": {
    "num_passages": 10000,
    "num_sentences": 41399,
    "num_entities": 65783,
    "num_es_edges": 141984,
    "num_sp_edges": 41399
  },
  "storage_time": 8.138792991638184,
  "embedding_time": 66.29140782356262,
  "index_time": 6.26908016204834,
  "hyperedge_time": 11.439210176467896,
  "num_hyperedges": 11706,
  "total_time": 206.63240098953247
}

## N = 15000
Peak footprint: 7.74 GB (RSS alone: 2.33 GB) across 198 samples
{
  "trigraph_time": 181.09617376327515,
  "trigraph_stats": {
    "num_passages": 15000,
    "num_sentences": 62344,
    "num_entities": 94685,
    "num_es_edges": 214274,
    "num_sp_edges": 62344
  },
  "storage_time": 14.251044750213623,
  "embedding_time": 98.47476100921631,
  "index_time": 6.5855491161346436,
  "hyperedge_time": 22.729968309402466,
  "num_hyperedges": 16659,
  "total_time": 323.1608929634094
}

## N = 20000
Peak footprint: 7.77 GB (RSS alone: 1.77 GB) across 194 sample
{
  "trigraph_time": 229.0640950202942,
  "trigraph_stats": {
    "num_passages": 20000,
    "num_sentences": 82863,
    "num_entities": 121242,
    "num_es_edges": 284027,
    "num_sp_edges": 82863
  },
  "storage_time": 25.112020015716553,
  "embedding_time": 130.52951288223267,
  "index_time": 7.048635959625244,
  "hyperedge_time": 38.04029417037964,
  "num_hyperedges": 21061,
  "total_time": 429.8239779472351
}

## N = 25000
Peak footprint: 7.84 GB (RSS alone: 1.87 GB) across 196 samples 
{
  "trigraph_time": 282.0170388221741,
  "trigraph_stats": {
    "num_passages": 25000,
    "num_sentences": 103021,
    "num_entities": 146497,
    "num_es_edges": 353972,
    "num_sp_edges": 103021
  },
  "storage_time": 28.594155073165894,
  "embedding_time": 157.20628905296326,
  "index_time": 7.119331121444702,
  "hyperedge_time": 55.27910614013672,
  "num_hyperedges": 25378,
  "total_time": 530.2577168941498
}
## N = 30000
Peak footprint: 7.91 GB (RSS alone: 2.78 GB) across 192 samples
{
  "trigraph_time": 346.1150999069214,
  "trigraph_stats": {
    "num_passages": 30000,
    "num_sentences": 123473,
    "num_entities": 171021,
    "num_es_edges": 424346,
    "num_sp_edges": 123473
  },
  "storage_time": 36.56094193458557,
  "embedding_time": 189.46760511398315,
  "index_time": 8.137299060821533,
  "hyperedge_time": 69.82848000526428,
  "num_hyperedges": 29425,
  "total_time": 650.1535637378693
}

## N = 35000
Peak footprint: 8.00 GB (RSS alone: 2.84 GB) across 196 samples 
{
  "trigraph_time": 400.85603404045105,
  "trigraph_stats": {
    "num_passages": 35000,
    "num_sentences": 143819,
    "num_entities": 194659,
    "num_es_edges": 493745,
    "num_sp_edges": 143819
  },
  "storage_time": 50.2752149105072,
  "embedding_time": 210.6258580684662,
  "index_time": 8.172528743743896,
  "hyperedge_time": 91.41892504692078,
  "num_hyperedges": 33405,
  "total_time": 761.3993339538574
}


## N = 40000
Peak footprint: 8.07 GB (RSS alone: 2.76 GB) across 204 samples 
{
  "trigraph_time": 445.8068380355835,
  "trigraph_stats": {
    "num_passages": 40000,
    "num_sentences": 164319,
    "num_entities": 217594,
    "num_es_edges": 563700,
    "num_sp_edges": 164319
  },
  "storage_time": 53.87275695800781,
  "embedding_time": 237.95460104942322,
  "index_time": 8.008273839950562,
  "hyperedge_time": 109.86572098731995,
  "num_hyperedges": 37202,
  "total_time": 855.5678200721741
}

## N = 45000
Peak footprint: 8.15 GB (RSS alone: 3.61 GB) across 225 samples 
{
  "trigraph_time": 506.1664078235626,
  "trigraph_stats": {
    "num_passages": 45000,
    "num_sentences": 184508,
    "num_entities": 240137,
    "num_es_edges": 633054,
    "num_sp_edges": 184508
  },
  "storage_time": 59.87282609939575,
  "embedding_time": 297.801078081131,
  "index_time": 8.269213199615479,
  "hyperedge_time": 136.74714398384094,
  "num_hyperedges": 40946,
  "total_time": 1008.9143190383911
}

## N = 50000
Peak footprint: 8.17 GB (RSS alone: 5.49 GB) across 279 samples
{
  "trigraph_time": 556.1695778369904,
  "trigraph_stats": {
    "num_passages": 50000,
    "num_sentences": 204427,
    "num_entities": 262184,
    "num_es_edges": 702204,
    "num_sp_edges": 204427
  },
  "storage_time": 67.61046600341797,
  "embedding_time": 291.78118109703064,
  "index_time": 8.520296096801758,
  "hyperedge_time": 157.92364501953125,
  "num_hyperedges": 44509,
  "total_time": 1082.0701172351837
}

## N = 55000
Peak footprint: 8.27 GB (RSS alone: 3.76 GB) across 200 samples

{
  "trigraph_time": 606.7467720508575,
  "trigraph_stats": {
    "num_passages": 55000,
    "num_sentences": 224449,
    "num_entities": 284172,
    "num_es_edges": 771088,
    "num_sp_edges": 224449
  },
  "storage_time": 62.79903602600098,
  "embedding_time": 320.26490092277527,
  "index_time": 8.809805154800415,
  "hyperedge_time": 185.2109558582306,
  "num_hyperedges": 48192,
  "total_time": 1183.9132978916168
}

## N = 60000
Peak footprint: 8.28 GB (RSS alone: 2.57 GB) across 204 samples
{
  "trigraph_time": 662.5666308403015,
  "trigraph_stats": {
    "num_passages": 60000,
    "num_sentences": 244562,
    "num_entities": 304936,
    "num_es_edges": 839425,
    "num_sp_edges": 244562
  },
  "storage_time": 103.28907704353333,
  "embedding_time": 348.4605357646942,
  "index_time": 9.033888101577759,
  "hyperedge_time": 211.78799891471863,
  "num_hyperedges": 51559,
  "total_time": 1335.2150609493256
}

## N = 70000
Peak footprint: 8.43 GB (RSS alone: 3.00 GB) across 206 samples

{
  "trigraph_time": 730.4100379943848,
  "trigraph_stats": {
    "num_passages": 66581,
    "num_sentences": 271027,
    "num_entities": 332161,
    "num_es_edges": 929323,
    "num_sp_edges": 271027
  },
  "storage_time": 89.30635976791382,
  "embedding_time": 381.9463438987732,
  "index_time": 9.276585102081299,
  "hyperedge_time": 248.18156480789185,
  "num_hyperedges": 56075,
  "total_time": 1459.205621957779
}


# Final Quality Index (trf)

I run the en_core_web_trf on the full corpus of Hotpotqa and here is the result

{
  "trigraph_time": 5081.79838180542,
  "trigraph_stats": {
    "num_passages": 66581,
    "num_sentences": 271027,
    "num_entities": 352255,
    "num_es_edges": 962015,
    "num_sp_edges": 271027
  },
  "storage_time": 95.82063698768616,
  "embedding_time": 389.8230838775635,
  "index_time": 9.468430042266846,
  "hyperedge_time": 282.905433177948,
  "num_hyperedges": 60747,
  "total_time": 5859.902626037598
}
Peak footprint: 13.82 GB (RSS alone: 6.14 GB) across 11422 samples

Wall - Clock:
{ started: 2026-07-17 15:58:32, 
  ended: 2026-07-17 17:36:21
  time lapse: 
}

# 4 tag eval
## Base
{
  "benchmark": "hotpotqa",
  "tag": "base",
  "model": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
  "use_hyperedges": false,
  "compression_enabled": false,
  "corpus_size": 66581,
  "ner_model": "en_core_web_trf",
  "git_hash": "0fb42da",
  "num_queries": 500,
  "wall_time_s": 12722.8,
  "em_f1": {
    "exact_match": 35.4,
    "f1": 43.108813963813965
  },
  "ragas_scores": null,
  "ragas_note": "not requested (--ragas off)",
  "source_jsonl": "results/queries/hotpotqa_base_llama-3.1-8b.jsonl"
}

## EHRAG
{
  "benchmark": "hotpotqa",
  "tag": "ehrag",
  "model": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
  "use_hyperedges": true,
  "compression_enabled": false,
  "corpus_size": 66581,
  "ner_model": "en_core_web_trf",
  "git_hash": "0fb42da",
  "num_queries": 500,
  "wall_time_s": 23158.6,
  "em_f1": {
    "exact_match": 28.2,
    "f1": 35.707067932067936
  },
  "ragas_scores": null,
  "ragas_note": "not requested (--ragas off)",
  "source_jsonl": "results/queries/hotpotqa_ehrag_llama-3.1-8b.jsonl"
}

## Compression
{
  "benchmark": "hotpotqa",
  "tag": "compression",
  "model": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
  "use_hyperedges": false,
  "compression_enabled": true,
  "corpus_size": 66581,
  "ner_model": "en_core_web_trf",
  "git_hash": "0fb42da",
  "num_queries": 500,
  "wall_time_s": 8676.2,
  "em_f1": {
    "exact_match": 30.0,
    "f1": 37.6221472971473
  },
  "ragas_scores": null,
  "ragas_note": "not requested (--ragas off)",
  "source_jsonl": "results/queries/hotpotqa_compression_llama-3.1-8b.jsonl"
}


# 4-tag matrix — AUTHORITATIVE (post-fix, regenerated 2026-08-12)

All four tags, HotpotQA full corpus (66,581 passages), 500 questions,
`Meta-Llama-3.1-8B-Instruct-4bit`, `en_core_web_trf`, git `4cac37d`, with the
2026-07-19 retrieval fixes active. Values copied verbatim from
`results/evaluation/hotpotqa_{tag}_llama-3.1-8b.json` — do not hand-transcribe.

| Tag | EM | F1 | wall time (500 q) |
|---|---|---|---|
| `base` | 36.0 | 44.63 | 34,746 s *(see latency note)* |
| `ehrag` | 35.2 | 44.54 | 13,305 s |
| `compression` | 35.8 | 44.27 | 12,138 s |
| `full` | 35.8 | 45.23 | 12,428 s |

`base` wall time is **sleep-contaminated and must not be quoted** — 69.5 s/q mean
against a 39.0 s p95. See `results/analysis/latency-notes.md`.

## Superseded: the pre-fix (legacy retrieval) matrix

Kept for the narrative only. These runs predate the 2026-07-19 seed/expansion/
budget fixes (git `0fb42da`).

| Tag | EM | F1 | wall time (500 q) |
|---|---|---|---|
| `base` | 35.4 | 43.11 | 12,723 s |
| `ehrag` | 28.2 | 35.71 | 23,159 s |
| `compression` | 30.0 | 37.62 | 8,676 s |
| `full` | 26.4 | 32.4 | 8,830 s |

### Reconciliation of pre-fix `full` (resolved 2026-08-12)

Three documents disagreed. Adjudication:

| Source | pre-fix `full` | verdict |
|---|---|---|
| `docs/tag-configurations.md` §8 | EM 26.4 / F1 32.4 (+ wall 8,830 s) | **adopted** |
| `docs/phase4-runbook.md` start-state | EM 26.4 / F1 32.4 | **adopted** (agrees) |
| `results/Phase 3.md` closing table (this file, before today) | EM 25.2 / F1 31.0 | **rejected** |

Reasons for adopting 26.4 / 32.4:

1. Two independent documents carry it, and one of them also carries a wall-time
   (8,830 s) that fits the compression-on tags (`compression` 8,676 s) — evidence it
   was transcribed from a real scored artifact.
2. 25.2 / 31.0 appeared exactly once, inside a table whose other three rows are
   **post-fix** values, i.e. a table assembled mid-transition. Its `full` cell is
   most consistent with a stale or interrupted figure carried over while the other
   three tags had already been re-run.

**Neither figure is recomputable.** The pre-fix `hotpotqa_full` 500-q jsonl was
destroyed by a filename overwrite on 2026-07-19 (`docs/tag-configurations.md` §9),
and `results/evaluation/hotpotqa_full_llama-3.1-8b.json` now holds the 2026-07-25
post-fix re-run. There is no artifact left to adjudicate against.

**Consequence for the paper:** do not print a precise pre-fix `full` EM/F1. State
the direction instead — "before the retrieval fixes, `full` trailed `base` by
roughly 9 EM points; the fixes closed that gap to a null" — which is supported by
`base` 35.4 -> 36.0 and the retrieval-only recall table in
`docs/tag-configurations.md` §9, both of which rest on surviving artifacts.
