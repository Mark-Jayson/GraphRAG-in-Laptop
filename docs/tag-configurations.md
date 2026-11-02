# How Each Tag Works: `base`, `ehrag`, `compression`, `full`

This document explains, technically and step by step, what each ablation tag actually
does inside the pipeline — what code runs, what problem each feature is supposed to
solve, and what it costs.

---

## 1. What a "tag" is

A tag is **not** a different system. It is the same pipeline with two independent
switches flipped on or off. The single source of truth is `TAG_FLAGS` in
[`src/config.py`](../src/config.py):

```python
TAG_FLAGS = {
    "base":        {"use_hyperedges": False, "compression": False},
    "ehrag":       {"use_hyperedges": True,  "compression": False},
    "compression": {"use_hyperedges": False, "compression": True},
    "full":        {"use_hyperedges": True,  "compression": True},
}
```

So the four tags are a classic **2×2 ablation matrix** over two features:

| | compression **off** | compression **on** |
|---|---|---|
| hyperedges **off** | `base` | `compression` |
| hyperedges **on** | `ehrag` | `full` |

`apply_tag()` overwrites `config.ehrag.use_hyperedges` and
`config.compression.enabled` from the tag — it is *authoritative* and overrides
whatever `config.yaml` says. This guarantees the tag recorded in a result file and
the configuration that actually ran can never drift apart (during development
they once did).

The point of the matrix: by comparing pairs of runs you can isolate the causal
effect of each feature.

- `ehrag` vs `base` → what do semantic hyperedges do?
- `compression` vs `base` → what does evidence compression do?
- `full` vs `base` → do the two features combine well, or interfere?

---

## 2. The shared backbone (what *every* tag runs)

Before the tags make sense, you need the machinery they all share. This project is a
**LinearRAG-style GraphRAG**: instead of asking an LLM to extract a knowledge graph
(expensive, slow, needs a big model), it builds a *structural* graph with zero LLM
calls, then retrieves by spreading activation over that graph.

### 2.1 Indexing (offline, once per corpus)

Implemented in [`src/pipeline/indexing_pipeline.py`](../src/pipeline/indexing_pipeline.py).

```
raw passages
   │  1. sentence splitting
   ▼
sentences
   │  2. spaCy NER (en_core_web_trf) on every sentence
   ▼
Tri-Graph:  Entity ↔ Sentence ↔ Passage        (trigraph_builder.py)
   │  3. embed every entity + passage (MiniLM, 384-dim)
   ▼
embeddings
   │  4. FAISS IVF-PQ index over ENTITY embeddings   (vector_index.py)
   ▼
searchable index + DuckDB store
   │  5. (only if hyperedges on) BIRCH clustering → semantic hyperedges
   ▼
done
```

The **Tri-Graph** ([`src/indexer/trigraph_builder.py`](../src/indexer/trigraph_builder.py))
has three node types and only *containment* edges:

- **Entity ↔ Sentence** — the entity's text appears in that sentence (found by NER).
- **Sentence ↔ Passage** — the sentence belongs to that passage.

Entities are deduplicated across the whole corpus by hashing their lowercased text
(`hash_entity_id`), so "Barack Obama" in passage 12 and "barack obama" in passage
40,000 become the *same node*. **This is the crucial trick: shared entities are the
bridges that connect passages to each other.** Collapsing the sentence layer gives
the entity↔passage bipartite graph that retrieval actually runs on
(`get_entity_passage_bipartite`).

On the full HotpotQA corpus (66,581 passages) this yields ~271K sentences, ~352K
entities, and ~962K entity–sentence edges — built purely with spaCy, no LLM.

### 2.2 Querying (online, per question)

Implemented in [`src/pipeline/query_pipeline.py`](../src/pipeline/query_pipeline.py).
Five steps, each timed separately:

**Step 1 — Query entity extraction.** Run the same spaCy NER on the *question*.
Each extracted entity is hashed with the same `hash_entity_id`, so if the question
mentions "Scott Derrickson", it maps to the exact same node the indexer created.

**Step 2 — Vector search for seed entities.** Embed the whole question with MiniLM
and search the FAISS entity index for the 10 nearest entities. This is the safety
net for when NER misses something or the question paraphrases an entity name.
The two sources are weighted differently when building the PPR restart
distribution:

```python
# exact NER matches: full activation
activation[ner_entity] = 1.0
# vector neighbours: cosine similarity × 0.5 (supplement, don't compete)
activation[seed_entity] = max(0, sim) * 0.5
```

**Step 3 — Personalized PageRank (PPR).** The heart of the system
([`src/retrieval/ppr_retrieval.py`](../src/retrieval/ppr_retrieval.py)). Think of it
as pouring "importance" onto the query entities and letting it diffuse through the
entity↔passage graph:

```
e_scores  ── A ──▶  passage scores        (entity splits its score over its passages)
p_scores  ── B ──▶  entity scores         (passage splits its score over its entities)
                    + α · teleport back to the query entities
repeat until convergence (≤50 iterations, L1 diff < 1e-6)
```

`alpha = 0.15` is the restart probability: at every step 15% of the mass jumps back
to the query entities, which keeps scores *personalized* to this question instead of
converging to global corpus popularity.

Why this beats plain vector search for multi-hop questions: a passage that shares
**no words** with the question can still score high if it is *graph-connected* to
the query entities through intermediate entities. For "What college did the director
of *Doctor Strange* attend?", mass flows: `Doctor Strange` → passage about the film
→ `Scott Derrickson` (co-occurring entity) → passage about Derrickson's education.
That second passage is unreachable by one-shot similarity but two hops away in the
graph. The top 20 passages by final PPR score are returned.

**Step 4 — Context assembly.** Either raw concatenation of the 20 passages
(compression off) or evidence-aware compression (compression on — see §4).

**Step 5 — Generation.** A local quantized Llama (Meta-Llama-3.1-8B-Instruct-4bit
via MLX) answers with a strict grounding prompt: answer *only* from the context, as
a short phrase.

Now the tags.

---

## 3. `base` — the structural GraphRAG floor

**Flags:** `use_hyperedges=False, compression=False`

`base` is the backbone above, unmodified. It is already a full GraphRAG — the graph
does the multi-hop reasoning via PPR — but it has two deliberate limitations, one
per disabled feature:

1. **Connections are purely lexical/structural.** Two entities are only connected
   if they literally co-occur in text, and two mentions only merge if their
   lowercased strings hash identically. "NYC" and "New York City" are *different
   nodes* with no edge between them unless some sentence happens to contain both.
   If a question uses a paraphrase or alias the corpus never uses, activation
   starts in the wrong place (Step 2's vector seeds mitigate this only at the
   query's starting points, not inside the graph).

2. **The LLM sees everything retrieved.** All 20 passages, full text, concatenated
   in score order. On HotpotQA that is routinely thousands of tokens, most of them
   irrelevant filler around one or two evidence sentences. For a local 8B model
   this costs prompt-processing time (memory-bandwidth-bound on an M1) and invites
   "lost in the middle" failures where the model overlooks evidence buried mid-context.

`base` is the control: every other tag must justify itself against these numbers.

**Measured (HotpotQA, 500 q, Llama-3.1-8B):** EM 35.4 / F1 43.1, ~25 s per query.

---

## 4. `ehrag` — semantic hyperedges (fixing limitation #1)

**Flags:** `use_hyperedges=True, compression=False`

EHRAG ("Entity-Hyperedge RAG") attacks the vocabulary-mismatch problem: it adds a
*semantic* layer of connections on top of the structural graph, so activation can
jump between entities that are conceptually related even if they never co-occur.

### At index time ([`src/indexer/ehrag_hyperedges.py`](../src/indexer/ehrag_hyperedges.py))

1. Take the MiniLM embedding of **every entity** in the corpus (~352K vectors).
2. Cluster them with **BIRCH** (`threshold=0.5, branching_factor=50`). BIRCH is
   chosen deliberately for the 16 GB constraint: it is single-pass and builds a
   compact in-memory tree of cluster summaries instead of holding an O(n²)
   distance matrix like agglomerative clustering would.
3. Every cluster with ≥ 2 members becomes a **semantic hyperedge** — a set of
   entities declared "mutually related". A hyperedge generalizes a normal edge:
   one edge connecting *k* entities at once, e.g.
   `{“heart attack”, “myocardial infarction”, “cardiac arrest”}`.
4. Each hyperedge gets a confidence weight from how tight the cluster is:
   `weight = exp(−avg_distance_to_centroid × 0.95)` — tight cluster → weight near
   1, loose cluster → weight decays toward 0. (Note: this weight is stored, but
   retrieval currently uses a flat 0.5 expansion factor — see below.)

On the full corpus this produces ~60K hyperedges and adds ~283 s of index time.

### At query time

The hyperedges are flattened into an **expansion map**
(`entity_id → set of cluster-mates`), loaded once at pipeline init. Then, in PPR
Stage 1 ([`ppr_retrieval.py`](../src/retrieval/ppr_retrieval.py)):

```python
for e, w in entity_weights.items():          # each activated query entity
    activated[e] = w
    for nbr in expansion_map[e]:             # its semantic cluster-mates
        activated[nbr] = max(activated[nbr], 0.5 * w)
```

Every query entity also activates its semantic neighbours **at half strength**,
*before* PPR runs. The restart distribution — the set of places importance
teleports back to — gets wider.

### Why this should help

For a question about "the Big Apple" over a corpus that says "New York City", base
PPR starts from a node with few or no edges and diffuses nowhere useful. With
hyperedges, "Big Apple" and "New York City" likely share a cluster (their
embeddings are close), so the *correct* node gets activated at 0.5 and PPR proceeds
from the right neighbourhood. The same mechanism helps synonyms, aliases,
abbreviations, and near-duplicate entity strings that hashing failed to merge.

### The trade-off (and what actually happened)

Widening the restart distribution is a **recall-for-precision trade**. Every unit
of restart mass sent to a cluster-mate is mass taken *away* from the exact query
entities (the restart vector is normalized to sum to 1). If BIRCH clusters are
loose — and at `threshold=0.5` on 384-dim MiniLM vectors some clusters lump merely
*similar-ish* entities together — a query entity with many cluster-mates sprays
activation over dozens of tangential nodes, and tangential passages start
outranking the correct ones.

The Phase 3 numbers show exactly this failure mode on HotpotQA: **EM dropped from
35.4 (base) to 28.2, F1 from 43.1 to 35.7, and wall time nearly doubled** (23,159 s
vs 12,723 s for 500 queries — the expanded activation set makes PPR converge more
slowly and drags in longer contexts). HotpotQA is a poor showcase for EHRAG:
its questions name their entities explicitly (they were written *from* the
supporting Wikipedia articles), so there is little vocabulary mismatch to fix —
only precision to lose. The mechanism's value has to come from corpora/benchmarks
where the query vocabulary genuinely diverges from the corpus vocabulary.

---

## 5. `compression` — evidence-aware context compression (fixing limitation #2)

**Flags:** `use_hyperedges=False, compression=True`

Retrieval is unchanged from `base` — identical passages come back. What changes is
**what the LLM gets to read**. Instead of dumping 20 full passages into the prompt,
[`src/retrieval/compression.py`](../src/retrieval/compression.py) performs
*hierarchical extractive compression*: keep the sentences that carry evidence,
drop the connective tissue, and stop at a hard 2,048-token budget.

### The algorithm (`EvidenceAwareCompressor.compress`)

1. **Extract query terms** — non-stopword, non-punctuation tokens of length > 2
   from the question.
2. **Walk passages in retrieval-score order** (best passage first, so if the
   budget runs out, it's the weakest passages that get cut).
3. **Within each passage, keep a sentence iff** it looks evidence-bearing:
   - it contains a named entity of an evidence type
     (`PERSON, ORG, GPE, DATE, MONEY, PERCENT, LOC, EVENT, WORK_OF_ART, CARDINAL`), **or**
   - it contains **any digit** (numbers are disproportionately answer material), **or**
   - it contains **any query term** (case-insensitive).
4. **Deduplicate** — an identical sentence (lowercased) seen in an earlier passage
   is skipped. HotpotQA passages overlap heavily, so this alone saves real tokens.
5. **Budget** — token cost is estimated at 0.25 tokens/char (~4 chars/token); once
   the 2,048-token budget would be exceeded, compression stops entirely.
6. **Keep sentences grouped by source passage, in document order**, so coreference
   chains ("Derrickson … He …") stay readable, rather than shuffling all evidence
   sentences into one soup.

Note what this is *not*: it is **not** an LLM summarizer. It is pure spaCy + regex +
string matching — fast, deterministic, and cheap enough to run per-query on a laptop.

### Why this should help

- **Speed.** Generation time on an M1 is dominated by prompt processing. Halving
  or quartering prompt tokens directly cuts per-query latency — the main pain
  point of a fully local pipeline (~25 s/query at base).
- **Focus.** Small local models are much weaker than frontier models at ignoring
  distractors. A prompt that is mostly evidence sentences raises the
  signal-to-noise ratio and reduces the chance the 8B model latches onto an
  irrelevant passage. This is the "lost in the middle" mitigation.
- **Budget safety.** A hard cap guarantees the prompt fits comfortably regardless
  of how long the retrieved passages are.

### The trade-off

The evidence heuristic is recall-oriented but not perfect. Failure modes to expect:

- A sentence with **no** entity, digit, or query term can still be the load-bearing
  one (e.g. "The film was his first collaboration with the studio." — pure
  pronouns). If it is dropped, the answer chain breaks.
- Dropping non-evidence sentences can strand a pronoun from its antecedent even
  though kept sentences stay in order.
- The 4-chars-per-token estimate is approximate, so the effective budget wobbles.

The bet is that on entity-centric benchmarks like HotpotQA (where answers *are*
entities, dates, and counts) the kept-sentence recall is high enough that the
speed/focus gains outweigh occasional evidence loss.

---

## 6. `full` — both features together

**Flags:** `use_hyperedges=True, compression=True`

`full` composes the two mechanisms, which touch *different stages* and so combine
cleanly in the code: hyperedges change **which passages are retrieved** (Step 3),
compression changes **how those passages are presented** (Step 4). Neither knows
about the other.

The interesting question `full` answers is whether they are complementary in
*effect*:

- **Optimistic story:** EHRAG casts a wider net (recall ↑, noise ↑), and
  compression then filters the noise back out at the sentence level (precision ↑)
  while keeping the extra evidence EHRAG found. The features cancel each other's
  weaknesses; `full` beats both singles.
- **Pessimistic story:** EHRAG's extra passages are *entity-dense* by construction
  (they were reached through entity clusters), so the compressor's
  entity-based heuristic happily keeps their sentences — the noise passes straight
  through the filter, and now consumes budget that correct-passage evidence needed.
  The features compound each other's weaknesses; `full` inherits EHRAG's precision
  drop *plus* compression's evidence loss.

Given that `ehrag` alone lost 7 EM points on HotpotQA, the pessimistic story is the
live risk for `full` on this benchmark; the `compression` and `full` runs will
settle it.

---

## 7. How a tagged run is executed and recorded

For every evaluation run
([`scripts/run_queries_batch.py`](../scripts/run_queries_batch.py)):

1. `Config.from_yaml()` loads `config.yaml`, then `apply_tag(config, tag)`
   **overwrites** the two flags — the tag always wins over the YAML.
2. The `QueryPipeline` constructor conditionally builds the optional components:
   the hyperedge expansion map is only loaded if `use_hyperedges` is true, and
   `self.compressor` is only instantiated if `compression.enabled` is true —
   otherwise those code paths simply don't exist for the run.
3. Answers stream to `results/queries/{benchmark}_{tag}_{model}.jsonl`, with
   per-step timings (`entity_extraction_time`, `vector_search_time`, `ppr_time`,
   `compression_time`, `generation_time`) so the latency cost of each feature is
   attributable.
4. A sidecar `.meta.json` records the resolved flags, corpus size, NER model, git
   hash, and wall time — the audit trail that makes the four runs comparable.
5. Scoring (deterministic EM/F1) happens in a separate second pass
   (`score_answers.py`), so a run can be re-scored without re-generating.

---

## 8. Cheat sheet

| Tag | Hyperedges | Compression | One-line mechanism | Fixes | Costs |
|---|---|---|---|---|---|
| `base` | ✗ | ✗ | Tri-Graph + PPR over exact entity matches; LLM sees full top-20 passages | — (control) | Vocabulary mismatch blind spot; long noisy prompts |
| `ehrag` | ✓ | ✗ | BIRCH-clustered entity embeddings → activate semantic cluster-mates at 0.5× before PPR | Synonyms / aliases / paraphrase gaps in the graph | Restart mass diluted → precision drop; ~2× query time; +hyperedge index time |
| `compression` | ✗ | ✓ | Keep only sentences with evidence entities / digits / query terms, dedup, 2,048-token cap | Prompt bloat, distractor noise, slow local generation | Heuristic may drop the load-bearing sentence |
| `full` | ✓ | ✓ | Both: wider retrieval net, then sentence-level evidence filter | Both of the above, *if* they compose | Both risks, *plus* possible interaction (EHRAG noise is entity-dense → sails through the filter) |

**Results so far (HotpotQA full corpus, 500 questions, Llama-3.1-8B-4bit, trf NER,
legacy retrieval — before the 2026-07-19 fixes below):**

| Tag | EM | F1 | Wall time (500 q) |
|---|---|---|---|
| `base` | 35.4 | 43.1 | 12,723 s (~25 s/q) |
| `ehrag` | 28.2 | 35.7 | 23,159 s (~46 s/q) |
| `compression` | 30.0 | 37.6 | 8,676 s (~17 s/q) |
| `full` | 26.4 | 32.4 | 8,830 s (~18 s/q) |

---

## 9. 2026-07-19 update: diagnosis and retrieval fixes

Per-question analysis of the matrix above (`results/analysis/`) found that **every
loss is a context-recall loss** — the model refuses ("I cannot answer…") ~61% of
the time when the gold passage misses the final context vs ~15% when it's present,
and EM|answer-in-context ≈ 50% for every tag. Three mechanisms, in causal order:

1. **Seed pathology (hurts all tags).** The 10 vector seeds are nearest neighbours
   in entity-*name* space (surname lookalikes). PPR splits an entity's mass over
   its passages, so a degree-1 junk seed puts ~5.4% of restart mass on its single
   passage while an exact query entity of degree 7 gives each true passage ~3.3%.
   Junk outranks gold; gold passages sit at ranks 10–20.
2. **Hyperedge restart dilution (`ehrag`, `full`).** Blanket 0.5× expansion of
   every activated entity (seeds included) sent a mean **49%** of restart mass to
   cluster-mates (+43 entities/question). BIRCH's biggest clusters are bags of
   DATE strings (762/757/594 members). Gold recall@20 fell 60.4% → 50.0%.
3. **Compression budget cutoff (`compression`, `full`).** The evidence filter is
   nearly lossless (3/40 losses); the 2,048-token budget is not — it exhausts on
   99% of queries after ~14/20 passages, cutting the low-ranked gold passages that
   mechanism #1 produced. 13.4% of answerable questions lost the answer.

**Fixes** (config-gated in `config.yaml`; legacy behaviour = all knobs off, kept
reproducible — the pre-fix pipeline was re-verified 500/500 identical):

- `ppr.seed_mass_fraction: 0.15` — seed pool capped at 15% of restart mass when
  exact NER matches exist.
- `ehrag.expand_seed_entities: false` — only exact NER query entities expand.
- `ehrag.expansion_max_cluster: 20`, `expansion_exclude_labels: [DATE, CARDINAL,
  ORDINAL, QUANTITY, TIME, PERCENT, MONEY]` — mega-cluster guard.
- `ehrag.expansion_weight: "inverse_size"` — a cluster shares 0.5·w instead of
  each member receiving it.
- `compression.max_tokens: 3072` (was 2048) — budget-sweep replay on the fixed
  retrieval: 2048 → 61.2% answerable, 3072 → 65.7%, uncompressed 67.2%.

**Validated retrieval-only on the same 500 questions** (generation pending re-run):

| Retrieval config | gold recall@20 | both-gold | answer-in-top-20 |
|---|---|---|---|
| base (legacy) | 60.4% | 42.4% | 64.0% |
| ehrag (legacy) | 50.0% | 28.4% | 53.1% |
| base + seed fix | **64.3%** | **48.6%** | **67.2%** |
| ehrag fixed (all knobs) | 64.5% | 48.6% | 66.8% |

Repaired hyperedges are now *neutral* rather than harmful (64.5 vs 64.3) —
consistent with §4's observation that HotpotQA has little vocabulary mismatch for
them to fix. The `hotpotqa_full` 500-q jsonl was lost to an overwrite on
2026-07-19 (see `results/evaluation/hotpotqa_full_llama-3.1-8b.json` note);
`run_queries_batch.py` now suffixes non-500-sample runs (`_n20`) so sanity runs
can't clobber full artifacts, and logs the exact post-compression prompt context
per question (`llm_context`).
