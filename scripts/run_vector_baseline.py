"""Pass 1 for the vanilla vector-RAG baseline — the 'no graph' control.

Mirrors run_queries_batch.py exactly, minus the graph: no NER, no entity FAISS
index, no PPR, no compressor. Loads DuckDBGraphStore + SentenceTransformer +
VectorBaselineRetriever + an unmodified LocalLLM, answers every question, and
writes one JSON object per line to
results/queries/{benchmark}_vector[_nN]_{model}.jsonl.

Deliberately standalone: query_pipeline.py stays byte-identical, so the eight
committed 500q runs keep their correspondence with HEAD.

Usage:
    python scripts/run_vector_baseline.py --benchmark hotpotqa --max-samples 20
    python scripts/run_vector_baseline.py --config config_musique.yaml \
        --benchmark musique --max-samples 500
"""
import sys, json, time, argparse, logging, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer

from src.config import Config, model_slug
from src.generation.local_llm import LocalLLM
from src.retrieval.vector_baseline import VectorBaselineRetriever
from src.storage.duckdb_store import DuckDBGraphStore

# Footgun F6: --config is what points db_path at the right DB. Passing
# config.yaml with --benchmark musique would happily retrieve MuSiQue questions
# against the HotpotQA graph and produce plausible-looking garbage.
EXPECTED_CORPUS_SIZE = {"hotpotqa": 66581, "2wikimultihop": 54957, "musique": 21100}


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--benchmark", default="hotpotqa",
                    choices=["hotpotqa", "2wikimultihop", "musique"])
    ap.add_argument("--max-samples", type=int, default=500)
    ap.add_argument("--questions-file", default=None,
                    help="Override ./data/raw/{benchmark}_val.json. Use for a "
                    "hop-stratified sample (scripts/make_stratified_sample.py). "
                    "Requires --out-suffix.")
    ap.add_argument("--out-suffix", default="",
                    help="Appended to the tag in the output filename. Mandatory with "
                    "--questions-file, and the clean way to name a same-config repeat "
                    "(e.g. --out-suffix _rep2) so it cannot overwrite the original.")
    ap.add_argument("--model", default=None,
                    help="Override config.llm.model (generation model). Adds a model "
                    "slug to the output filename so ablation runs don't overwrite.")
    ap.add_argument("--top-k", type=int, default=None,
                    help="Passages to retrieve. Default: config.ppr.top_k_passages, so "
                    "the control tracks the graph arm automatically if that knob moves.")
    args = ap.parse_args()
    if args.questions_file and not args.out_suffix:
        ap.error("--questions-file requires --out-suffix: without it a non-default "
                 "sample resolves to the same filename as the canonical run "
                 "(this destroyed a 500-question artifact on 2026-07-19)")

    config = Config.from_yaml(args.config)
    if args.model:
        config.llm.model = args.model
    slug = model_slug(config.llm.model)
    top_k = args.top_k if args.top_k is not None else config.ppr.top_k_passages

    # An overridden top_k is a DIFFERENT arm (the token-matched robustness
    # control), so it gets its own tag. Without this both arms resolve to the
    # same filename and the second run silently destroys the first (F3), and
    # experiments.csv would carry two indistinguishable tag=vector rows.
    tag = "vector" if top_k == config.ppr.top_k_passages else f"vector_k{top_k}"

    qpath = (Path(args.questions_file) if args.questions_file
             else Path(f"./data/raw/{args.benchmark}_val.json"))
    with open(qpath) as f:
        questions = json.load(f)[: args.max_samples]

    # Non-standard sample counts get a filename suffix so a sanity run can
    # never clobber a full 500-question artifact (it did once: 2026-07-19).
    n_suffix = "" if args.max_samples == 500 else f"_n{args.max_samples}"
    out = (Path(config.paths.results_dir) / "queries"
           / f"{args.benchmark}_{tag}{n_suffix}{args.out_suffix}_{slug}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[pass 1/2] {len(questions)} {args.benchmark} questions [{tag}] "
          f"with {config.llm.model} "
          f"(db={config.paths.db_path}, top_k={top_k}) -> {out}")

    store = DuckDBGraphStore(config.paths.db_path)
    embedder = SentenceTransformer(config.embeddings.model)
    retriever = VectorBaselineRetriever(store, embedder, top_k=top_k,
                                        dim=config.embeddings.dimension)

    expected = EXPECTED_CORPUS_SIZE[args.benchmark]
    corpus_size = len(retriever.passage_ids)
    assert corpus_size == expected, (
        f"F6: {args.benchmark} expects {expected} passages but {config.paths.db_path} "
        f"has {corpus_size}. Wrong --config for this --benchmark?")
    print(f"  corpus_size={corpus_size} matches {args.benchmark} (F6 OK)")

    llm = LocalLLM(
        model_path=config.llm.model, max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature, top_p=config.llm.top_p,
        repetition_penalty=config.llm.repetition_penalty)

    t_start = time.time()
    with open(out, "w") as fout:
        for i, q in enumerate(questions):
            q_start = time.time()

            t0 = time.time()
            retrieved = retriever.retrieve(q["question"])
            t_retrieval = time.time() - t0

            # Byte-identical to the uncompressed branch at query_pipeline.py:150.
            context = "\n\n".join(text for _, _, text in retrieved)

            t0 = time.time()
            gen = llm.generate(q["question"], context)
            t_gen = time.time() - t0
            t_total = time.time() - q_start

            record = {
                "question": q["question"],
                "answer": gen["answer"],
                "contexts": [text for _, _, text in retrieved],
                # No compressor in this arm, so llm_context == contexts joined.
                "llm_context": context,
                "ground_truth": q.get("ground_truth", q.get("answer", "")),
                "type": q.get("type"),
                "level": q.get("level"),
                "retrieved_ids": [p_id for p_id, _, _ in retrieved],
                # Zero-filled rather than omitted (F2) so the latency-breakdown
                # figure and the stage-summing analysis work on this arm unchanged.
                "timings": {
                    "entity_extraction_time": 0.0,
                    "vector_search_time": t_retrieval,
                    "ppr_time": 0.0,
                    "compression_time": 0.0,
                    "generation_time": t_gen,
                    "total_time": t_total,
                },
            }
            fout.write(json.dumps(record, default=str) + "\n")
            fout.flush()
            print(f"  [{i+1}/{len(questions)}] {t_total:.1f}s | "
                  f"{q['question'][:55]}")

    meta = {"benchmark": args.benchmark, "tag": tag, "model": config.llm.model,
            "use_hyperedges": False, "compression_enabled": False,
            "corpus_size": corpus_size,
            # Honest: this arm never loads spaCy, so recording en_core_web_trf
            # would make the ner_model column in experiments.csv misleading.
            "ner_model": None,
            "retriever": "dense_passage_bruteforce_cosine",
            "embedding_model": config.embeddings.model,
            "top_k": top_k,
            "git_hash": _git_hash(),
            "num_queries": len(questions), "wall_time_s": round(time.time() - t_start, 1),
            "run_date": time.strftime("%Y-%m-%d")}
    with open(str(out) + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[pass 1/2] done in {meta['wall_time_s']}s -> {out}\n"
          f"Now score with: python scripts/score_answers.py --input {out}")


if __name__ == "__main__":
    main()
