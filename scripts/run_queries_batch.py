"""Pass 1 of the two-pass evaluation: generate answers, write them to disk, exit.

Loads the pipeline (+ generation LLM) ONCE, answers every question, and writes
one JSON object per line to results/queries/{benchmark}_{tag}[_{model}].jsonl:
    {question, answer, contexts, ground_truth, type, level, retrieved_ids, timings}

Kept deliberately separate from scoring (score_answers.py), so that generation
and evaluation never compete for the 16 GB of unified memory and a run can be
re-scored without re-generating.

Usage:
    python scripts/run_queries_batch.py --benchmark hotpotqa --tag full --max-samples 500
    python scripts/run_queries_batch.py --benchmark hotpotqa --tag base --max-samples 5 \
        --model mlx-community/Llama-3.2-3B-Instruct-4bit
"""
import sys, json, time, argparse, logging, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, TAG_FLAGS, apply_tag, model_slug
from src.pipeline.query_pipeline import QueryPipeline


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
    ap.add_argument("--tag", default="base", choices=list(TAG_FLAGS),
                    help="Ablation tag. AUTHORITATIVE: sets ehrag.use_hyperedges and "
                    "compression.enabled (via config.apply_tag), overriding config.yaml.")
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
    args = ap.parse_args()
    if args.questions_file and not args.out_suffix:
        ap.error("--questions-file requires --out-suffix: without it a non-default "
                 "sample resolves to the same filename as the canonical run "
                 "(this destroyed a 500-question artifact on 2026-07-19)")

    config = Config.from_yaml(args.config)
    apply_tag(config, args.tag)
    if args.model:
        config.llm.model = args.model
    slug = model_slug(config.llm.model)

    qpath = (Path(args.questions_file) if args.questions_file
             else Path(f"./data/raw/{args.benchmark}_val.json"))
    with open(qpath) as f:
        questions = json.load(f)[: args.max_samples]

    # Non-standard sample counts get a filename suffix so a sanity run can
    # never clobber a full 500-question artifact (it did once: 2026-07-19).
    n_suffix = "" if args.max_samples == 500 else f"_n{args.max_samples}"
    out = (Path(config.paths.results_dir) / "queries"
           / f"{args.benchmark}_{args.tag}{n_suffix}{args.out_suffix}_{slug}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[pass 1/2] {len(questions)} {args.benchmark} questions [{args.tag}] "
          f"with {config.llm.model} "
          f"(hyperedges={config.ehrag.use_hyperedges}, "
          f"compression={config.compression.enabled}) -> {out}")

    pipeline = QueryPipeline(config, load_llm=True)
    t_start = time.time()
    with open(out, "w") as fout:
        for i, q in enumerate(questions):
            result = pipeline.query(q["question"])
            # Full passage texts (query() truncates them for its own log).
            contexts = [pipeline.store.get_passage_text(p["id"])
                        for p in result.get("retrieved_passages", [])]
            record = {
                "question": q["question"],
                "answer": result.get("answer", ""),
                "contexts": contexts,
                # What the LLM actually saw (post-compression when enabled;
                # `contexts` above always holds the full retrieved passages).
                "llm_context": result.get("context", ""),
                "ground_truth": q.get("ground_truth", q.get("answer", "")),
                "type": q.get("type"),
                "level": q.get("level"),
                "retrieved_ids": [p["id"] for p in result.get("retrieved_passages", [])],
                "timings": {k: result[k] for k in
                            ("entity_extraction_time", "vector_search_time", "ppr_time",
                             "compression_time", "generation_time", "total_time")
                            if k in result},
            }
            fout.write(json.dumps(record, default=str) + "\n")
            fout.flush()
            print(f"  [{i+1}/{len(questions)}] {result.get('total_time', 0):.1f}s | "
                  f"{q['question'][:55]}")

    meta = {"benchmark": args.benchmark, "tag": args.tag, "model": config.llm.model,
            "use_hyperedges": config.ehrag.use_hyperedges,
            "compression_enabled": config.compression.enabled,
            "corpus_size": len(pipeline.passage_ids),
            "ner_model": config.nlp.spacy_model,
            "git_hash": _git_hash(),
            "num_queries": len(questions), "wall_time_s": round(time.time() - t_start, 1),
            "run_date": time.strftime("%Y-%m-%d")}
    with open(str(out) + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[pass 1/2] done in {meta['wall_time_s']}s -> {out}\n"
          f"Now score with: python scripts/score_answers.py --input {out}")


if __name__ == "__main__":
    main()
