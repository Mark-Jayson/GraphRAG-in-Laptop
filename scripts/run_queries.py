"""Interactive / batch query runner.

Usage:
    python scripts/run_queries.py --question "Who founded the company that makes the iPhone?"
    python scripts/run_queries.py --interactive
    python scripts/run_queries.py --no-llm --question "..."   # retrieval only (fast dev)
"""
import sys, json, argparse, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.pipeline.query_pipeline import QueryPipeline

logging.basicConfig(level=logging.INFO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--question", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="Skip LLM (retrieval only)")
    args = ap.parse_args()

    config = Config.from_yaml(args.config)
    pipeline = QueryPipeline(config, load_llm=not args.no_llm)

    def run(q):
        result = pipeline.query(q)
        print(json.dumps(result, indent=2, default=str))
        # Append to query log
        log_path = Path(config.paths.results_dir) / "queries" / "query_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")

    if args.question:
        run(args.question)
    if args.interactive:
        while True:
            q = input("\nQuestion (blank to quit): ").strip()
            if not q:
                break
            run(q)


if __name__ == "__main__":
    main()
