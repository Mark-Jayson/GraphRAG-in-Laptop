"""Run the indexing pipeline on a JSON list of passages.

Usage:
    python scripts/run_indexing.py --passages data/raw/hotpotqa_corpus.json --limit 1000
"""
import sys, json, argparse, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.pipeline.indexing_pipeline import IndexingPipeline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--passages", required=True, help="JSON file: list of passage strings")
    ap.add_argument("--limit", type=int, default=None, help="Max passages (dev runs)")
    args = ap.parse_args()

    config = Config.from_yaml(args.config)
    with open(args.passages) as f:
        passages = json.load(f)
    if args.limit:
        passages = passages[: args.limit]

    pipeline = IndexingPipeline(config)
    stats = pipeline.index(passages)
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
