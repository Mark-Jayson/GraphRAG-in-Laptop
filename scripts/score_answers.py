"""Pass 2 of the two-pass evaluation: score a jsonl of answers, no pipeline reload.

Reads results/queries/*.jsonl from run_queries_batch.py / run_vector_baseline.py
and writes results/evaluation/{stem}.json with deterministic exact-match / F1
(SQuAD normalisation, list-valued gold honoured). No judge model is involved.

Usage:
    python scripts/score_answers.py --input results/queries/hotpotqa_full_llama-3.1-8b.jsonl
"""
import sys, json, argparse, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.em_f1 import compute_em_f1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jsonl from run_queries_batch.py")
    args = ap.parse_args()

    records = [json.loads(line) for line in open(args.input) if line.strip()]
    answers = [r["answer"] for r in records]
    gts = [r["ground_truth"] for r in records]

    em_f1 = compute_em_f1(answers, gts)
    print(f"EM/F1 (0-100) over {len(records)} answers: {em_f1}")

    # Carry forward run metadata (tag/model/flags) if the pass-1 sidecar exists.
    meta = {}
    meta_path = Path(str(args.input) + ".meta.json")
    if meta_path.exists():
        meta = json.load(open(meta_path))

    out = Path("./results/evaluation") / (Path(args.input).stem + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {**meta, "num_queries": len(records), "em_f1": em_f1,
              "source_jsonl": str(args.input)}
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
