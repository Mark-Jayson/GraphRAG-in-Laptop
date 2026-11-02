"""Download the three benchmarks and write data/raw/{bench}_val.json + _corpus.json.

Dataset revisions are pinned to the Hugging Face commits the paper used, so the
written files are byte-identical to the ones scored (verify with
`shasum -a 256 -c results/provenance/data-sha256.txt`). 2WikiMultihopQA and
MuSiQue come from community mirrors of the official releases.
"""
from datasets import load_dataset
from pathlib import Path
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hugging Face dataset commits used for every run in the paper.
REVISIONS = {
    "hotpotqa/hotpot_qa": "1908d6afbbead072334abe2965f91bd2709910ab",
    "voidful/2WikiMultihopQA": "16852fde9d85cba158cf7e6517e7a3f9415a28c0",
    "dgslibisey/MuSiQue": "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321",
}

DATA_DIR = Path("./data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_hotpotqa():
    logger.info("Downloading HotpotQA (distractor)...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor",
                      revision=REVISIONS["hotpotqa/hotpot_qa"])
    val_data, corpus = [], {}
    for item in ds["validation"]:
        val_data.append({
            "question": item["question"], "answer": item["answer"],
            "type": item["type"], "level": item["level"],
            "supporting_facts": item["supporting_facts"],
        })
        # Build corpus from context paragraphs (title -> joined sentences)
        for title, sents in zip(item["context"]["title"], item["context"]["sentences"]):
            corpus.setdefault(title, " ".join(sents))
    with open(DATA_DIR / "hotpotqa_val.json", "w") as f:
        json.dump(val_data, f, indent=2)
    with open(DATA_DIR / "hotpotqa_corpus.json", "w") as f:
        json.dump(list(corpus.values()), f, indent=2)
    logger.info(f"Saved {len(val_data)} questions and {len(corpus)} corpus passages")


def download_2wikimultihop():
    logger.info("Downloading 2WikiMultiHopQA...")
    ds = load_dataset("voidful/2WikiMultihopQA",
                      revision=REVISIONS["voidful/2WikiMultihopQA"])
    split = "validation" if "validation" in ds else list(ds.keys())[0]
    val_data, corpus = [], {}
    for item in ds[split]:
        # supporting_facts is a list of [title, sent_id] pairs; normalize to the
        # same {"title": [...], "sent_id": [...]} shape HotpotQA uses so shared
        # analysis code works across benchmarks.
        sf = item.get("supporting_facts", []) or []
        val_data.append({
            "question": item["question"], "answer": item["answer"],
            "type": item.get("type", ""),
            "supporting_facts": {"title": [t for t, _ in sf],
                                 "sent_id": [s for _, s in sf]},
        })
        # context is a list of [title, [sentences]] — same as HotpotQA. Build the
        # corpus from it (title -> joined sentences) so gold passages are present.
        for title, sents in item.get("context", []):
            corpus.setdefault(title, " ".join(sents))
    with open(DATA_DIR / "2wikimultihop_val.json", "w") as f:
        json.dump(val_data, f, indent=2)
    with open(DATA_DIR / "2wikimultihop_corpus.json", "w") as f:
        json.dump(list(corpus.values()), f, indent=2)
    logger.info(f"Saved {len(val_data)} questions and {len(corpus)} corpus passages")


def download_musique():
    logger.info("Downloading MuSiQue...")
    ds = load_dataset("dgslibisey/MuSiQue", revision=REVISIONS["dgslibisey/MuSiQue"])
    split = "validation" if "validation" in ds else list(ds.keys())[0]
    val_data, corpus = [], set()
    for item in ds[split]:
        # MuSiQue's official eval scores against `answer` PLUS `answer_aliases`
        # (28% of validation items have at least one alias, e.g. gold
        # "Bombardier Inc." accepts "Bombardier"). Dropping them understated EM,
        # so carry them as a `ground_truth` list — run_queries_batch.py prefers
        # that key and compute_em_f1 takes the max over the list.
        aliases = item.get("answer_aliases") or []
        val_data.append({"question": item["question"], "answer": item["answer"],
                         "ground_truth": [item["answer"], *aliases]})
        for p in item.get("paragraphs", []):
            corpus.add(p.get("paragraph_text", ""))
    with open(DATA_DIR / "musique_val.json", "w") as f:
        json.dump(val_data, f, indent=2)
    with open(DATA_DIR / "musique_corpus.json", "w") as f:
        json.dump(sorted(corpus), f, indent=2)
    logger.info(f"Saved {len(val_data)} MuSiQue questions, {len(corpus)} passages")


DOWNLOADERS = {
    "hotpotqa": download_hotpotqa,
    "2wikimultihop": download_2wikimultihop,
    "musique": download_musique,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(DOWNLOADERS), default=None,
                    help="Download just one benchmark (default: all three).")
    args = ap.parse_args()
    names = [args.only] if args.only else list(DOWNLOADERS)
    for name in names:
        DOWNLOADERS[name]()
