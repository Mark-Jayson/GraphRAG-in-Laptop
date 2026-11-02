"""Deterministic, judge-free EM/F1 metrics (standard SQuAD/HotpotQA scoring)."""
import re
import string
from collections import Counter
from typing import Dict, List, Union

GroundTruths = Union[str, List[str]]


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: GroundTruths) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def compute_em_f1(predictions: List[str], references: List[GroundTruths]) -> Dict[str, float]:
    """Corpus-level exact-match and F1 against gold answer(s) per question.

    `references[i]` may be a single gold answer string or a list of
    acceptable gold answers (max score over the list is taken, matching the
    standard HotpotQA/2WikiMultiHop/MuSiQue evaluation scripts).
    """
    n = len(predictions)
    if n == 0:
        return {"exact_match": 0.0, "f1": 0.0}
    em_total = sum(_metric_max_over_ground_truths(exact_match_score, p, r)
                   for p, r in zip(predictions, references))
    f1_total = sum(_metric_max_over_ground_truths(f1_score, p, r)
                   for p, r in zip(predictions, references))
    return {"exact_match": 100.0 * em_total / n, "f1": 100.0 * f1_total / n}
