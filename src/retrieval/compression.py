"""Hierarchical extractive compression for retrieved passages.

Strategy: preserve evidence-bearing spans (entities, numbers, query-relevant
noun phrases), prune connective tissue.
"""
import re
import spacy
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class EvidenceAwareCompressor:
    """Compress retrieved passages while preserving evidence-bearing spans."""

    EVIDENCE_LABELS = {"PERSON", "ORG", "GPE", "DATE", "MONEY", "PERCENT",
                       "LOC", "EVENT", "WORK_OF_ART", "CARDINAL"}

    def __init__(self, max_tokens: int = 2048, spacy_model: str = "en_core_web_sm"):
        self.max_tokens = max_tokens
        self.nlp = spacy.load(spacy_model)

    def _extract_evidence_spans(self, text: str, query_terms: List[str]) -> List[Dict]:
        doc = self.nlp(text)
        spans = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            is_evidence = (
                any(ent.label_ in self.EVIDENCE_LABELS for ent in sent.ents)
                or re.search(r"\d", sent_text) is not None
                or any(t.lower() in sent_text.lower() for t in query_terms)
            )
            if is_evidence:
                spans.append({"text": sent_text, "start": sent.start_char,
                              "end": sent.end_char, "type": "evidence"})
        return spans

    def compress(self, passages: List[Tuple[str, float, str]], query: str) -> str:
        """Keep evidence sentences grouped by source passage, in document
        order, so coreference chains and multi-fact passages stay readable.
        Passages are consumed in retrieval-score order until the token budget
        is exhausted.
        """
        query_doc = self.nlp(query)
        query_terms = [t.text for t in query_doc
                       if not t.is_stop and not t.is_punct and len(t.text) > 2]

        seen, blocks, total_tokens, n_spans = set(), [], 0.0, 0
        tokens_per_char = 0.25  # ~4 chars/token
        budget_exhausted = False
        for p_id, score, text in passages:  # already ranked by retrieval score
            kept = []
            for span in self._extract_evidence_spans(text, query_terms):
                key = span["text"].lower().strip()
                if key in seen:
                    continue
                est = len(span["text"]) * tokens_per_char
                if total_tokens + est > self.max_tokens:
                    budget_exhausted = True
                    break
                seen.add(key)
                kept.append(span["text"])
                total_tokens += est
            if kept:
                blocks.append(" ".join(kept))
                n_spans += len(kept)
            if budget_exhausted:
                break

        logger.info(f"Compressed {len(passages)} passages -> ~{total_tokens:.0f} tokens "
                    f"({n_spans} evidence spans, {len(blocks)} passages)")
        return "\n\n".join(blocks)
