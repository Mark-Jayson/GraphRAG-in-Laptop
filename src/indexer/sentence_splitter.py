"""Sentence segmentation for passage splitting."""
import spacy
from typing import List


class SentenceSplitter:
    """Split passages into sentences using a lightweight spaCy sentencizer."""

    def __init__(self):
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

    def split(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def split_batch(self, texts: List[str]) -> List[List[str]]:
        return [self.split(t) for t in texts]
