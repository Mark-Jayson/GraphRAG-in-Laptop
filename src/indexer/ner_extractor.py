"""Named Entity Recognition using spaCy."""
import spacy
from typing import List, Dict, Set
import logging

logger = logging.getLogger(__name__)


class NERExtractor:
    """Extract entities from passages using a spaCy pipeline."""

    def __init__(self, model_name: str = "en_core_web_trf"):
        logger.info(f"Loading spaCy NER model: {model_name}")
        self.nlp = spacy.load(model_name)
        # Keep only components needed for NER
        keep = [p for p in ("transformer", "tok2vec", "ner") if p in self.nlp.pipe_names]
        self.nlp.select_pipes(enable=keep)

    @staticmethod
    def _dedup_ents(doc) -> List[Dict]:
        entities, seen = [], set()
        for ent in doc.ents:
            key = (ent.text.lower(), ent.label_)
            if key not in seen:
                seen.add(key)
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                })
        return entities

    def extract_entities(self, text: str) -> List[Dict]:
        return self._dedup_ents(self.nlp(text))

    def extract_entities_batch(self, texts: List[str], batch_size: int = 32) -> List[List[Dict]]:
        return [self._dedup_ents(doc) for doc in self.nlp.pipe(texts, batch_size=batch_size)]
