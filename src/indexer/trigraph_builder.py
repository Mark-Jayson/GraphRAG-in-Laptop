"""Build the LinearRAG Tri-Graph: Entities <-> Sentences <-> Passages."""
from typing import List, Dict, Tuple, Set
import hashlib
import logging

logger = logging.getLogger(__name__)


class TriGraphBuilder:
    """Construct the Tri-Graph data structure.

    Node types: entities, sentences, passages.
    Edges (structural containment only, zero LLM calls):
      Entity <-> Sentence (entity appears in sentence)
      Sentence <-> Passage (sentence belongs to passage)
    """

    def __init__(self):
        self.passages: Dict[str, Dict] = {}
        self.sentences: Dict[str, Dict] = {}
        self.entities: Dict[str, Dict] = {}
        self.entity_sentence_edges: Set[Tuple[str, str]] = set()
        self.sentence_passage_edges: List[Tuple[str, str]] = []

    @staticmethod
    def hash_entity_id(text: str) -> str:
        return "e_" + hashlib.md5(text.lower().strip().encode()).hexdigest()[:16]

    def add_passage(self, passage_text: str, passage_idx: int) -> str:
        passage_id = f"p_{passage_idx}"
        self.passages[passage_id] = {"text": passage_text, "idx": passage_idx}
        return passage_id

    def add_sentence(self, sentence_text: str, passage_id: str, sent_idx: int) -> str:
        sentence_id = f"s_{passage_id}_{sent_idx}"
        self.sentences[sentence_id] = {
            "text": sentence_text, "passage_id": passage_id, "idx": sent_idx,
        }
        self.sentence_passage_edges.append((sentence_id, passage_id))
        return sentence_id

    def add_entity(self, entity_text: str, entity_label: str) -> str:
        entity_id = self.hash_entity_id(entity_text)
        if entity_id not in self.entities:
            self.entities[entity_id] = {"text": entity_text, "label": entity_label}
        return entity_id

    def link_entity_to_sentence(self, entity_id: str, sentence_id: str):
        self.entity_sentence_edges.add((entity_id, sentence_id))

    def build_from_passages(self, passages: List[str], sentence_splitter, ner_extractor,
                            ner_batch_size: int = 32) -> Dict:
        logger.info(f"Building Tri-Graph from {len(passages)} passages...")

        all_sentences = sentence_splitter.split_batch(passages)

        flat_sentences: List[str] = []
        flat_sentence_ids: List[str] = []

        for p_idx, (passage, sents) in enumerate(zip(passages, all_sentences)):
            passage_id = self.add_passage(passage, p_idx)
            for s_idx, sent_text in enumerate(sents):
                sentence_id = self.add_sentence(sent_text, passage_id, s_idx)
                flat_sentences.append(sent_text)
                flat_sentence_ids.append(sentence_id)

        logger.info(f"Extracting entities from {len(flat_sentences)} sentences...")
        entities_batch = ner_extractor.extract_entities_batch(flat_sentences, batch_size=ner_batch_size)

        for sentence_id, entities in zip(flat_sentence_ids, entities_batch):
            for ent in entities:
                entity_id = self.add_entity(ent["text"], ent["label"])
                self.link_entity_to_sentence(entity_id, sentence_id)

        stats = {
            "num_passages": len(self.passages),
            "num_sentences": len(self.sentences),
            "num_entities": len(self.entities),
            "num_es_edges": len(self.entity_sentence_edges),
            "num_sp_edges": len(self.sentence_passage_edges),
        }
        logger.info(f"Tri-Graph built: {stats}")
        return stats

    def get_entity_sentence_adjacency(self) -> Dict[str, List[str]]:
        adj: Dict[str, List[str]] = {}
        for e_id, s_id in self.entity_sentence_edges:
            adj.setdefault(e_id, []).append(s_id)
        return adj

    def get_sentence_passage_adjacency(self) -> Dict[str, str]:
        return {s_id: p_id for s_id, p_id in self.sentence_passage_edges}

    def get_entity_passage_bipartite(self) -> Dict[str, List[str]]:
        es_adj = self.get_entity_sentence_adjacency()
        sp_map = self.get_sentence_passage_adjacency()
        ep_adj = {}
        for e_id, s_ids in es_adj.items():
            ep_adj[e_id] = list({sp_map[s] for s in s_ids if s in sp_map})
        return ep_adj
