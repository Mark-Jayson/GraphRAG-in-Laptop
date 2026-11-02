"""EHRAG: Semantic hyperedge construction via BIRCH clustering."""
import numpy as np
from sklearn.cluster import Birch
from typing import List, Dict, Set
import logging

logger = logging.getLogger(__name__)


class EHRAGHyperedges:
    """Semantic hyperedges connecting conceptually related entities.

    Structural hyperedges: entities co-occurring in the same sentence (implicit
    in the Tri-Graph). Semantic hyperedges: entities clustered by embedding
    similarity using BIRCH (memory-friendly, single-pass).
    """

    def __init__(self, birch_threshold: float = 0.5, branching_factor: int = 50,
                 semantic_top_k: int = 5, decay_factor: float = 0.95):
        self.birch_threshold = birch_threshold
        self.branching_factor = branching_factor
        self.semantic_top_k = semantic_top_k
        self.decay_factor = decay_factor

        self.semantic_hyperedges: List[Set[str]] = []
        self.hyperedge_weights: List[float] = []

    def build_semantic_hyperedges(self, entities: Dict[str, Dict],
                                  entity_embeddings: np.ndarray) -> List[Set[str]]:
        logger.info(f"Building semantic hyperedges for {len(entities)} entities...")
        if len(entities) < 10:
            logger.warning("Too few entities for clustering; skipping semantic hyperedges.")
            return []

        birch = Birch(threshold=self.birch_threshold,
                      branching_factor=self.branching_factor,
                      n_clusters=None)
        birch.fit(entity_embeddings)
        labels = birch.labels_

        entity_ids = list(entities.keys())
        clusters: Dict[int, Set[str]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, set()).add(entity_ids[idx])

        self.semantic_hyperedges, self.hyperedge_weights = [], []
        for cluster_id, cluster_entities in clusters.items():
            if len(cluster_entities) < 2:
                continue
            self.semantic_hyperedges.append(cluster_entities)
            mask = labels == cluster_id
            cluster_embs = entity_embeddings[mask]
            centroid = cluster_embs.mean(axis=0)
            avg_dist = np.linalg.norm(cluster_embs - centroid, axis=1).mean()
            self.hyperedge_weights.append(float(np.exp(-avg_dist * self.decay_factor)))

        logger.info(f"Built {len(self.semantic_hyperedges)} semantic hyperedges "
                    f"from {len(clusters)} clusters")
        return self.semantic_hyperedges

    def get_entity_expansion_map(self) -> Dict[str, Set[str]]:
        """entity_id -> set of semantically related entity_ids (for query-time expansion)."""
        expansion: Dict[str, Set[str]] = {}
        for hyperedge in self.semantic_hyperedges:
            for e in hyperedge:
                expansion.setdefault(e, set()).update(hyperedge - {e})
        return expansion
