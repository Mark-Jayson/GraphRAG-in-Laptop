"""Personalized PageRank for global importance aggregation."""
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Tuple, Set, Optional, Union
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class PPRRetriever:
    """Dual-stage retrieval (LinearRAG-style):
    1. Local Semantic Bridging: activate query entities (+ hyperedge expansion)
    2. Global Importance Aggregation: PPR on entity-passage bipartite graph
    """

    def __init__(self, alpha: float = 0.15, max_iterations: int = 50,
                 convergence_threshold: float = 1e-6, top_k: int = 10):
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.top_k = top_k

    def build_entity_passage_matrix(self, ep_adj: Dict[str, List[str]],
                                    entity_ids: List[str],
                                    passage_ids: List[str]) -> csr_matrix:
        entity_idx = {e: i for i, e in enumerate(entity_ids)}
        passage_idx = {p: i for i, p in enumerate(passage_ids)}
        rows, cols, data = [], [], []
        for e_id, p_list in ep_adj.items():
            if e_id not in entity_idx or not p_list:
                continue
            e_i = entity_idx[e_id]
            w = 1.0 / len(p_list)
            for p_id in p_list:
                if p_id in passage_idx:
                    rows.append(e_i)
                    cols.append(passage_idx[p_id])
                    data.append(w)
        return csr_matrix((data, (rows, cols)),
                          shape=(len(entity_ids), len(passage_ids)))

    def build_passage_entity_matrix(self, ep_adj: Dict[str, List[str]],
                                    entity_ids: List[str],
                                    passage_ids: List[str]) -> csr_matrix:
        passage_to_entities = defaultdict(list)
        for e_id, p_list in ep_adj.items():
            for p_id in p_list:
                passage_to_entities[p_id].append(e_id)
        entity_idx = {e: i for i, e in enumerate(entity_ids)}
        passage_idx = {p: i for i, p in enumerate(passage_ids)}
        rows, cols, data = [], [], []
        for p_id, e_list in passage_to_entities.items():
            if p_id not in passage_idx or not e_list:
                continue
            p_i = passage_idx[p_id]
            w = 1.0 / len(e_list)
            for e_id in e_list:
                if e_id in entity_idx:
                    rows.append(p_i)
                    cols.append(entity_idx[e_id])
                    data.append(w)
        return csr_matrix((data, (rows, cols)),
                          shape=(len(passage_ids), len(entity_ids)))

    def personalized_pagerank(self, A: csr_matrix, B: csr_matrix,
                              restart_entities: np.ndarray) -> np.ndarray:
        """Iterate PPR over the bipartite graph; return passage scores.

        Entity scores diffuse to passages via A^T (each entity splits its
        score evenly across its own passages), then back to entities via B^T
        (each passage splits its score evenly across the entities it
        contains), with teleport back to the query-biased entity restart
        distribution.
        """
        e_scores = restart_entities.copy()
        for it in range(self.max_iterations):
            p_scores = A.T @ e_scores                      # entities -> passages
            s = p_scores.sum()
            if s > 0:
                p_scores = p_scores / s
            e_new = (1 - self.alpha) * (B.T @ p_scores) + self.alpha * restart_entities
            s = e_new.sum()
            if s > 0:
                e_new = e_new / s
            diff = np.linalg.norm(e_new - e_scores, ord=1)
            e_scores = e_new
            if diff < self.convergence_threshold:
                logger.debug(f"PPR converged at iteration {it}")
                break
        final_p = A.T @ e_scores
        s = final_p.sum()
        return final_p / s if s > 0 else final_p

    def retrieve(self, query_entities: Union[List[str], Dict[str, float]],
                 ep_adj: Dict[str, List[str]],
                 entity_ids: List[str],
                 passage_ids: List[str],
                 passage_texts: Dict[str, str],
                 expansion_map: Optional[Dict[str, Set[str]]] = None
                 ) -> List[Tuple[str, float, str]]:
        entity_pos = {e: i for i, e in enumerate(entity_ids)}

        # Accept either a flat list (uniform weight 1.0) or per-entity
        # activation weights (e.g. exact NER matches at 1.0, vector-similarity
        # seeds scaled down so noisy nearest-neighbor entities don't dominate
        # the restart distribution).
        if isinstance(query_entities, dict):
            entity_weights = query_entities
        else:
            entity_weights = {e: 1.0 for e in query_entities}

        # Stage 1: activation (query entities + hyperedge expansion)
        activated: Dict[str, float] = {}
        for e, w in entity_weights.items():
            if e in entity_pos and w > 0:
                activated[e] = max(activated.get(e, 0.0), w)
                if expansion_map and e in expansion_map:
                    for nbr in expansion_map[e]:
                        activated[nbr] = max(activated.get(nbr, 0.0), 0.5 * w)

        # Stage 2: PPR
        A = self.build_entity_passage_matrix(ep_adj, entity_ids, passage_ids)
        B = self.build_passage_entity_matrix(ep_adj, entity_ids, passage_ids)
        restart = np.zeros(len(entity_ids))
        for e, w in activated.items():
            restart[entity_pos[e]] = w
        if restart.sum() == 0:
            logger.warning("No query entities found in graph; uniform restart.")
            restart = np.ones(len(entity_ids))
        restart = restart / restart.sum()

        passage_scores = self.personalized_pagerank(A, B, restart)
        ranked = np.argsort(passage_scores)[::-1][: self.top_k]

        return [(passage_ids[i], float(passage_scores[i]),
                 passage_texts.get(passage_ids[i], ""))
                for i in ranked if passage_scores[i] > 0 or True]
