"""Vanilla dense-passage retrieval — the 'no graph' control.

No entities, no NER, no hyperedges, no PPR. Embed the question with the same
MiniLM model used at index time, cosine against passage embeddings, return the
top-k passages. The ONLY difference from the `base` tag is how passages are
ranked; corpus, embedder, top_k, prompt, generator and sample set are identical.

Brute force rather than FAISS on purpose: 66,581 x 384 float32 is 102 MB and a
single matmul is ~25 MFLOP, so exact search costs nothing here and makes this a
*stronger* control — no IVF-PQ quantization error confounding the graph delta.
"""
from typing import Dict, List, Tuple

import numpy as np

from src.retrieval.vector_index import l2_normalize  # same cosine semantics as the graph path


class VectorBaselineRetriever:
    """Exact cosine top-k over passage embeddings. Loads once; queries are cheap."""

    def __init__(self, store, embedder, top_k: int = 20, dim: int = 384):
        self.passage_ids, mat = store.get_passage_embeddings(dim=dim)
        self.mat = l2_normalize(mat)              # (n, dim), unit rows
        self.passage_texts: Dict[str, str] = store.get_all_passages()
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, question: str) -> List[Tuple[str, float, str]]:
        """-> [(passage_id, score, text), ...] — same shape as PPRRetriever.retrieve."""
        q = self.embedder.encode([question])[0].astype(np.float32)
        q = l2_normalize(q.reshape(1, -1))[0]
        scores = self.mat @ q                                   # exact cosine, ~0.5ms at 66K
        k = min(self.top_k, scores.shape[0])
        idx = np.argpartition(-scores, k - 1)[:k]
        # deterministic tie-break: score desc, then passage_id asc
        idx = sorted(idx, key=lambda i: (-float(scores[i]), self.passage_ids[i]))
        return [(self.passage_ids[i], float(scores[i]),
                 self.passage_texts[self.passage_ids[i]]) for i in idx]
