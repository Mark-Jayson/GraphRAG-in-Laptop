"""FAISS-based approximate nearest neighbor index."""
import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Avoid a segfault on macOS where FAISS's OpenMP-threaded k-means clustering
# crashes when PyTorch (imported elsewhere in the pipeline) has already
# initialized its own OpenMP runtime in the same process.
faiss.omp_set_num_threads(1)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class VectorIndex:
    """IVF-PQ vector index (falls back to flat index for small corpora).

    Uses inner product over L2-normalized vectors = cosine similarity.
    """

    def __init__(self, dimension: int, nlist: int = 100, m: int = 16, nbits: int = 8):
        self.dimension = dimension
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.index = None
        self.id_map: Dict[int, str] = {}
        self.reverse_map: Dict[str, int] = {}

    def build_index(self, embeddings: np.ndarray, ids: List[str]):
        n = len(ids)
        logger.info(f"Building FAISS index: {n} vectors, dim={self.dimension}")
        for i, id_str in enumerate(ids):
            self.id_map[i] = id_str
            self.reverse_map[id_str] = i

        x = l2_normalize(embeddings.astype(np.float32))

        # IVF-PQ needs enough training points (~39 * nlist); fall back to flat.
        if n < max(39 * self.nlist, 1000):
            logger.info("Small corpus: using IndexFlatIP (exact search).")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(x)
        else:
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, self.nlist,
                                          self.m, self.nbits, faiss.METRIC_INNER_PRODUCT)
            logger.info("Training FAISS IVF-PQ index...")
            self.index.train(x)
            self.index.add(x)
        logger.info(f"Index built. Total vectors: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() or load() first.")
        q = l2_normalize(query_embedding.reshape(1, -1).astype(np.float32))
        distances, indices = self.index.search(q, k)
        return [(self.id_map[int(idx)], float(dist))
                for dist, idx in zip(distances[0], indices[0])
                if idx >= 0 and int(idx) in self.id_map]

    def set_nprobe(self, nprobe: int):
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

    def save(self, path: str):
        faiss.write_index(self.index, path)
        with open(path + ".ids.json", "w") as f:
            json.dump({str(k): v for k, v in self.id_map.items()}, f)
        logger.info(f"Index saved to {path}")

    def load(self, path: str):
        self.index = faiss.read_index(path)
        ids_path = Path(path + ".ids.json")
        if ids_path.exists():
            with open(ids_path) as f:
                self.id_map = {int(k): v for k, v in json.load(f).items()}
            self.reverse_map = {v: k for k, v in self.id_map.items()}
        logger.info(f"Index loaded from {path}. Total vectors: {self.index.ntotal}")
