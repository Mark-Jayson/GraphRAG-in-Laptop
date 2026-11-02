"""End-to-end indexing pipeline: raw text -> searchable graph."""
import time
import json
import logging
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer

from src.config import Config
from src.indexer.ner_extractor import NERExtractor
from src.indexer.sentence_splitter import SentenceSplitter
from src.indexer.trigraph_builder import TriGraphBuilder
from src.indexer.ehrag_hyperedges import EHRAGHyperedges
from src.storage.duckdb_store import DuckDBGraphStore
from src.retrieval.vector_index import VectorIndex

logger = logging.getLogger(__name__)


class IndexingPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.ner = NERExtractor(config.nlp.spacy_model)
        self.splitter = SentenceSplitter()
        self.embedder = SentenceTransformer(config.embeddings.model)
        self.store = DuckDBGraphStore(config.paths.db_path)
        self.ehrag = EHRAGHyperedges(
            birch_threshold=config.ehrag.birch_threshold,
            branching_factor=config.ehrag.birch_branching_factor,
            semantic_top_k=config.ehrag.semantic_top_k,
            decay_factor=config.ehrag.decay_factor,
        ) if config.ehrag.use_hyperedges else None
        self.vector_index = VectorIndex(
            dimension=config.embeddings.dimension,
            nlist=config.vector_index.nlist,
            m=config.vector_index.m,
            nbits=config.vector_index.nbits,
        )

    def index(self, passages: List[str]) -> Dict:
        start_time = time.time()
        results: Dict = {}

        # Step 1: Tri-Graph
        logger.info("=== Step 1: Building Tri-Graph ===")
        t0 = time.time()
        trigraph = TriGraphBuilder()
        stats = trigraph.build_from_passages(passages, self.splitter, self.ner)
        results["trigraph_time"] = time.time() - t0
        results["trigraph_stats"] = stats

        # Step 2: Persist graph
        logger.info("=== Step 2: Saving to DuckDB ===")
        t0 = time.time()
        self.store.save_trigraph(trigraph)
        results["storage_time"] = time.time() - t0

        # Step 3: Embeddings
        logger.info("=== Step 3: Generating embeddings ===")
        t0 = time.time()
        entity_ids = list(trigraph.entities.keys())
        entity_texts = [trigraph.entities[e]["text"] for e in entity_ids]
        entity_embeddings = self.embedder.encode(
            entity_texts, batch_size=self.config.embeddings.batch_size,
            show_progress_bar=True, convert_to_numpy=True)

        passage_ids = list(trigraph.passages.keys())
        passage_texts = [trigraph.passages[p]["text"] for p in passage_ids]
        passage_embeddings = self.embedder.encode(
            passage_texts, batch_size=self.config.embeddings.batch_size,
            show_progress_bar=True, convert_to_numpy=True)

        self.store.save_embeddings(
            {e: emb.tolist() for e, emb in zip(entity_ids, entity_embeddings)},
            {p: emb.tolist() for p, emb in zip(passage_ids, passage_embeddings)})
        results["embedding_time"] = time.time() - t0

        # Step 4: FAISS index over entities
        logger.info("=== Step 4: Building FAISS vector index ===")
        t0 = time.time()
        self.vector_index.build_index(entity_embeddings, entity_ids)
        index_path = Path(self.config.paths.models_dir) / "entity_faiss.index"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_index.save(str(index_path))
        results["index_time"] = time.time() - t0

        # Step 5: EHRAG hyperedges
        if self.ehrag is not None:
            logger.info("=== Step 5: Building EHRAG semantic hyperedges ===")
            t0 = time.time()
            hyperedges = self.ehrag.build_semantic_hyperedges(
                trigraph.entities, entity_embeddings)
            self.store.save_hyperedges(hyperedges, self.ehrag.hyperedge_weights)
            results["hyperedge_time"] = time.time() - t0
            results["num_hyperedges"] = len(hyperedges)

        # Persist metadata for the query pipeline
        meta = {"entity_ids": entity_ids, "passage_ids": passage_ids}
        meta_path = Path(self.config.paths.models_dir) / "index_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        results["total_time"] = time.time() - start_time
        logger.info(f"=== Indexing complete in {results['total_time']:.1f}s ===")

        results_path = Path(self.config.paths.results_dir) / "indexing" / "indexing_stats.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        return results
