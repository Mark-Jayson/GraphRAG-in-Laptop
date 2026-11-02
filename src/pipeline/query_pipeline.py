"""End-to-end query pipeline: question -> retrieved evidence -> answer."""
import time
import json
import logging
from pathlib import Path
from typing import Dict
from sentence_transformers import SentenceTransformer

from src.config import Config
from src.indexer.ner_extractor import NERExtractor
from src.indexer.trigraph_builder import TriGraphBuilder
from src.retrieval.vector_index import VectorIndex
from src.retrieval.ppr_retrieval import PPRRetriever
from src.retrieval.compression import EvidenceAwareCompressor
from src.generation.local_llm import LocalLLM
from src.storage.duckdb_store import DuckDBGraphStore

logger = logging.getLogger(__name__)


class QueryPipeline:
    """Loads everything ONCE at init (models, index, adjacency); queries are cheap."""

    def __init__(self, config: Config, store: DuckDBGraphStore = None,
                 load_llm: bool = True):
        self.config = config
        self.store = store or DuckDBGraphStore(config.paths.db_path)

        # Graph metadata
        meta_path = Path(config.paths.models_dir) / "index_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        self.entity_ids = meta["entity_ids"]
        self.passage_ids = meta["passage_ids"]

        # Loaded once (NOT per query)
        self.ner = NERExtractor(config.nlp.spacy_model)
        self.embedder = SentenceTransformer(config.embeddings.model)

        self.vector_index = VectorIndex(
            dimension=config.embeddings.dimension,
            nlist=config.vector_index.nlist,
            m=config.vector_index.m, nbits=config.vector_index.nbits)
        self.vector_index.load(f"{config.paths.models_dir}/entity_faiss.index")
        self.vector_index.set_nprobe(config.vector_index.nprobe)

        self.ppr = PPRRetriever(
            alpha=config.ppr.alpha, max_iterations=config.ppr.max_iterations,
            convergence_threshold=config.ppr.convergence_threshold,
            top_k=config.ppr.top_k_passages)

        # Cache heavy graph structures
        self.ep_adj = self.store.get_entity_passage_adjacency()
        self.passage_texts = self.store.get_all_passages()
        self.entity_pos = {e: i for i, e in enumerate(self.entity_ids)}
        self.expansion_map = (self.store.load_hyperedge_expansion()
                              if config.ehrag.use_hyperedges else None)
        self.entity_labels = (self.store.get_entity_labels()
                              if (config.ehrag.use_hyperedges
                                  and config.ehrag.expansion_exclude_labels)
                              else {})

        self.compressor = (EvidenceAwareCompressor(max_tokens=config.compression.max_tokens)
                           if config.compression.enabled else None)
        self.llm = LocalLLM(
            model_path=config.llm.model, max_tokens=config.llm.max_tokens,
            temperature=config.llm.temperature, top_p=config.llm.top_p,
            repetition_penalty=config.llm.repetition_penalty) if load_llm else None

    def query(self, question: str) -> Dict:
        result: Dict = {"question": question}
        start_time = time.time()

        # Step 1: query entity extraction
        t0 = time.time()
        query_entities = self.ner.extract_entities(question)
        query_entity_ids = [TriGraphBuilder.hash_entity_id(e["text"])
                            for e in query_entities]
        result["entity_extraction_time"] = time.time() - t0
        result["query_entities"] = [e["text"] for e in query_entities]

        # Step 2: vector search for seed entities
        t0 = time.time()
        query_emb = self.embedder.encode([question])[0]
        seed_results = self.vector_index.search(query_emb, k=10)
        seed_entity_ids = [e_id for e_id, _ in seed_results]
        result["vector_search_time"] = time.time() - t0
        result["seed_entities"] = seed_entity_ids[:5]

        # Exact NER matches get full activation; vector seeds are weighted by
        # cosine similarity and halved, so approximate neighbors supplement
        # the restart distribution instead of competing with real query
        # entities on equal footing.
        activation: Dict[str, float] = {}
        for e_id, sim in seed_results:
            w = max(0.0, sim) * 0.5
            activation[e_id] = max(activation.get(e_id, 0.0), w)

        # Optional seed-pool cap: PPR splits an entity's mass across its
        # passages, so a degree-1 lookalike seed can put more mass on its one
        # passage than an exact query entity spreads over its several. When
        # exact matches exist, rescale the whole seed pool to a fixed fraction
        # of restart mass instead of letting each seed keep 0.5*cosine.
        ner_in_graph = {e for e in query_entity_ids if e in self.entity_pos}
        f = self.config.ppr.seed_mass_fraction
        if f > 0 and ner_in_graph:
            seed_mass = sum(activation.values())
            if seed_mass > 0:
                scale = (f / (1.0 - f)) * len(ner_in_graph) / seed_mass
                activation = {e: w * min(scale, 1.0) for e, w in activation.items()}

        for e_id in query_entity_ids:
            activation[e_id] = 1.0

        # Hyperedge expansion (moved out of PPRRetriever so it can distinguish
        # exact NER entities from vector seeds and apply shaping knobs).
        if self.expansion_map:
            eh = self.config.ehrag
            sources = (list(activation.items()) if eh.expand_seed_entities
                       else [(e, activation[e]) for e in ner_in_graph])
            exclude = set(eh.expansion_exclude_labels)
            for e, w in sources:
                if e not in self.entity_pos or e not in self.expansion_map:
                    continue
                nbrs = self.expansion_map[e]
                if eh.expansion_max_cluster and len(nbrs) > eh.expansion_max_cluster:
                    continue
                w_flat = 0.5 * w
                w_exp = w_flat / len(nbrs) if eh.expansion_weight == "inverse_size" else w_flat
                for nbr in nbrs:
                    if exclude and self.entity_labels.get(nbr) in exclude:
                        continue
                    activation[nbr] = max(activation.get(nbr, 0.0), w_exp)

        # Step 3: PPR retrieval (expansion already applied above)
        t0 = time.time()
        retrieved = self.ppr.retrieve(
            activation, self.ep_adj, self.entity_ids, self.passage_ids,
            self.passage_texts, None)
        result["ppr_time"] = time.time() - t0
        result["retrieved_passages"] = [
            {"id": p_id, "score": score, "text": text[:200] + "..."}
            for p_id, score, text in retrieved]

        # Step 4: compression
        t0 = time.time()
        if self.compressor:
            context = self.compressor.compress(retrieved, question)
        else:
            context = "\n\n".join(text for _, _, text in retrieved)
        result["compression_time"] = time.time() - t0
        result["context_length_chars"] = len(context)
        # The exact context string the LLM sees. Logged because the 2026-07-19
        # diagnosis needed a full compressor replay to learn what compression
        # actually dropped — retrieved_passages alone can't tell you that.
        result["context"] = context

        # Step 5: local LLM generation
        if self.llm is not None:
            t0 = time.time()
            gen = self.llm.generate(question, context)
            result["generation_time"] = time.time() - t0
            result.update({k: gen[k] for k in
                           ("answer", "prompt_tokens", "completion_tokens", "total_tokens")})

        result["total_time"] = time.time() - start_time
        logger.info(f"Query answered in {result['total_time']:.2f}s")
        return result
