"""DuckDB-based graph storage for the Tri-Graph."""
import duckdb
import numpy as np
import pyarrow as pa
from typing import List, Dict, Set, Tuple
import logging

logger = logging.getLogger(__name__)

# All tables owned by an index() run. save_trigraph() clears these at the start
# of a fresh index so a re-index never inherits stale rows from a previous run
# (previously INSERT OR REPLACE left rows whose PK wasn't in the new corpus).
_ALL_TABLES = ["passages", "sentences", "entities", "entity_sentence_edges",
               "sentence_passage_edges", "entity_embeddings", "passage_embeddings",
               "semantic_hyperedges"]


class DuckDBGraphStore:
    """Embeddable graph storage using DuckDB (single file, zero server)."""

    def __init__(self, db_path: str = "./data/graph.duckdb"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""CREATE TABLE IF NOT EXISTS passages (
            passage_id VARCHAR PRIMARY KEY, text TEXT, idx INTEGER)""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS sentences (
            sentence_id VARCHAR PRIMARY KEY, text TEXT, passage_id VARCHAR, idx INTEGER)""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS entities (
            entity_id VARCHAR PRIMARY KEY, text TEXT, label VARCHAR)""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS entity_sentence_edges (
            entity_id VARCHAR, sentence_id VARCHAR, PRIMARY KEY (entity_id, sentence_id))""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS sentence_passage_edges (
            sentence_id VARCHAR, passage_id VARCHAR, PRIMARY KEY (sentence_id, passage_id))""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS entity_embeddings (
            entity_id VARCHAR PRIMARY KEY, embedding FLOAT[])""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS passage_embeddings (
            passage_id VARCHAR PRIMARY KEY, embedding FLOAT[])""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS semantic_hyperedges (
            hyperedge_id INTEGER, entity_id VARCHAR, weight FLOAT,
            PRIMARY KEY (hyperedge_id, entity_id))""")

    def _bulk_insert(self, table: str, columns: Dict[str, pa.Array]):
        """Insert an Arrow-backed batch in one statement (INSERT ... SELECT).

        Far faster than row-by-row executemany on large batches: it avoids
        N Python->C round-trips and lets DuckDB build the primary-key index in
        one pass. Column order in `columns` must match the table's schema.
        Tables are cleared by save_trigraph() before a fresh index, so a plain
        INSERT (no OR REPLACE) is correct and lets DuckDB skip conflict checks.
        """
        tbl = pa.table(columns)
        if tbl.num_rows == 0:
            return
        self.conn.register("_bulk_src", tbl)
        try:
            self.conn.execute(f"INSERT INTO {table} SELECT * FROM _bulk_src")
        finally:
            self.conn.unregister("_bulk_src")

    def save_trigraph(self, trigraph):
        """Save a TriGraphBuilder instance to DuckDB using bulk Arrow inserts.

        Resets ALL index tables first: an index() run always starts here, so
        clearing guarantees a clean slate (no stale rows from a prior corpus).
        """
        logger.info("Saving Tri-Graph to DuckDB...")
        c = self.conn
        for t in _ALL_TABLES:
            c.execute(f"DELETE FROM {t}")

        passages = trigraph.passages
        self._bulk_insert("passages", {
            "passage_id": pa.array(list(passages.keys()), type=pa.string()),
            "text": pa.array([d["text"] for d in passages.values()], type=pa.string()),
            "idx": pa.array([d["idx"] for d in passages.values()], type=pa.int32()),
        })
        sentences = trigraph.sentences
        self._bulk_insert("sentences", {
            "sentence_id": pa.array(list(sentences.keys()), type=pa.string()),
            "text": pa.array([d["text"] for d in sentences.values()], type=pa.string()),
            "passage_id": pa.array([d["passage_id"] for d in sentences.values()], type=pa.string()),
            "idx": pa.array([d["idx"] for d in sentences.values()], type=pa.int32()),
        })
        entities = trigraph.entities
        self._bulk_insert("entities", {
            "entity_id": pa.array(list(entities.keys()), type=pa.string()),
            "text": pa.array([d["text"] for d in entities.values()], type=pa.string()),
            "label": pa.array([d["label"] for d in entities.values()], type=pa.string()),
        })
        # Dedupe edges in Python (tables are empty, so plain INSERT needs unique
        # keys); preserves the original INSERT OR IGNORE dedup semantics.
        es_edges = list(dict.fromkeys(tuple(e) for e in trigraph.entity_sentence_edges))
        self._bulk_insert("entity_sentence_edges", {
            "entity_id": pa.array([e for e, _ in es_edges], type=pa.string()),
            "sentence_id": pa.array([s for _, s in es_edges], type=pa.string()),
        })
        sp_edges = list(dict.fromkeys(tuple(e) for e in trigraph.sentence_passage_edges))
        self._bulk_insert("sentence_passage_edges", {
            "sentence_id": pa.array([s for s, _ in sp_edges], type=pa.string()),
            "passage_id": pa.array([p for _, p in sp_edges], type=pa.string()),
        })
        c.commit()
        logger.info("Tri-Graph saved successfully.")

    def get_passage_text(self, passage_id: str) -> str:
        row = self.conn.execute(
            "SELECT text FROM passages WHERE passage_id = ?", (passage_id,)).fetchone()
        return row[0] if row else ""

    def get_all_passages(self) -> Dict[str, str]:
        rows = self.conn.execute("SELECT passage_id, text FROM passages").fetchall()
        return {r[0]: r[1] for r in rows}

    def get_all_entity_ids(self) -> List[str]:
        return [r[0] for r in self.conn.execute("SELECT entity_id FROM entities").fetchall()]

    def get_all_passage_ids(self) -> List[str]:
        return [r[0] for r in
                self.conn.execute("SELECT passage_id FROM passages ORDER BY idx").fetchall()]

    def get_passage_embeddings(self, dim: int = 384,
                               batch_size: int = 10_000) -> Tuple[List[str], np.ndarray]:
        """passage_ids (ordered by passages.idx) + an (n, dim) float32 matrix.

        Chunked into a preallocated array on purpose: fetching 66K FLOAT[384]
        rows as Python lists in one shot transiently allocates ~600MB of boxed
        floats (66581 * 384 * 24B), which is a real risk on 16GB alongside an
        8B model. ORDER BY p.idx matches get_all_passage_ids(), so row order is
        deterministic and reproducible across runs.
        """
        n = self.conn.execute("SELECT COUNT(*) FROM passage_embeddings").fetchone()[0]
        ids: List[str] = []
        mat = np.empty((n, dim), dtype=np.float32)
        off = 0
        cur = self.conn.execute("""
            SELECT pe.passage_id, pe.embedding
            FROM passage_embeddings pe
            JOIN passages p ON p.passage_id = pe.passage_id
            ORDER BY p.idx""")
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            for pid, emb in rows:
                mat[off] = emb
                ids.append(pid)
                off += 1
        assert off == n, f"passage_embeddings: expected {n} rows, wrote {off}"
        return ids, mat

    def get_entity_passage_adjacency(self) -> Dict[str, List[str]]:
        rows = self.conn.execute("""
            SELECT DISTINCT ese.entity_id, spe.passage_id
            FROM entity_sentence_edges ese
            JOIN sentence_passage_edges spe ON ese.sentence_id = spe.sentence_id
        """).fetchall()
        adj: Dict[str, List[str]] = {}
        for e_id, p_id in rows:
            adj.setdefault(e_id, []).append(p_id)
        return adj

    def save_embeddings(self, entity_embeddings: Dict[str, List[float]],
                        passage_embeddings: Dict[str, List[float]]):
        # Tables were cleared by save_trigraph() earlier in this index() run.
        # embedding is FLOAT[] -> Arrow list<float32>.
        self._bulk_insert("entity_embeddings", {
            "entity_id": pa.array(list(entity_embeddings.keys()), type=pa.string()),
            "embedding": pa.array(list(entity_embeddings.values()),
                                  type=pa.list_(pa.float32())),
        })
        self._bulk_insert("passage_embeddings", {
            "passage_id": pa.array(list(passage_embeddings.keys()), type=pa.string()),
            "embedding": pa.array(list(passage_embeddings.values()),
                                  type=pa.list_(pa.float32())),
        })
        self.conn.commit()

    def save_hyperedges(self, hyperedges: List[Set[str]], weights: List[float]):
        rows = [(h_id, e_id, float(w))
                for h_id, (hyperedge, w) in enumerate(zip(hyperedges, weights))
                for e_id in hyperedge]
        if rows:
            self._bulk_insert("semantic_hyperedges", {
                "hyperedge_id": pa.array([r[0] for r in rows], type=pa.int32()),
                "entity_id": pa.array([r[1] for r in rows], type=pa.string()),
                "weight": pa.array([r[2] for r in rows], type=pa.float32()),
            })
            self.conn.commit()

    def get_entity_labels(self) -> Dict[str, str]:
        """entity_id -> NER label (PERSON/DATE/...), for expansion filtering."""
        return dict(self.conn.execute(
            "SELECT entity_id, label FROM entities").fetchall())

    def load_hyperedge_expansion(self) -> Dict[str, Set[str]]:
        """entity_id -> semantically related entity_ids, reconstructed from DB."""
        rows = self.conn.execute(
            "SELECT hyperedge_id, entity_id FROM semantic_hyperedges").fetchall()
        groups: Dict[int, Set[str]] = {}
        for h_id, e_id in rows:
            groups.setdefault(h_id, set()).add(e_id)
        expansion: Dict[str, Set[str]] = {}
        for members in groups.values():
            for e in members:
                expansion.setdefault(e, set()).update(members - {e})
        return expansion

    def close(self):
        self.conn.close()
