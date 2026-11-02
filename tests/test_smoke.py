"""Smoke test: Tri-Graph + PPR retrieval end-to-end on a toy corpus (no LLM).

Run: pytest tests/test_smoke.py -v
Uses en_core_web_sm so it runs fast; install with:
    python -m spacy download en_core_web_sm
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

PASSAGES = [
    "Apple Inc. was founded by Steve Jobs and Steve Wozniak in 1976. The company is headquartered in Cupertino, California.",
    "Steve Jobs also founded NeXT in 1985 after leaving Apple. NeXT was later acquired by Apple in 1997.",
    "Microsoft was founded by Bill Gates and Paul Allen in 1975. It is based in Redmond, Washington.",
    "The iPhone was introduced by Apple in 2007 and revolutionized the smartphone industry.",
]


@pytest.fixture(scope="module")
def trigraph():
    from src.indexer.ner_extractor import NERExtractor
    from src.indexer.sentence_splitter import SentenceSplitter
    from src.indexer.trigraph_builder import TriGraphBuilder
    tg = TriGraphBuilder()
    tg.build_from_passages(PASSAGES, SentenceSplitter(), NERExtractor("en_core_web_sm"))
    return tg


def test_trigraph_builds(trigraph):
    assert len(trigraph.passages) == 4
    assert len(trigraph.sentences) >= 4
    assert len(trigraph.entities) > 0
    assert len(trigraph.entity_sentence_edges) > 0


def test_ppr_retrieval(trigraph):
    from src.retrieval.ppr_retrieval import PPRRetriever
    from src.indexer.trigraph_builder import TriGraphBuilder

    ep_adj = trigraph.get_entity_passage_bipartite()
    entity_ids = list(trigraph.entities.keys())
    passage_ids = list(trigraph.passages.keys())
    passage_texts = {p: d["text"] for p, d in trigraph.passages.items()}

    query_entity = TriGraphBuilder.hash_entity_id("Steve Jobs")
    ppr = PPRRetriever(top_k=2)
    results = ppr.retrieve([query_entity], ep_adj, entity_ids, passage_ids, passage_texts)

    assert len(results) > 0
    top_texts = " ".join(r[2] for r in results)
    assert "Steve Jobs" in top_texts  # Jobs-related passages should rank first


def test_duckdb_roundtrip(tmp_path, trigraph):
    from src.storage.duckdb_store import DuckDBGraphStore
    store = DuckDBGraphStore(str(tmp_path / "test.duckdb"))
    store.save_trigraph(trigraph)
    assert len(store.get_all_passages()) == 4
    assert len(store.get_entity_passage_adjacency()) > 0
    store.close()
