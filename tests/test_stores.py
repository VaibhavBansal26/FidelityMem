import pytest

from fidelitymem.stores import ArchivalStore, BM25Index, GraphStore, TrieIndex, VectorIndex
from fidelitymem.types import Episode, Provenance, Relation


def test_bm25_add_remove_search():
    ix = BM25Index()
    ix.add("a", "billing-api runs on PostgreSQL")
    ix.add("b", "the cafeteria serves tacos on Tuesdays")
    assert ix.search("billing-api PostgreSQL")[0][0] == "a"
    ix.remove("a")
    assert all(d != "a" for d, _ in ix.search("billing-api"))
    assert len(ix) == 1


def test_vector_index_fuzzy_match_and_allowed_filter():
    vi = VectorIndex()
    vi.add_many([("a", "billing-api connection pool exhausted"), ("b", "quarterly roadmap planning offsite")])
    assert vi.search("biling-api conection pool")[0][0] == "a"  # typo tolerant via char n-grams
    assert [d for d, _ in vi.search("billing-api", allowed={"b"})] == ["b"]
    vi.remove("a")
    assert len(vi) == 1


def test_graph_relation_following_beats_co_mention():
    g = GraphStore()
    g.add_memory("hop1", ["billing-api", "auth-service"], [Relation(subject="billing-api", predicate="depends on", object="auth-service")])
    g.add_memory("hop2", ["auth-service", "PostgreSQL"], [Relation(subject="auth-service", predicate="runs on", object="PostgreSQL")])
    for i in range(10):
        g.add_memory(f"noise{i}", ["auth-service", "Mondays"], [])
    ranked = dict(g.expand(["billing-api"], hops=2))
    assert ranked["hop1"] > ranked["hop2"] > ranked["noise0"]
    assert g.successors("billing-api") == ["auth-service"]


def test_trie_exact_prefix_and_phrase():
    t = TrieIndex()
    t.add("a", "billing-api is capped at 4821 and owned by Priya")
    t.add("b", "billing-worker runs on Redis")
    assert t.prefix("billing-") == {"a", "b"}
    assert t.search("4821")[0][0] == "a"
    assert t.search("capped at 4821")[0][0] == "a"
    t.remove("a")
    assert t.prefix("4821") == set()


def test_archival_is_immutable(tmp_path):
    store = ArchivalStore(tmp_path / "ep.jsonl")
    ep = Episode(text="raw", provenance=Provenance(source_id="s"))
    store.put(ep)
    with pytest.raises(ValueError):
        store.put(ep)
    again = ArchivalStore(tmp_path / "ep.jsonl")
    assert again.get(ep.id).text == "raw"
