import json

from fidelitymem.compression import build_probes, compress_with_gate
from fidelitymem.eval.datasets import load_locomo
from fidelitymem.eval.pareto import frontier
from fidelitymem.eval.synthetic import from_real
from fidelitymem.llm import ClaudeCompressor, ClaudeExtractor, ClaudeJudge, MockLLM
from fidelitymem.stores.embedders import best_available, embedder_name
from fidelitymem.types import Message


def _locomo_fixture(tmp_path):
    data = [
        {
            "sample_id": "conv-1",
            "conversation": {
                "speaker_a": "A", "speaker_b": "B",
                "session_1_date_time": "1:56 pm on 8 May, 2023",
                "session_1": [
                    {"speaker": "A", "dia_id": "D1:1", "text": "Hey! Good to see you."},
                    {"speaker": "B", "dia_id": "D1:2", "text": "I went to the LGBTQ support group yesterday, it was great."},
                ],
                "session_2_date_time": "3:10 pm on 25 May, 2023",
                "session_2": [{"speaker": "A", "dia_id": "D2:1", "text": "I ran a charity race on Sunday for mental health."}],
            },
            "qa": [
                {"question": "When did B go to the support group?", "answer": "7 May 2023", "evidence": ["D1:2"], "category": 2},
                {"question": "What did A run?", "answer": "charity race", "evidence": ["D2:1"], "category": 4},
                {"question": "What did B run?", "adversarial_answer": "no information", "evidence": ["D2:1"], "category": 5},
            ],
        }
    ]
    p = tmp_path / "locomo.json"
    p.write_text(json.dumps(data))
    return p


def test_locomo_loader_turn_and_session(tmp_path):
    p = _locomo_fixture(tmp_path)
    b = load_locomo(p)
    assert len(b.episodes) == 3 and len(b.questions) == 3
    assert b.questions[0].gold == ["conv-1:D1:2"] and b.questions[0].qtype == "temporal"
    assert b.episodes[0][2].year == 2023 and b.episodes[0][2].month == 5
    s = load_locomo(p, per_turn=False)
    assert len(s.episodes) == 2 and s.questions[0].gold == ["conv-1:session_1"]
    bench = from_real(b, seed=1)
    assert bench.name == "locomo" and all(bench.gold_indices(it) for it in bench.items)
    assert {it.qtype for it in bench.items} == {"temporal", "single_hop", "adversarial"}


def test_content_gate_profile_is_stricter():
    raw = "Hey Mel, good to see you. After the race I realised that self-care is important. Nothing else happened."
    facts = build_probes(raw, "facts")
    content = build_probes(raw, "content")
    assert len(content.probes) > len(facts.probes)
    gist = "Hey Mel, good to see you."
    assert facts.score(gist) >= content.score(gist)
    res = compress_with_gate(raw, 3, 0.95, profile="content")
    assert "self-care" in res.text


def test_mock_llm_components_round_trip():
    llm = MockLLM()
    c = ClaudeCompressor(llm)
    out = c.compress("Lunch was fine. billing-api is capped at 4821 on 2026-03-04. The weather was nice today.", 3)
    assert "4821" in out and llm.calls == 1
    assert c.compress("Lunch was fine. billing-api is capped at 4821 on 2026-03-04. The weather was nice today.", 3) == out and llm.calls == 1  # cached
    cands = ClaudeExtractor(llm).extract([Message(content="The deployment of ledger-core failed on 2026-02-02 because migration 7 removed a column.")])
    assert cands and "ledger-core" in cands[0].entities
    assert ClaudeJudge(llm).judge("when", "7 May 2023", "On 7 May 2023.") is True
    assert ClaudeJudge(llm).judge("when", "7 May 2023", "I don't know") is False


def test_pareto_frontier():
    rows = [
        {"system": "a", "recall": 1.0, "storage_tokens": 100, "tokens_per_query": 50, "p95_ms": 1.0},
        {"system": "b", "recall": 1.0, "storage_tokens": 200, "tokens_per_query": 60, "p95_ms": 2.0},  # dominated by a
        {"system": "c", "recall": 0.9, "storage_tokens": 50, "tokens_per_query": 20, "p95_ms": 0.5},  # cheaper, on frontier
    ]
    f = {r["system"]: r["pareto"] for r in frontier(rows)}
    assert f == {"a": True, "b": False, "c": True}


def test_embedder_fallback_is_unit_norm():
    e = best_available("hashing")
    v = e.embed(["billing-api runs on PostgreSQL"])
    assert abs(float((v[0] ** 2).sum()) - 1.0) < 1e-4 and "Hashing" in embedder_name(e)
