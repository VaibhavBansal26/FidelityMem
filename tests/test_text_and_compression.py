from fidelitymem.compression import ExtractiveCompressor, build_probes, compress_with_gate
from fidelitymem.text import extract_dates, extract_entities, extract_negations, extract_numbers, stem

RAW = (
    "The team spent the first part of the sync going over the roadmap. "
    "On 2026-03-04 the deployment of billing-api failed because migration 18 removed a required column. "
    "The connection pool must not be raised above 240 without load testing. "
    "Someone mentioned the coffee machine is broken again."
)


def test_extractors():
    assert "billing-api" in extract_entities(RAW)
    assert "18" in extract_numbers(RAW) and "240" in extract_numbers(RAW)
    assert "2026-03-04" in extract_dates(RAW)
    negs = extract_negations(RAW)
    assert any("must not be raised" in n for n in negs)
    # sentence-initial capitalised common words are not entities
    assert "Someone" not in extract_entities(RAW)
    assert stem("runs") == "run" and stem("deployed") == "deploy" and stem("class") == "class"


def test_probes_fail_on_lossy_gist():
    probes = build_probes(RAW)
    kinds = {p.kind for p in probes.probes}
    assert {"entity", "number", "date", "negation"} <= kinds
    gist = "The team discussed the roadmap and the coffee machine."
    assert probes.score(gist) < 0.5
    assert probes.score(RAW) == 1.0


def test_gate_backs_off_to_level_that_preserves_facts():
    res = compress_with_gate(RAW, target_level=3, min_fidelity=0.95)
    assert res.fidelity >= 0.95
    assert "2026-03-04" in res.text and "240" in res.text and "not" in res.text
    assert res.ratio < 1.0  # filler was removed
    assert res.attempts[0][0] == 3  # tried the aggressive level first


def test_gate_nothing_to_lose_passes_unchanged():
    res = compress_with_gate("x", target_level=3, min_fidelity=1.0)
    assert res.text == "x" and res.fidelity == 1.0


def test_gate_falls_back_to_raw_when_every_level_loses_facts():
    class Lossy:
        def compress(self, text, level):
            return "nothing here"

    raw = "billing-api is capped at 4821 on 2026-03-04 and must not exceed it."
    res = compress_with_gate(raw, target_level=3, min_fidelity=0.95, compressor=Lossy())
    assert res.level == 0 and res.text == raw and len(res.attempts) == 4


def test_extractive_levels_monotone():
    c = ExtractiveCompressor()
    l1, l2, l3 = (len(c.compress(RAW, lvl)) for lvl in (1, 2, 3))
    assert l3 <= l2 <= l1 <= len(RAW)
