"""Reversible compression ladder with deterministic fidelity probes.

Levels
  0  raw text (no loss)
  1  extractive: keep the most salient sentences (~60%)
  2  key facts: keep only sentences carrying entities / numbers / dates / negations (~35%)
  3  gist: a single salient sentence (~15%)

An optional `Compressor` protocol lets an LLM provide abstractive text at any
level; the fidelity gate applies regardless of who produced the text.

Fidelity probes are *generated from the raw source*, not from the compressed
text, so the gate measures what was lost rather than what was kept.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .text import (
    content_tokens,
    estimate_tokens,
    extract_dates,
    extract_entities,
    extract_negations,
    extract_numbers,
    sentences,
    tf_vector,
    cosine_counter,
)

MAX_LEVEL = 3


class Compressor(Protocol):
    def compress(self, text: str, level: int) -> str: ...


@dataclass
class Probe:
    kind: str  # entity | number | date | negation | evidence
    target: str
    weight: float = 1.0

    def satisfied_by(self, text: str) -> bool:
        low = text.lower()
        t = self.target.lower()
        if self.kind == "negation":
            # the negation sentence's content words + a negation cue must survive
            words = set(content_tokens(self.target))
            cues = ("not", "no", "never", "n't", "none", "cannot", "without", "fail")
            keep = sum(1 for w in words if w in low) / max(1, len(words))
            return keep >= 0.6 and any(c in low for c in cues)
        return t in low


@dataclass
class ProbeSet:
    probes: list[Probe] = field(default_factory=list)

    def score(self, text: str) -> float:
        if not self.probes:
            return 1.0
        tot = sum(p.weight for p in self.probes)
        ok = sum(p.weight for p in self.probes if p.satisfied_by(text))
        return ok / tot

    def failures(self, text: str) -> list[Probe]:
        return [p for p in self.probes if not p.satisfied_by(text)]


def build_probes(raw: str) -> ProbeSet:
    ps: list[Probe] = []
    for e in extract_entities(raw):
        ps.append(Probe("entity", e, 1.0))
    for n in extract_numbers(raw):
        ps.append(Probe("number", n, 1.5))
    for d in extract_dates(raw):
        ps.append(Probe("date", d, 1.5))
    for s in extract_negations(raw):
        ps.append(Probe("negation", s, 2.0))
    return ProbeSet(ps)


def _salience(sent: str, doc_vec, idx: int, n: int) -> float:
    v = tf_vector(sent)
    central = cosine_counter(v, doc_vec)
    facts = len(extract_numbers(sent)) + len(extract_dates(sent)) + 0.5 * len(extract_entities(sent))
    neg = 1.0 if extract_negations(sent) else 0.0
    pos = 1.0 - (idx / max(1, n))  # lead bias
    return central + 0.25 * facts + 0.5 * neg + 0.1 * pos


class ExtractiveCompressor:
    """Deterministic compressor; the default when no LLM is configured."""

    ratios = {1: 0.6, 2: 0.35, 3: 0.15}

    def compress(self, text: str, level: int) -> str:
        if level <= 0:
            return text
        sents = sentences(text)
        if len(sents) <= 1:
            return text
        doc_vec = tf_vector(text)
        n = len(sents)
        scored = [(_salience(s, doc_vec, i, n), i, s) for i, s in enumerate(sents)]
        if level == 2:
            # key facts: sentences with hard facts, fall back to top salience
            keep = [i for _, i, s in scored if extract_numbers(s) or extract_dates(s) or extract_negations(s) or extract_entities(s)]
            if not keep:
                keep = [i for _, i, _ in sorted(scored, reverse=True)[: max(1, int(n * 0.35))]]
            keep = keep[: max(1, int(n * 0.6))]
        else:
            k = max(1, round(n * self.ratios[level]))
            keep = [i for _, i, _ in sorted(scored, key=lambda x: x[0], reverse=True)[:k]]
        keep = sorted(set(keep))
        return " ".join(sents[i] for i in keep)


@dataclass
class CompressionResult:
    text: str
    level: int
    fidelity: float
    ratio: float  # tokens_compressed / tokens_raw
    attempts: list[tuple[int, float]]
    failures: list[Probe]


def compress_with_gate(
    raw: str,
    target_level: int,
    min_fidelity: float,
    compressor: Compressor | None = None,
    probes: ProbeSet | None = None,
) -> CompressionResult:
    """Compress at `target_level`, backing off to lower levels until the fidelity
    gate passes. Level 0 always passes."""
    compressor = compressor or ExtractiveCompressor()
    probes = probes or build_probes(raw)
    raw_tokens = max(1, estimate_tokens(raw))
    attempts: list[tuple[int, float]] = []
    for level in range(min(target_level, MAX_LEVEL), -1, -1):
        text = raw if level == 0 else compressor.compress(raw, level)
        fid = probes.score(text)
        attempts.append((level, fid))
        if fid >= min_fidelity or level == 0:
            return CompressionResult(
                text=text,
                level=level,
                fidelity=fid,
                ratio=estimate_tokens(text) / raw_tokens,
                attempts=attempts,
                failures=probes.failures(text),
            )
    raise RuntimeError("unreachable")
