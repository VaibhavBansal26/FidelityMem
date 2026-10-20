"""Three-stage memory validation.

Stage 1  source & consistency: provenance, trust, contradiction, versioning.
Stage 2  relevance & utility: is this worth storing at all?
Stage 3  post-compression fidelity gate (delegated to compression.compress_with_gate).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .extraction import Candidate
from .text import content_tokens, jaccard
from .types import Memory, MemoryType, Relation, ValidationIssue


@dataclass
class Contradiction:
    existing_id: str
    kind: str  # negation | relation | value
    detail: str


def _relation_conflict(a: Relation, b: Relation) -> bool:
    same = a.subject.lower() == b.subject.lower() and a.predicate == b.predicate
    if not same:
        return False
    if a.object.lower() == b.object.lower():
        return a.negated != b.negated
    # same subject+predicate, different object: only a conflict for exclusive predicates
    return a.predicate in {"runs on", "belongs to", "reports to", "is part of"}


def find_contradictions(cand: Candidate, existing: Iterable[Memory]) -> list[Contradiction]:
    out: list[Contradiction] = []
    c_toks = set(content_tokens(cand.text))
    c_neg = bool(cand.negations)
    for mem in existing:
        if not mem.active:
            continue
        for r in cand.relations:
            for r2 in mem.relations:
                if _relation_conflict(r, r2):
                    out.append(Contradiction(mem.id, "relation", f"{r} vs {r2}"))
        # same proposition, flipped polarity
        m_toks = set(content_tokens(mem.compressed_text))
        if jaccard(c_toks, m_toks) >= 0.6 and c_neg != bool(mem.negations):
            out.append(Contradiction(mem.id, "negation", "near-identical statement with opposite polarity"))
        # same entities, conflicting numbers
        if cand.numbers and mem.numbers and set(cand.entities) and set(cand.entities) == set(mem.entities):
            if set(cand.numbers) != set(mem.numbers) and jaccard(c_toks - set(cand.numbers), m_toks - set(mem.numbers)) >= 0.7:
                out.append(Contradiction(mem.id, "value", f"{cand.numbers} vs {mem.numbers}"))
    return out


@dataclass
class Stage1Result:
    issues: list[ValidationIssue] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    confidence: float = 0.7
    types: list[MemoryType] = field(default_factory=list)


def stage1_source_consistency(cand: Candidate, existing: Iterable[Memory], authority: float) -> Stage1Result:
    res = Stage1Result(types=list(cand.types))
    conf = 0.5 + 0.3 * authority + 0.1 * cand.importance
    if not cand.trusted:
        res.issues.append(ValidationIssue(stage=1, code="untrusted_source", detail="content from untrusted source"))
        conf -= 0.2
        if MemoryType.PROCEDURAL in res.types:
            # instructions embedded in untrusted content must never become procedures
            res.types = [t for t in res.types if t != MemoryType.PROCEDURAL] or [MemoryType.EPISODIC]
            res.issues.append(ValidationIssue(stage=1, code="procedural_blocked", detail="untrusted instruction demoted"))
    res.contradictions = find_contradictions(cand, existing)
    for c in res.contradictions:
        res.issues.append(ValidationIssue(stage=1, code=f"contradiction_{c.kind}", detail=f"{c.existing_id}: {c.detail}"))
    if res.contradictions:
        # newer, trusted, >= authority wins: mark old ones superseded (never deleted)
        if cand.trusted:
            res.supersedes = [c.existing_id for c in res.contradictions]
        else:
            conf -= 0.2
    res.confidence = max(0.05, min(0.99, conf))
    return res


@dataclass
class Stage2Result:
    accept: bool
    score: float
    issues: list[ValidationIssue] = field(default_factory=list)


def stage2_relevance_utility(cand: Candidate, existing: Iterable[Memory], min_score: float = 0.25) -> Stage2Result:
    """Score durability of a candidate; drop near-duplicates and chatter."""
    issues: list[ValidationIssue] = []
    toks = set(content_tokens(cand.text))
    if len(toks) < 3:
        return Stage2Result(False, 0.0, [ValidationIssue(stage=2, code="too_short", detail="no content")])
    # near-duplicate
    for mem in existing:
        if (
            mem.active
            and jaccard(toks, set(content_tokens(mem.compressed_text))) >= 0.85
            and set(cand.entities) == set(mem.entities)
            and set(cand.numbers) == set(mem.numbers)
            and set(cand.dates) == set(mem.dates)
        ):
            return Stage2Result(False, 0.0, [ValidationIssue(stage=2, code="duplicate", detail=mem.id)])
    facts = len(cand.entities) + len(cand.numbers) + len(cand.dates) + len(cand.relations) + len(cand.negations)
    score = 0.3 * cand.importance + 0.4 * min(1.0, facts / 4) + 0.3 * min(1.0, len(toks) / 12)
    if MemoryType.PROCEDURAL in cand.types or MemoryType.RELATIONAL in cand.types:
        score += 0.15
    if cand.role == "assistant":
        score -= 0.1
    if score < min_score:
        issues.append(ValidationIssue(stage=2, code="low_utility", detail=f"score={score:.2f}"))
    return Stage2Result(score >= min_score, score, issues)
