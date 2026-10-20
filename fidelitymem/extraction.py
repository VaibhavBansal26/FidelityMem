"""Candidate memory extraction from a window segment.

Rule-based by default (deterministic, cheap, testable). An LLM extractor can
be plugged in via the `Extractor` protocol and must return the same schema.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .text import extract_dates, extract_entities, extract_negations, extract_numbers, sentences
from .types import Message, MemoryType, Relation

_PROCEDURAL = re.compile(
    r"\b(always|never|must|should|make sure|before (?:you|we|deploying|merging)|rule:|convention|policy|checklist|step \d)\b",
    re.I,
)
_EPISODIC = re.compile(
    r"\b(failed|succeeded|deployed|merged|broke|crashed|fixed|happened|ran|resolved|incident|error|outage|rejected|accepted)\b",
    re.I,
)
_SEMANTIC = re.compile(r"\b(is|are|uses|requires|depends on|runs on|stores|has|contains|supports|was built)\b", re.I)
_RELATION = re.compile(
    r"(?P<s>[A-Za-z][\w\-\.]*(?: [A-Z][\w\-]*)*)\s+(?P<p>depends on|uses|calls|owns|requires|runs on|stores (?:in|to)|is part of|belongs to|reports to|maintains|imports)\s+(?P<o>[A-Za-z][\w\-\.]*(?: [A-Z][\w\-]*)*)",
)


@dataclass
class Candidate:
    text: str
    types: list[MemoryType]
    message_ids: list[str]
    trusted: bool
    importance: float
    entities: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    negations: list[str] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    role: str = "user"


class Extractor(Protocol):
    def extract(self, messages: list[Message]) -> list[Candidate]: ...


def classify_types(text: str) -> list[MemoryType]:
    types: list[MemoryType] = []
    if _EPISODIC.search(text):
        types.append(MemoryType.EPISODIC)
    if _PROCEDURAL.search(text):
        types.append(MemoryType.PROCEDURAL)
    if _RELATION.search(text):
        types.append(MemoryType.RELATIONAL)
    if _SEMANTIC.search(text) and MemoryType.EPISODIC not in types:
        types.append(MemoryType.SEMANTIC)
    if not types:
        types.append(MemoryType.EPISODIC if extract_dates(text) else MemoryType.SEMANTIC)
    return types


def extract_relations(text: str) -> list[Relation]:
    rels = []
    for s in sentences(text):
        for m in _RELATION.finditer(s):
            neg = bool(re.search(r"\b(not|no longer|never)\b", s[: m.start("p")], re.I)) or bool(
                re.search(r"\bdoes(?:n't| not)\b", s, re.I)
            )
            rels.append(Relation(subject=m.group("s").strip(), predicate=m.group("p").lower(), object=m.group("o").strip(), negated=neg))
    return rels


class RuleExtractor:
    """One candidate per message that carries durable content."""

    min_chars: int = 20

    def extract(self, messages: list[Message]) -> list[Candidate]:
        out: list[Candidate] = []
        for m in messages:
            text = m.content.strip()
            if len(text) < self.min_chars:
                continue
            if m.role == "assistant" and not (_EPISODIC.search(text) or _PROCEDURAL.search(text)):
                # assistant chatter is rarely durable unless it records an outcome/decision
                if not m.metadata.get("decision"):
                    continue
            out.append(
                Candidate(
                    text=text,
                    types=classify_types(text),
                    message_ids=[m.id],
                    trusted=m.trusted,
                    importance=m.importance,
                    entities=extract_entities(text),
                    numbers=extract_numbers(text),
                    dates=extract_dates(text),
                    negations=extract_negations(text),
                    relations=extract_relations(text),
                    role=m.role,
                )
            )
        return out
