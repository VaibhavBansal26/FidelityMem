"""Core data types for FidelityMem."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class MemoryType(str, Enum):
    WORKING = "working"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    RELATIONAL = "relational"
    ARCHIVAL = "archival"


class QueryType(str, Enum):
    FACTUAL = "factual"
    TEMPORAL = "temporal"
    MULTI_HOP = "multi_hop"
    EXACT = "exact"
    PROCEDURAL = "procedural"
    CODE_REVIEW = "code_review"


class Provenance(BaseModel):
    """Where a memory came from. Always attached; never discarded."""

    source_id: str
    source_type: str = "message"  # message | tool | document | user | system
    tenant: str = "default"
    namespace: str = "default"
    timestamp: datetime = Field(default_factory=now)
    trusted: bool = True
    authority: float = 0.5  # 0..1, how authoritative the source is
    version: int = 1


class Message(BaseModel):
    """An interaction event entering the working window."""

    id: str = Field(default_factory=lambda: new_id("msg"))
    role: str = "user"
    content: str
    timestamp: datetime = Field(default_factory=now)
    task_id: Optional[str] = None
    importance: float = 0.5
    trusted: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class Episode(BaseModel):
    """Immutable raw evidence stored in the archival tier."""

    id: str = Field(default_factory=lambda: new_id("ep"))
    text: str
    provenance: Provenance
    message_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now)


class Relation(BaseModel):
    subject: str
    predicate: str
    object: str
    negated: bool = False


class Memory(BaseModel):
    """A derived, compressed, searchable memory with a pointer back to its raw source."""

    id: str = Field(default_factory=lambda: new_id("mem"))
    episode_id: str
    types: list[MemoryType]
    compressed_text: str
    compression_level: int = 0
    entities: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    negations: list[str] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    provenance: Provenance
    confidence: float = 0.7
    valid_from: datetime = Field(default_factory=now)
    valid_to: Optional[datetime] = None
    superseded_by: Optional[str] = None
    utility: float = 0.0  # learned from feedback
    retrieval_count: int = 0
    used_count: int = 0
    fidelity_score: float = 1.0
    created_at: datetime = Field(default_factory=now)

    @property
    def active(self) -> bool:
        return self.valid_to is None and self.superseded_by is None


class RetrievedMemory(BaseModel):
    memory: Memory
    score: float
    sources: dict[str, float] = Field(default_factory=dict)  # per-route scores
    expanded: bool = False
    raw_text: Optional[str] = None

    @property
    def context_text(self) -> str:
        return self.raw_text if self.expanded and self.raw_text else self.memory.compressed_text


class ValidationIssue(BaseModel):
    stage: int
    code: str
    detail: str
    fatal: bool = False


class ValidationReport(BaseModel):
    accepted: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    confidence: float = 0.7
    fidelity_score: float = 1.0
    compression_level: int = 0


class ContextBundle(BaseModel):
    """What the agent actually sees."""

    query: str
    query_type: QueryType
    working: list[Message]
    retrieved: list[RetrievedMemory]
    tokens_used: int
    token_budget: int
    expansions: int
    plan: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        parts = []
        if self.retrieved:
            parts.append("## Retrieved memory")
            for r in self.retrieved:
                tag = "raw" if r.expanded else f"L{r.memory.compression_level}"
                parts.append(f"- [{r.memory.id}|{tag}|conf={r.memory.confidence:.2f}] {r.context_text}")
        if self.working:
            parts.append("## Working context")
            for m in self.working:
                parts.append(f"{m.role}: {m.content}")
        return "\n".join(parts)
