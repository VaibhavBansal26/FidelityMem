"""Query classification and per-request retrieval planning."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..policy import MemoryPolicy, Mode
from ..text import extract_dates, extract_numbers
from ..types import QueryType

_TEMPORAL = re.compile(r"\b(when|before|after|since|until|last|first|earlier|later|ago|recent|timeline|date|how long|still)\b", re.I)
_MULTIHOP = re.compile(r"\b(why|cause|caused|because|related|depend|connected|chain|which .* (?:uses|depends)|through|via|affect)\b", re.I)
_EXACT = re.compile(r"\b(exact|exactly|how many|how much|number|count|version|value|quote|verbatim|precise|id|line \d+)\b", re.I)
_PROCEDURAL = re.compile(r"\b(how (?:do|should|to)|steps?|procedure|process|rule|convention|checklist|workflow|policy|must)\b", re.I)
_CODE = re.compile(r"\b(pull request|pr\b|diff|review|finding|lint|test(?:s|ing)?|commit|merge|function|class|module|bug)\b", re.I)


def classify_query(q: str) -> QueryType:
    if _CODE.search(q):
        return QueryType.CODE_REVIEW
    if _EXACT.search(q) or (extract_numbers(q) and not _TEMPORAL.search(q)):
        return QueryType.EXACT
    if _TEMPORAL.search(q) or extract_dates(q):
        return QueryType.TEMPORAL
    if _PROCEDURAL.search(q):
        return QueryType.PROCEDURAL
    if _MULTIHOP.search(q):
        return QueryType.MULTI_HOP
    return QueryType.FACTUAL


@dataclass
class RetrievalPlan:
    query_type: QueryType
    routes: list[str]
    top_k: int
    per_route_k: int
    weights: dict[str, float]
    expand_budget: int
    graph_hops: int
    expand_always: bool  # e.g. exact numeric tasks need raw evidence
    recency_halflife_days: float
    memory_token_budget: int
    type_boost: dict[str, float] = field(default_factory=dict)


_BASE_WEIGHTS = {"lexical": 1.0, "vector": 1.0, "trie": 0.8, "graph": 0.6, "temporal": 0.5, "procedural": 0.5, "recency": 0.3, "utility": 0.4, "authority": 0.4}


def plan_for(query: str, policy: MemoryPolicy, overrides: dict | None = None) -> RetrievalPlan:
    qt = classify_query(query)
    d = policy.mode_defaults()
    w = dict(_BASE_WEIGHTS)
    routes = list(d["routes"])
    hops, expand_always, halflife, boost = 1, False, 30.0, {}
    if qt == QueryType.TEMPORAL:
        w["temporal"] = 1.2
        w["recency"] = 0.6
        routes = sorted(set(routes) | {"temporal"})
        boost = {"episodic": 0.3}
    elif qt == QueryType.MULTI_HOP:
        w["graph"] = 1.2
        hops = 2 if policy.mode != Mode.FAST else 1
        routes = sorted(set(routes) | {"graph"})
        boost = {"relational": 0.3, "episodic": 0.1}
    elif qt == QueryType.EXACT:
        w["lexical"] = 1.4
        w["trie"] = 1.5
        routes = sorted(set(routes) | {"trie"})
        expand_always = True
    elif qt == QueryType.PROCEDURAL:
        w["procedural"] = 1.3
        routes = sorted(set(routes) | {"procedural"})
        boost = {"procedural": 0.5}
    elif qt == QueryType.CODE_REVIEW:
        w["lexical"] = 1.2
        w["procedural"] = 0.8
        w["trie"] = 1.2
        routes = sorted(set(routes) | {"procedural", "graph", "trie"})
        boost = {"procedural": 0.3, "semantic": 0.2, "episodic": 0.2}
    top_k = d["top_k"]
    expand = d["expand_budget"]
    if policy.mode == Mode.HIGH_RECALL:
        expand_always = expand_always or qt in (QueryType.EXACT, QueryType.MULTI_HOP)
    plan = RetrievalPlan(
        query_type=qt,
        routes=routes,
        top_k=top_k,
        per_route_k=top_k * 3,
        weights=w,
        expand_budget=expand,
        graph_hops=hops,
        expand_always=expand_always,
        recency_halflife_days=halflife,
        memory_token_budget=max(256, policy.max_context_tokens - policy.working_window_min_tokens),
        type_boost=boost,
    )
    for k, v in (overrides or {}).items():
        if hasattr(plan, k):
            setattr(plan, k, v)
    return plan
