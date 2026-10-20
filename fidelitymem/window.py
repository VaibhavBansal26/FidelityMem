"""Adaptive working-memory window.

Each message gets a query-conditioned retention score
    R(m,q) = w_s*semantic + w_r*recency + w_i*importance + w_t*task_continuity
           + w_c*confidence + w_u*utility
The window grows while older messages stay relevant to an unresolved task and
shrinks when a boundary fires. Boundaries emit *segments* that the extraction
pipeline turns into episodes asynchronously (here: a callback).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

from .text import cosine_counter, estimate_tokens, extract_entities, tf_vector
from .types import Message


@dataclass
class RetentionWeights:
    semantic: float = 0.25
    recency: float = 0.15
    importance: float = 0.20
    task: float = 0.30
    confidence: float = 0.05
    utility: float = 0.05


@dataclass
class Boundary:
    reason: str
    messages: list[Message]


@dataclass
class AdaptiveWindow:
    min_tokens: int = 256
    max_tokens: int = 2000
    weights: RetentionWeights = field(default_factory=RetentionWeights)
    topic_shift_threshold: float = 0.15
    inactivity: timedelta = timedelta(minutes=30)
    on_boundary: Optional[Callable[[Boundary], None]] = None
    _messages: list[Message] = field(default_factory=list)
    _active_tasks: set[str] = field(default_factory=set)
    _last_time: Optional[datetime] = None
    _topic_vec: Counter = field(default_factory=Counter)
    _entity_set: set[str] = field(default_factory=set)
    boundaries_fired: list[str] = field(default_factory=list)

    # ---- public API -------------------------------------------------
    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def tokens(self) -> int:
        return sum(estimate_tokens(m.content) for m in self._messages)

    def start_task(self, task_id: str) -> None:
        self._active_tasks.add(task_id)

    def complete_task(self, task_id: str) -> None:
        self._active_tasks.discard(task_id)
        done = [m for m in self._messages if m.task_id == task_id]
        if done:
            self._emit("task_complete", done)

    def add(self, msg: Message) -> list[Boundary]:
        fired: list[Boundary] = []
        # inactivity boundary
        if self._last_time and msg.timestamp - self._last_time > self.inactivity and self._messages:
            fired.append(self._emit("inactivity", list(self._messages)))
        # topic / entity shift
        if self._messages and self._is_topic_shift(msg):
            fired.append(self._emit("topic_shift", list(self._messages)))
        # decision / tool result signals (importance spike)
        if msg.role == "tool" or msg.metadata.get("decision"):
            msg.importance = max(msg.importance, 0.8)
        self._messages.append(msg)
        self._last_time = msg.timestamp
        self._topic_vec = self._topic_vec + tf_vector(msg.content)  # running topic model
        self._entity_set |= set(extract_entities(msg.content))
        if msg.task_id:
            self._active_tasks.add(msg.task_id)
        # token pressure
        if self.tokens > self.max_tokens:
            fired.extend(self._evict(query=msg.content))
        return fired

    def select(self, query: str, budget_tokens: int) -> list[Message]:
        """Choose which working messages enter context for this query, under budget."""
        if not self._messages:
            return []
        scored = sorted(
            ((self.retention(m, query), m) for m in self._messages),
            key=lambda x: x[0],
            reverse=True,
        )
        chosen, used = [], 0
        for s, m in scored:
            t = estimate_tokens(m.content)
            if used + t <= budget_tokens:
                chosen.append(m)
                used += t
        chosen.sort(key=lambda m: m.timestamp)
        return chosen

    # ---- scoring ------------------------------------------------------
    def retention(self, m: Message, query: str) -> float:
        w = self.weights
        q = tf_vector(query)
        sem = cosine_counter(q, tf_vector(m.content))
        idx = self._messages.index(m) if m in self._messages else 0
        age = len(self._messages) - 1 - idx
        rec = math.exp(-age / 8.0)
        task = 1.0 if (m.task_id and m.task_id in self._active_tasks) else 0.0
        conf = 1.0 if m.trusted else 0.3
        util = float(m.metadata.get("utility", 0.0))
        return (
            w.semantic * sem
            + w.recency * rec
            + w.importance * m.importance
            + w.task * task
            + w.confidence * conf
            + w.utility * util
        )

    # ---- internals ----------------------------------------------------
    def _is_topic_shift(self, msg: Message) -> bool:
        if len(self._messages) < 3:
            return False
        sim = cosine_counter(self._topic_vec, tf_vector(msg.content))
        ents = set(extract_entities(msg.content))
        ent_new = bool(ents) and not (ents & self._entity_set)
        return sim < self.topic_shift_threshold and ent_new

    def _evict(self, query: str) -> list[Boundary]:
        """Drop lowest-retention messages until under budget; emit them as a segment."""
        evicted: list[Message] = []
        while self.tokens > self.max_tokens and len(self._messages) > 1:
            # active-task messages are evicted last, and never below min_tokens
            scored = sorted(self._messages, key=lambda m: (m.task_id in self._active_tasks, self.retention(m, query)))
            victim = scored[0]
            if victim.task_id in self._active_tasks and self.tokens - estimate_tokens(victim.content) < self.min_tokens:
                break
            self._messages.remove(victim)
            evicted.append(victim)
        if evicted:
            evicted.sort(key=lambda m: m.timestamp)
            return [self._emit("token_pressure", evicted)]
        return []

    def _emit(self, reason: str, msgs: list[Message]) -> Boundary:
        b = Boundary(reason=reason, messages=msgs)
        self.boundaries_fired.append(reason)
        if reason in ("topic_shift", "inactivity", "task_complete"):
            for m in msgs:
                if m in self._messages:
                    self._messages.remove(m)
            if reason != "task_complete":
                self._topic_vec = Counter()
                self._entity_set = set()
        if self.on_boundary:
            self.on_boundary(b)
        return b
