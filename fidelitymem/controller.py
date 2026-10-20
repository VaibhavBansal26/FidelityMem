"""Recall-constrained controller.

The controller chooses a *configuration*
    (compression_level, top_k, expand_budget, routes)
to minimise   cost = alpha*storage + beta*latency + gamma*tokens + delta*noise
subject to    RecallRetention@K >= rho,  contradiction_rate <= eps.

Recall retention cannot be measured online per query (no gold labels), so the
controller works with a *calibration set* of probe queries with known gold
episode ids (built from the archive itself: for each sampled episode, generate
a query from its raw facts). Compressed-index recall is measured against raw-
index recall on the same queries. Policies are versioned; a candidate policy is
promoted only if it satisfies the constraints on the held-out split, and the
previous policy is kept for rollback.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Callable

from .policy import MemoryPolicy
from .text import extract_dates, extract_entities, extract_numbers, sentences

if TYPE_CHECKING:  # pragma: no cover
    from .memory import FidelityMem


@dataclass(frozen=True)
class Config:
    compression_level: int = 2
    top_k: int = 10
    expand_budget: int = 3
    routes: tuple[str, ...] = ("lexical", "vector", "graph", "temporal")

    def as_overrides(self) -> dict:
        return {"top_k": self.top_k, "expand_budget": self.expand_budget, "routes": list(self.routes), "per_route_k": self.top_k * 3}


@dataclass
class Measurement:
    config: Config
    recall_raw: float
    recall_compressed: float
    retention: float
    contradiction_rate: float
    tokens: float
    latency_ms: float
    storage_ratio: float
    noise: float  # 1 - precision proxy

    def cost(self, p: MemoryPolicy) -> float:
        return p.alpha_storage * self.storage_ratio + p.beta_latency * self.latency_ms / 100.0 + p.gamma_tokens * self.tokens / 1000.0 + p.delta_noise * self.noise

    def feasible(self, p: MemoryPolicy) -> bool:
        return self.retention >= p.minimum_recall_retention and self.contradiction_rate <= p.maximum_contradiction_rate


@dataclass
class ProbeQuery:
    query: str
    gold_episode_ids: list[str]


def make_probe_queries(mem: "FidelityMem", n: int, seed: int = 0) -> list[ProbeQuery]:
    """Self-supervised probes: a fact-bearing sentence from a raw episode, paraphrased
    by dropping function words, must retrieve that episode."""
    rng = random.Random(seed)
    eps = list(mem.archive)
    rng.shuffle(eps)
    out: list[ProbeQuery] = []
    for ep in eps:
        if len(out) >= n:
            break
        # pick the most fact-dense sentence: that is what a user would ask about.
        # A probe needs a named entity *and* either a value (number/date) or a second
        # entity; boilerplate with a weekday in it does not qualify.
        def density(x: str) -> int:
            ents, vals = extract_entities(x), extract_numbers(x) + extract_dates(x)
            if not ents or (len(ents) < 2 and not vals):
                return 0
            return len(ents) + 2 * len(vals)

        sents = sorted(sentences(ep.text), key=density, reverse=True)
        if not sents or density(sents[0]) == 0:
            continue
        s = sents[0]
        toks = s.split()
        if len(toks) > 6:
            toks = rng.sample(toks, max(4, len(toks) * 2 // 3))
        out.append(ProbeQuery(" ".join(toks), [ep.id]))
    return out


@dataclass
class PolicyVersion:
    version: int
    config: Config
    measurement: Measurement | None
    rejected: bool = False  # a candidate that failed the hold-out gate (kept for audit)


@dataclass
class Controller:
    policy: MemoryPolicy
    current: Config = field(default_factory=Config)
    history: list[PolicyVersion] = field(default_factory=list)
    search_space: dict = field(
        default_factory=lambda: {
            "compression_level": [0, 1, 2, 3],
            "top_k": [5, 10, 20],
            "expand_budget": [0, 2, 5],
            "routes": [("lexical", "vector"), ("lexical", "vector", "trie"), ("lexical", "vector", "graph", "temporal"), ("lexical", "vector", "trie", "graph", "temporal", "procedural")],
        }
    )

    def __post_init__(self):
        d = self.policy.mode_defaults()
        self.current = Config(
            compression_level=min(2, self.policy.max_compression_level), top_k=d["top_k"], expand_budget=d["expand_budget"], routes=tuple(d["routes"])
        )
        self.history.append(PolicyVersion(0, self.current, None))

    def candidates(self) -> list[Config]:
        keys = list(self.search_space)
        out = []
        for combo in itertools.product(*(self.search_space[k] for k in keys)):
            c = Config(**dict(zip(keys, combo)))
            if c.compression_level <= self.policy.max_compression_level:
                out.append(c)
        # group by level so the store is recompressed once per level, not once per candidate
        out.sort(key=lambda c: c.compression_level, reverse=True)
        return out

    def optimise(self, evaluate: Callable[[Config, list[ProbeQuery]], Measurement], probes: list[ProbeQuery], holdout_frac: float = 0.5, seed: int = 0, max_confirmations: int = 8) -> Measurement | None:
        """Guarded offline search.

        1. Evaluate every candidate on the tuning split; keep those that satisfy the
           recall-retention and contradiction constraints.
        2. Walk the feasible set in ascending cost and confirm on the held-out split;
           the first candidate that is also feasible there is promoted.
        3. If none confirms, fall back to the safest config (raw, wide). Rejected
           candidates stay in the history for audit; the previous policy is never lost,
           so `rollback()` always has a target.
        Returns the measurement of the policy that is active afterwards."""
        rng = random.Random(seed)
        probes = list(probes)
        rng.shuffle(probes)
        cut = max(1, int(len(probes) * (1 - holdout_frac)))
        tune, hold = probes[:cut], probes[cut:] or probes[:cut]
        feasible: list[tuple[float, Config]] = []
        for c in self.candidates():
            m = evaluate(c, tune)
            if m.feasible(self.policy):
                feasible.append((m.cost(self.policy), c))
        feasible.sort(key=lambda x: x[0])
        rejected_levels: set[int] = set()
        tried = 0
        for _, cfg in feasible:
            if tried >= max_confirmations:
                break
            if cfg.compression_level in rejected_levels:
                continue  # retention is level-driven; do not re-confirm a level that failed
            tried += 1
            confirm = evaluate(cfg, hold)
            if confirm.feasible(self.policy):
                self._promote(cfg, confirm)
                return confirm
            rejected_levels.add(cfg.compression_level)
            self.history.append(PolicyVersion(self.history[-1].version, cfg, confirm, rejected=True))
        safe = Config(0, max(self.search_space["top_k"]), max(self.search_space["expand_budget"]), tuple(self.search_space["routes"][-1]))
        m = evaluate(safe, hold)
        self._promote(safe, m)
        return m

    def _promote(self, cfg: Config, m: Measurement) -> None:
        self.current = cfg
        self.history.append(PolicyVersion(self.history[-1].version + 1, cfg, m))

    def rollback(self) -> Config:
        """Reinstate the previously promoted policy (rejected candidates are skipped)."""
        promoted = [h for h in self.history if not h.rejected]
        if len(promoted) >= 2:
            prev = promoted[-2]
            self.current = prev.config
            self.history.append(PolicyVersion(prev.version, prev.config, prev.measurement))
        return self.current

    def describe(self) -> dict:
        active = [h for h in self.history if not h.rejected][-1]
        return {"version": active.version, "config": asdict(self.current), "rejected_candidates": sum(1 for h in self.history if h.rejected)}
