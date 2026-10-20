"""Synthetic long-history benchmark with gold evidence ids and realistic distractors.

Why synthetic: public benchmarks (LoCoMo, LongMemEval) need an LLM to answer
and judge. This generator produces *retrieval-level* ground truth (which raw
episodes carry the evidence) so Recall@K and RecallRetention can be computed
exactly, offline, at any scale, with the same evidence set and progressively
more distractors. It covers four question families: factual, temporal,
multi-hop (two linked episodes) and exact-value.

The generated world is *consistent* (one database per service, one owner),
so the contradiction detector does not legitimately supersede gold evidence.
Episodes are multi-sentence (fact + filler) so compression is a real test:
a gist that keeps the filler and drops the fact must fail the fidelity gate.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

PREFIX = ["billing", "auth", "search", "payments", "notify", "ledger", "report", "ingest", "profile", "rate", "cache", "audit", "sched", "export", "scoring", "fraud", "catalog", "pricing", "inventory", "shipping", "email", "sms", "geo", "media", "feed"]
SUFFIX = ["api", "service", "worker", "indexer", "core", "gateway", "pipeline", "store"]
DBS = ["PostgreSQL", "Redis", "Kafka", "ClickHouse", "S3", "DynamoDB", "Elasticsearch", "MongoDB", "Cassandra"]
PEOPLE = ["Priya", "Marcus", "Elena", "Tomas", "Aiko", "Dmitri", "Sofia", "Kwame", "Lena", "Rafael", "Ingrid", "Yusuf", "Mei", "Olu", "Bianca"]
ACTIONS = ["deployment", "migration", "rollback", "hotfix", "config change", "schema change", "dependency upgrade"]
OUTCOMES = ["failed", "succeeded", "timed out", "was rolled back", "caused an outage", "degraded latency"]
REASONS = ["migration {n} removed a required column", "the connection pool was capped at {n}", "a null check was missing", "the retry budget of {n} was exhausted", "the {db} index was never created", "the feature flag defaulted to off"]
RULES = ["Always run migrations in a transaction before deploying {svc}.", "Never merge changes to {svc} without a regression test.", "Make sure {svc} has a rollback plan before any {action}.", "Do not raise the pool size of {svc} above {n} without load testing."]
FILLER = [
    "The team spent the first part of the sync going over the roadmap for the quarter.",
    "There was a long discussion about whether the retro should move to Thursdays.",
    "Someone mentioned that the coffee machine on the second floor is broken again.",
    "We agreed to revisit the onboarding checklist once the new hires start.",
    "A few people were out, so the design review was pushed to next week.",
    "The hackathon planning thread is getting long and nobody has volunteered to lead it.",
    "As usual the dashboard colours came up and as usual nothing was decided.",
    "The parking situation near the office is still a mess on Mondays.",
    "Notes from the vendor call will be shared in the channel later.",
    "Several people asked for the wellness budget to be clarified before the offsite.",
]
DISTRACTOR_TOPICS = ["Lunch options near the office include {x} and {y}.", "Reminder: the {x} meeting moved to {day}.", "The {x} team is hiring two engineers this quarter.", "Someone left a {x} on the third floor.", "Weekly metrics: {n} tickets closed, {m} opened.", "Reading list: a paper on {x} and another on {y}.", "The {x} dashboard needs a new colour scheme.", "Parking lot {x} is closed on {day}."]
WORDS = ["coffee", "kanban", "retro", "offsite", "wellness", "budget", "design", "onboarding", "latency", "roadmap", "hiring", "umbrella", "scooter", "notebook", "hackathon"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


@dataclass
class BenchItem:
    query: str
    gold: list[str]  # raw texts of gold episodes
    qtype: str
    answer: str = ""
    gold_idx: list[int] | None = None  # episode indices; authoritative when set (texts may repeat)


@dataclass
class Benchmark:
    episodes: list[tuple[str, datetime]]  # (text, timestamp) in ingestion order
    items: list[BenchItem]
    text_to_idx: dict[str, int] = field(default_factory=dict)
    name: str = "synthetic"

    def gold_indices(self, item: BenchItem) -> list[int]:
        if item.gold_idx is not None:
            return list(item.gold_idx)
        return [self.text_to_idx[g] for g in item.gold]


def from_real(bench, seed: int = 0, shuffle: bool = True) -> Benchmark:
    """Adapt a RealBench (eval.datasets) to the Benchmark schema used by every runner.
    Ingestion order is shuffled per seed so seeds measure order sensitivity too."""
    import random

    eps = list(bench.episodes)
    if shuffle:
        random.Random(seed).shuffle(eps)
    idx_of = {eid: i for i, (eid, _, _) in enumerate(eps)}
    episodes = [(text, ts) for _, text, ts in eps]
    items = []
    for q in bench.questions:
        gi = [idx_of[g] for g in q.gold if g in idx_of]
        if not gi and q.qtype != "adversarial":
            continue
        items.append(BenchItem(q.question, [episodes[i][0] for i in gi], q.qtype, q.answer, gi))
    return Benchmark(episodes, items, {}, name=bench.name)


def _wrap(rng: random.Random, fact: str, n_filler: tuple[int, int] = (2, 4)) -> str:
    """Embed a fact sentence in filler so compression has something to lose."""
    k = rng.randint(*n_filler)
    fill = rng.sample(FILLER, k)
    pos = rng.randint(0, k)
    fill.insert(pos, fact)
    return " ".join(fill)


def generate(n_memories: int = 1000, n_queries: int = 100, seed: int = 0, filler: bool = True) -> Benchmark:
    rng = random.Random(seed)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    services = [f"{p}-{s}" for p in PREFIX for s in SUFFIX]
    rng.shuffle(services)
    world_db = {s: rng.choice(DBS) for s in services}
    world_owner = {s: rng.choice(PEOPLE) for s in services}
    world_dep = {s: rng.choice([x for x in services if x != s]) for s in services}
    used_ids: set[int] = set()

    def uid() -> int:
        while True:
            n = rng.randint(100, 9999)
            if n not in used_ids:
                used_ids.add(n)
                return n

    def wrap(fact: str) -> str:
        return _wrap(rng, fact) if filler else fact

    episodes: list[tuple[str, datetime]] = []
    items: list[BenchItem] = []
    svc_iter = iter(services * 10)  # enough unique-ish services; each (svc, kind) pair used once
    used_pairs: set[tuple[str, str]] = set()
    for i in range(n_queries):
        kind = ["factual", "temporal", "multi_hop", "exact"][i % 4]
        svc = next(svc_iter)
        while (svc, kind) in used_pairs:
            svc = next(svc_iter)
        used_pairs.add((svc, kind))
        db, who = world_db[svc], world_owner[svc]
        ts = t0 + timedelta(days=rng.randint(0, 200), hours=rng.randint(0, 23))
        if kind == "factual":
            fact = f"{svc} stores its session data in {db} and is maintained by {who}."
            txt = wrap(fact)
            episodes.append((txt, ts))
            items.append(BenchItem(f"where does {svc} store session data", [txt], kind, db))
        elif kind == "temporal":
            action, outcome = rng.choice(ACTIONS), rng.choice(OUTCOMES)
            reason = rng.choice(REASONS).format(n=uid(), db=db)
            fact = f"On {ts.strftime('%Y-%m-%d')} the {action} of {svc} {outcome} because {reason}. {who} resolved it."
            txt = wrap(fact)
            episodes.append((txt, ts))
            items.append(BenchItem(f"when did the {action} of {svc} {outcome.split()[0]}", [txt], kind, ts.strftime("%Y-%m-%d")))
        elif kind == "multi_hop":
            svc2 = world_dep[svc]
            t1 = wrap(f"{svc} depends on {svc2} for request validation.")
            t2 = wrap(f"{svc2} runs on {world_db[svc2]} and was last upgraded by {world_owner[svc2]}.")
            episodes.append((t1, ts))
            episodes.append((t2, ts + timedelta(days=1)))
            items.append(BenchItem(f"which database does the service that {svc} depends on run on", [t1, t2], kind, world_db[svc2]))
        else:
            n, m = uid(), uid()
            fact = f"The connection pool for {svc} is capped at {n} and the p95 latency target is {m}ms. It must not be raised above {n} without load testing."
            txt = wrap(fact)
            episodes.append((txt, ts))
            items.append(BenchItem(f"what is the exact connection pool cap for {svc}", [txt], kind, str(n)))
    for _ in range(max(1, n_memories // 50)):
        svc = rng.choice(services)
        episodes.append((wrap(rng.choice(RULES).format(svc=svc, action=rng.choice(ACTIONS), n=uid())), t0 + timedelta(days=rng.randint(0, 200))))
    while len(episodes) < n_memories:
        if rng.random() < 0.3:
            svc = rng.choice(services)
            fact = f"{rng.choice(PEOPLE)} asked whether {svc} could also read from {rng.choice(DBS)} for the {rng.choice(WORDS)} experiment; no decision yet."
        else:
            fact = rng.choice(DISTRACTOR_TOPICS).format(x=rng.choice(WORDS), y=rng.choice(WORDS), n=uid(), m=uid(), day=rng.choice(DAYS))
        episodes.append((wrap(fact), t0 + timedelta(days=rng.randint(0, 200), minutes=rng.randint(0, 1000))))
    rng.shuffle(episodes)
    return Benchmark(episodes, items, {t: i for i, (t, _) in enumerate(episodes)})
