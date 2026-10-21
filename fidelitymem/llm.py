"""LLM-backed components: compressor, extractor, reader and judge.

All four speak to Claude through the official Anthropic SDK and plug into the
same protocols the deterministic defaults implement, so the fidelity gate and
the controller apply unchanged. `MockLLM` lets the pipeline and tests run
without credentials.

Credentials resolve from the environment (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN
or an `ant auth login` profile). `available()` reports whether a call can be made.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from .extraction import Candidate, RuleExtractor, classify_types, extract_relations
from .text import extract_dates, extract_entities, extract_negations, extract_numbers
from .types import Message, MemoryType, Relation

DEFAULT_MODEL = "claude-opus-5-5"
JUDGE_MODEL = "claude-sonnet-5-5"


class LLM(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str: ...


@dataclass
class ClaudeLLM:
    model: str = DEFAULT_MODEL
    effort: str = "low"
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _client: object = field(default=None, repr=False)

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        client = self._get_client()
        resp = client.beta.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        self.calls += 1
        self.input_tokens += getattr(resp.usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(resp.usage, "output_tokens", 0) or 0
        if resp.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in resp.content if b.type == "text")


@dataclass
class MockLLM:
    """Deterministic stand-in: compresses extractively, extracts by rule, judges by containment."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        self.calls += 1
        if "COMPRESS" in system:
            from .compression import ExtractiveCompressor

            level = int(re.search(r"level (\d)", user).group(1))
            text = user.split("TEXT:\n", 1)[1]
            return ExtractiveCompressor().compress(text, level)
        if "EXTRACT" in system:
            text = user.split("TEXT:\n", 1)[1]
            return json.dumps([{"text": s.strip(), "types": [t.value for t in classify_types(s)]} for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20])
        if "JUDGE" in system:
            gold = re.search(r"GOLD ANSWER:\n(.*?)\n", user, re.S).group(1).strip().lower()
            pred = user.split("CANDIDATE ANSWER:\n", 1)[1].lower()
            return json.dumps({"correct": gold in pred, "reason": "containment"})
        if "ANSWER" in system:
            return "unknown"
        return ""


def available() -> bool:
    """True only if a credential resolves and a trivial request succeeds."""
    try:
        import anthropic

        anthropic.Anthropic().messages.count_tokens(model=JUDGE_MODEL, messages=[{"role": "user", "content": "ping"}])
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- compressor
_COMPRESS_SYSTEM = (
    "COMPRESS. You compress a memory record for later retrieval. Keep every named entity, number, date, "
    "identifier, negation and causal link exactly as written; drop pleasantries and filler. Output only the compressed text."
)
_LEVEL_GUIDE = {
    1: "level 1: keep about 60% of the content as faithful extractive sentences.",
    2: "level 2: keep only the sentences that state facts (who/what/when/how much); about 35% of the content.",
    3: "level 3: one or two sentences stating the core fact(s) with all names, numbers and dates preserved.",
}


class ClaudeCompressor:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm or ClaudeLLM()
        self.cache: dict[tuple[str, int], str] = {}

    def compress(self, text: str, level: int) -> str:
        if level <= 0:
            return text
        key = (text, level)
        if key not in self.cache:
            out = self.llm.complete(_COMPRESS_SYSTEM, f"Compress at {_LEVEL_GUIDE[min(level, 3)]}\nTEXT:\n{text}", max_tokens=1024).strip()
            self.cache[key] = out or text
        return self.cache[key]


# --------------------------------------------------------------------------- extractor
_EXTRACT_SYSTEM = (
    "EXTRACT. From the conversation segment, list durable memories worth keeping: stable facts, events with outcomes, "
    "procedures or rules, and relationships. Return a JSON list of objects {\"text\": <self-contained sentence>, "
    "\"types\": [semantic|episodic|procedural|relational]}. Preserve names, numbers and dates verbatim. Return [] if nothing durable."
)


class ClaudeExtractor:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm or ClaudeLLM()
        self._fallback = RuleExtractor()

    def extract(self, messages: list[Message]) -> list[Candidate]:
        if not messages:
            return []
        segment = "\n".join(f"{m.role}: {m.content}" for m in messages)
        raw = self.llm.complete(_EXTRACT_SYSTEM, f"TEXT:\n{segment}", max_tokens=2048)
        try:
            items = json.loads(raw[raw.index("[") : raw.rindex("]") + 1])
        except Exception:  # noqa: BLE001
            return self._fallback.extract(messages)
        trusted = all(m.trusted for m in messages)
        importance = max((m.importance for m in messages), default=0.5)
        out: list[Candidate] = []
        for it in items:
            text = str(it.get("text", "")).strip()
            if len(text) < 10:
                continue
            types = [MemoryType(t) for t in it.get("types", []) if t in MemoryType._value2member_map_] or classify_types(text)
            out.append(
                Candidate(
                    text=text, types=types, message_ids=[m.id for m in messages], trusted=trusted, importance=importance,
                    entities=extract_entities(text), numbers=extract_numbers(text), dates=extract_dates(text),
                    negations=extract_negations(text), relations=extract_relations(text), role="user",
                )
            )
        return out


# --------------------------------------------------------------------------- reader + judge
_ANSWER_SYSTEM = (
    "ANSWER. Answer the question using only the provided memory context. Be brief: a name, date, number or short phrase. "
    "If the context does not contain the answer, reply exactly: I don't know."
)
_JUDGE_SYSTEM = (
    "JUDGE. Decide whether the candidate answer is correct given the gold answer. Minor wording or formatting differences are fine; "
    "a different date, number, entity or an 'I don't know' is wrong unless the gold answer is also an abstention. "
    "Return JSON {\"correct\": true|false, \"reason\": <short>}."
)


class ClaudeReader:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm or ClaudeLLM()

    def answer(self, question: str, context: str) -> str:
        return self.llm.complete(_ANSWER_SYSTEM, f"CONTEXT:\n{context}\n\nQUESTION:\n{question}", max_tokens=256).strip()


class ClaudeJudge:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm or ClaudeLLM(model=JUDGE_MODEL)

    def judge(self, question: str, gold: str, candidate: str) -> bool:
        raw = self.llm.complete(_JUDGE_SYSTEM, f"QUESTION:\n{question}\nGOLD ANSWER:\n{gold}\nCANDIDATE ANSWER:\n{candidate}", max_tokens=128)
        try:
            return bool(json.loads(raw[raw.index("{") : raw.rindex("}") + 1]).get("correct"))
        except Exception:  # noqa: BLE001
            return gold.strip().lower() in candidate.lower()


__all__ = ["LLM", "ClaudeLLM", "MockLLM", "ClaudeCompressor", "ClaudeExtractor", "ClaudeReader", "ClaudeJudge", "available", "Relation"]
