"""Lightweight, dependency-free text analysis used across the system.

Everything here is deterministic. These extractors power the fidelity probes,
so they are intentionally conservative (high precision over recall).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-\.']*")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")

STOPWORDS = set(
    """a an the and or but if then else of to in on at by for with from as is are was were be been
    being this that these those it its it's i you he she we they them his her our their your my me
    not no nor do does did done has have had having will would can could should may might must
    about into over under after before between through during without within up down out off
    so than too very just also only own same such which who whom whose what when where why how
    all any both each few more most other some because while until against""".split()
)

NEGATION_CUES = ("not", "no", "never", "n't", "none", "cannot", "without", "neither", "nor", "fails", "failed")

_NUMBER = re.compile(
    r"(?<![A-Za-z])(?:\$|€|£)?\d[\d,]*(?:\.\d+)?(?:%|x|k|K|M|B|ms|s|GB|MB|TB)?(?![A-Za-z])"
)
_DATE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}(?:T[\d:]+Z?)?|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.? \d{1,2}(?:,? \d{4})?|"
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)s?|"
    r"(?:yesterday|today|tomorrow|last week|next week|last month|next month|last year|next year))\b",
    re.IGNORECASE,
)
# Capitalised tokens not at sentence start, snake/kebab identifiers, dotted identifiers, versions.
_ENTITY = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*|[a-z]+(?:[-_][a-z0-9]+)+|[a-z]+\.[a-z_]+(?:\.[a-z_]+)*|v\d+(?:\.\d+)+)\b"
)


def tokenize(text: str) -> list[str]:
    return [t.lower().strip(".'") for t in _WORD.findall(text) if t.strip(".'")]


def stem(t: str) -> str:
    """Very light suffix stripping; enough to align 'runs'/'run', 'deployed'/'deploy'."""
    if len(t) > 5 and t.endswith("ing"):
        return t[:-3]
    if len(t) > 4 and t.endswith("ed"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


@lru_cache(maxsize=200_000)
def _content_tokens_cached(text: str) -> tuple[str, ...]:
    return tuple(stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 1)


def content_tokens(text: str) -> list[str]:
    return list(_content_tokens_cached(text))


def estimate_tokens(text: str) -> int:
    """Rough LLM token estimate (~4 chars/token, floor on word count)."""
    if not text:
        return 0
    return max(len(text) // 4, len(text.split()))


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT.split(text) if s and s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def extract_numbers(text: str) -> list[str]:
    return sorted({m.group(0) for m in _NUMBER.finditer(text)})


def extract_dates(text: str) -> list[str]:
    return sorted({m.group(0) for m in _DATE.finditer(text)})


def extract_entities(text: str) -> list[str]:
    found = set()
    for sent in sentences(text):
        words = sent.split()
        for i, w in enumerate(words):
            w_clean = w.strip(",.;:()[]\"'!?")
            if not w_clean:
                continue
            m = _ENTITY.fullmatch(w_clean)
            if not m:
                continue
            # skip sentence-initial plainly-capitalised words ("Notes", "Several") unless
            # they look like identifiers or recur capitalised mid-sentence; a word after a
            # speaker label ("Caroline: Hey") is sentence-initial too
            after_label = i > 0 and words[i - 1].endswith(":")
            if (i == 0 or after_label) and w_clean[0].isupper() and not re.search(r"[-_.\d]", w_clean):
                if not re.search(rf"(?<!^)(?<![.!?]\s){re.escape(w_clean)}\b", text[text.find(sent) + 1 :] if False else " ".join(words[1:])):
                    continue
            if w_clean.lower() in STOPWORDS:
                continue
            if len(w_clean) < 3:
                continue
            found.add(w_clean)
    return sorted(found)


def extract_negations(text: str) -> list[str]:
    """Return sentences containing a negation cue; these must survive compression."""
    out = []
    for s in sentences(text):
        low = s.lower()
        if any(re.search(rf"(?:^|\W){re.escape(c)}(?:\W|$)", low) for c in NEGATION_CUES):
            out.append(s)
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap(a: set, b: set) -> float:
    """Fraction of a covered by b."""
    if not a:
        return 1.0
    return len(a & b) / len(a)


def tf_vector(text: str) -> Counter:
    return Counter(content_tokens(text))


def cosine_counter(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    num = sum(a[k] * b[k] for k in a if k in b)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0
