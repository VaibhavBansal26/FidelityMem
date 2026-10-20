"""Token trie for exact / prefix lookup of identifiers and entity spans.

Motivation (user suggestion): a sliding window handles recency, a trie
handles *exact* recall of identifiers, versions, error codes and multi-word
entity spans with prefix matching ("billing-" -> billing-api, billing-worker),
which BM25 and dense retrieval both handle poorly. The trie is tiny, O(len)
per lookup, and fully deterministic.

Two tries are kept:
  * a character trie over normalised identifier tokens (entities, numbers,
    dates, kebab/snake/dotted names) -> memory ids, for prefix lookup;
  * a word-level trie over 2..4-gram phrases from the text -> memory ids, for
    exact span matching (a quoted phrase or a multi-word entity).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..text import extract_dates, extract_entities, extract_numbers, tokenize


@dataclass
class _Node:
    children: dict[str, "_Node"] = field(default_factory=dict)
    ids: set[str] = field(default_factory=set)


class TrieIndex:
    def __init__(self, max_ngram: int = 4):
        self._chars = _Node()
        self._words = _Node()
        self.max_ngram = max_ngram
        self._doc_keys: dict[str, tuple[list[str], list[tuple[str, ...]]]] = {}

    # ---- build -------------------------------------------------------
    @staticmethod
    def _identifiers(text: str) -> list[str]:
        ids = set(e.lower() for e in extract_entities(text))
        ids |= set(n.lower() for n in extract_numbers(text))
        ids |= set(d.lower() for d in extract_dates(text))
        # any token with a separator or a digit is identifier-like
        ids |= {t for t in tokenize(text) if any(c in t for c in "-_.") or any(c.isdigit() for c in t)}
        return sorted(ids)

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._doc_keys:
            self.remove(doc_id)
        idents = self._identifiers(text)
        for ident in idents:
            node = self._chars
            for ch in ident:
                node = node.children.setdefault(ch, _Node())
                node.ids.add(doc_id)  # every prefix node knows its docs
        toks = tokenize(text)
        grams: list[tuple[str, ...]] = []
        for n in range(2, self.max_ngram + 1):
            for i in range(len(toks) - n + 1):
                g = tuple(toks[i : i + n])
                grams.append(g)
                node = self._words
                for w in g:
                    node = node.children.setdefault(w, _Node())
                node.ids.add(doc_id)
        self._doc_keys[doc_id] = (idents, grams)

    def remove(self, doc_id: str) -> None:
        keys = self._doc_keys.pop(doc_id, None)
        if not keys:
            return
        idents, grams = keys
        for ident in idents:
            node = self._chars
            for ch in ident:
                node = node.children.get(ch)
                if node is None:
                    break
                node.ids.discard(doc_id)
        for g in grams:
            node = self._words
            for w in g:
                node = node.children.get(w)
                if node is None:
                    break
            if node is not None:
                node.ids.discard(doc_id)

    def __len__(self) -> int:
        return len(self._doc_keys)

    # ---- query -------------------------------------------------------
    def prefix(self, p: str) -> set[str]:
        node = self._chars
        for ch in p.lower():
            node = node.children.get(ch)
            if node is None:
                return set()
        return set(node.ids)

    def phrase(self, words: list[str]) -> set[str]:
        node = self._words
        for w in words:
            node = node.children.get(w.lower())
            if node is None:
                return set()
        return set(node.ids)

    def search(self, query: str, k: int = 10, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """Score = exact identifier hits (weight 2) + longest exact phrase hits (weight 1
        per word) + prefix hits (weight 0.5). Only identifier-like query tokens are used
        for the char trie, so common words do not fire."""
        scores: dict[str, float] = defaultdict(float)
        q_idents = self._identifiers(query)
        for ident in q_idents:
            exact = self.prefix(ident)
            full = {d for d in exact if ident in self._doc_keys.get(d, ([], []))[0]}
            for d in full:
                scores[d] += 2.0
            for d in exact - full:
                scores[d] += 0.5
        toks = tokenize(query)
        for n in range(self.max_ngram, 1, -1):
            for i in range(len(toks) - n + 1):
                for d in self.phrase(toks[i : i + n]):
                    scores[d] += 0.5 * n
        if allowed is not None:
            scores = {d: s for d, s in scores.items() if d in allowed}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
