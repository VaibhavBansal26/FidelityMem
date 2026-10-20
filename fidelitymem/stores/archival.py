"""Immutable raw-evidence store. In-memory with optional JSONL persistence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from ..types import Episode


class ArchivalStore:
    def __init__(self, path: Optional[str | Path] = None):
        self._items: dict[str, Episode] = {}
        self._path = Path(path) if path else None
        if self._path and self._path.exists():
            for line in self._path.read_text().splitlines():
                if line.strip():
                    ep = Episode.model_validate_json(line)
                    self._items[ep.id] = ep

    def put(self, ep: Episode) -> Episode:
        if ep.id in self._items:
            raise ValueError(f"episode {ep.id} is immutable and already exists")
        self._items[ep.id] = ep
        if self._path:
            with self._path.open("a") as f:
                f.write(ep.model_dump_json() + "\n")
        return ep

    def get(self, ep_id: str) -> Optional[Episode]:
        return self._items.get(ep_id)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Episode]:
        return iter(self._items.values())

    def storage_bytes(self) -> int:
        return sum(len(e.text.encode()) for e in self._items.values())
