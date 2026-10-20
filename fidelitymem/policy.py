"""User-facing memory policy: constraints, not vague percentages."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Mode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    HIGH_RECALL = "high_recall"


class MemoryPolicy(BaseModel):
    max_context_tokens: int = 4000
    working_window_min_tokens: int = 256
    working_window_max_tokens: int = 2000
    minimum_recall_retention: float = Field(0.95, ge=0.0, le=1.0)
    maximum_contradiction_rate: float = Field(0.01, ge=0.0, le=1.0)
    latency_budget_ms: float = 300.0
    mode: Mode = Mode.BALANCED
    max_compression_level: int = 3
    tenant: str = "default"
    namespace: str = "default"

    # cost weights for the controller objective
    alpha_storage: float = 1.0
    beta_latency: float = 1.0
    gamma_tokens: float = 1.0
    delta_noise: float = 0.5

    def mode_defaults(self) -> dict:
        if self.mode == Mode.FAST:
            return {"top_k": 5, "expand_budget": 1, "routes": ["lexical", "vector"], "rerank": False}
        if self.mode == Mode.HIGH_RECALL:
            return {"top_k": 20, "expand_budget": 8, "routes": ["lexical", "vector", "graph", "temporal", "procedural"], "rerank": True}
        return {"top_k": 10, "expand_budget": 3, "routes": ["lexical", "vector", "graph", "temporal"], "rerank": True}
