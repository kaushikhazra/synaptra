"""Pydantic models for the synaptra system."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime
from typing import NamedTuple, Optional

from pydantic import BaseModel, Field


class MemoryType(str, enum.Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    IDENTITY = "identity"
    PERSON = "person"


class MemoryState(str, enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class RelType(str, enum.Enum):
    CAUSES = "causes"
    FOLLOWS = "follows"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    RELATES_TO = "relates_to"
    SUPERSEDES = "supersedes"
    PART_OF = "part_of"
    DESCRIBES = "describes"


# Reserved rater for scores produced by `score_importance`'s heuristic rather
# than by a model.  A heuristic score and a model score are not comparable, so
# the heuristic gets named as the rater it is.  See issue #8.
AUTO_RATER = "auto:heuristic"


class Memory(BaseModel):
    id: str
    content: str
    memory_type: MemoryType
    state: MemoryState = MemoryState.ACTIVE
    importance: float = Field(ge=0.0, le=1.0)
    stability: float
    retrievability: float
    access_count: int = 0
    created_at: datetime
    updated_at: datetime
    last_accessed: datetime
    source: Optional[str] = None
    conversation_id: Optional[str] = None
    # Who set `importance`, and when.  Importance is not an absolute quantity —
    # different models score the same memory differently — so a score is only
    # comparable to another score from the same rater.  NULL means the row
    # predates rater tracking; it is a closed set, because `rater` is mandatory
    # on every write path from here on.  See issue #8.
    rater: Optional[str] = None
    rated_at: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list)


class MemoryVersion(BaseModel):
    id: str
    memory_id: str
    content: str
    metadata: dict | None = None
    created_at: datetime


class Relationship(BaseModel):
    id: str
    source_id: str
    target_id: str
    rel_type: RelType
    strength: float = 1.0
    created_at: datetime


class ConsolidationLogEntry(BaseModel):
    id: str
    action: str  # promote | merge | archive | flag_contradiction
    source_ids: list[str]
    target_id: Optional[str] = None
    reason: str
    created_at: datetime


class ContradictionInfo(BaseModel):
    memory_id: str
    content_preview: str
    strength: float


class RecallResult(BaseModel):
    id: str
    content: str
    memory_type: MemoryType
    importance: float
    retrievability: float
    score: float
    found_by: list[str]
    tags: list[str]
    created_at: datetime
    last_accessed: datetime
    contradictions: list[ContradictionInfo] = Field(default_factory=list)


class RelationshipInfo(BaseModel):
    memory_id: str
    rel_type: RelType
    strength: float
    direction: str  # "outgoing" or "incoming"


class MemoryGetResponse(BaseModel):
    memory: Memory
    relationships: list[RelationshipInfo] = Field(default_factory=list)
    versions: list[MemoryVersion] = Field(default_factory=list)


class StatsResponse(BaseModel):
    counts: dict
    decay: dict
    consolidation: dict
    storage: dict


class ToolResponse(BaseModel):
    success: bool = True
    data: dict | list | None = None
    meta: dict = Field(default_factory=dict)


class SpreadingActivationRow(NamedTuple):
    neighbor_id: str
    depth: int
    rel_strength: float
    current_stability: float
    state: str


@dataclass
class ReinforceUpdate:
    memory_id: str
    stability: float
    last_accessed: datetime
    access_count: int
