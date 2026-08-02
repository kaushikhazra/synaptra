"""Memory type classification and importance scoring — heuristic-based with agent override."""

from __future__ import annotations

import re
from .models import MemoryType


# Keyword/pattern sets for classification
_WORKING_KEYWORDS = {
    "todo", "task", "remind", "note to self", "currently", "right now",
    "in progress", "working on", "need to", "don't forget",
}

_EPISODIC_PATTERNS = [
    r"\byesterday\b", r"\btoday\b", r"\blast (?:week|month|year|night|time)\b",
    r"\bwhen (?:we|i|he|she|they)\b", r"\bremember when\b",
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",  # date patterns
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d",
    r"\bhappened\b", r"\bexperienced\b", r"\bmet\b", r"\bvisited\b",
    r"\bwe (?:went|did|saw|had|made)\b",
]

_SEMANTIC_PATTERNS = [
    r"\bis (?:a|an|the|defined as)\b", r"\bmeans\b", r"\brefers to\b",
    r"\bis known as\b", r"\bfact:\b", r"\bdefinition:\b",
    r"\balways\b", r"\bnever\b", r"\bevery\b",
    r"(?:^|\. )[A-Z][a-z]+ (?:is|are|was|were) ",  # declarative statements
]

_PROCEDURAL_PATTERNS = [
    r"\bhow to\b", r"\bsteps?\b", r"\bworkflow\b", r"\bprocedure\b",
    r"\bwhen .+ do\b", r"\bfirst .+ then\b", r"\binstruction\b",
    r"\brecipe\b", r"\bif .+ then\b", r"\bto (?:do|make|create|build|run|set up)\b",
    r"^\d+\.\s",  # numbered steps
]

_IDENTITY_PATTERNS = [
    r"\b(?:i am|my name is|i was (?:built|created|designed))\b",
    r"\b(?:my (?:role|purpose|values?|capabilities?|personality|origin|identity))\b",
    r"\b(?:as an? (?:agent|ai|assistant))\b",
    r"\b(?:i (?:believe|value|prefer|specialize))\b",
]

_PERSON_PATTERNS = [
    r"\bperson:\w+\b",  # explicit person tag in content
    r"\b(?:(?:he|she|they) (?:is|are|prefers?|likes?|works?))\b",
    r"\b(?:(?:'s|'s) (?:wife|husband|partner|health|preference|style))\b",
]


def classify(content: str, source: str | None = None) -> tuple[MemoryType, float]:
    """Classify memory content into a type. Returns (type, confidence 0.0-1.0).

    Agent should override when they know the type — this is a fallback.
    """
    content_lower = content.lower()
    scores: dict[MemoryType, float] = {
        MemoryType.WORKING: 0.0,
        MemoryType.EPISODIC: 0.0,
        MemoryType.SEMANTIC: 0.0,
        MemoryType.PROCEDURAL: 0.0,
        MemoryType.IDENTITY: 0.0,
        MemoryType.PERSON: 0.0,
    }

    # Working memory signals
    if source == "conversation":
        scores[MemoryType.WORKING] += 0.3
    if len(content) < 100:
        scores[MemoryType.WORKING] += 0.1
    for kw in _WORKING_KEYWORDS:
        if kw in content_lower:
            scores[MemoryType.WORKING] += 0.2
            break

    # Episodic signals
    for pattern in _EPISODIC_PATTERNS:
        if re.search(pattern, content_lower):
            scores[MemoryType.EPISODIC] += 0.2
    # Cap episodic score contribution from patterns
    scores[MemoryType.EPISODIC] = min(scores[MemoryType.EPISODIC], 0.8)

    # Semantic signals
    for pattern in _SEMANTIC_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            scores[MemoryType.SEMANTIC] += 0.2
    scores[MemoryType.SEMANTIC] = min(scores[MemoryType.SEMANTIC], 0.8)

    # Procedural signals
    for pattern in _PROCEDURAL_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            scores[MemoryType.PROCEDURAL] += 0.25
    scores[MemoryType.PROCEDURAL] = min(scores[MemoryType.PROCEDURAL], 0.8)

    # Identity signals
    for pattern in _IDENTITY_PATTERNS:
        if re.search(pattern, content_lower):
            scores[MemoryType.IDENTITY] += 0.25
    scores[MemoryType.IDENTITY] = min(scores[MemoryType.IDENTITY], 0.8)

    # Person signals
    for pattern in _PERSON_PATTERNS:
        if re.search(pattern, content_lower):
            scores[MemoryType.PERSON] += 0.25
    scores[MemoryType.PERSON] = min(scores[MemoryType.PERSON], 0.8)

    # Pick the highest
    best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_type]

    # If confidence is too low, default to episodic
    if best_score < 0.2:
        return MemoryType.EPISODIC, 0.3

    return best_type, min(best_score, 1.0)


def score_importance(
    content: str,
    memory_type: MemoryType,
    agent_importance: float | None = None,
    config: dict | None = None,
) -> float:
    """Score importance heuristically. Agent can override with agent_importance."""
    if agent_importance is not None:
        return max(0.1, min(1.0, agent_importance))

    cfg = config or {}
    base = cfg.get("base_score", 0.5)
    score = base

    # Named entity bonus — words starting with uppercase that aren't sentence starters
    words = content.split()
    if len(words) > 1:
        # Check for capitalized words mid-sentence (rough named entity detection)
        mid_caps = sum(
            1 for i, w in enumerate(words)
            if i > 0 and w[0].isupper() and not words[i - 1].endswith(".")
        )
        if mid_caps >= 1:
            score += cfg.get("named_entity_bonus", 0.1)

    # Agent marked important — handled by agent_importance override above

    # Relational context bonus — keyword hints
    relational_keywords = ["related to", "same as", "similar to", "connected to", "links to", "part of"]
    if any(kw in content.lower() for kw in relational_keywords):
        score += cfg.get("relational_bonus", 0.1)

    # Length bonus
    length_threshold = cfg.get("length_threshold", 200)
    if len(content) > length_threshold:
        score += cfg.get("length_bonus", 0.1)

    # Working memory penalty
    if memory_type == MemoryType.WORKING:
        score -= cfg.get("working_penalty", 0.1)

    # Identity bonus — self-knowledge is inherently high-value
    if memory_type == MemoryType.IDENTITY:
        score += cfg.get("identity_bonus", 0.2)

    # Person bonus — social knowledge is high-value
    if memory_type == MemoryType.PERSON:
        score += cfg.get("person_bonus", 0.15)

    # Clamp
    min_val = cfg.get("min", 0.1)
    max_val = cfg.get("max", 1.0)
    return max(min_val, min(max_val, score))
