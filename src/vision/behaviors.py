"""
Multi-stage behavioral sequence analysis engine.

Tracks micro-behaviors over time and classifies behavioral intent
by analyzing temporal patterns rather than single-frame decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MicroBehavior(str, Enum):
    ENTERED_ZONE = "entered_zone"
    LINGERING = "lingering"
    ITEM_PICKUP = "item_pickup"
    LOOKING_AROUND = "looking_around"
    ITEM_CONCEALED = "item_concealed"
    MOVED_TO_CHECKOUT = "moved_to_checkout"
    MOVED_TO_EXIT = "moved_to_exit"
    ITEM_RETURNED = "item_returned"
    IDLE = "idle"


@dataclass(frozen=True)
class BehaviorSignal:
    frame_index: int
    timestamp_utc: str
    behavior: MicroBehavior
    zone: str
    confidence: float
    metadata: dict[str, object] = field(default_factory=dict)


SUSPICIOUS_SEQUENCES: list[tuple[list[MicroBehavior], float, str]] = [
    (
        [MicroBehavior.LINGERING, MicroBehavior.ITEM_PICKUP, MicroBehavior.LOOKING_AROUND, MicroBehavior.ITEM_CONCEALED],
        0.92,
        "Classic concealment pattern: linger, pick up, scan surroundings, conceal.",
    ),
    (
        [MicroBehavior.ITEM_PICKUP, MicroBehavior.ITEM_CONCEALED, MicroBehavior.MOVED_TO_EXIT],
        0.88,
        "Grab-and-go pattern: item picked, concealed, moved directly to exit.",
    ),
    (
        [MicroBehavior.LINGERING, MicroBehavior.ITEM_PICKUP, MicroBehavior.ITEM_CONCEALED, MicroBehavior.MOVED_TO_EXIT],
        0.95,
        "High-intent shrinkage: prolonged loitering followed by concealment and exit trajectory.",
    ),
    (
        [MicroBehavior.ITEM_PICKUP, MicroBehavior.LOOKING_AROUND, MicroBehavior.ITEM_CONCEALED],
        0.85,
        "Surveillance-aware concealment: pickup with visual scanning before concealing.",
    ),
]


@dataclass
class BehaviorSequenceResult:
    matched: bool
    pattern_name: str
    pattern_confidence: float
    matched_behaviors: list[BehaviorSignal]
    explanation: str


class BehaviorSequenceAnalyzer:
    """Tracks micro-behaviors over a sliding window and matches against
    known suspicious behavioral sequences using subsequence matching."""

    def __init__(self, window_size: int = 30) -> None:
        self._history: list[BehaviorSignal] = []
        self._window_size = window_size

    def record(self, signal: BehaviorSignal) -> None:
        self._history.append(signal)
        if len(self._history) > self._window_size * 3:
            self._history = self._history[-self._window_size * 2 :]

    def analyze_window(self) -> BehaviorSequenceResult | None:
        window = self._history[-self._window_size :]
        if len(window) < 3:
            return None

        best_match: BehaviorSequenceResult | None = None
        best_conf = 0.0

        for pattern, base_conf, explanation in SUSPICIOUS_SEQUENCES:
            matched_signals = self._subsequence_match(window, pattern)
            if matched_signals is not None:
                time_bonus = self._temporal_density_bonus(matched_signals)
                final_conf = min(0.99, base_conf + time_bonus)
                if final_conf > best_conf:
                    best_conf = final_conf
                    best_match = BehaviorSequenceResult(
                        matched=True,
                        pattern_name=explanation.split(":")[0].strip(),
                        pattern_confidence=round(final_conf, 3),
                        matched_behaviors=matched_signals,
                        explanation=explanation,
                    )

        return best_match

    @property
    def history(self) -> list[BehaviorSignal]:
        return list(self._history)

    def _subsequence_match(
        self, window: list[BehaviorSignal], pattern: list[MicroBehavior]
    ) -> list[BehaviorSignal] | None:
        matched: list[BehaviorSignal] = []
        pattern_idx = 0
        for signal in window:
            if pattern_idx >= len(pattern):
                break
            if signal.behavior == pattern[pattern_idx]:
                matched.append(signal)
                pattern_idx += 1
        return matched if pattern_idx == len(pattern) else None

    def _temporal_density_bonus(self, signals: list[BehaviorSignal]) -> float:
        if len(signals) < 2:
            return 0.0
        frame_span = signals[-1].frame_index - signals[0].frame_index
        if frame_span <= 0:
            return 0.0
        density = len(signals) / frame_span
        return min(0.05, density * 0.1)
