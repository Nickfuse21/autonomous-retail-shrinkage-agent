"""
Zone-aware spatial intelligence engine.

Models store layout as named zones and classifies person trajectories
to determine whether movement patterns indicate checkout intent or exit intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ZoneType(str, Enum):
    ENTRANCE = "entrance"
    AISLE = "aisle"
    SHELF = "shelf"
    CHECKOUT = "checkout"
    EXIT = "exit"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StoreZone:
    zone_id: str
    zone_type: ZoneType
    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float


DEFAULT_STORE_LAYOUT: list[StoreZone] = [
    StoreZone("z_entrance", ZoneType.ENTRANCE, "Main Entrance", 0.0, 0.0, 0.2, 0.3),
    StoreZone("z_aisle_1", ZoneType.AISLE, "Aisle 1 - Electronics", 0.2, 0.0, 0.5, 0.5),
    StoreZone("z_aisle_2", ZoneType.AISLE, "Aisle 2 - Groceries", 0.2, 0.5, 0.5, 1.0),
    StoreZone("z_shelf_high", ZoneType.SHELF, "High-Value Shelf", 0.5, 0.0, 0.7, 0.4),
    StoreZone("z_checkout", ZoneType.CHECKOUT, "Checkout Lane", 0.7, 0.0, 1.0, 0.5),
    StoreZone("z_exit", ZoneType.EXIT, "Store Exit", 0.7, 0.5, 1.0, 1.0),
]


@dataclass(frozen=True)
class PositionUpdate:
    frame_index: int
    timestamp_utc: str
    person_id: str
    x: float
    y: float


@dataclass
class TrajectoryVerdict:
    person_id: str
    trajectory_zones: list[str]
    heading_toward: ZoneType
    checkout_probability: float
    exit_probability: float
    is_suspicious: bool
    explanation: str


class ZoneIntelligenceEngine:
    """Tracks person positions, resolves zone occupancy, and classifies
    whether a trajectory leads toward checkout or exit."""

    def __init__(self, layout: list[StoreZone] | None = None) -> None:
        self._layout = layout or DEFAULT_STORE_LAYOUT
        self._trajectories: dict[str, list[PositionUpdate]] = {}

    def update_position(self, pos: PositionUpdate) -> None:
        if pos.person_id not in self._trajectories:
            self._trajectories[pos.person_id] = []
        self._trajectories[pos.person_id].append(pos)
        if len(self._trajectories[pos.person_id]) > 200:
            self._trajectories[pos.person_id] = self._trajectories[pos.person_id][-150:]

    def resolve_zone(self, x: float, y: float) -> StoreZone:
        for zone in self._layout:
            if zone.x_min <= x <= zone.x_max and zone.y_min <= y <= zone.y_max:
                return zone
        return StoreZone("z_unknown", ZoneType.UNKNOWN, "Unknown", 0, 0, 0, 0)

    def classify_trajectory(self, person_id: str) -> TrajectoryVerdict | None:
        positions = self._trajectories.get(person_id)
        if not positions or len(positions) < 3:
            return None

        zone_sequence: list[str] = []
        for pos in positions:
            zone = self.resolve_zone(pos.x, pos.y)
            if not zone_sequence or zone_sequence[-1] != zone.label:
                zone_sequence.append(zone.label)

        recent = positions[-5:]
        avg_x = sum(p.x for p in recent) / len(recent)
        avg_y = sum(p.y for p in recent) / len(recent)

        dx = recent[-1].x - recent[0].x
        dy = recent[-1].y - recent[0].y

        checkout_zones = [z for z in self._layout if z.zone_type == ZoneType.CHECKOUT]
        exit_zones = [z for z in self._layout if z.zone_type == ZoneType.EXIT]

        checkout_prob = self._proximity_score(avg_x, avg_y, dx, dy, checkout_zones)
        exit_prob = self._proximity_score(avg_x, avg_y, dx, dy, exit_zones)

        total = checkout_prob + exit_prob
        if total > 0:
            checkout_prob /= total
            exit_prob /= total
        else:
            checkout_prob = 0.5
            exit_prob = 0.5

        heading = ZoneType.CHECKOUT if checkout_prob > exit_prob else ZoneType.EXIT
        is_suspicious = exit_prob > 0.6

        visited_checkout = any(
            z.zone_type == ZoneType.CHECKOUT
            for pos in positions
            for z in [self.resolve_zone(pos.x, pos.y)]
        )

        if visited_checkout:
            is_suspicious = False
            heading = ZoneType.CHECKOUT
            explanation = f"Person visited checkout zone. Trajectory appears normal across {len(zone_sequence)} zones."
        elif is_suspicious:
            explanation = (
                f"Exit-directed trajectory detected. Person moved through {len(zone_sequence)} zones "
                f"with {exit_prob:.0%} exit probability without approaching checkout."
            )
        else:
            explanation = f"Trajectory inconclusive across {len(zone_sequence)} zones. Monitoring continues."

        return TrajectoryVerdict(
            person_id=person_id,
            trajectory_zones=zone_sequence,
            heading_toward=heading,
            checkout_probability=round(checkout_prob, 3),
            exit_probability=round(exit_prob, 3),
            is_suspicious=is_suspicious,
            explanation=explanation,
        )

    @property
    def layout(self) -> list[StoreZone]:
        return list(self._layout)

    @property
    def tracked_persons(self) -> list[str]:
        return list(self._trajectories.keys())

    def _proximity_score(
        self, x: float, y: float, dx: float, dy: float, zones: list[StoreZone]
    ) -> float:
        if not zones:
            return 0.0
        best = 0.0
        for zone in zones:
            cx = (zone.x_min + zone.x_max) / 2
            cy = (zone.y_min + zone.y_max) / 2
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            proximity = max(0.0, 1.0 - dist)
            direction_bonus = 0.0
            if dx != 0 or dy != 0:
                vec_len = (dx**2 + dy**2) ** 0.5
                to_zone_x, to_zone_y = cx - x, cy - y
                to_zone_len = (to_zone_x**2 + to_zone_y**2) ** 0.5
                if vec_len > 0 and to_zone_len > 0:
                    dot = (dx * to_zone_x + dy * to_zone_y) / (vec_len * to_zone_len)
                    direction_bonus = max(0.0, dot) * 0.3
            best = max(best, proximity + direction_bonus)
        return best
