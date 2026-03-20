from src.vision.zones import PositionUpdate, ZoneIntelligenceEngine, ZoneType


def test_exit_trajectory_flagged_suspicious() -> None:
    engine = ZoneIntelligenceEngine()
    positions = [
        PositionUpdate(1, "2026-03-20T10:00:01Z", "p1", 0.1, 0.1),
        PositionUpdate(2, "2026-03-20T10:00:02Z", "p1", 0.3, 0.2),
        PositionUpdate(3, "2026-03-20T10:00:03Z", "p1", 0.5, 0.4),
        PositionUpdate(4, "2026-03-20T10:00:04Z", "p1", 0.7, 0.6),
        PositionUpdate(5, "2026-03-20T10:00:05Z", "p1", 0.85, 0.75),
        PositionUpdate(6, "2026-03-20T10:00:06Z", "p1", 0.92, 0.85),
    ]
    for p in positions:
        engine.update_position(p)
    verdict = engine.classify_trajectory("p1")
    assert verdict is not None
    assert verdict.is_suspicious is True
    assert verdict.heading_toward == ZoneType.EXIT
    assert verdict.exit_probability > 0.5


def test_checkout_trajectory_not_suspicious() -> None:
    engine = ZoneIntelligenceEngine()
    positions = [
        PositionUpdate(1, "2026-03-20T10:00:01Z", "p2", 0.1, 0.1),
        PositionUpdate(2, "2026-03-20T10:00:02Z", "p2", 0.3, 0.2),
        PositionUpdate(3, "2026-03-20T10:00:03Z", "p2", 0.5, 0.15),
        PositionUpdate(4, "2026-03-20T10:00:04Z", "p2", 0.75, 0.2),
        PositionUpdate(5, "2026-03-20T10:00:05Z", "p2", 0.85, 0.25),
    ]
    for p in positions:
        engine.update_position(p)
    verdict = engine.classify_trajectory("p2")
    assert verdict is not None
    assert verdict.is_suspicious is False


def test_zone_resolution() -> None:
    engine = ZoneIntelligenceEngine()
    zone = engine.resolve_zone(0.1, 0.15)
    assert zone.zone_type == ZoneType.ENTRANCE
    zone = engine.resolve_zone(0.85, 0.25)
    assert zone.zone_type == ZoneType.CHECKOUT
    zone = engine.resolve_zone(0.85, 0.75)
    assert zone.zone_type == ZoneType.EXIT
