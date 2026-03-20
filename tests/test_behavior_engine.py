from src.vision.behaviors import (
    BehaviorSequenceAnalyzer,
    BehaviorSignal,
    MicroBehavior,
)


def test_detects_classic_concealment_sequence() -> None:
    analyzer = BehaviorSequenceAnalyzer(window_size=20)
    signals = [
        BehaviorSignal(1, "2026-03-20T10:00:01Z", MicroBehavior.ENTERED_ZONE, "Aisle 1", 0.1),
        BehaviorSignal(2, "2026-03-20T10:00:02Z", MicroBehavior.LINGERING, "Aisle 1", 0.2),
        BehaviorSignal(5, "2026-03-20T10:00:05Z", MicroBehavior.ITEM_PICKUP, "Aisle 1", 0.4),
        BehaviorSignal(7, "2026-03-20T10:00:07Z", MicroBehavior.LOOKING_AROUND, "Aisle 1", 0.6),
        BehaviorSignal(9, "2026-03-20T10:00:09Z", MicroBehavior.ITEM_CONCEALED, "Aisle 1", 0.8),
    ]
    for s in signals:
        analyzer.record(s)
    result = analyzer.analyze_window()
    assert result is not None
    assert result.matched is True
    assert result.pattern_confidence > 0.85
    assert "concealment" in result.explanation.lower()


def test_no_match_for_normal_shopping() -> None:
    analyzer = BehaviorSequenceAnalyzer(window_size=20)
    signals = [
        BehaviorSignal(1, "2026-03-20T10:00:01Z", MicroBehavior.ENTERED_ZONE, "Aisle 1", 0.1),
        BehaviorSignal(2, "2026-03-20T10:00:02Z", MicroBehavior.ITEM_PICKUP, "Aisle 1", 0.3),
        BehaviorSignal(3, "2026-03-20T10:00:03Z", MicroBehavior.MOVED_TO_CHECKOUT, "Checkout", 0.2),
    ]
    for s in signals:
        analyzer.record(s)
    result = analyzer.analyze_window()
    assert result is None


def test_grab_and_go_pattern() -> None:
    analyzer = BehaviorSequenceAnalyzer(window_size=20)
    signals = [
        BehaviorSignal(1, "2026-03-20T10:00:01Z", MicroBehavior.ITEM_PICKUP, "Shelf", 0.5),
        BehaviorSignal(3, "2026-03-20T10:00:03Z", MicroBehavior.ITEM_CONCEALED, "Aisle", 0.7),
        BehaviorSignal(5, "2026-03-20T10:00:05Z", MicroBehavior.MOVED_TO_EXIT, "Exit", 0.6),
    ]
    for s in signals:
        analyzer.record(s)
    result = analyzer.analyze_window()
    assert result is not None
    assert result.matched is True
    assert result.pattern_confidence > 0.8
