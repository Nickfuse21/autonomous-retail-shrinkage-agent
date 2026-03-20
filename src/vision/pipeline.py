from src.vision.behaviors import (
    BehaviorSequenceAnalyzer,
    BehaviorSequenceResult,
    BehaviorSignal,
    MicroBehavior,
)
from src.vision.detector import BaselineBehaviorDetector, SuspiciousEvent
from src.vision.frame_buffer import BufferedFrame, FrameBuffer
from src.vision.schemas import ObservationIn
from src.vision.zones import (
    PositionUpdate,
    TrajectoryVerdict,
    ZoneIntelligenceEngine,
)


class VisionPipeline:
    def __init__(self, buffer_size: int = 300) -> None:
        self._buffer = FrameBuffer(max_frames=buffer_size)
        self._detector = BaselineBehaviorDetector()
        self._behavior_analyzer = BehaviorSequenceAnalyzer()
        self._zone_engine = ZoneIntelligenceEngine()
        self._events: list[SuspiciousEvent] = []
        self._last_behavior_result: BehaviorSequenceResult | None = None
        self._last_zone_verdict: TrajectoryVerdict | None = None

    def ingest_observation(self, observation: ObservationIn) -> SuspiciousEvent | None:
        frame = BufferedFrame(
            source_frame_index=observation.source_frame_index,
            timestamp_utc=observation.timestamp_utc,
            item_sku=observation.item_sku,
            item_visible=observation.item_visible,
            hand_near_item=observation.hand_near_item,
            motion_score=observation.motion_score,
        )
        self._buffer.add(frame)

        self._zone_engine.update_position(
            PositionUpdate(
                frame_index=observation.source_frame_index,
                timestamp_utc=observation.timestamp_utc,
                person_id=observation.person_id,
                x=observation.person_x,
                y=observation.person_y,
            )
        )

        micro = self._classify_micro_behavior(observation)
        zone = self._zone_engine.resolve_zone(observation.person_x, observation.person_y)
        self._behavior_analyzer.record(
            BehaviorSignal(
                frame_index=observation.source_frame_index,
                timestamp_utc=observation.timestamp_utc,
                behavior=micro,
                zone=zone.label,
                confidence=observation.motion_score,
            )
        )

        self._last_behavior_result = self._behavior_analyzer.analyze_window()
        self._last_zone_verdict = self._zone_engine.classify_trajectory(observation.person_id)

        event = self._detector.process(frame)
        if event is not None:
            self._events.append(event)
        return event

    @property
    def last_behavior_result(self) -> BehaviorSequenceResult | None:
        return self._last_behavior_result

    @property
    def last_zone_verdict(self) -> TrajectoryVerdict | None:
        return self._last_zone_verdict

    @property
    def zone_engine(self) -> ZoneIntelligenceEngine:
        return self._zone_engine

    @property
    def behavior_analyzer(self) -> BehaviorSequenceAnalyzer:
        return self._behavior_analyzer

    def recent_events(self, count: int = 20) -> list[SuspiciousEvent]:
        return self._events[-count:]

    def _classify_micro_behavior(self, obs: ObservationIn) -> MicroBehavior:
        if obs.linger_seconds > 15:
            return MicroBehavior.LINGERING
        if obs.hand_near_item and obs.item_visible:
            return MicroBehavior.ITEM_PICKUP
        if obs.head_rotation_score > 0.5:
            return MicroBehavior.LOOKING_AROUND
        if not obs.item_visible and obs.motion_score > 0.3:
            return MicroBehavior.ITEM_CONCEALED
        zone = self._zone_engine.resolve_zone(obs.person_x, obs.person_y)
        if zone.zone_type.value == "checkout":
            return MicroBehavior.MOVED_TO_CHECKOUT
        if zone.zone_type.value == "exit":
            return MicroBehavior.MOVED_TO_EXIT
        return MicroBehavior.IDLE
