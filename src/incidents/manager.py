from datetime import datetime, timezone
from uuid import uuid4

from src.alerts.slack import SlackNotifier
from src.incidents.clipper import IncidentClipper
from src.incidents.models import Incident
from src.pos.client import POSClient
from src.vision.behaviors import BehaviorSequenceResult
from src.vision.detector import SuspiciousEvent
from src.vision.reasoning import ReasoningChainBuilder
from src.vision.zones import TrajectoryVerdict


class IncidentManager:
    def __init__(self) -> None:
        self._pos_client = POSClient()
        self._clipper = IncidentClipper()
        self._notifier = SlackNotifier()
        self._reasoning_builder = ReasoningChainBuilder()
        self._incidents: list[Incident] = []

    def process_event(
        self,
        event: SuspiciousEvent,
        behavior_result: BehaviorSequenceResult | None = None,
        zone_verdict: TrajectoryVerdict | None = None,
    ) -> Incident:
        incident_id = f"inc_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        pos_match = self._pos_client.check_scan_match(
            sku=event.observed_sku,
            observed_at_utc=event.timestamp_utc,
        )

        behavior_conf = behavior_result.pattern_confidence if behavior_result else None
        zone_exit_prob = zone_verdict.exit_probability if zone_verdict else None
        final_confidence = self._compute_combined_confidence(
            event.confidence, behavior_conf, zone_exit_prob, pos_match.matched
        )

        status = "resolved" if pos_match.matched else "escalated"
        clip_path = self._clipper.create_clip(
            incident_id=incident_id,
            event_timestamp_utc=event.timestamp_utc,
            seconds=5,
        )

        chain = self._reasoning_builder.build(
            incident_id=incident_id,
            behavior_explanation=behavior_result.explanation if behavior_result else None,
            behavior_confidence=behavior_conf,
            zone_explanation=zone_verdict.explanation if zone_verdict else None,
            zone_exit_probability=zone_exit_prob,
            pos_matched=pos_match.matched,
            pos_reason=pos_match.reason,
            observed_sku=event.observed_sku,
            final_confidence=final_confidence,
        )

        incident = Incident(
            incident_id=incident_id,
            event_id=event.event_id,
            event_type=event.event_type,
            observed_sku=event.observed_sku,
            observed_at_utc=event.timestamp_utc,
            confidence=round(final_confidence, 3),
            pos_match=pos_match.matched,
            transaction_id=pos_match.transaction_id,
            decision_reason=chain.narrative,
            status=status,
            clip_path=clip_path,
            slack_delivery="pending",
            behavior_pattern=behavior_result.pattern_name if behavior_result else None,
            zone_heading=zone_verdict.heading_toward.value if zone_verdict else None,
            zone_exit_probability=round(zone_exit_prob, 3) if zone_exit_prob is not None else None,
            reasoning_narrative=chain.narrative,
            reasoning_chain=chain.to_dict(),
        )
        delivery = self._notifier.notify(incident)
        incident.slack_delivery = delivery
        self._incidents.append(incident)
        return incident

    def list_incidents(self, count: int = 50) -> list[Incident]:
        return self._incidents[-count:]

    def metrics(self) -> dict[str, int]:
        total = len(self._incidents)
        escalated = len([item for item in self._incidents if item.status == "escalated"])
        resolved = len([item for item in self._incidents if item.status == "resolved"])
        return {
            "total_incidents": total,
            "escalated_incidents": escalated,
            "resolved_incidents": resolved,
        }

    def _compute_combined_confidence(
        self,
        detection_conf: float,
        behavior_conf: float | None,
        zone_exit_prob: float | None,
        pos_matched: bool,
    ) -> float:
        score = detection_conf * 0.4
        if behavior_conf is not None:
            score += behavior_conf * 0.3
        else:
            score += detection_conf * 0.15
        if zone_exit_prob is not None:
            score += zone_exit_prob * 0.2
        else:
            score += 0.1
        if pos_matched:
            score *= 0.3
        else:
            score += 0.1
        return min(0.99, max(0.0, score))
