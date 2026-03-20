from enum import Enum

from pydantic import BaseModel, Field


class IncidentStatus(str, Enum):
    escalated = "escalated"
    resolved = "resolved"


class ReviewStatus(str, Enum):
    unreviewed = "unreviewed"
    approved = "approved"
    dismissed = "dismissed"
    escalated_security = "escalated_security"
    reviewed = "reviewed"


class ReviewAction(str, Enum):
    approve = "approve"
    false_positive = "false_positive"
    escalate_security = "escalate_security"
    mark_reviewed = "mark_reviewed"


class AuditEvent(BaseModel):
    event_type: str
    actor: str = "system"
    timestamp_utc: str
    details: dict[str, object] = Field(default_factory=dict)


class EvidencePackage(BaseModel):
    clip_path: str
    detector_snapshot: dict[str, object] = Field(default_factory=dict)
    reasoning_chain: dict[str, object] = Field(default_factory=dict)
    pos_correlation: dict[str, object] = Field(default_factory=dict)
    reviewer_notes: str | None = None


class Incident(BaseModel):
    incident_id: str
    event_id: str
    event_type: str
    observed_sku: str
    observed_at_utc: str
    confidence: float
    pos_match: bool
    transaction_id: str | None
    decision_reason: str
    status: IncidentStatus
    clip_path: str
    slack_delivery: str
    behavior_pattern: str | None = None
    zone_heading: str | None = None
    zone_exit_probability: float | None = None
    reasoning_narrative: str | None = None
    reasoning_chain: dict[str, object] | None = None
    review_status: ReviewStatus = ReviewStatus.unreviewed
    review_action: ReviewAction | None = None
    review_notes: str | None = None
    reviewed_at_utc: str | None = None
    store_id: str = "store-001"
    camera_id: str = "cam-01"
    evidence_package: EvidencePackage | None = None
    audit_timeline: list[AuditEvent] = Field(default_factory=list)
