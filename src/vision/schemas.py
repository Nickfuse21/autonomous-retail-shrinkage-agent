from pydantic import BaseModel, Field


class ObservationIn(BaseModel):
    source_frame_index: int = Field(ge=0)
    timestamp_utc: str
    item_sku: str = "sku-apple-001"
    item_visible: bool
    hand_near_item: bool
    motion_score: float = Field(ge=0.0, le=1.0)
    person_id: str = "person-01"
    person_x: float = Field(default=0.3, ge=0.0, le=1.0)
    person_y: float = Field(default=0.3, ge=0.0, le=1.0)
    head_rotation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    linger_seconds: float = Field(default=0.0, ge=0.0)


class SuspiciousEventOut(BaseModel):
    event_id: str
    event_type: str
    observed_sku: str
    source_frame_index: int
    timestamp_utc: str
    confidence: float
    reason: str
    behavior_pattern: str | None = None
    zone_verdict: str | None = None
    reasoning_narrative: str | None = None


class ObservationResponse(BaseModel):
    processed: bool
    event: SuspiciousEventOut | None
