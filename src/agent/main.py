from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.incidents.manager import IncidentManager
from src.vision.detector import SuspiciousEvent
from src.vision.pipeline import VisionPipeline
from src.vision.schemas import ObservationIn, ObservationResponse, SuspiciousEventOut

app = FastAPI(title="Retail Loss Prevention Intelligence Platform", version="0.4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
pipeline = VisionPipeline()
incident_manager = IncidentManager()
frame_counter: dict[str, int] = {"total": 0}
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "agent", "ui": "enabled"}


def _serialize_event(event: SuspiciousEvent | None) -> SuspiciousEventOut | None:
    if event is None:
        return None
    behavior = pipeline.last_behavior_result
    zone = pipeline.last_zone_verdict
    return SuspiciousEventOut(
        event_id=event.event_id,
        event_type=event.event_type,
        observed_sku=event.observed_sku,
        source_frame_index=event.source_frame_index,
        timestamp_utc=event.timestamp_utc,
        confidence=event.confidence,
        reason=event.reason,
        behavior_pattern=behavior.pattern_name if behavior else None,
        zone_verdict=zone.explanation if zone else None,
    )


@app.post("/vision/observations", response_model=ObservationResponse)
def ingest_observation(observation: ObservationIn) -> ObservationResponse:
    frame_counter["total"] += 1
    event = pipeline.ingest_observation(observation)
    if event is not None:
        incident_manager.process_event(
            event,
            behavior_result=pipeline.last_behavior_result,
            zone_verdict=pipeline.last_zone_verdict,
        )
    return ObservationResponse(processed=True, event=_serialize_event(event))


@app.get("/vision/events", response_model=list[SuspiciousEventOut])
def list_recent_events() -> list[SuspiciousEventOut]:
    return [_serialize_event(event) for event in pipeline.recent_events() if event is not None]


@app.get("/incidents")
def list_incidents() -> list[dict[str, object]]:
    return [item.model_dump() for item in incident_manager.list_incidents()]


@app.get("/metrics")
def metrics() -> dict[str, int]:
    base = incident_manager.metrics()
    base["frames_processed"] = frame_counter["total"]
    return base


@app.get("/zones")
def get_zones() -> list[dict[str, object]]:
    return [
        {
            "zone_id": z.zone_id,
            "zone_type": z.zone_type.value,
            "label": z.label,
            "x_min": z.x_min,
            "y_min": z.y_min,
            "x_max": z.x_max,
            "y_max": z.y_max,
        }
        for z in pipeline.zone_engine.layout
    ]


@app.get("/behavior/history")
def behavior_history() -> list[dict[str, object]]:
    return [
        {
            "frame_index": s.frame_index,
            "timestamp_utc": s.timestamp_utc,
            "behavior": s.behavior.value,
            "zone": s.zone,
            "confidence": s.confidence,
        }
        for s in pipeline.behavior_analyzer.history[-50:]
    ]


@app.post("/demo/run")
def run_demo_scenario() -> dict[str, object]:
    """Simulates a realistic multi-stage theft scenario with behavioral
    progression through store zones, triggering the full reasoning pipeline."""
    now = datetime.now(timezone.utc)
    scenario = [
        {"t": 0, "x": 0.1, "y": 0.15, "vis": True, "hand": False, "motion": 0.1, "head": 0.1, "linger": 0},
        {"t": 1, "x": 0.25, "y": 0.2, "vis": True, "hand": False, "motion": 0.15, "head": 0.1, "linger": 5},
        {"t": 2, "x": 0.35, "y": 0.25, "vis": True, "hand": False, "motion": 0.12, "head": 0.2, "linger": 18},
        {"t": 3, "x": 0.4, "y": 0.3, "vis": True, "hand": True, "motion": 0.35, "head": 0.3, "linger": 22},
        {"t": 4, "x": 0.42, "y": 0.3, "vis": True, "hand": True, "motion": 0.28, "head": 0.7, "linger": 25},
        {"t": 5, "x": 0.44, "y": 0.32, "vis": False, "hand": False, "motion": 0.72, "head": 0.6, "linger": 28},
        {"t": 6, "x": 0.5, "y": 0.35, "vis": False, "hand": False, "motion": 0.65, "head": 0.3, "linger": 0},
        {"t": 7, "x": 0.6, "y": 0.5, "vis": False, "hand": False, "motion": 0.55, "head": 0.2, "linger": 0},
        {"t": 8, "x": 0.72, "y": 0.6, "vis": False, "hand": False, "motion": 0.48, "head": 0.15, "linger": 0},
        {"t": 9, "x": 0.82, "y": 0.7, "vis": False, "hand": False, "motion": 0.42, "head": 0.1, "linger": 0},
        {"t": 10, "x": 0.9, "y": 0.8, "vis": False, "hand": False, "motion": 0.38, "head": 0.1, "linger": 0},
    ]
    incidents_created = 0
    for step in scenario:
        payload = ObservationIn(
            source_frame_index=5000 + step["t"],
            timestamp_utc=(now + timedelta(seconds=step["t"])).isoformat(),
            item_sku="sku-electronics-042",
            item_visible=step["vis"],
            hand_near_item=step["hand"],
            motion_score=step["motion"],
            person_id="suspect-alpha",
            person_x=step["x"],
            person_y=step["y"],
            head_rotation_score=step["head"],
            linger_seconds=step["linger"],
        )
        frame_counter["total"] += 1
        event = pipeline.ingest_observation(payload)
        if event is not None:
            incident_manager.process_event(
                event,
                behavior_result=pipeline.last_behavior_result,
                zone_verdict=pipeline.last_zone_verdict,
            )
            incidents_created += 1

    return {
        "observations_processed": len(scenario),
        "incidents_created": incidents_created,
        "scenario": "multi-stage-theft-with-exit-trajectory",
    }
