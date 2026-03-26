import base64
import binascii
import json
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent.copilot import AgenticCopilotService
from src.incidents.manager import IncidentManager
from src.vision.detector import SuspiciousEvent
from src.vision.detection_service import DetectionService
from src.vision.pipeline import VisionPipeline
from src.vision.schemas import ObservationIn, ObservationResponse, SuspiciousEventOut

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_origins(raw: str) -> list[str]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return parts if parts else [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8081",
        "http://127.0.0.1:8081",
    ]


APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
API_TOKEN = os.getenv("API_TOKEN", "").strip()
CORS_ALLOWED_ORIGINS = _parse_origins(
    os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:8080,http://127.0.0.1:8080,http://localhost:8081,http://127.0.0.1:8081",
    )
)
MAX_IMAGE_BASE64_CHARS = _env_int("MAX_IMAGE_BASE64_CHARS", 4_000_000)
RATE_LIMIT_WINDOW_SECONDS = max(10, _env_int("RATE_LIMIT_WINDOW_SECONDS", 60))
RATE_LIMIT_DETECT_PER_WINDOW = max(5, _env_int("RATE_LIMIT_DETECT_PER_WINDOW", 45))
RATE_LIMIT_CHAT_PER_WINDOW = max(3, _env_int("RATE_LIMIT_CHAT_PER_WINDOW", 20))
REQUIRE_API_TOKEN_IN_NON_DEV = os.getenv("REQUIRE_API_TOKEN_IN_NON_DEV", "true").strip().lower() == "true"
AUTHZ_REVIEW_ROLES = {x.strip() for x in os.getenv("AUTHZ_REVIEW_ROLES", "manager,admin").split(",") if x.strip()}
AUTHZ_EXPORT_ROLES = {x.strip() for x in os.getenv("AUTHZ_EXPORT_ROLES", "manager,admin,auditor").split(",") if x.strip()}
_rate_buckets: dict[tuple[str, str], deque[float]] = defaultdict(deque)
_endpoint_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "total_ms": 0, "max_ms": 0})
_logger = logging.getLogger("agent.api")
if not _logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI(title="Retail Loss Prevention Intelligence Platform", version="0.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
pipeline = VisionPipeline()
incident_manager = IncidentManager()
detection_service = DetectionService()
copilot_service = AgenticCopilotService()
frame_counter: dict[str, int] = {"total": 0}
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

if APP_ENV != "development" and REQUIRE_API_TOKEN_IN_NON_DEV and not API_TOKEN:
    raise RuntimeError("API_TOKEN is required in non-development environments")


@app.middleware("http")
async def request_context_middleware(request: Request, call_next) -> Response:
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid4())
    started = time.perf_counter()
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    key = f"{request.method} {request.url.path}"
    stat = _endpoint_stats[key]
    stat["count"] += 1
    stat["total_ms"] += elapsed_ms
    stat["max_ms"] = max(stat["max_ms"], elapsed_ms)
    response.headers["X-Correlation-ID"] = correlation_id
    _logger.info(
        json.dumps(
            {
                "msg": "request_complete",
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(elapsed_ms, 2),
            }
        )
    )
    return response


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "retail-loss-prevention-agent",
        "product": "theft-detection",
        "ui": "enabled",
        "app_env": APP_ENV,
        "version": app.version,
        "capabilities": [
            "vision_observations",
            "incident_lifecycle",
            "pos_correlation",
            "copilot",
            "evidence_export",
            "theft_hot_spots",
        ],
    }


def _require_api_token(x_api_token: str | None) -> None:
    if API_TOKEN and x_api_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="invalid_api_token")


def _require_authorized_role(x_actor_role: str | None, allowed_roles: set[str]) -> None:
    role = (x_actor_role or "").strip().lower()
    if APP_ENV == "development":
        return
    if role not in allowed_roles:
        raise HTTPException(status_code=403, detail="insufficient_role")


def _enforce_rate_limit(scope: str, request: Request, max_requests: int) -> None:
    client_id = request.client.host if request.client else "unknown"
    key = (scope, client_id)
    now = time.time()
    bucket = _rate_buckets[key]
    while bucket and (now - bucket[0]) > RATE_LIMIT_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= max_requests:
        raise HTTPException(status_code=429, detail="rate_limit_exceeded")
    bucket.append(now)


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
            store_id=observation.store_id.strip() or "store-001",
            camera_id=observation.camera_id.strip() or "cam-01",
        )
    return ObservationResponse(processed=True, event=_serialize_event(event))


@app.get("/vision/events", response_model=list[SuspiciousEventOut])
def list_recent_events() -> list[SuspiciousEventOut]:
    return [_serialize_event(event) for event in pipeline.recent_events() if event is not None]


@app.get("/incidents")
def list_incidents(
    status: str | None = Query(default=None),
    sku: str | None = Query(default=None),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    max_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    review_status: str | None = Query(default=None),
    store_id: str | None = Query(default=None),
    camera_id: str | None = Query(default=None),
    zone_heading: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[dict[str, object]]:
    return [
        item.model_dump()
        for item in incident_manager.list_incidents(
            count=limit,
            status=status,
            sku=sku,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            review_status=review_status,
            store_id=store_id,
            camera_id=camera_id,
            zone_heading=zone_heading,
        )
    ]


class ReviewActionIn(BaseModel):
    action: str
    notes: str | None = None


class FrameDetectIn(BaseModel):
    image_base64: str
    conf_threshold: float = 0.3
    max_detections: int = 20
    model_variant: str = "pretrained"


class CopilotQuestionIn(BaseModel):
    question: str


@app.post("/incidents/{incident_id}/review")
def review_incident(
    incident_id: str,
    payload: ReviewActionIn,
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
    x_actor_role: str | None = Header(default=None, alias="X-Actor-Role"),
) -> dict[str, object]:
    _require_api_token(x_api_token)
    _require_authorized_role(x_actor_role, AUTHZ_REVIEW_ROLES)
    updated = incident_manager.update_review(
        incident_id=incident_id,
        action=payload.action,
        notes=payload.notes,
    )
    if updated is None:
        return {"ok": False, "error": "incident_not_found"}
    return {"ok": True, "incident": updated.model_dump()}


@app.post("/vision/detect-frame")
def detect_frame(
    payload: FrameDetectIn,
    request: Request,
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
) -> dict[str, object]:
    _require_api_token(x_api_token)
    _enforce_rate_limit("detect_frame", request, RATE_LIMIT_DETECT_PER_WINDOW)
    if len(payload.image_base64) > MAX_IMAGE_BASE64_CHARS:
        raise HTTPException(status_code=413, detail="image_payload_too_large")
    try:
        image_bytes = base64.b64decode(payload.image_base64, validate=True)
    except (binascii.Error, ValueError):
        return {
            "ok": False,
            "detections": [],
            "model_ready": False,
            "message": "invalid_image_base64",
            "model_file": None,
        }
    result = detection_service.detect(
        image_bytes=image_bytes,
        conf_threshold=payload.conf_threshold,
        max_detections=payload.max_detections,
        model_variant=payload.model_variant,
    )
    return {
        "ok": True,
        "detections": result.detections,
        "model_ready": result.model_ready,
        "message": result.message,
        "device": result.device,
        "model_file": result.model_file,
    }


def _copilot_context() -> dict[str, object]:
    incidents = [x.model_dump() for x in incident_manager.list_incidents(count=10)]
    latest = incidents[-1] if incidents else None
    events = [_serialize_event(e).model_dump() for e in pipeline.recent_events() if e is not None]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics(),
        "latest_incident": latest,
        "recent_incidents": incidents,
        "recent_events": events[-8:],
        "behavior_tail": behavior_history()[-12:],
    }


@app.get("/copilot/status")
def copilot_status() -> dict[str, object]:
    return copilot_service.status()


@app.get("/copilot/brief")
def copilot_brief() -> dict[str, object]:
    result = copilot_service.generate_brief(_copilot_context())
    return {
        "narrative": result.narrative,
        "risk_level": result.risk_level,
        "recommended_action": result.recommended_action,
        "possibilities": result.possibilities,
        "source": result.source,
    }


@app.post("/copilot/chat")
def copilot_chat(
    payload: CopilotQuestionIn,
    request: Request,
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
) -> dict[str, object]:
    _require_api_token(x_api_token)
    _enforce_rate_limit("copilot_chat", request, RATE_LIMIT_CHAT_PER_WINDOW)
    result = copilot_service.answer_question(payload.question, _copilot_context())
    return {
        "answer": result.narrative,
        "risk_level": result.risk_level,
        "recommended_action": result.recommended_action,
        "possibilities": result.possibilities,
        "source": result.source,
    }


@app.get("/incidents/export.csv")
def export_incidents_csv(
    status: str | None = Query(default=None),
    sku: str | None = Query(default=None),
    review_status: str | None = Query(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
    api_token: str | None = Query(default=None),
    x_actor_role: str | None = Header(default=None, alias="X-Actor-Role"),
) -> PlainTextResponse:
    _require_api_token(x_api_token or api_token)
    _require_authorized_role(x_actor_role, AUTHZ_EXPORT_ROLES)
    csv_text = incident_manager.export_incidents_csv(
        status=status,
        sku=sku,
        review_status=review_status,
    )
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=incidents_export.csv"},
    )


@app.get("/incidents/{incident_id}/evidence")
def export_incident_evidence(
    incident_id: str,
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
    api_token: str | None = Query(default=None),
    x_actor_role: str | None = Header(default=None, alias="X-Actor-Role"),
) -> JSONResponse:
    _require_api_token(x_api_token or api_token)
    _require_authorized_role(x_actor_role, AUTHZ_EXPORT_ROLES)
    bundle = incident_manager.export_incident_evidence_bundle(incident_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="incident_not_found")
    return JSONResponse(content=json.loads(bundle))


@app.get("/metrics")
def metrics() -> dict[str, int]:
    base = incident_manager.metrics()
    base["frames_processed"] = frame_counter["total"]
    return base


@app.get("/theft/hot-spots")
def theft_hot_spots(top_n: int = Query(default=8, ge=1, le=50)) -> dict[str, object]:
    """Camera / store pairs ranked by escalations plus unreviewed queue (ops heat map)."""
    return {"items": incident_manager.theft_hot_spots(top_n=top_n)}


@app.get("/metrics/extended")
def metrics_extended() -> dict[str, object]:
    endpoint_latency: dict[str, dict[str, float]] = {}
    for key, stat in _endpoint_stats.items():
        avg = (stat["total_ms"] / stat["count"]) if stat["count"] else 0.0
        endpoint_latency[key] = {
            "count": stat["count"],
            "avg_ms": round(avg, 2),
            "max_ms": round(stat["max_ms"], 2),
        }
    return {
        "app_env": APP_ENV,
        "api_token_configured": bool(API_TOKEN),
        "rate_limit_window_seconds": RATE_LIMIT_WINDOW_SECONDS,
        "endpoint_latency": endpoint_latency,
    }


@app.get("/health/dependencies")
def health_dependencies() -> dict[str, object]:
    copilot_status = copilot_service.status()
    return {
        "status": "ok",
        "dependencies": {
            "incident_repository": "ok",
            "copilot": copilot_status,
            "detector": detection_service.status(),
        },
    }


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
def run_demo_scenario(x_api_token: str | None = Header(default=None, alias="X-API-Token")) -> dict[str, object]:
    """Simulates a realistic multi-stage theft scenario with behavioral
    progression through store zones, triggering the full reasoning pipeline."""
    _require_api_token(x_api_token)
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
