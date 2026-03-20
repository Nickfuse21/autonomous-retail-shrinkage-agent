from datetime import datetime, timezone

from fastapi.testclient import TestClient

from src.agent import main as agent_main
from src.incidents.models import (
    AuditEvent,
    EvidencePackage,
    Incident,
    IncidentStatus,
    ReviewStatus,
)
from src.incidents.repository import IncidentRepository


def _seed_incident(repo: IncidentRepository, incident_id: str = "inc_test_001") -> Incident:
    incident = Incident(
        incident_id=incident_id,
        event_id="evt-1",
        event_type="suspected_item_disappearance",
        observed_sku="sku-apple-001",
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        confidence=0.82,
        pos_match=False,
        transaction_id=None,
        decision_reason="test",
        status=IncidentStatus.escalated,
        clip_path="clips/inc_test_001.mp4",
        slack_delivery="pending",
        review_status=ReviewStatus.unreviewed,
        evidence_package=EvidencePackage(
            clip_path="clips/inc_test_001.mp4",
            detector_snapshot={"label": "bottle"},
            reasoning_chain={"narrative": "test"},
            pos_correlation={"matched": False},
        ),
        audit_timeline=[
            AuditEvent(
                event_type="incident_created",
                actor="system",
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                details={"seeded": True},
            )
        ],
    )
    repo.upsert_incident(incident)
    return incident


def test_csv_export_requires_token_when_configured(monkeypatch) -> None:
    client = TestClient(agent_main.app)
    monkeypatch.setattr(agent_main, "API_TOKEN", "token-123")
    denied = client.get("/incidents/export.csv")
    assert denied.status_code == 401
    ok = client.get("/incidents/export.csv", headers={"X-API-Token": "token-123", "X-Actor-Role": "manager"})
    assert ok.status_code == 200


def test_chat_rate_limit_is_enforced(monkeypatch) -> None:
    client = TestClient(agent_main.app)
    monkeypatch.setattr(agent_main, "RATE_LIMIT_WINDOW_SECONDS", 600)
    monkeypatch.setattr(agent_main, "RATE_LIMIT_CHAT_PER_WINDOW", 1)
    agent_main._rate_buckets.clear()
    first = client.post("/copilot/chat", json={"question": "hello"})
    assert first.status_code == 200
    second = client.post("/copilot/chat", json={"question": "again"})
    assert second.status_code == 429


def test_dependency_and_extended_metrics_endpoints() -> None:
    client = TestClient(agent_main.app)
    deps = client.get("/health/dependencies")
    assert deps.status_code == 200
    body = deps.json()
    assert "dependencies" in body
    ext = client.get("/metrics/extended")
    assert ext.status_code == 200
    assert "endpoint_latency" in ext.json()


def test_repository_persists_across_instances(tmp_path) -> None:
    db_path = tmp_path / "incidents.db"
    repo_one = IncidentRepository(db_path=str(db_path))
    seeded = _seed_incident(repo_one, "inc_restart_001")
    repo_two = IncidentRepository(db_path=str(db_path))
    loaded = repo_two.get_incident(seeded.incident_id)
    assert loaded is not None
    assert loaded.incident_id == seeded.incident_id
    assert loaded.evidence_package is not None
