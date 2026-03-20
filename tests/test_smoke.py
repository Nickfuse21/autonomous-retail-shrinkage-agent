from fastapi.testclient import TestClient

from src.agent.main import app as agent_app
from src.api.mock_pos_api import app as pos_app


def test_agent_health() -> None:
    response = TestClient(agent_app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_mock_pos_health() -> None:
    response = TestClient(pos_app).get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "mock-pos"


def test_vision_observation_generates_event_on_disappearance_pattern() -> None:
    client = TestClient(agent_app)

    seed_payloads = [
        {
            "source_frame_index": 1,
            "timestamp_utc": "2026-03-20T12:00:01Z",
            "item_visible": True,
            "hand_near_item": False,
            "motion_score": 0.12,
        },
        {
            "source_frame_index": 2,
            "timestamp_utc": "2026-03-20T12:00:02Z",
            "item_visible": True,
            "hand_near_item": True,
            "motion_score": 0.38,
        },
        {
            "source_frame_index": 3,
            "timestamp_utc": "2026-03-20T12:00:03Z",
            "item_visible": False,
            "hand_near_item": False,
            "motion_score": 0.61,
        },
    ]

    last_response = None
    for payload in seed_payloads:
        last_response = client.post("/vision/observations", json=payload)
        assert last_response.status_code == 200

    assert last_response is not None
    body = last_response.json()
    assert body["processed"] is True
    assert body["event"] is not None
    assert body["event"]["event_type"] == "suspected_item_disappearance"
    assert body["event"]["observed_sku"] == "sku-apple-001"


def test_vision_observation_no_event_without_recent_hand_proximity() -> None:
    client = TestClient(agent_app)

    response_one = client.post(
        "/vision/observations",
        json={
            "source_frame_index": 500,
            "timestamp_utc": "2026-03-20T12:08:20Z",
            "item_visible": True,
            "hand_near_item": False,
            "motion_score": 0.2,
        },
    )
    assert response_one.status_code == 200

    response_two = client.post(
        "/vision/observations",
        json={
            "source_frame_index": 501,
            "timestamp_utc": "2026-03-20T12:08:21Z",
            "item_visible": False,
            "hand_near_item": False,
            "motion_score": 0.7,
        },
    )
    assert response_two.status_code == 200
    assert response_two.json()["event"] is None


def test_demo_scenario_creates_dashboard_incident() -> None:
    client = TestClient(agent_app)
    demo_response = client.post("/demo/run")
    assert demo_response.status_code == 200
    assert demo_response.json()["observations_processed"] == 11

    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    assert metrics_response.json()["total_incidents"] >= 1


def test_copilot_brief_and_chat_endpoints() -> None:
    client = TestClient(agent_app)
    status = client.get("/copilot/status")
    assert status.status_code == 200
    assert "mode" in status.json()

    brief = client.get("/copilot/brief")
    assert brief.status_code == 200
    brief_body = brief.json()
    assert "narrative" in brief_body
    assert "risk_level" in brief_body
    assert "recommended_action" in brief_body

    chat = client.post("/copilot/chat", json={"question": "What is happening now?"})
    assert chat.status_code == 200
    chat_body = chat.json()
    assert "answer" in chat_body
    assert "recommended_action" in chat_body
