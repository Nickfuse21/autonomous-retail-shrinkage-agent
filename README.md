# Autonomous Retail Shrinkage Agent

> Agentic computer vision system that detects suspicious retail behavior, validates against POS data, and autonomously escalates with evidence.

<p align="center">
  <img src="assets/hero-banner.svg" alt="Autonomous Retail Shrinkage Agent Banner" width="100%" />
</p>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Production%20API-009688?logo=fastapi&logoColor=white)
![CI](https://img.shields.io/badge/CI-Ready-111827)
![MLOps](https://img.shields.io/badge/MLOps-Agentic%20Pipeline-6D28D9)
![Status](https://img.shields.io/badge/Status-Active%20Build-059669)

## Business and engineering value

This project models a practical retail operations workflow instead of a standalone ML demo:
- Detects suspicious shelf behavior and converts it into structured events
- Validates events against POS scans before escalation to reduce false alarms
- Produces incident artifacts (timeline, evidence reference, alert payload) for auditability
- Exposes service APIs and a live dashboard for operations visibility

The architecture emphasizes production concerns: deterministic workflows, test coverage, explicit service boundaries, and reproducible local deployment.

## System architecture

```mermaid
flowchart LR
  cameraFeed[CameraFeed] --> streamProcessor[StreamProcessor]
  streamProcessor --> behaviorDetector[BehaviorDetector]
  behaviorDetector --> candidateEvent[CandidateEvent]
  candidateEvent --> decisionEngine[DecisionEngine]
  decisionEngine --> posClient[POSClient]
  posClient --> mockPOS[MockPOSAPI]
  decisionEngine -->|NoPOSMatch| incidentManager[IncidentManager]
  incidentManager --> clipGenerator[ClipGenerator]
  clipGenerator --> evidenceStore[EvidenceStore]
  incidentManager --> slackNotifier[SlackNotifier]
  slackNotifier --> managerChannel[StoreManagerChannel]
  decisionEngine --> auditLog[AuditLog]
  incidentManager --> auditLog
```

## What the agent does

1. Watches webcam/video stream for suspicious concealment behavior.
2. Creates a candidate event with timestamp and confidence.
3. Queries POS scans in a configurable time window.
4. Flags mismatch between observed item behavior and scanned inventory.
5. Generates an incident-centered 5-second clip.
6. Sends a structured Slack alert with evidence and reason code.

## Visual dashboard

Open `http://localhost:8080/` to access a polished operations dashboard with:
- Live incident metrics (total, escalated, resolved)
- Real-time incident feed table
- Event stream log
- One-click `Run Demo Scenario` button for instant end-to-end demo

## Tech stack

- **Core:** Python, FastAPI, Pydantic
- **Vision/data plane (next build steps):** OpenCV, FFmpeg
- **Transport:** HTTPX for POS and webhook calls
- **Quality:** Pytest, Ruff, MyPy, GitHub Actions
- **Deployment:** Docker Compose (agent + POS mock)

## Project layout

```text
autonomous-retail-shrinkage-agent/
  docs/
    architecture.md
  src/
    agent/main.py
    api/mock_pos_api.py
    vision/
    pos/
    incidents/
    alerts/
  tests/test_smoke.py
  docker-compose.yml
  pyproject.toml
  README.md
```

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
uvicorn src.agent.main:app --reload --port 8080
```

Run mock POS in another terminal:

```powershell
uvicorn src.api.mock_pos_api:app --reload --port 8081
```

Health checks:

- Agent: `http://localhost:8080/health`
- Mock POS: `http://localhost:8081/health`

Demo endpoints:
- `GET /` dashboard UI
- `POST /demo/run` generate sample behavior sequence
- `GET /vision/events` suspicious event stream
- `GET /incidents` processed incident objects
- `GET /metrics` dashboard counters

## Docker run

```powershell
docker compose up --build
```

## Engineering roadmap (7-day sprint)

- Day 1: project scaffold, quality gates, architecture
- Day 2: video ingestion pipeline + event schema
- Day 3: POS mock/service client + temporal correlation logic
- Day 4: decision engine state machine + incident lifecycle
- Day 5: deterministic 5-second clip generation + evidence package
- Day 6: Slack incident cards + operational hardening
- Day 7: polishing, tests, benchmark notes, and demo assets

## Phase 2 progress

- Added observation ingestion endpoint: `POST /vision/observations`
- Added event listing endpoint: `GET /vision/events`
- Implemented in-memory frame buffer and baseline disappearance detector
- Added tests for both event and no-event behavior patterns

## Key implementation highlights

- Multi-signal decisioning pipeline combining vision events and POS transaction checks
- Incident lifecycle management with escalation and resolution states
- Evidence packaging and alert dispatch simulation for downstream operations tooling
- Operator-facing dashboard for real-time status, metrics, and incident review
