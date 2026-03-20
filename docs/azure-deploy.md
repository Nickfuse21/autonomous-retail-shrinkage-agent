# Azure Deployment Checklist

This checklist is optimized for Azure App Service (Container) using free credits.

## 1) Container image

Build and push image to Azure Container Registry (ACR) or Docker Hub:

```powershell
docker build -t retail-guard:latest .
```

## 2) App Service setup

- Create **Web App for Containers** (Linux)
- Configure container image source (ACR/Docker Hub)
- Set health check path to `/health`

## 3) Required environment variables

```text
APP_ENV=production
AGENT_PORT=8080
POS_API_URL=https://<your-pos-service-url>
CORS_ALLOWED_ORIGINS=https://<your-app>.azurewebsites.net
COPILOT_PROVIDER=local
DETECTOR_PRETRAINED_MODEL=yolo11m.pt
DETECTOR_DEVICE=cpu
API_TOKEN=<strong-random-token>
REQUIRE_API_TOKEN_IN_NON_DEV=true
INCIDENT_DB_PATH=/home/site/wwwroot/data/incidents.db
AUTHZ_REVIEW_ROLES=manager,admin
AUTHZ_EXPORT_ROLES=manager,admin,auditor
```

Notes:
- `DETECTOR_DEVICE=cpu` is recommended for standard App Service plans.
- Use `X-API-Token` header for protected write/compute routes.
- Use `X-Actor-Role` for review/export policy checks in production.
- Persist `/home/site/wwwroot/data` via App Service storage or mount.

## 4) Verification

After deployment:

- `GET /health` returns status `ok`
- Dashboard loads from root `/`
- `POST /vision/detect-frame` works with valid token header
- `GET /health/dependencies` returns dependency details
- `GET /metrics/extended` shows endpoint latency counters

## 5) Optional next upgrades

- Move incident persistence to managed DB
- Store clips to Azure Blob Storage
- Add App Insights monitoring and alerting
