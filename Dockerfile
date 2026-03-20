FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV AGENT_PORT=8080

WORKDIR /app

COPY . /app
RUN pip install --upgrade pip && pip install -e .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn src.agent.main:app --host 0.0.0.0 --port ${AGENT_PORT}"]
