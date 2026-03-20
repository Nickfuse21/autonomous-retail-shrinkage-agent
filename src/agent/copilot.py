from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class CopilotResult:
    narrative: str
    risk_level: str
    recommended_action: str
    possibilities: list[str]
    source: str


class AgenticCopilotService:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._last_error: str = ""
        self._active_model: str | None = None

    def status(self) -> dict[str, object]:
        return {
            "enabled": bool(self.api_key),
            "provider": "gemini",
            "model": self.model,
            "mode": "live" if self.api_key else "fallback",
            "active_model": self._active_model,
            "last_error": self._last_error or None,
        }

    def generate_brief(self, context: dict[str, Any]) -> CopilotResult:
        if not self.api_key:
            return self._fallback_brief(context, "No API key configured. Using local reasoning mode.")
        prompt = self._brief_prompt(context)
        response_text = self._call_gemini(prompt)
        if response_text is None:
            return self._fallback_brief(context, "Gemini unavailable. Using local reasoning mode.")
        return self._parse_response(response_text, source="gemini")

    def answer_question(self, question: str, context: dict[str, Any]) -> CopilotResult:
        if not self.api_key:
            return self._fallback_qa(question, context, "No API key configured. Using local reasoning mode.")
        prompt = self._qa_prompt(question, context)
        response_text = self._call_gemini(prompt)
        if response_text is None:
            return self._fallback_qa(question, context, "Gemini unavailable. Using local reasoning mode.")
        return self._parse_response(response_text, source="gemini")

    def _call_gemini(self, prompt: str) -> str | None:
        if not self.api_key:
            self._last_error = "api_key_missing"
            return None
        candidate_models = [
            self.model,
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ]
        try:
            with httpx.Client(timeout=12.0) as client:
                for model_name in candidate_models:
                    endpoint = (
                        "https://generativelanguage.googleapis.com/v1beta/models/"
                        f"{model_name}:generateContent"
                    )
                    resp = client.post(
                        f"{endpoint}?key={self.api_key}",
                        json={
                            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                            "generationConfig": {
                                "temperature": 0.2,
                                "maxOutputTokens": 500,
                                "responseMimeType": "application/json",
                            },
                        },
                    )
                    if resp.status_code != 200:
                        self._last_error = f"{model_name}:{resp.status_code}"
                        continue
                    payload = resp.json()
                    text = (
                        payload.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text")
                    )
                    if text:
                        self._active_model = model_name
                        self._last_error = ""
                        return text
                return None
        except Exception:
            self._last_error = "network_or_provider_error"
            return None

    def _brief_prompt(self, context: dict[str, Any]) -> str:
        compact = self._compact_context(context)
        return (
            "You are an autonomous retail loss prevention copilot. "
            "Analyze the context and produce a concrete operator briefing. "
            "Use factual details from context (incident id, sku, confidence, status, trajectory, POS match) "
            "instead of generic advice.\n\n"
            "Return strict JSON only with keys:\n"
            '{"narrative":"...", "risk_level":"low|medium|high|critical", '
            '"recommended_action":"...", "possibilities":["...","...","..."]}\n\n'
            "Rules:\n"
            "- If no active incident, explicitly say system is stable and what to monitor next.\n"
            "- Mention at least 2 concrete signals when incidents exist.\n"
            "- Keep narrative to 2-5 sentences.\n\n"
            f"CONTEXT_JSON:\n{json.dumps(compact, ensure_ascii=True)}"
        )

    def _qa_prompt(self, question: str, context: dict[str, Any]) -> str:
        compact = self._compact_context(context)
        return (
            "You are an autonomous retail loss prevention copilot. "
            "Answer user question using only supplied context. Keep response practical.\n\n"
            "Return strict JSON only with keys:\n"
            '{"narrative":"...", "risk_level":"low|medium|high|critical", '
            '"recommended_action":"...", "possibilities":["...","...","..."]}\n\n'
            "Rules:\n"
            "- Directly answer the question first.\n"
            "- Reference concrete context values where available.\n"
            "- If context is insufficient, state what is missing.\n\n"
            f"QUESTION: {question}\n"
            f"CONTEXT_JSON:\n{json.dumps(compact, ensure_ascii=True)}"
        )

    def _parse_response(self, text: str, source: str) -> CopilotResult:
        narrative = "No narrative provided."
        risk = "medium"
        action = "Continue monitoring."
        possibilities: list[str] = []

        parsed_json = self._try_parse_json(text)
        if parsed_json is not None:
            narrative = str(parsed_json.get("narrative") or narrative).strip()
            risk = str(parsed_json.get("risk_level") or risk).strip().lower()
            action = str(parsed_json.get("recommended_action") or action).strip()
            raw_poss = parsed_json.get("possibilities")
            if isinstance(raw_poss, list):
                possibilities = [str(x).strip() for x in raw_poss if str(x).strip()]
        else:
            n_match = re.search(r"NARRATIVE:\s*(.*?)(?:\n[A-Z_]+:|\Z)", text, re.IGNORECASE | re.DOTALL)
            r_match = re.search(r"RISK_LEVEL:\s*([^\n]+)", text, re.IGNORECASE)
            a_match = re.search(r"RECOMMENDED_ACTION:\s*(.*?)(?:\n[A-Z_]+:|\Z)", text, re.IGNORECASE | re.DOTALL)
            p_match = re.search(r"POSSIBILITIES:\s*(.*?)(?:\n[A-Z_]+:|\Z)", text, re.IGNORECASE | re.DOTALL)
            if n_match:
                narrative = n_match.group(1).strip()
            elif text.strip():
                # If model ignores schema, still show a useful answer.
                narrative = text.strip()[:900]
            if r_match:
                risk = r_match.group(1).strip().lower()
            if a_match:
                action = a_match.group(1).strip()
            if p_match:
                raw = p_match.group(1).strip()
                possibilities = [p.strip("- ").strip() for p in re.split(r"[|\n]", raw) if p.strip()]

        if risk not in {"low", "medium", "high", "critical"}:
            risk = "medium"
        if not possibilities:
            possibilities = [
                "False positive likely if POS scan appears shortly.",
                "Concealment risk increases if movement shifts toward exit.",
                "Human review can validate intent from trajectory and behavior chain.",
            ]
        return CopilotResult(
            narrative=narrative,
            risk_level=risk,
            recommended_action=action,
            possibilities=possibilities[:5],
            source=source,
        )

    def _fallback_brief(self, context: dict[str, Any], reason: str) -> CopilotResult:
        escalated = int(context.get("metrics", {}).get("escalated_incidents", 0))
        risk = "high" if escalated > 0 else "medium"
        latest = context.get("latest_incident")
        latest_msg = (
            f"Latest incident {latest.get('incident_id')} with confidence {latest.get('confidence')}."
            if latest
            else "No recent incidents in current window."
        )
        narrative = (
            f"{reason} System continues deterministic analysis. "
            f"{latest_msg} Escalated incidents in window: {escalated}."
        )
        return CopilotResult(
            narrative=narrative,
            risk_level=risk,
            recommended_action="Review top incident panel and verify POS mismatch before escalation.",
            possibilities=[
                "If POS scan arrives late, status may resolve automatically.",
                "If exit trajectory remains high, escalate to floor response.",
                "If similar patterns repeat, tighten aisle surveillance policy.",
            ],
            source="fallback",
        )

    def _fallback_qa(self, question: str, context: dict[str, Any], reason: str) -> CopilotResult:
        base = self._fallback_brief(context, reason)
        base.narrative = f"{reason} Question: {question}. {base.narrative}"
        return base

    def _try_parse_json(self, text: str) -> dict[str, Any] | None:
        candidate = text.strip()
        if not candidate:
            return None
        # Remove fenced markdown wrappers if present.
        if candidate.startswith("```"):
            candidate = re.sub(r"^```[a-zA-Z]*\s*", "", candidate)
            candidate = re.sub(r"\s*```$", "", candidate)
        # Direct JSON parse.
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # Extract first JSON object block if extra text exists.
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(candidate[start : end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
        return None

    def _compact_context(self, context: dict[str, Any]) -> dict[str, Any]:
        metrics = context.get("metrics", {})
        latest = context.get("latest_incident") or {}
        incidents = context.get("recent_incidents") or []
        compact_incidents: list[dict[str, Any]] = []
        for item in incidents[-3:]:
            compact_incidents.append(
                {
                    "incident_id": item.get("incident_id"),
                    "status": item.get("status"),
                    "sku": item.get("observed_sku"),
                    "confidence": item.get("confidence"),
                    "pos_match": item.get("pos_match"),
                    "behavior_pattern": item.get("behavior_pattern"),
                    "zone_heading": item.get("zone_heading"),
                    "zone_exit_probability": item.get("zone_exit_probability"),
                    "review_status": item.get("review_status"),
                }
            )
        return {
            "timestamp_utc": context.get("timestamp_utc"),
            "metrics": {
                "total_incidents": metrics.get("total_incidents", 0),
                "escalated_incidents": metrics.get("escalated_incidents", 0),
                "resolved_incidents": metrics.get("resolved_incidents", 0),
                "frames_processed": metrics.get("frames_processed", 0),
            },
            "latest_incident": {
                "incident_id": latest.get("incident_id"),
                "status": latest.get("status"),
                "sku": latest.get("observed_sku"),
                "confidence": latest.get("confidence"),
                "pos_match": latest.get("pos_match"),
                "behavior_pattern": latest.get("behavior_pattern"),
                "zone_heading": latest.get("zone_heading"),
                "zone_exit_probability": latest.get("zone_exit_probability"),
            }
            if latest
            else None,
            "recent_incidents": compact_incidents,
            "recent_events_count": len(context.get("recent_events") or []),
            "behavior_tail_count": len(context.get("behavior_tail") or []),
        }
