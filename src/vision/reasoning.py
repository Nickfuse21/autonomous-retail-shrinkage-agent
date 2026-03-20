"""
Explainable reasoning chain generator.

Produces human-readable, step-by-step decision narratives that explain
exactly why the system escalated (or resolved) an incident. Designed
for auditability and transparency in loss prevention operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class ReasoningStep(str, Enum):
    OBSERVATION = "observation"
    BEHAVIOR_ANALYSIS = "behavior_analysis"
    ZONE_ANALYSIS = "zone_analysis"
    POS_VALIDATION = "pos_validation"
    CONFIDENCE_ASSESSMENT = "confidence_assessment"
    VERDICT = "verdict"


@dataclass
class ChainLink:
    step: ReasoningStep
    description: str
    data: dict[str, object] = field(default_factory=dict)
    timestamp_utc: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp_utc:
            self.timestamp_utc = datetime.now(timezone.utc).isoformat()


@dataclass
class ReasoningChain:
    incident_id: str
    links: list[ChainLink] = field(default_factory=list)
    final_verdict: str = ""
    final_confidence: float = 0.0
    narrative: str = ""

    def add(self, step: ReasoningStep, description: str, data: dict[str, object] | None = None) -> None:
        self.links.append(ChainLink(step=step, description=description, data=data or {}))

    def finalize(self, verdict: str, confidence: float) -> None:
        self.final_verdict = verdict
        self.final_confidence = confidence
        self.narrative = self._build_narrative()

    def _build_narrative(self) -> str:
        parts: list[str] = []
        for i, link in enumerate(self.links, 1):
            parts.append(f"Step {i} [{link.step.value}]: {link.description}")
        parts.append(f"VERDICT: {self.final_verdict} (confidence: {self.final_confidence:.1%})")
        return " → ".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "incident_id": self.incident_id,
            "links": [
                {
                    "step": link.step.value,
                    "description": link.description,
                    "data": link.data,
                    "timestamp_utc": link.timestamp_utc,
                }
                for link in self.links
            ],
            "final_verdict": self.final_verdict,
            "final_confidence": self.final_confidence,
            "narrative": self.narrative,
        }


class ReasoningChainBuilder:
    """Constructs a reasoning chain by accumulating evidence from
    the behavior analyzer, zone engine, and POS client."""

    def build(
        self,
        incident_id: str,
        behavior_explanation: str | None,
        behavior_confidence: float | None,
        zone_explanation: str | None,
        zone_exit_probability: float | None,
        pos_matched: bool,
        pos_reason: str,
        observed_sku: str,
        final_confidence: float,
    ) -> ReasoningChain:
        chain = ReasoningChain(incident_id=incident_id)

        chain.add(
            ReasoningStep.OBSERVATION,
            f"Suspicious activity detected for item {observed_sku}.",
            {"sku": observed_sku},
        )

        if behavior_explanation:
            chain.add(
                ReasoningStep.BEHAVIOR_ANALYSIS,
                behavior_explanation,
                {"pattern_confidence": behavior_confidence or 0.0},
            )
        else:
            chain.add(
                ReasoningStep.BEHAVIOR_ANALYSIS,
                "No known behavioral sequence matched. Using baseline detection signals.",
            )

        if zone_explanation:
            chain.add(
                ReasoningStep.ZONE_ANALYSIS,
                zone_explanation,
                {"exit_probability": zone_exit_probability or 0.0},
            )
        else:
            chain.add(
                ReasoningStep.ZONE_ANALYSIS,
                "Zone trajectory data not available for this incident.",
            )

        chain.add(
            ReasoningStep.POS_VALIDATION,
            f"POS cross-check: {'Match found' if pos_matched else 'No matching scan'}. {pos_reason}",
            {"pos_matched": pos_matched},
        )

        risk_factors: list[str] = []
        if behavior_confidence and behavior_confidence > 0.8:
            risk_factors.append("high-confidence behavioral pattern")
        if zone_exit_probability and zone_exit_probability > 0.6:
            risk_factors.append("exit-directed trajectory")
        if not pos_matched:
            risk_factors.append("no POS scan match")

        chain.add(
            ReasoningStep.CONFIDENCE_ASSESSMENT,
            f"Risk factors: {', '.join(risk_factors) if risk_factors else 'none identified'}. "
            f"Combined confidence: {final_confidence:.1%}.",
            {"risk_factors": risk_factors, "combined_confidence": final_confidence},
        )

        verdict = "ESCALATED" if not pos_matched and final_confidence > 0.5 else "RESOLVED"
        chain.add(
            ReasoningStep.VERDICT,
            f"Final decision: {verdict}.",
            {"verdict": verdict},
        )
        chain.finalize(verdict, final_confidence)
        return chain
