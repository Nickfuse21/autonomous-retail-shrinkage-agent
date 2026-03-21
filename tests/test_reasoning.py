from src.vision.reasoning import ReasoningChainBuilder


def test_escalated_chain_with_all_signals() -> None:
    builder = ReasoningChainBuilder()
    chain = builder.build(
        incident_id="inc_test_001",
        behavior_explanation="Classic concealment: linger, pickup, scan, conceal.",
        behavior_confidence=0.92,
        zone_explanation="Exit-directed trajectory with 78% exit probability.",
        zone_exit_probability=0.78,
        pos_matched=False,
        pos_reason="No matching POS scan in 60s window.",
        observed_sku="sku-electronics-042",
        final_confidence=0.88,
    )
    assert chain.final_verdict == "ESCALATED"
    assert chain.final_confidence == 0.88
    assert len(chain.links) == 6
    assert "observation" in chain.links[0].step.value
    assert "ESCALATED" in chain.narrative
    d = chain.to_dict()
    assert d["incident_id"] == "inc_test_001"
    assert len(d["links"]) == 6


def test_resolved_chain_when_pos_matches() -> None:
    builder = ReasoningChainBuilder()
    chain = builder.build(
        incident_id="inc_test_002",
        behavior_explanation=None,
        behavior_confidence=None,
        zone_explanation=None,
        zone_exit_probability=None,
        pos_matched=True,
        pos_reason="Matching scan found for tx-5001.",
        observed_sku="sku-grocery-010",
        final_confidence=0.25,
    )
    assert chain.final_verdict == "RESOLVED"
    assert "RESOLVED" in chain.narrative


def test_chain_includes_risk_factors() -> None:
    builder = ReasoningChainBuilder()
    chain = builder.build(
        incident_id="inc_test_003",
        behavior_explanation="Grab-and-go pattern detected.",
        behavior_confidence=0.88,
        zone_explanation="Exit trajectory detected.",
        zone_exit_probability=0.72,
        pos_matched=False,
        pos_reason="No match.",
        observed_sku="sku-phone-001",
        final_confidence=0.91,
    )
    conf_step = [link for link in chain.links if link.step.value == "confidence_assessment"][0]
    assert "high-confidence behavioral pattern" in conf_step.description
    assert "exit-directed trajectory" in conf_step.description
    assert "no POS scan match" in conf_step.description
