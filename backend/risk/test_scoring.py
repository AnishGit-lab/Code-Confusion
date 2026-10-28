"""
Tests for the PriviScan sector-free risk scoring algorithm.
"""

from backend.risk.scoring import (
    analyze_risk,
    calculate_average_risk,
    calculate_individual_risk,
    calculate_impact,
    calculate_likelihood,
)


def test_empty_document():
    result = analyze_risk([])

    assert result["risk_score"] == 0.0
    assert result["risk_level"] == "LOW"
    assert result["total_detections"] == 0
    assert result["average_risk_score"] == 0.0


def test_individual_risk_formula():
    """
    Impact = 1.5
    Likelihood = 2

    Expected:

        (1.5 * 2 / 25) * 100
        = 12
    """

    result = calculate_individual_risk(
        1.5,
        2.0,
    )

    assert result == 12.0


def test_impact_has_four_factors():
    factors = calculate_impact(
        "NAME",
        "Anish Kandel",
    )

    assert "identifiability" in factors
    assert "uniqueness" in factors
    assert "potential_harm" in factors
    assert "reusability" in factors
    assert "impact" in factors

    assert 1.0 <= factors["identifiability"] <= 5.0
    assert 1.0 <= factors["uniqueness"] <= 5.0
    assert 1.0 <= factors["potential_harm"] <= 5.0
    assert 1.0 <= factors["reusability"] <= 5.0
    assert 1.0 <= factors["impact"] <= 5.0


def test_likelihood_range():
    result = calculate_likelihood(
        confidence=0.99,
        linkability=3.0,
    )

    assert 1.0 <= result <= 5.0


def test_name_only():
    detections = [
        {
            "type": "NAME",
            "value": "Anish Kandel",
            "confidence": 0.99,
        }
    ]

    result = analyze_risk(
        detections
    )

    assert result["total_detections"] == 1
    assert result["risk_score"] >= 0.0
    assert result["risk_score"] <= 100.0

    assert len(
        result["detections"]
    ) == 1

    detection = result["detections"][0]

    assert detection["type"] == "NAME"
    assert detection["risk_score"] >= 0.0
    assert detection["risk_score"] <= 100.0


def test_name_email_linkability():
    detections = [
        {
            "type": "NAME",
            "value": "Anish Kandel",
            "confidence": 0.99,
        },
        {
            "type": "EMAIL",
            "value": "anish@example.com",
            "confidence": 0.99,
        },
    ]

    result = analyze_risk(
        detections
    )

    assert (
        result["linkability_adjustment"]
        > 0
    )

    assert (
        result["risk_score"]
        >= result["average_risk_score"]
    )


def test_name_email_phone_id_combination():
    detections = [
        {
            "type": "NAME",
            "value": "Anish Kandel",
            "confidence": 0.99,
        },
        {
            "type": "EMAIL",
            "value": "anish@example.com",
            "confidence": 0.99,
        },
        {
            "type": "PHONE",
            "value": "+1 202-555-0123",
            "confidence": 0.98,
        },
        {
            "type": "ID_NUMBER",
            "value": "STU-2026-A123",
            "confidence": 0.99,
        },
    ]

    result = analyze_risk(
        detections
    )

    assert (
        result["total_detections"]
        == 4
    )

    assert (
        result["linkability_adjustment"]
        > 0
    )

    assert (
        result["risk_score"]
        <= 100
    )


def test_government_id_has_high_impact():
    detections = [
        {
            "type": "GOVERNMENT_ID",
            "value": "GOV-AB123456789",
            "confidence": 0.99,
        }
    ]

    result = analyze_risk(
        detections
    )

    detection = result["detections"][0]

    assert (
        detection["impact"]
        >= 4.0
    )


def test_high_confidence_password_critical_override():
    detections = [
        {
            "type": "PASSWORD",
            "value": "MySecretPassword123!",
            "confidence": 0.99,
        }
    ]

    result = analyze_risk(
        detections
    )

    assert (
        result["credential_override"]
        is True
    )

    assert (
        result["risk_score"]
        == 100.0
    )

    assert (
        result["risk_level"]
        == "CRITICAL"
    )

    detection = result["detections"][0]

    assert (
        detection["credential_override"]
        is True
    )

    assert (
        detection["risk_score"]
        == 100.0
    )


def test_high_confidence_api_key_critical_override():
    detections = [
        {
            "type": "API_KEY",
            "value": "sk-test-123456789",
            "confidence": 0.99,
        }
    ]

    result = analyze_risk(
        detections
    )

    assert (
        result["risk_score"]
        == 100.0
    )

    assert (
        result["risk_level"]
        == "CRITICAL"
    )


def test_low_confidence_password_no_override():
    detections = [
        {
            "type": "PASSWORD",
            "value": "PossiblePassword",
            "confidence": 0.70,
        }
    ]

    result = analyze_risk(
        detections
    )

    assert (
        result["credential_override"]
        is False
    )

    assert (
        result["risk_score"]
        < 100.0
    )


def test_risk_is_capped_at_100():
    detections = [
        {
            "type": "NAME",
            "value": "Anish Kandel",
            "confidence": 0.99,
        },
        {
            "type": "EMAIL",
            "value": "anish@example.com",
            "confidence": 0.99,
        },
        {
            "type": "PHONE",
            "value": "+1 202-555-0123",
            "confidence": 0.99,
        },
        {
            "type": "ADDRESS",
            "value": "123 Main Street, London",
            "confidence": 0.99,
        },
        {
            "type": "GOVERNMENT_ID",
            "value": "GOV-AB123456789",
            "confidence": 0.99,
        },
        {
            "type": "BANK_ACCOUNT",
            "value": "1234567890123456",
            "confidence": 0.99,
        },
    ]

    result = analyze_risk(
        detections
    )

    assert (
        result["risk_score"]
        <= 100.0
    )


def test_average_risk_compatibility_function():
    detections = [
        {
            "type": "NAME",
            "value": "Anish Kandel",
            "confidence": 0.99,
        },
        {
            "type": "EMAIL",
            "value": "anish@example.com",
            "confidence": 0.99,
        },
    ]

    average = calculate_average_risk(
        detections
    )

    assert 0.0 <= average <= 100.0


def test_bounding_box_is_preserved():
    detections = [
        {
            "type": "EMAIL",
            "value": "anish@example.com",
            "confidence": 0.99,
            "box": [
                10,
                20,
                200,
                50,
            ],
        }
    ]

    result = analyze_risk(
        detections
    )

    scored_detection = (
        result["detections"][0]
    )

    assert (
        scored_detection["box"]
        == [
            10,
            20,
            200,
            50,
        ]
    )


def test_no_sector_required():
    """
    The finalized implementation must work without any
    sector argument.

    This test intentionally provides only:

        type
        value
        confidence
    """

    detections = [
        {
            "type": "NAME",
            "value": "Anish Kandel",
            "confidence": 0.99,
        },
        {
            "type": "EMAIL",
            "value": "anish@example.com",
            "confidence": 0.99,
        },
    ]

    result = analyze_risk(
        detections
    )

    assert (
        "risk_score"
        in result
    )

    assert (
        "risk_level"
        in result
    )

    assert (
        "linkability_adjustment"
        in result
    )

    assert (
        "sector"
        not in result
    )