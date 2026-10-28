"""
PriviScan Risk Scoring Engine
================================

Final sector-free PriviScan risk algorithm.

Core model:

    Impact
        = Average(
            Identifiability,
            Uniqueness,
            Potential Harm,
            Reusability
          )

    Likelihood
        = based on Detection Confidence + Linkability

    Individual Risk
        = Impact * Likelihood

    Normalized Individual Risk
        = (Impact * Likelihood / 25) * 100

    Final Document Risk
        = Aggregate(Item Risks)
        + Linkability Adjustment

    Final Risk
        = min(100, Final Document Risk)

Risk levels:

    0 - 20    LOW
    21 - 40   MEDIUM
    41 - 70   HIGH
    71 - 100  CRITICAL

Credentials:

    PASSWORD / API_KEY / SECRET

    If confidence >= 0.85:
        Final Risk = 100
        Risk Level = CRITICAL

Important design principle:

    PriviScan does NOT assign a permanent score such as:

        NAME = 20
        EMAIL = 40
        PHONE = 60

    Instead, each detection is evaluated using multiple
    explainable factors.

Sector/context classification is intentionally NOT used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional
import re


# ============================================================
# CONSTANTS
# ============================================================

MIN_FACTOR = 1.0
MAX_FACTOR = 5.0

MIN_RISK = 0.0
MAX_RISK = 100.0

CREDENTIAL_CONFIDENCE_THRESHOLD = 0.85

RISK_LEVEL_LOW = "LOW"
RISK_LEVEL_MEDIUM = "MEDIUM"
RISK_LEVEL_HIGH = "HIGH"
RISK_LEVEL_CRITICAL = "CRITICAL"


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class RiskAssessment:
    """
    Complete risk assessment for one detected item.
    """

    detection_type: str
    value: str
    confidence: float

    identifiability: float
    uniqueness: float
    potential_harm: float
    reusability: float

    impact: float

    linkability: float
    likelihood: float

    individual_risk: float

    risk_level: str
    explanation: str

    credential_override: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert assessment to a dictionary suitable for
        FastAPI / HTML / JSON output.
        """

        return {
            "type": self.detection_type,
            "value": self.value,
            "confidence": self.confidence,

            "identifiability": round(self.identifiability, 2),
            "uniqueness": round(self.uniqueness, 2),
            "potential_harm": round(self.potential_harm, 2),
            "reusability": round(self.reusability, 2),

            "impact": round(self.impact, 2),

            "linkability": round(self.linkability, 2),
            "likelihood": round(self.likelihood, 2),

            "risk_score": round(self.individual_risk, 2),
            "risk_level": self.risk_level,

            "credential_override": self.credential_override,

            "explanation": self.explanation,
        }


# ============================================================
# BASIC HELPERS
# ============================================================

def _clamp(
    value: float,
    minimum: float = MIN_FACTOR,
    maximum: float = MAX_FACTOR,
) -> float:
    """
    Keep a numerical factor inside the required range.
    """

    return max(minimum, min(maximum, float(value)))


def _clamp_risk(value: float) -> float:
    """
    Keep a final risk score between 0 and 100.
    """

    return max(MIN_RISK, min(MAX_RISK, float(value)))


def _normalise_type(detection_type: Any) -> str:
    """
    Normalize detection type names.

    Examples:

        "email"          -> "EMAIL"
        "ID_NUMBER"      -> "ID_NUMBER"
        "government id"  -> "GOVERNMENT_ID"
        "api key"        -> "API_KEY"
    """

    if detection_type is None:
        return "UNKNOWN"

    text = str(detection_type).strip().upper()

    text = text.replace("-", "_")
    text = re.sub(r"\s+", "_", text)

    aliases = {
        "ID": "ID_NUMBER",
        "IDENTIFIER": "ID_NUMBER",
        "STUDENT_ID": "ID_NUMBER",
        "STAFF_ID": "ID_NUMBER",
        "EMPLOYEE_ID": "ID_NUMBER",
        "FACULTY_ID": "ID_NUMBER",
        "MEMBER_ID": "ID_NUMBER",

        "GOV_ID": "GOVERNMENT_ID",
        "GOVERNMENTID": "GOVERNMENT_ID",

        "BANK_ACCOUNT": "BANK_ACCOUNT",
        "ACCOUNT_NUMBER": "BANK_ACCOUNT",
        "BANK_ACCOUNT_NUMBER": "BANK_ACCOUNT",

        "BANK_CARD": "BANK_CARD",
        "CARD_NUMBER": "BANK_CARD",

        "APIKEY": "API_KEY",
        "API_TOKEN": "API_KEY",

        "SECRET_KEY": "SECRET",
        "PASSWORD": "PASSWORD",
    }

    return aliases.get(text, text)


def _safe_confidence(value: Any) -> float:
    """
    Convert confidence to a safe 0.0 - 1.0 value.
    """

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0

    return max(0.0, min(1.0, confidence))


def _risk_level(score: float) -> str:
    """
    Convert a 0-100 risk score to the PriviScan risk level.
    """

    score = _clamp_risk(score)

    if score <= 20:
        return RISK_LEVEL_LOW

    if score <= 40:
        return RISK_LEVEL_MEDIUM

    if score <= 70:
        return RISK_LEVEL_HIGH

    return RISK_LEVEL_CRITICAL


def _clean_value(value: Any) -> str:
    """
    Safely convert a detection value to text.
    """

    if value is None:
        return ""

    return str(value).strip()


# ============================================================
# CREDENTIAL DETECTION
# ============================================================

CREDENTIAL_TYPES = {
    "PASSWORD",
    "API_KEY",
    "SECRET",
    "ACCESS_TOKEN",
    "AUTH_TOKEN",
    "PRIVATE_KEY",
    "TOKEN",
}


def _is_credential_type(detection_type: str) -> bool:
    """
    Determine whether a detection is a security credential.
    """

    return _normalise_type(detection_type) in CREDENTIAL_TYPES


# ============================================================
# IMPACT FACTORS
# ============================================================

def _base_impact_profile(
    detection_type: str,
) -> Dict[str, float]:
    """
    Return explainable baseline impact factors.

    IMPORTANT:

    These are NOT final risk scores.

    They are starting values for the four impact dimensions.
    The final score is still affected by the actual detected
    value, confidence, uniqueness, linkability, and other
    adjustments.

    The exact numerical rubrics are PriviScan design parameters
    and should be validated with the project's dataset.
    """

    detection_type = _normalise_type(detection_type)

    profiles = {

        # ----------------------------------------------------
        # NAME
        # ----------------------------------------------------
        "NAME": {
            "identifiability": 2.0,
            "uniqueness": 1.5,
            "potential_harm": 1.5,
            "reusability": 1.5,
        },

        # ----------------------------------------------------
        # EMAIL
        # ----------------------------------------------------
        "EMAIL": {
            "identifiability": 3.0,
            "uniqueness": 3.0,
            "potential_harm": 2.5,
            "reusability": 3.0,
        },

        # ----------------------------------------------------
        # PHONE
        # ----------------------------------------------------
        "PHONE": {
            "identifiability": 3.0,
            "uniqueness": 3.5,
            "potential_harm": 3.0,
            "reusability": 3.0,
        },

        # ----------------------------------------------------
        # ADDRESS
        # ----------------------------------------------------
        "ADDRESS": {
            "identifiability": 3.0,
            "uniqueness": 3.0,
            "potential_harm": 3.0,
            "reusability": 2.5,
        },

        # ----------------------------------------------------
        # GENERAL ID
        # ----------------------------------------------------
        "ID_NUMBER": {
            "identifiability": 4.0,
            "uniqueness": 4.0,
            "potential_harm": 3.5,
            "reusability": 3.5,
        },

        # ----------------------------------------------------
        # GOVERNMENT ID
        # ----------------------------------------------------
        "GOVERNMENT_ID": {
            "identifiability": 5.0,
            "uniqueness": 5.0,
            "potential_harm": 4.5,
            "reusability": 4.5,
        },

        # ----------------------------------------------------
        # DATE OF BIRTH
        # ----------------------------------------------------
        "DATE_OF_BIRTH": {
            "identifiability": 3.0,
            "uniqueness": 2.5,
            "potential_harm": 2.5,
            "reusability": 3.0,
        },

        # ----------------------------------------------------
        # BANK ACCOUNT
        # ----------------------------------------------------
        "BANK_ACCOUNT": {
            "identifiability": 4.5,
            "uniqueness": 4.5,
            "potential_harm": 4.5,
            "reusability": 4.0,
        },

        # ----------------------------------------------------
        # BANK CARD
        # ----------------------------------------------------
        "BANK_CARD": {
            "identifiability": 4.5,
            "uniqueness": 4.5,
            "potential_harm": 5.0,
            "reusability": 4.5,
        },

        # ----------------------------------------------------
        # QR CODE
        # ----------------------------------------------------
        "QR": {
            "identifiability": 3.0,
            "uniqueness": 3.5,
            "potential_harm": 3.0,
            "reusability": 3.5,
        },

        "QR_CODE": {
            "identifiability": 3.0,
            "uniqueness": 3.5,
            "potential_harm": 3.0,
            "reusability": 3.5,
        },

        # ----------------------------------------------------
        # FACE
        # ----------------------------------------------------
        "FACE": {
            "identifiability": 4.0,
            "uniqueness": 4.0,
            "potential_harm": 3.5,
            "reusability": 3.5,
        },

        # ----------------------------------------------------
        # CREDENTIALS
        # ----------------------------------------------------
        "PASSWORD": {
            "identifiability": 5.0,
            "uniqueness": 5.0,
            "potential_harm": 5.0,
            "reusability": 5.0,
        },

        "API_KEY": {
            "identifiability": 5.0,
            "uniqueness": 5.0,
            "potential_harm": 5.0,
            "reusability": 5.0,
        },

        "SECRET": {
            "identifiability": 5.0,
            "uniqueness": 5.0,
            "potential_harm": 5.0,
            "reusability": 5.0,
        },

        "ACCESS_TOKEN": {
            "identifiability": 5.0,
            "uniqueness": 5.0,
            "potential_harm": 5.0,
            "reusability": 5.0,
        },

        "AUTH_TOKEN": {
            "identifiability": 5.0,
            "uniqueness": 5.0,
            "potential_harm": 5.0,
            "reusability": 5.0,
        },

        "PRIVATE_KEY": {
            "identifiability": 5.0,
            "uniqueness": 5.0,
            "potential_harm": 5.0,
            "reusability": 5.0,
        },
    }

    return profiles.get(
        detection_type,
        {
            "identifiability": 2.5,
            "uniqueness": 2.5,
            "potential_harm": 2.5,
            "reusability": 2.5,
        },
    ).copy()


# ============================================================
# VALUE-BASED IMPACT ADJUSTMENT
# ============================================================

def _value_characteristics(
    detection_type: str,
    value: str,
) -> Dict[str, bool]:
    """
    Examine characteristics of the actual detected value.

    This prevents the algorithm from behaving as a pure
    fixed category lookup.

    Example:

        A generic name and a highly unique full name do not
        necessarily have identical privacy characteristics.

        A generic ID and a government identifier do not have
        identical consequences.
    """

    detection_type = _normalise_type(detection_type)
    value = _clean_value(value)

    lower_value = value.lower()

    return {
        "has_domain": (
            detection_type == "EMAIL"
            and "@" in value
            and "." in value.split("@")[-1]
        ),

        "has_country_code": (
            detection_type == "PHONE"
            and bool(re.search(r"^\s*\+\d{1,3}", value))
        ),

        "long_identifier": (
            detection_type in {
                "ID_NUMBER",
                "GOVERNMENT_ID",
                "BANK_ACCOUNT",
                "BANK_CARD",
            }
            and len(re.sub(r"\W", "", value)) >= 10
        ),

        "very_long_identifier": (
            detection_type in {
                "ID_NUMBER",
                "GOVERNMENT_ID",
                "BANK_ACCOUNT",
                "BANK_CARD",
            }
            and len(re.sub(r"\W", "", value)) >= 14
        ),

        "credential_like": (
            detection_type in CREDENTIAL_TYPES
            or any(
                marker in lower_value
                for marker in (
                    "sk-",
                    "token",
                    "secret",
                    "password",
                    "api_key",
                    "apikey",
                )
            )
        ),

        "full_name": (
            detection_type == "NAME"
            and len(value.split()) >= 2
        ),

        "specific_address": (
            detection_type == "ADDRESS"
            and len(value) >= 15
        ),
    }


def _adjust_impact_for_value(
    detection_type: str,
    value: str,
    factors: Dict[str, float],
) -> Dict[str, float]:
    """
    Adjust the four impact factors based on characteristics
    of the actual detected value.

    The adjustment is intentionally small.

    The purpose is NOT to create arbitrary scores.

    The purpose is to allow the same detection category to
    behave differently when the actual detected information
    has different characteristics.
    """

    characteristics = _value_characteristics(
        detection_type,
        value,
    )

    # --------------------------------------------------------
    # Name
    # --------------------------------------------------------

    if characteristics["full_name"]:
        factors["identifiability"] += 0.5
        factors["uniqueness"] += 0.5

    # --------------------------------------------------------
    # Email
    # --------------------------------------------------------

    if characteristics["has_domain"]:
        factors["uniqueness"] += 0.25
        factors["reusability"] += 0.25

    # --------------------------------------------------------
    # International phone
    # --------------------------------------------------------

    if characteristics["has_country_code"]:
        factors["identifiability"] += 0.25
        factors["uniqueness"] += 0.25

    # --------------------------------------------------------
    # Long IDs / account numbers
    # --------------------------------------------------------

    if characteristics["long_identifier"]:
        factors["uniqueness"] += 0.25
        factors["reusability"] += 0.25

    if characteristics["very_long_identifier"]:
        factors["potential_harm"] += 0.25

    # --------------------------------------------------------
    # Address
    # --------------------------------------------------------

    if characteristics["specific_address"]:
        factors["identifiability"] += 0.25
        factors["potential_harm"] += 0.25

    # --------------------------------------------------------
    # Credential-like values
    # --------------------------------------------------------

    if characteristics["credential_like"]:
        factors["potential_harm"] = max(
            factors["potential_harm"],
            5.0,
        )

        factors["reusability"] = max(
            factors["reusability"],
            5.0,
        )

    # --------------------------------------------------------
    # Keep every factor valid.
    # --------------------------------------------------------

    for key in factors:
        factors[key] = _clamp(factors[key])

    return factors


# ============================================================
# IMPACT CALCULATION
# ============================================================

def calculate_impact(
    detection_type: str,
    value: str = "",
) -> Dict[str, float]:
    """
    Calculate the four impact factors and overall impact.

    Formula:

        Impact =
            Average(
                Identifiability,
                Uniqueness,
                Potential Harm,
                Reusability
            )
    """

    factors = _base_impact_profile(
        detection_type
    )

    factors = _adjust_impact_for_value(
        detection_type,
        value,
        factors,
    )

    impact = (
        factors["identifiability"]
        + factors["uniqueness"]
        + factors["potential_harm"]
        + factors["reusability"]
    ) / 4.0

    factors["impact"] = _clamp(
        impact
    )

    return factors


# ============================================================
# LINKABILITY
# ============================================================

def _detection_group(
    detection_type: str,
) -> str:
    """
    Group detection types according to the kind of
    relationship they provide.

    This is used only for linkability.

    It does NOT assign risk scores.
    """

    detection_type = _normalise_type(
        detection_type
    )

    if detection_type == "NAME":
        return "identity"

    if detection_type in {
        "EMAIL",
        "PHONE",
        "ADDRESS",
    }:
        return "contact"

    if detection_type in {
        "ID_NUMBER",
        "GOVERNMENT_ID",
        "DATE_OF_BIRTH",
        "FACE",
    }:
        return "identity"

    if detection_type in {
        "BANK_ACCOUNT",
        "BANK_CARD",
    }:
        return "financial"

    if detection_type in CREDENTIAL_TYPES:
        return "credential"

    if detection_type in {
        "QR",
        "QR_CODE",
    }:
        return "encoded"

    return "other"


def calculate_linkability(
    detection_type: str,
    value: str,
    all_detections: Iterable[Mapping[str, Any]],
) -> float:
    """
    Calculate a 1-5 linkability factor for one detection.

    Linkability becomes higher when the detected item can
    reasonably be connected with other detected information.

    Important:

        We do not simply increase risk because there are
        many detections.

        The algorithm looks for complementary information.
    """

    detection_type = _normalise_type(
        detection_type
    )

    current_value = _clean_value(
        value
    ).lower()

    current_group = _detection_group(
        detection_type
    )

    other_types: List[str] = []
    other_values: List[str] = []

    for item in all_detections:

        item_type = _normalise_type(
            item.get("type")
            or item.get("detection_type")
            or item.get("label")
        )

        item_value = _clean_value(
            item.get("value")
        )

        # Ignore the current detection itself.
        if (
            item_type == detection_type
            and item_value.lower() == current_value
        ):
            continue

        other_types.append(
            item_type
        )

        other_values.append(
            item_value.lower()
        )

    if not other_types:
        return 1.0

    groups = {
        _detection_group(item_type)
        for item_type in other_types
    }

    score = 1.0

    # --------------------------------------------------------
    # Name + contact information
    # --------------------------------------------------------

    if (
        detection_type == "NAME"
        and any(
            item in other_types
            for item in (
                "EMAIL",
                "PHONE",
                "ADDRESS",
            )
        )
    ):
        score += 1.5

    if (
        detection_type in {
            "EMAIL",
            "PHONE",
            "ADDRESS",
        }
        and "NAME" in other_types
    ):
        score += 1.5

    # --------------------------------------------------------
    # Identity combinations
    # --------------------------------------------------------

    if detection_type in {
        "ID_NUMBER",
        "GOVERNMENT_ID",
    }:
        if any(
            item in other_types
            for item in (
                "NAME",
                "EMAIL",
                "PHONE",
                "ADDRESS",
                "DATE_OF_BIRTH",
            )
        ):
            score += 1.5

    # --------------------------------------------------------
    # Date of birth + identity
    # --------------------------------------------------------

    if detection_type == "DATE_OF_BIRTH":
        if any(
            item in other_types
            for item in (
                "NAME",
                "ID_NUMBER",
                "GOVERNMENT_ID",
            )
        ):
            score += 1.0

    # --------------------------------------------------------
    # Financial combinations
    # --------------------------------------------------------

    if detection_type in {
        "BANK_ACCOUNT",
        "BANK_CARD",
    }:
        if any(
            item in other_types
            for item in (
                "NAME",
                "PHONE",
                "EMAIL",
                "ADDRESS",
            )
        ):
            score += 1.5

    # --------------------------------------------------------
    # Name + financial information
    # --------------------------------------------------------

    if detection_type == "NAME":
        if any(
            item in other_types
            for item in (
                "BANK_ACCOUNT",
                "BANK_CARD",
            )
        ):
            score += 1.5

    # --------------------------------------------------------
    # Multiple complementary groups
    # --------------------------------------------------------

    if len(groups) >= 3:
        score += 0.5

    # --------------------------------------------------------
    # Credential
    # --------------------------------------------------------

    if _is_credential_type(
        detection_type
    ):
        if other_types:
            score += 0.5

    return _clamp(
        score
    )


# ============================================================
# LIKELIHOOD
# ============================================================

def calculate_likelihood(
    confidence: float,
    linkability: float,
) -> float:
    """
    Calculate likelihood from detection confidence and
    linkability.

    Both inputs are converted to a 1-5 scale.

    Confidence:

        0.00 -> 1
        1.00 -> 5

    Linkability:

        already represented as 1-5

    Final likelihood:

        Average(Confidence Factor, Linkability)
    """

    confidence = _safe_confidence(
        confidence
    )

    confidence_factor = (
        1.0
        + (confidence * 4.0)
    )

    confidence_factor = _clamp(
        confidence_factor
    )

    linkability = _clamp(
        linkability
    )

    likelihood = (
        confidence_factor
        + linkability
    ) / 2.0

    return _clamp(
        likelihood
    )


# ============================================================
# INDIVIDUAL RISK
# ============================================================

def calculate_individual_risk(
    impact: float,
    likelihood: float,
) -> float:
    """
    Calculate normalized individual risk.

    Formula:

        (Impact * Likelihood / 25) * 100
    """

    impact = _clamp(
        impact
    )

    likelihood = _clamp(
        likelihood
    )

    raw_risk = (
        impact
        * likelihood
    )

    normalized_risk = (
        raw_risk
        / 25.0
    ) * 100.0

    return _clamp_risk(
        normalized_risk
    )


# ============================================================
# LINKABILITY ADJUSTMENT
# ============================================================

def calculate_linkability_adjustment(
    detections: Iterable[Mapping[str, Any]],
) -> float:
    """
    Calculate a document-level linkability adjustment.

    The adjustment is based on complementary information,
    not simply the number of detections.

    Examples:

        Name
        Name + Email
        Name + Email + Phone
        Name + Email + Phone + ID

    become increasingly linkable.
    """

    items = list(
        detections
    )

    if len(items) < 2:
        return 0.0

    types = {
        _normalise_type(
            item.get("type")
            or item.get("detection_type")
            or item.get("label")
        )
        for item in items
    }

    adjustment = 0.0

    # --------------------------------------------------------
    # Identity + contact
    # --------------------------------------------------------

    has_identity = bool(
        types.intersection(
            {
                "NAME",
                "ID_NUMBER",
                "GOVERNMENT_ID",
                "DATE_OF_BIRTH",
                "FACE",
            }
        )
    )

    has_contact = bool(
        types.intersection(
            {
                "EMAIL",
                "PHONE",
                "ADDRESS",
            }
        )
    )

    if has_identity and has_contact:
        adjustment += 8.0

    # --------------------------------------------------------
    # Strong identity combination
    # --------------------------------------------------------

    strong_identity_count = len(
        types.intersection(
            {
                "NAME",
                "EMAIL",
                "PHONE",
                "ADDRESS",
                "ID_NUMBER",
                "GOVERNMENT_ID",
                "DATE_OF_BIRTH",
            }
        )
    )

    if strong_identity_count >= 3:
        adjustment += 7.0

    if strong_identity_count >= 4:
        adjustment += 5.0

    # --------------------------------------------------------
    # Financial combination
    # --------------------------------------------------------

    has_financial = bool(
        types.intersection(
            {
                "BANK_ACCOUNT",
                "BANK_CARD",
            }
        )
    )

    if has_financial and (
        has_identity or has_contact
    ):
        adjustment += 8.0

    # --------------------------------------------------------
    # Credential combination
    # --------------------------------------------------------

    has_credential = bool(
        types.intersection(
            CREDENTIAL_TYPES
        )
    )

    if has_credential and len(types) > 1:
        adjustment += 3.0

    return max(
        0.0,
        adjustment
    )


# ============================================================
# SINGLE DETECTION ASSESSMENT
# ============================================================

def assess_detection(
    detection: Mapping[str, Any],
    all_detections: Optional[
        Iterable[Mapping[str, Any]]
    ] = None,
) -> RiskAssessment:
    """
    Assess one detection.

    Accepted detection fields:

        type
        value
        confidence

    The function also accepts:

        detection_type
        label

    for compatibility with different PriviScan modules.
    """

    detection_type = _normalise_type(
        detection.get("type")
        or detection.get("detection_type")
        or detection.get("label")
    )

    value = _clean_value(
        detection.get("value")
    )

    confidence = _safe_confidence(
        detection.get("confidence", 0.0)
    )

    if all_detections is None:
        all_detections = [detection]

    all_detections_list = list(
        all_detections
    )

    # --------------------------------------------------------
    # Credential critical override
    # --------------------------------------------------------

    credential_override = (
        _is_credential_type(
            detection_type
        )
        and confidence >= CREDENTIAL_CONFIDENCE_THRESHOLD
    )

    # --------------------------------------------------------
    # Impact
    # --------------------------------------------------------

    impact_data = calculate_impact(
        detection_type,
        value,
    )

    identifiability = impact_data[
        "identifiability"
    ]

    uniqueness = impact_data[
        "uniqueness"
    ]

    potential_harm = impact_data[
        "potential_harm"
    ]

    reusability = impact_data[
        "reusability"
    ]

    impact = impact_data[
        "impact"
    ]

    # --------------------------------------------------------
    # Linkability
    # --------------------------------------------------------

    linkability = calculate_linkability(
        detection_type,
        value,
        all_detections_list,
    )

    # --------------------------------------------------------
    # Likelihood
    # --------------------------------------------------------

    likelihood = calculate_likelihood(
        confidence,
        linkability,
    )

    # --------------------------------------------------------
    # Individual risk
    # --------------------------------------------------------

    individual_risk = calculate_individual_risk(
        impact,
        likelihood,
    )

    # --------------------------------------------------------
    # Credential override
    # --------------------------------------------------------

    if credential_override:
        individual_risk = 100.0
        risk_level = RISK_LEVEL_CRITICAL
    else:
        risk_level = _risk_level(
            individual_risk
        )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    explanation_parts = [
        (
            f"Impact is {impact:.2f}/5, based on "
            f"identifiability {identifiability:.2f}, "
            f"uniqueness {uniqueness:.2f}, "
            f"potential harm {potential_harm:.2f}, "
            f"and reusability {reusability:.2f}."
        ),
        (
            f"Likelihood is {likelihood:.2f}/5, "
            f"using detection confidence "
            f"{confidence:.2f} and linkability "
            f"{linkability:.2f}."
        ),
        (
            f"Individual risk is "
            f"{individual_risk:.2f}/100."
        ),
    ]

    if credential_override:
        explanation_parts.append(
            "Credential confidence is at least "
            "0.85, so the PriviScan Critical "
            "credential override sets the risk to "
            "100/100."
        )

    explanation = " ".join(
        explanation_parts
    )

    return RiskAssessment(
        detection_type=detection_type,
        value=value,
        confidence=confidence,

        identifiability=identifiability,
        uniqueness=uniqueness,
        potential_harm=potential_harm,
        reusability=reusability,

        impact=impact,

        linkability=linkability,
        likelihood=likelihood,

        individual_risk=individual_risk,

        risk_level=risk_level,
        explanation=explanation,

        credential_override=credential_override,
    )


# ============================================================
# DOCUMENT-LEVEL RISK
# ============================================================

def calculate_document_risk(
    detections: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate the complete PriviScan document-level risk.

    Steps:

        1. Assess every detection.
        2. Aggregate individual risks.
        3. Calculate linkability adjustment.
        4. Cap the final result at 100.
        5. Determine final risk level.

    No sector/context classification is used.
    """

    items = list(
        detections
    )

    if not items:
        return {
            "risk_score": 0.0,
            "risk_level": RISK_LEVEL_LOW,
            "total_detections": 0,
            "average_risk_score": 0.0,
            "aggregate_item_risk": 0.0,
            "linkability_adjustment": 0.0,
            "credential_override": False,
            "detections": [],
        }

    assessments: List[
        RiskAssessment
    ] = []

    for detection in items:
        assessment = assess_detection(
            detection,
            items,
        )

        assessments.append(
            assessment
        )

    # --------------------------------------------------------
    # Aggregate item risks
    # --------------------------------------------------------

    item_scores = [
        assessment.individual_risk
        for assessment in assessments
    ]

    aggregate_item_risk = sum(
        item_scores
    )

    # --------------------------------------------------------
    # Average score
    # --------------------------------------------------------

    average_risk_score = (
        aggregate_item_risk
        / len(assessments)
    )

    # --------------------------------------------------------
    # Linkability adjustment
    # --------------------------------------------------------

    linkability_adjustment = (
        calculate_linkability_adjustment(
            items
        )
    )

    # --------------------------------------------------------
    # Credential override
    # --------------------------------------------------------

    credential_override = any(
        assessment.credential_override
        for assessment in assessments
    )

    # --------------------------------------------------------
    # Final document risk
    # --------------------------------------------------------

    if credential_override:
        final_risk = 100.0
        final_level = RISK_LEVEL_CRITICAL

    else:
        final_risk = _clamp_risk(
            aggregate_item_risk
            + linkability_adjustment
        )

        final_level = _risk_level(
            final_risk
        )

    # --------------------------------------------------------
    # Convert detections to output dictionaries
    # --------------------------------------------------------

    result_detections = []

    for assessment in assessments:

        assessment_dict = (
            assessment.to_dict()
        )

        # Preserve common original fields
        # such as bounding box when available.
        original_detection = next(
            (
                item
                for item in items
                if _normalise_type(
                    item.get("type")
                    or item.get("detection_type")
                    or item.get("label")
                )
                == assessment.detection_type
                and _clean_value(
                    item.get("value")
                )
                == assessment.value
            ),
            None,
        )

        if original_detection is not None:

            if "box" in original_detection:
                assessment_dict["box"] = (
                    original_detection["box"]
                )

            if "bbox" in original_detection:
                assessment_dict["bbox"] = (
                    original_detection["bbox"]
                )

        result_detections.append(
            assessment_dict
        )

    return {
        "risk_score": round(
            final_risk,
            2,
        ),

        "risk_level": final_level,

        "total_detections": len(
            assessments
        ),

        "average_risk_score": round(
            average_risk_score,
            2,
        ),

        "aggregate_item_risk": round(
            aggregate_item_risk,
            2,
        ),

        "linkability_adjustment": round(
            linkability_adjustment,
            2,
        ),

        "credential_override": (
            credential_override
        ),

        "detections": (
            result_detections
        ),
    }


# ============================================================
# COMPATIBILITY FUNCTION
# ============================================================

def calculate_risk(
    detection: Mapping[str, Any],
    all_detections: Optional[
        Iterable[Mapping[str, Any]]
    ] = None,
) -> float:
    """
    Compatibility helper.

    Returns only the individual risk score.

    This is useful if older PriviScan code calls:

        calculate_risk(detection)
    """

    assessment = assess_detection(
        detection,
        all_detections,
    )

    return round(
        assessment.individual_risk,
        2,
    )


# ============================================================
# AVERAGE RISK
# ============================================================

def calculate_average_risk(
    results: Iterable[Mapping[str, Any]],
) -> float:
    """
    Calculate the average individual risk score.

    This function is intentionally kept because the existing
    PriviScan main.py imports it:

        from backend.risk.scoring import calculate_average_risk

    If the detections have not yet received risk_score values,
    the function calculates them using the new algorithm.

    This returns the average of individual detection risks,
    not the final document risk.

    Example:

        Email risk = 40
        Phone risk = 60
        ID risk = 30

        Average = (40 + 60 + 30) / 3
                = 43.33
    """

    items = list(
        results
    )

    if not items:
        return 0.0

    # --------------------------------------------------------
    # If scores already exist, use them.
    # --------------------------------------------------------

    existing_scores = []

    all_have_scores = True

    for item in items:

        score = item.get(
            "risk_score"
        )

        if score is None:
            all_have_scores = False
            break

        try:
            existing_scores.append(
                float(score)
            )
        except (
            TypeError,
            ValueError,
        ):
            all_have_scores = False
            break

    if all_have_scores:
        return round(
            sum(existing_scores)
            / len(existing_scores),
            2,
        )

    # --------------------------------------------------------
    # Otherwise calculate using the new algorithm.
    # --------------------------------------------------------

    assessments = [
        assess_detection(
            item,
            items,
        )
        for item in items
    ]

    total = sum(
        assessment.individual_risk
        for assessment in assessments
    )

    return round(
        total / len(assessments),
        2,
    )


# ============================================================
# ENRICH EXISTING DETECTIONS
# ============================================================

def score_detections(
    detections: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Score every detection and return a new list.

    This is useful for integration with analyze.py.

    The original detection fields are preserved.

    Additional fields include:

        risk_score
        risk_level
        impact
        likelihood
        identifiability
        uniqueness
        potential_harm
        reusability
        linkability
        explanation
        credential_override
    """

    items = list(
        detections
    )

    scored = []

    for item in items:

        assessment = assess_detection(
            item,
            items,
        )

        output = dict(
            item
        )

        output.update(
            assessment.to_dict()
        )

        scored.append(
            output
        )

    return scored


# ============================================================
# FULL ANALYSIS HELPER
# ============================================================

def analyze_risk(
    detections: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Main public function for complete risk analysis.

    Use this when you want:

        individual detection scores
        +
        final document score
        +
        risk level
        +
        explanation
    """

    return calculate_document_risk(
        detections
    )


# ============================================================
# SIMPLE TEST / DEMONSTRATION
# ============================================================

if __name__ == "__main__":

    sample_detections = [
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
        sample_detections
    )

    print("=" * 70)
    print("PriviScan Risk Scoring Test")
    print("=" * 70)

    for detection in result["detections"]:

        print()
        print(
            f"Type: {detection['type']}"
        )

        print(
            f"Value: {detection['value']}"
        )

        print(
            f"Confidence: "
            f"{detection['confidence']:.2f}"
        )

        print(
            f"Impact: "
            f"{detection['impact']:.2f}/5"
        )

        print(
            f"Likelihood: "
            f"{detection['likelihood']:.2f}/5"
        )

        print(
            f"Linkability: "
            f"{detection['linkability']:.2f}/5"
        )

        print(
            f"Risk: "
            f"{detection['risk_score']:.2f}/100"
        )

        print(
            f"Level: "
            f"{detection['risk_level']}"
        )

        print(
            f"Explanation: "
            f"{detection['explanation']}"
        )

    print()
    print("-" * 70)

    print(
        f"Total detections: "
        f"{result['total_detections']}"
    )

    print(
        f"Average individual risk: "
        f"{result['average_risk_score']:.2f}"
    )

    print(
        f"Aggregate item risk: "
        f"{result['aggregate_item_risk']:.2f}"
    )

    print(
        f"Linkability adjustment: "
        f"{result['linkability_adjustment']:.2f}"
    )

    print(
        f"Final document risk: "
        f"{result['risk_score']:.2f}/100"
    )

    print(
        f"Final risk level: "
        f"{result['risk_level']}"
    )

    print(
        f"Credential override: "
        f"{result['credential_override']}"
    )

    print("=" * 70)