"""
PriviScan Context and Sector Detection

This module determines the sector/context of the uploaded
content.

Supported sectors:

    HEALTHCARE
    FINANCE
    EDUCATION
    BUSINESS_EMPLOYMENT
    GOVERNMENT_LEGAL
    CONSUMER_SERVICES
    UNKNOWN

The detected sector is later passed to the risk engine.

Example:

    OCR text
        ↓
    Context detection
        ↓
    FINANCE
        ↓
    Risk engine
"""


from typing import Any, Dict, List, Tuple


# ============================================================
# SUPPORTED SECTORS
# ============================================================

SUPPORTED_SECTORS = {
    "HEALTHCARE",
    "FINANCE",
    "EDUCATION",
    "BUSINESS_EMPLOYMENT",
    "GOVERNMENT_LEGAL",
    "CONSUMER_SERVICES",
    "UNKNOWN",
}


# ============================================================
# SECTOR KEYWORDS
# ============================================================

# These are initial context indicators.
#
# They are NOT the final AI classifier.
#
# Later, the AI model can replace/augment this keyword layer.

SECTOR_KEYWORDS: Dict[str, List[str]] = {

    # --------------------------------------------------------
    # Healthcare / Hospital
    # --------------------------------------------------------

    "HEALTHCARE": [
        "hospital",
        "clinic",
        "medical",
        "medicine",
        "doctor",
        "patient",
        "patient id",
        "patient number",
        "diagnosis",
        "treatment",
        "prescription",
        "pharmacy",
        "health",
        "healthcare",
        "blood",
        "blood type",
        "medical record",
        "medical history",
        "laboratory",
        "lab result",
        "radiology",
        "insurance claim",
        "symptoms",
    ],

    # --------------------------------------------------------
    # Finance / Bank
    # --------------------------------------------------------

    "FINANCE": [
        "bank",
        "banking",
        "account",
        "account number",
        "bank account",
        "balance",
        "transaction",
        "transaction id",
        "transaction number",
        "payment",
        "payment id",
        "transfer",
        "wire transfer",
        "deposit",
        "withdrawal",
        "credit",
        "debit",
        "credit card",
        "debit card",
        "card number",
        "billing",
        "invoice",
        "financial",
        "finance",
        "loan",
        "mortgage",
        "interest",
        "statement",
        "routing number",
        "swift",
        "iban",
    ],

    # --------------------------------------------------------
    # Education / University
    # --------------------------------------------------------

    "EDUCATION": [
        "university",
        "college",
        "school",
        "student",
        "student id",
        "student number",
        "student identification",
        "faculty",
        "department",
        "professor",
        "teacher",
        "course",
        "class",
        "semester",
        "grade",
        "gpa",
        "transcript",
        "enrollment",
        "admission",
        "campus",
        "academic",
        "education",
        "university id",
        "school id",
    ],

    # --------------------------------------------------------
    # Business / Employment
    # --------------------------------------------------------

    "BUSINESS_EMPLOYMENT": [
        "company",
        "business",
        "employee",
        "employee id",
        "employee number",
        "staff",
        "staff id",
        "staff number",
        "employment",
        "employer",
        "job",
        "position",
        "department",
        "manager",
        "human resources",
        "hr",
        "payroll",
        "salary",
        "workplace",
        "office",
        "corporate",
        "organization",
        "employee record",
        "employment record",
    ],

    # --------------------------------------------------------
    # Government / Legal
    # --------------------------------------------------------

    "GOVERNMENT_LEGAL": [
        "government",
        "government id",
        "national id",
        "citizen",
        "citizenship",
        "passport",
        "passport number",
        "driver license",
        "driver's license",
        "driving licence",
        "license number",
        "resident",
        "resident card",
        "residence card",
        "immigration",
        "visa",
        "visa number",
        "tax",
        "tax id",
        "tax number",
        "social security",
        "legal",
        "court",
        "case number",
        "police",
        "law",
        "lawyer",
        "attorney",
        "permit",
        "registration",
        "official document",
    ],

    # --------------------------------------------------------
    # Consumer / Services
    # --------------------------------------------------------

    "CONSUMER_SERVICES": [
        "customer",
        "customer id",
        "customer number",
        "membership",
        "membership id",
        "member number",
        "loyalty",
        "loyalty card",
        "order",
        "order number",
        "booking",
        "reservation",
        "delivery",
        "shipping",
        "service",
        "subscription",
        "account",
        "support",
        "customer service",
        "retail",
        "shopping",
        "store",
        "hotel",
        "restaurant",
        "airline",
        "ticket",
        "ticket number",
    ],
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text before sector analysis.
    """

    if not text:
        return ""

    return " ".join(
        str(text)
        .lower()
        .strip()
        .split()
    )


# ============================================================
# KEYWORD SCORING
# ============================================================

def calculate_sector_scores(
    text: str,
) -> Dict[str, float]:
    """
    Calculate a simple keyword-based score for every sector.

    The score is based on how many sector-specific indicators
    appear in the supplied text.
    """

    normalized_text = normalize_text(text)

    scores: Dict[str, float] = {
        sector: 0.0
        for sector in SECTOR_KEYWORDS
    }

    if not normalized_text:
        return scores

    for sector, keywords in SECTOR_KEYWORDS.items():

        for keyword in keywords:

            keyword_normalized = normalize_text(
                keyword
            )

            if not keyword_normalized:
                continue

            if keyword_normalized in normalized_text:

                # Longer/more specific phrases receive
                # slightly more weight.
                if len(keyword_normalized.split()) >= 2:
                    scores[sector] += 2.0
                else:
                    scores[sector] += 1.0

    return scores


# ============================================================
# SECTOR DETECTION
# ============================================================

def detect_sector(
    text: str,
) -> Dict[str, Any]:
    """
    Detect the most likely sector from text.

    Returns:

        {
            "sector": "FINANCE",
            "confidence": 0.85,
            "scores": {...},
            "matched_keywords": [...]
        }
    """

    normalized_text = normalize_text(text)

    if not normalized_text:

        return {
            "sector": "UNKNOWN",
            "confidence": 0.0,
            "scores": {},
            "matched_keywords": [],
        }

    scores = calculate_sector_scores(
        normalized_text
    )

    # --------------------------------------------------------
    # Find highest-scoring sector
    # --------------------------------------------------------

    sorted_sectors = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_sector, best_score = sorted_sectors[0]

    # --------------------------------------------------------
    # No useful context
    # --------------------------------------------------------

    if best_score <= 0:

        return {
            "sector": "UNKNOWN",
            "confidence": 0.0,
            "scores": scores,
            "matched_keywords": [],
        }

    # --------------------------------------------------------
    # Calculate a simple confidence
    # --------------------------------------------------------

    total_score = sum(
        scores.values()
    )

    if total_score <= 0:

        confidence = 0.0

    else:

        confidence = best_score / total_score

    confidence = round(
        min(confidence, 1.0),
        3,
    )

    # --------------------------------------------------------
    # Find matched keywords
    # --------------------------------------------------------

    matched_keywords = []

    for keyword in SECTOR_KEYWORDS.get(
        best_sector,
        [],
    ):

        if normalize_text(keyword) in normalized_text:

            matched_keywords.append(
                keyword
            )

    return {
        "sector": best_sector,
        "confidence": confidence,
        "scores": scores,
        "matched_keywords": matched_keywords,
    }


# ============================================================
# OCR CONTEXT BUILDER
# ============================================================

def build_context_from_ocr(
    ocr_results: List[Dict[str, Any]],
) -> str:
    """
    Combine OCR text into one context string.

    Example OCR:

        [
            {"text": "Bank Account"},
            {"text": "Account Number"},
            {"text": "Balance"},
        ]

    becomes:

        "Bank Account Account Number Balance"
    """

    texts = []

    for item in ocr_results:

        text = item.get(
            "text",
            "",
        )

        if text:

            texts.append(
                str(text)
            )

    return " ".join(texts)


# ============================================================
# CONTEXT ANALYSIS FROM OCR
# ============================================================

def detect_sector_from_ocr(
    ocr_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Detect sector using OCR results.
    """

    context_text = build_context_from_ocr(
        ocr_results
    )

    result = detect_sector(
        context_text
    )

    result["context_text"] = context_text

    return result


# ============================================================
# DETECTION-AWARE CONTEXT
# ============================================================

def enhance_sector_with_detections(
    sector_result: Dict[str, Any],
    detections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Improve sector interpretation using detected
    sensitive-information categories.

    Example:

        ACCOUNT_NUMBER
        BANK_CARD

    strongly supports FINANCE.

    PASSPORT / GOVERNMENT_ID
    supports GOVERNMENT_LEGAL.

    STUDENT ID
    supports EDUCATION.

    This is an additional context signal.
    """

    scores = dict(
        sector_result.get(
            "scores",
            {}
        )
    )

    # Make sure all sectors exist.
    for sector in SECTOR_KEYWORDS:

        scores.setdefault(
            sector,
            0.0
        )

    for detection in detections:

        detection_type = str(
            detection.get(
                "type",
                ""
            )
        ).upper()

        subtype = str(
            detection.get(
                "subtype",
                ""
            )
        ).upper()

        # ----------------------------------------------------
        # Financial information
        # ----------------------------------------------------

        if detection_type in {
            "ACCOUNT_NUMBER",
            "BANK_CARD",
        }:

            scores["FINANCE"] += 3.0

        # ----------------------------------------------------
        # Government identity
        # ----------------------------------------------------

        if detection_type == "GOVERNMENT_ID":

            scores["GOVERNMENT_LEGAL"] += 3.0

        # ----------------------------------------------------
        # General ID
        # ----------------------------------------------------

        if detection_type == "ID":

            if any(
                word in subtype
                for word in [
                    "STUDENT",
                    "UNIVERSITY",
                    "SCHOOL",
                ]
            ):

                scores["EDUCATION"] += 3.0

            elif any(
                word in subtype
                for word in [
                    "EMPLOYEE",
                    "STAFF",
                    "WORK",
                    "COMPANY",
                ]
            ):

                scores[
                    "BUSINESS_EMPLOYMENT"
                ] += 3.0

            elif any(
                word in subtype
                for word in [
                    "PASSPORT",
                    "DRIVER",
                    "GOVERNMENT",
                    "RESIDENT",
                ]
            ):

                scores[
                    "GOVERNMENT_LEGAL"
                ] += 3.0

        # ----------------------------------------------------
        # Healthcare indicators
        # ----------------------------------------------------

        if any(
            word in subtype
            for word in [
                "PATIENT",
                "MEDICAL",
                "HEALTH",
            ]
        ):

            scores["HEALTHCARE"] += 3.0

    # --------------------------------------------------------
    # Determine new winner
    # --------------------------------------------------------

    sorted_sectors = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_sector, best_score = (
        sorted_sectors[0]
    )

    total_score = sum(
        scores.values()
    )

    if total_score > 0:

        confidence = round(
            min(
                best_score / total_score,
                1.0,
            ),
            3,
        )

    else:

        confidence = 0.0

    result = dict(
        sector_result
    )

    result["sector"] = best_sector
    result["confidence"] = confidence
    result["scores"] = scores

    return result


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def analyze_context(
    ocr_results: List[Dict[str, Any]],
    detections: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Complete context analysis.

    1. Analyze OCR text.
    2. Determine sector.
    3. Optionally improve sector using AI detections.
    """

    result = detect_sector_from_ocr(
        ocr_results
    )

    if detections:

        result = enhance_sector_with_detections(
            sector_result=result,
            detections=detections,
        )

    return result