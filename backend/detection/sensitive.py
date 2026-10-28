from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

try:
    import phonenumbers
    from phonenumbers import PhoneNumberMatcher
except ImportError:
    phonenumbers = None
    PhoneNumberMatcher = None


# ============================================================
# REGEX PATTERNS
# ============================================================

EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

DATE_RE = re.compile(
    r"\b(?:"
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|"
    r"\d{1,2}[-/.]\d{1,2}[-/.]\d{4}"
    r"|"
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}"
    r")\b",
    re.IGNORECASE,
)

PASSWORD_RE = re.compile(
    r"(?i)\b(?:password|passwd|pwd|passcode)\s*[:=]\s*([^\s,;]+)"
)

API_KEY_RE = re.compile(
    r"(?i)\b(?:api[_ -]?key|api[_ -]?token|access[_ -]?token|secret[_ -]?key)"
    r"\s*[:=]\s*([A-Za-z0-9_\-./+=]{8,})"
)

ACCOUNT_RE = re.compile(
    r"(?i)\b(?:account|account[_ -]?number|bank[_ -]?account)"
    r"\s*(?:number)?\s*[:#=-]?\s*(\d{6,30})\b"
)

GOV_ID_RE = re.compile(
    r"(?i)\b(?:"
    r"passport|passport[_ -]?no|passport[_ -]?number|"
    r"national[_ -]?id|government[_ -]?id|"
    r"government[_ -]?identification|"
    r"driver'?s?\s*licen[cs]e|"
    r"license[_ -]?number|"
    r"resident[_ -]?card|"
    r"social[_ -]?security|ssn"
    r")\s*[:#=-]?\s*([A-Z0-9][A-Z0-9\-]{4,30})\b"
)

GENERAL_ID_RE = re.compile(
    r"(?i)\b(?:"
    r"student[_ -]?id|student[_ -]?number|"
    r"staff[_ -]?id|staff[_ -]?number|"
    r"employee[_ -]?id|employee[_ -]?number|"
    r"faculty[_ -]?id|faculty[_ -]?number|"
    r"member[_ -]?id|member[_ -]?number|"
    r"registration[_ -]?number|registration[_ -]?id|"
    r"identifier|id[_ -]?number|id"
    r")\s*[:#=-]\s*([A-Z0-9][A-Z0-9\-]{3,30})\b"
)

NAME_RE = re.compile(
    r"(?i)^\s*(?:"
    r"name|full\s+name|person\s+name|customer\s+name|"
    r"student\s+name|employee\s+name|staff\s+name|"
    r"faculty\s+name"
    r")\s*:\s*([A-Za-zÀ-ÖØ-öø-ÿ'`.-]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'`.-]+){1,4})\s*$"
)

ADDRESS_RE = re.compile(
    r"(?i)^\s*(?:"
    r"address|home\s+address|street\s+address|"
    r"residential\s+address"
    r")\s*:\s*(.+?)\s*$"
)

LABELED_PHONE_RE = re.compile(
    r"(?i)^\s*(?:"
    r"phone|phone\s+number|mobile|mobile\s+number|"
    r"telephone|telephone\s+number|contact\s+number"
    r")\s*:\s*(.+?)\s*$"
)

USERNAME_RE = re.compile(
    r"(?i)^\s*(?:username|user\s+name)\s*:"
)


# ============================================================
# HELPERS
# ============================================================

def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _box_from_match(match: re.Match) -> List[int]:
    return [match.start(), 0, match.end(), 1]


def _make_detection(
    detection_type: str,
    value: str,
    confidence: float = 0.95,
    box: Optional[Any] = None,
    source: str = "regex",
) -> Dict[str, Any]:
    return {
        "type": detection_type,
        "value": value,
        "confidence": max(0.0, min(1.0, float(confidence))),
        "box": box,
        "source": source,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_username_field(text: str) -> bool:
    return bool(USERNAME_RE.match(_text(text)))


def _iter_ocr(ocr_results: Iterable[Any]) -> Iterable[Dict[str, Any]]:
    for item in ocr_results or []:
        if isinstance(item, dict):
            yield item


# ============================================================
# DETECTORS
# ============================================================

def detect_emails(text: str, box: Any = None) -> List[Dict[str, Any]]:
    results = []

    for match in EMAIL_RE.finditer(text):
        results.append(
            _make_detection(
                "EMAIL",
                match.group(0),
                0.99,
                box if box is not None else _box_from_match(match),
            )
        )

    return results


def detect_international_phones(
    text: str,
    box: Any = None,
) -> List[Dict[str, Any]]:
    results = []

    if phonenumbers is None or PhoneNumberMatcher is None:
        return results

    try:
        matcher = PhoneNumberMatcher(text, None)

        for match in matcher:
            raw = match.raw_string

            try:
                number = phonenumbers.parse(raw, None)

                if phonenumbers.is_possible_number(number):
                    if phonenumbers.is_valid_number(number):
                        results.append(
                            _make_detection(
                                "PHONE",
                                raw,
                                0.98,
                                box if box is not None else [
                                    match.start,
                                    0,
                                    match.end,
                                    1,
                                ],
                            )
                        )
            except Exception:
                continue

    except Exception:
        pass

    return results


def detect_labeled_phone(text: str, box: Any = None) -> List[Dict[str, Any]]:
    match = LABELED_PHONE_RE.match(text)

    if not match:
        return []

    value = match.group(1).strip()

    if len(re.sub(r"\D", "", value)) < 7:
        return []

    return [
        _make_detection(
            "PHONE",
            value,
            0.96,
            box if box is not None else _box_from_match(match),
        )
    ]


def detect_name(text: str, box: Any = None) -> List[Dict[str, Any]]:
    if is_username_field(text):
        return []

    match = NAME_RE.match(text)

    if not match:
        return []

    value = match.group(1).strip()

    return [
        _make_detection(
            "NAME",
            value,
            0.96,
            box if box is not None else _box_from_match(match),
        )
    ]


def detect_date_of_birth(text: str, box: Any = None) -> List[Dict[str, Any]]:
    lowered = text.lower()

    if not any(
        keyword in lowered
        for keyword in (
            "dob",
            "date of birth",
            "birth date",
            "born",
        )
    ):
        return []

    results = []

    for match in DATE_RE.finditer(text):
        results.append(
            _make_detection(
                "DATE_OF_BIRTH",
                match.group(0),
                0.96,
                box if box is not None else _box_from_match(match),
            )
        )

    return results


def detect_address(text: str, box: Any = None) -> List[Dict[str, Any]]:
    match = ADDRESS_RE.match(text)

    if not match:
        return []

    value = match.group(1).strip()

    if len(value) < 5:
        return []

    return [
        _make_detection(
            "ADDRESS",
            value,
            0.94,
            box if box is not None else _box_from_match(match),
        )
    ]


def detect_id_number(text: str, box: Any = None) -> List[Dict[str, Any]]:
    results = []

    match = GENERAL_ID_RE.search(text)

    if match:
        results.append(
            _make_detection(
                "ID",
                match.group(1),
                0.97,
                box if box is not None else _box_from_match(match),
            )
        )

    return results


def detect_government_id(text: str, box: Any = None) -> List[Dict[str, Any]]:
    results = []

    match = GOV_ID_RE.search(text)

    if match:
        results.append(
            _make_detection(
                "GOVERNMENT_ID",
                match.group(1),
                0.98,
                box if box is not None else _box_from_match(match),
            )
        )

    return results


def detect_password(text: str, box: Any = None) -> List[Dict[str, Any]]:
    results = []

    match = PASSWORD_RE.search(text)

    if match:
        results.append(
            _make_detection(
                "PASSWORD",
                match.group(1),
                0.99,
                box if box is not None else _box_from_match(match),
            )
        )

    return results


def detect_api_key(text: str, box: Any = None) -> List[Dict[str, Any]]:
    results = []

    match = API_KEY_RE.search(text)

    if match:
        results.append(
            _make_detection(
                "API_KEY",
                match.group(1),
                0.99,
                box if box is not None else _box_from_match(match),
            )
        )

    return results


def detect_bank_account(text: str, box: Any = None) -> List[Dict[str, Any]]:
    results = []

    match = ACCOUNT_RE.search(text)

    if match:
        results.append(
            _make_detection(
                "ACCOUNT_NUMBER",
                match.group(1),
                0.97,
                box if box is not None else _box_from_match(match),
            )
        )

    return results


# ============================================================
# TEXT DETECTION
# ============================================================

def detect_text(
    text: str,
    box: Any = None,
) -> List[Dict[str, Any]]:
    text = _text(text)

    if not text.strip():
        return []

    detections: List[Dict[str, Any]] = []

    detectors = (
        detect_emails,
        detect_international_phones,
        detect_labeled_phone,
        detect_name,
        detect_date_of_birth,
        detect_address,
        detect_government_id,
        detect_id_number,
        detect_password,
        detect_api_key,
        detect_bank_account,
    )

    for detector in detectors:
        try:
            detections.extend(detector(text, box))
        except Exception:
            continue

    return detections


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_detections(
    detections: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    normalized = []

    for item in detections or []:
        if not isinstance(item, dict):
            continue

        detection_type = (
            item.get("type")
            or item.get("label")
            or item.get("category")
            or ""
        )

        value = (
            item.get("value")
            or item.get("text")
            or item.get("entity")
            or ""
        )

        detection_type = str(detection_type).upper().strip()
        value = str(value).strip()

        if not detection_type or not value:
            continue

        confidence = _safe_float(
            item.get("confidence", item.get("score", 0.9)),
            0.9,
        )

        box = item.get("box", item.get("bbox"))

        normalized.append(
            {
                **item,
                "type": detection_type,
                "value": value,
                "confidence": max(0.0, min(1.0, confidence)),
                "box": box,
            }
        )

    return normalized


# ============================================================
# DEDUPLICATION
# ============================================================

def _dedup_key(item: Dict[str, Any]) -> tuple:
    return (
        str(item.get("type", "")).upper(),
        str(item.get("value", "")).strip().lower(),
    )


def deduplicate_detections(
    detections: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    output: Dict[tuple, Dict[str, Any]] = {}

    for item in normalize_detections(detections):
        key = _dedup_key(item)

        if key not in output:
            output[key] = item
            continue

        old = output[key]

        if float(item.get("confidence", 0)) > float(
            old.get("confidence", 0)
        ):
            output[key] = item

    return list(output.values())


# ============================================================
# OCR DETECTION
# ============================================================

def detect_from_ocr_results(
    ocr_results: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    detections = []

    for item in _iter_ocr(ocr_results):
        text = item.get("text", "")
        confidence = _safe_float(item.get("confidence", 0.9), 0.9)
        box = item.get("box", item.get("bbox"))

        for detection in detect_text(text, box):
            detection["confidence"] = min(
                float(detection.get("confidence", 0.9)),
                max(0.0, confidence),
            )
            detection["source"] = "ocr_regex"
            detections.append(detection)

    return deduplicate_detections(detections)


# ============================================================
# MAIN DETECTOR
# ============================================================

def detect_sensitive(
    text: Optional[str] = None,
    ocr_results: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    detections = []

    if text:
        detections.extend(detect_text(text))

    if ocr_results:
        detections.extend(
            detect_from_ocr_results(ocr_results)
        )

    return deduplicate_detections(detections)


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

detect_sensitive_data = detect_sensitive
detect_sensitive_text = detect_text


def score_detections(detections):
    from backend.risk.scoring import score_detections as _score

    return _score(detections)


def analyze_sensitive_data(
    text: Optional[str] = None,
    ocr_results: Optional[Iterable[Dict[str, Any]]] = None,
):
    detections = detect_sensitive(
        text=text,
        ocr_results=ocr_results,
    )

    return score_detections(detections)


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":
    sample = """
    Name: Anish Kandel
    Email: anish@example.com
    Phone: +82 10-1234-5678
    Student ID: 2024891090
    Password: MySecretPassword123
    API Key: sk-test-123456789
    """

    results = detect_sensitive(text=sample)

    for item in results:
        print(item)