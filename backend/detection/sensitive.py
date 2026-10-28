
from __future__ import annotations

import json
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
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"[a-z]*\s+\d{4}"
    r")\b",
    re.IGNORECASE,
)

FIELD_SEPARATOR = r"\s*(?::|=|\|)\s*"

PASSWORD_RE = re.compile(
    r"(?i)\b(?:password|passwd|pwd|passcode)"
    + FIELD_SEPARATOR
    + r"([^\s,;|\"']+)"
)

API_KEY_RE = re.compile(
    r"(?i)\b(?:api[_ -]?key|api[_ -]?token|access[_ -]?token|secret[_ -]?key)"
    + FIELD_SEPARATOR
    + r"([A-Za-z0-9_\-./+=]{8,})"
)

ACCOUNT_RE = re.compile(
    r"(?i)\b(?:account(?:[_ -]?number)?|bank[_ -]?account)"
    + r"\s*(?:number)?\s*(?::|=|\||#|-)?\s*"
    + r"([0-9][0-9 -]{4,35}[0-9])\b"
)

GOV_ID_RE = re.compile(
    r"(?i)\b(?:"
    r"passport(?:[_ -]?(?:no|number))?|"
    r"national[_ -]?id|government[_ -]?id|"
    r"government[_ -]?identification|"
    r"driver'?s?\s*licen[cs]e|"
    r"license[_ -]?number|resident[_ -]?card|"
    r"social[_ -]?security|ssn"
    r")\s*(?::|=|\||#|-)\s*"
    r"([A-Z0-9][A-Z0-9 -]{3,30})",
)

GENERAL_ID_RE = re.compile(
    r"(?i)\b(?:"
    r"student[_ -]?(?:id|number)|"
    r"staff[_ -]?(?:id|number)|"
    r"employee[_ -]?(?:id|number)|"
    r"faculty[_ -]?(?:id|number)|"
    r"member[_ -]?(?:id|number)|"
    r"registration[_ -]?(?:number|id)|"
    r"identifier|id[_ -]?number|id"
    r")\s*(?::|=|\||#)\s*"
    r"([A-Z0-9][A-Z0-9-]{3,30})",
)

NAME_RE = re.compile(
    r"(?i)^\s*(?:"
    r"name|full\s+name|person\s+name|customer\s+name|"
    r"student\s+name|employee\s+name|staff\s+name|faculty\s+name"
    r")\s*(?::|=|\|)\s*[\"']?"
    r"([A-Za-zÀ-ÖØ-öø-ÿ'`.-]+"
    r"(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'`.-]+){1,4})"
    r"[\"']?\s*,?\s*$"
)

ADDRESS_RE = re.compile(
    r"(?i)^\s*(?:address|home\s+address|street\s+address|"
    r"residential\s+address)\s*(?::|=|\|)\s*[\"']?(.+?)[\"']?\s*,?\s*$"
)

LABELED_PHONE_RE = re.compile(
    r"(?i)^\s*(?:phone|phone\s+number|mobile|mobile\s+number|"
    r"telephone|telephone\s+number|contact\s+number)"
    r"\s*(?::|=|\|)\s*[\"']?(.+?)[\"']?\s*,?\s*$"
)

USERNAME_RE = re.compile(
    r"(?i)^\s*(?:username|user\s+name)\s*(?::|=|\|)"
)


# ============================================================
# HELPERS
# ============================================================

def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _box_from_match(match: re.Match) -> List[int]:
    # These are text offsets, not PDF coordinates.
    # PDF redaction needs a real PDF bounding box.
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
        "value": value.strip(),
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


def _clean_value(value: Any) -> str:
    value = _text(value).strip()
    value = value.strip(" \t\r\n\"'")
    return value.rstrip(",;").strip()


def _normalise_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


def _detect_labeled_value(
    label: str,
    raw_value: str,
    box: Any = None,
) -> List[Dict[str, Any]]:
    """Detect a value when its field label is available."""
    label_key = _normalise_label(label)
    value = _clean_value(raw_value)

    if not label_key or not value:
        return []

    def result(kind: str, confidence: float = 0.96):
        return [_make_detection(kind, value, confidence, box)]

    # Credentials remain separate from personal information.
    if label_key in {"password", "passwd", "pwd", "passcode"}:
        return result("PASSWORD", 0.99)

    if label_key in {
        "api key", "api token", "access token", "secret key",
        "secret", "api secret",
    }:
        if len(value) >= 8:
            return result("API_KEY", 0.99)
        return []

    if label_key in {
        "email", "email address", "e mail", "e mail address",
    }:
        return detect_emails(value, box)

    if label_key in {
        "phone", "phone number", "mobile", "mobile number",
        "telephone", "telephone number", "contact number",
    }:
        return detect_labeled_phone(f"Phone: {value}", box)

    if label_key in {
        "name", "full name", "person name", "customer name",
        "student name", "employee name", "staff name", "faculty name",
    }:
        if is_username_field(label):
            return []

        if re.fullmatch(
            r"[A-Za-zÀ-ÖØ-öø-ÿ'`.-]+"
            r"(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'`.-]+){1,4}",
            value,
        ):
            return result("NAME")

        return []

    if label_key in {
        "address", "home address", "street address",
        "residential address", "mailing address",
    }:
        if len(value) >= 5:
            return result("ADDRESS", 0.94)
        return []

    if label_key in {
        "dob", "date of birth", "birth date", "birthdate",
    }:
        dates = DATE_RE.findall(value)
        return [
            _make_detection("DATE_OF_BIRTH", date, 0.96, box)
            for date in dates
        ]

    government_labels = {
        "government id", "government identification", "national id",
        "national identification", "passport", "passport no",
        "passport number", "drivers license", "driver license",
        "drivers license number", "license number", "resident card",
        "resident card number", "social security", "ssn",
    }

    if label_key in government_labels:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 -]{3,30}", value):
            return result("GOVERNMENT_ID", 0.98)
        return []

    general_id_labels = {
        "id", "id number", "identifier", "student id", "student number",
        "staff id", "staff number", "employee id", "employee number",
        "faculty id", "faculty number", "member id", "member number",
        "registration id", "registration number",
    }

    if label_key in general_id_labels:
        # Reject plain words such as "Account" or "Number".
        # A general identifier must contain at least one digit.
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{3,30}", value)
            and any(character.isdigit() for character in value)
        ):
            return result("ID", 0.97)
        return []

    account_labels = {
        "account", "account number", "bank account",
        "bank account number", "bank account no",
    }

    if label_key in account_labels:
        digits = re.sub(r"\D", "", value)
        if 6 <= len(digits) <= 30 and re.fullmatch(r"[0-9 -]+", value):
            return result("ACCOUNT_NUMBER", 0.97)
        return []

    # A generic label alone is not enough to classify arbitrary data.
    return []


def _extract_labeled_fields(text: str) -> List[Dict[str, Any]]:
    """
    Handle labelled values, JSON-style pairs, and simple pipe-separated
    tables. Examples: Name: Anish Kandel or Email | anish@example.com.
    """
    detections: List[Dict[str, Any]] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Parse JSON first when the entire text is valid JSON.
    try:
        parsed = json.loads(text)

        def walk_json(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (str, int, float)):
                        detections.extend(
                            _detect_labeled_value(str(key), str(value))
                        )
                    elif isinstance(value, (dict, list)):
                        walk_json(value)
            elif isinstance(obj, list):
                for value in obj:
                    walk_json(value)

        walk_json(parsed)
    except (ValueError, TypeError):
        pass

    for line in lines:
        json_pair = re.match(
            r"""^\s*["']([^"']+)["']\s*:\s*["']?(.*?)["']?\s*,?\s*$""",
            line,
        )

        if json_pair:
            detections.extend(
                _detect_labeled_value(
                    json_pair.group(1),
                    json_pair.group(2),
                )
            )
            continue

        pair = re.match(
            r"^\s*([A-Za-z][A-Za-z0-9 _-]{0,50}?)"
            r"\s*(?::|=|\|)\s*(.*?)\s*$",
            line,
        )

        if pair:
            label = pair.group(1).strip()
            value = pair.group(2).strip()

            if value and _normalise_label(label) not in {
                "field", "column", "key",
            }:
                detections.extend(_detect_labeled_value(label, value))

    # Parse simple pipe-separated tables.
    rows = [
        [cell.strip() for cell in line.split("|")]
        for line in lines
        if "|" in line
    ]

    for index, row in enumerate(rows):
        if index + 1 < len(rows):
            next_row = rows[index + 1]

            if len(row) >= 2 and len(row) == len(next_row):
                recognised_headers = sum(
                    _normalise_label(cell) in {
                        "name", "full name", "email", "phone",
                        "phone number", "address", "student id",
                        "employee id", "staff id", "government id",
                        "account number", "password", "api key",
                    }
                    for cell in row
                )

                if recognised_headers >= 2:
                    for label, value in zip(row, next_row):
                        detections.extend(
                            _detect_labeled_value(label, value)
                        )

        if (
            len(row) == 2
            and _normalise_label(row[0]) in {"field", "key"}
            and _normalise_label(row[1]) == "value"
        ):
            continue

        if len(row) == 2:
            detections.extend(
                _detect_labeled_value(row[0], row[1])
            )

    return detections


# ============================================================
# DETECTORS
# ============================================================

def detect_emails(
    text: str,
    box: Any = None,
) -> List[Dict[str, Any]]:
    return [
        _make_detection(
            "EMAIL",
            match.group(0),
            0.99,
            box if box is not None else _box_from_match(match),
        )
        for match in EMAIL_RE.finditer(text)
    ]


def detect_international_phones(
    text: str,
    box: Any = None,
) -> List[Dict[str, Any]]:
    results = []

    if phonenumbers is None or PhoneNumberMatcher is None:
        return results

    try:
        for match in PhoneNumberMatcher(text, None):
            raw = match.raw_string

            try:
                number = phonenumbers.parse(raw, None)

                if (
                    phonenumbers.is_possible_number(number)
                    and phonenumbers.is_valid_number(number)
                ):
                    results.append(
                        _make_detection(
                            "PHONE",
                            raw,
                            0.98,
                            box if box is not None else [
                                match.start, 0, match.end, 1
                            ],
                        )
                    )
            except Exception:
                continue
    except Exception:
        pass

    return results


def detect_labeled_phone(
    text: str,
    box: Any = None,
) -> List[Dict[str, Any]]:
    match = LABELED_PHONE_RE.match(text)

    if not match:
        return []

    value = _clean_value(match.group(1))

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


def detect_name(
    text: str,
    box: Any = None,
) -> List[Dict[str, Any]]:
    if is_username_field(text):
        return []

    match = NAME_RE.match(text)

    if not match:
        return []

    return [
        _make_detection(
            "NAME",
            _clean_value(match.group(1)),
            0.96,
            box if box is not None else _box_from_match(match),
        )
    ]


def detect_date_of_birth(
    text: str,
    box: Any = None,
) -> List[Dict[str, Any]]:
    lowered = text.lower()

    if not any(
        keyword in lowered
        for keyword in ("dob", "date of birth", "birth date", "born")
    ):
        return []

    return [
        _make_detection(
            "DATE_OF_BIRTH",
            match.group(0),
            0.96,
            box if box is not None else _box_from_match(match),
        )
        for match in DATE_RE.finditer(text)
    ]


def detect_address(
    text: str,
    box: Any = None,
) -> List[Dict[str, Any]]:
    match = ADDRESS_RE.match(text)

    if not match:
        return []

    value = _clean_value(match.group(1))

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


def detect_id_number(
    text: str,
    box: Any = None,
) -> List[Dict[str, Any]]:
    match = GENERAL_ID_RE.search(text)

    if not match:
        return []

    value = match.group(1).strip()

    # Important false-positive fix:
    # Reject values containing no digits, for example a table heading
    # accidentally captured as "ID | Account".
    if not any(character.isdigit() for character in value):
        return []

    return [
        _make_detection(
            "ID",
            value,
            0.97,
            box if box is not None else _box_from_match(match),
        )
    ]


def detect_government_id(
    text: str,
    box: Any = None,
) -> List[Dict[str, Any]]:
    match = GOV_ID_RE.search(text)

    if not match:
        return []

    return [
        _make_detection(
            "GOVERNMENT_ID",
            match.group(1).strip(),
            0.98,
            box if box is not None else _box_from_match(match),
        )
    ]


def detect_password(
    text: str,
    box: Any = None,
) -> List[Dict[str, Any]]:
    match = PASSWORD_RE.search(text)

    if not match:
        return []

    return [
        _make_detection(
            "PASSWORD",
            match.group(1),
            0.99,
            box if box is not None else _box_from_match(match),
        )
    ]


def detect_api_key(
    text: str,
    box: Any = None,
) -> List[Dict[str, Any]]:
    match = API_KEY_RE.search(text)

    if not match:
        return []

    return [
        _make_detection(
            "API_KEY",
            match.group(1),
            0.99,
            box if box is not None else _box_from_match(match),
        )
    ]


def detect_bank_account(
    text: str,
    box: Any = None,
) -> List[Dict[str, Any]]:
    match = ACCOUNT_RE.search(text)

    if not match:
        return []

    value = re.sub(r"\s+", " ", match.group(1)).strip()

    return [
        _make_detection(
            "ACCOUNT_NUMBER",
            value,
            0.97,
            box if box is not None else _box_from_match(match),
        )
    ]


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

    try:
        detections.extend(_extract_labeled_fields(text))
    except Exception:
        pass

    return deduplicate_detections(detections)


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

        normalized.append({
            **item,
            "type": detection_type,
            "value": value,
            "confidence": max(0.0, min(1.0, confidence)),
            "box": item.get("box", item.get("bbox")),
        })

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
        elif float(item.get("confidence", 0)) > float(
            output[key].get("confidence", 0)
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
        detections.extend(detect_from_ocr_results(ocr_results))

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

    for item in detect_sensitive(text=sample):
        print(item)
