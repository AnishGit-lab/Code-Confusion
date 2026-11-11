"""
PriviScan - rule-based sensitive-data detection (value-first).

What changed compared with the old version
------------------------------------------
* Values are recognised from their OWN shape (email, phone, card + Luhn,
  API keys, national IDs with checksum, ...). A label such as "Email:" is a
  bonus, never a requirement.
* A label may be on the same line ("Student ID: 123"), without a separator
  ("Student ID 123"), or on the previous OCR line (label on one line,
  value on the next / next box).
* Unlabeled numbers are scored from context (document words, shape) and are
  reported with a LOWER confidence instead of being ignored.
* Order / invoice / tracking / ticket numbers are no longer reported as
  personal IDs (negative-label list).
* Deduplication keeps every LOCATION of a value (type + value + box), so a
  value that appears twice in an image is blurred twice.
* Public names used by the rest of the project are kept:
  detect_text, detect_sensitive, detect_from_ocr_results,
  deduplicate_detections, normalize_detections, is_username_field,
  score_detections, analyze_sensitive_data, ...

Environment variables
---------------------
PRIVISCAN_PHONE_REGIONS   comma separated default regions, default "KR"
                          (example: "KR,US,GB,IN,JP,NP")
"""

from __future__ import annotations

import json
import math
import os
import re
from difflib import get_close_matches
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import phonenumbers
    from phonenumbers import Leniency, PhoneNumberMatcher
except ImportError:  # the project still works, with a simpler phone detector
    phonenumbers = None
    Leniency = None
    PhoneNumberMatcher = None


# ============================================================
# CONSTANTS
# ============================================================

PHONE_REGIONS = tuple(
    region.strip().upper()
    for region in os.environ.get("PRIVISCAN_PHONE_REGIONS", "KR").split(",")
    if region.strip()
) or ("KR",)

# Higher number wins when two detections describe the same value.
TYPE_PRIORITY = {
    "PASSWORD": 12,
    "API_KEY": 11,
    "SECRET": 10,
    "GOVERNMENT_ID": 9,
    "BANK_CARD": 8,
    "ACCOUNT_NUMBER": 7,
    "EMAIL": 6,
    "PHONE": 5,
    "DATE_OF_BIRTH": 4,
    "ID": 3,
    "ADDRESS": 2,
    "NAME": 1,
}

# ------------------------------------------------------------
# Label vocabulary (used by classify_label)
# ------------------------------------------------------------

# Words that mean "this number is NOT a personal identifier".
NEGATIVE_TOKENS = {
    "order", "invoice", "tracking", "ticket", "reference", "ref", "transaction",
    "txn", "serial", "sku", "item", "product", "model", "zip", "postal",
    "postcode", "amount", "total", "price", "qty", "quantity", "version",
    "page", "year", "date", "time", "batch", "lot", "job", "case", "shipment",
    "booking", "confirmation", "receipt", "document", "file", "isbn", "barcode",
    "coupon", "voucher", "promo", "build", "release", "issue", "ip", "mac",
}

PERSON_NAME_PREFIX = {
    "name", "full", "person", "customer", "student", "employee", "staff",
    "faculty", "patient", "applicant", "holder", "cardholder", "card",
    "account", "first", "last", "surname", "given", "family", "client",
    "member", "of", "legal", "middle",
}

NUMBER_TOKENS = {"no", "number", "num", "nr", "id", "ident", "identifier", "code"}

PASSWORD_TOKENS = {"password", "passwd", "passcode", "pwd", "pin"}
PASSWORD_LABEL_EXTRA = {
    "new", "old", "current", "confirm", "login", "account", "user", "admin", "root",
    "master", "wifi", "wi", "fi", "email", "mail", "database", "db", "app", "your",
    "enter", "retype", "re", "type", "repeat", "temporary", "temp", "one", "time",
    "otp", "code", "number", "no", "card", "bank", "phone",
}

PERSONAL_ID_PREFIX = {
    "student", "staff", "employee", "faculty", "member", "membership",
    "registration", "reg", "roll", "matric", "patient", "customer", "client",
    "citizen", "resident", "alumni", "teacher", "professor", "user",
}

# Non-Latin / exact labels
EXACT_LABELS = {
    "이름": "NAME", "성명": "NAME", "이메일": "EMAIL", "전화번호": "PHONE",
    "휴대폰": "PHONE", "핸드폰": "PHONE", "연락처": "PHONE", "주소": "ADDRESS",
    "학번": "ID", "사번": "ID", "계좌번호": "ACCOUNT_NUMBER",
    "카드번호": "BANK_CARD", "비밀번호": "PASSWORD", "주민등록번호": "GOVERNMENT_ID",
    "여권번호": "GOVERNMENT_ID", "운전면허번호": "GOVERNMENT_ID",
    "생년월일": "DATE_OF_BIRTH",
    "名前": "NAME", "氏名": "NAME", "メール": "EMAIL", "電話番号": "PHONE",
    "住所": "ADDRESS", "パスワード": "PASSWORD", "生年月日": "DATE_OF_BIRTH",
    "नाम": "NAME", "ईमेल": "EMAIL", "फोन": "PHONE", "ठेगाना": "ADDRESS",
    "पासवर्ड": "PASSWORD",
}

# Words used to repair OCR typos in labels (Emai1 -> email, Passw0rd -> password)
LABEL_VOCAB = (
    "email", "mail", "phone", "mobile", "telephone", "contact", "address",
    "password", "passcode", "passwd", "account", "student", "employee", "staff",
    "faculty", "member", "membership", "registration", "passport", "license",
    "licence", "national", "government", "identification", "identifier",
    "number", "name", "card", "credit", "debit", "secret", "token", "birth",
    "resident", "social", "security", "patient", "customer",
)

# ------------------------------------------------------------
# Context words for ambiguous, unlabeled numbers
# ------------------------------------------------------------

CONTEXT_WORDS = {
    "education": ("student", "university", "college", "school", "campus",
                  "faculty", "academic", "semester", "course", "transcript"),
    "employment": ("employee", "staff", "payroll", "company", "department",
                   "human resources", "salary", "employer"),
    "finance": ("bank", "account", "balance", "branch", "swift", "iban",
                "transfer", "deposit", "savings", "statement", "routing"),
    "health": ("patient", "hospital", "clinic", "medical", "diagnosis",
               "prescription", "insurance"),
    "government": ("passport", "license", "licence", "national", "resident",
                   "government", "citizen", "immigration", "tax"),
}


# ============================================================
# REGEX PATTERNS
# ============================================================

EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

# OCR often adds a space around "@" or reads the last dot as a comma.
EMAIL_LOOSE_RE = re.compile(
    r"[A-Za-z0-9._%+-]+\s?@\s?[A-Za-z0-9-]+(?:[.,][A-Za-z0-9-]+)*[.,][A-Za-z]{2,}"
)

DATE_RE = re.compile(
    r"\b(?:"
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r")\b",
    re.IGNORECASE,
)

# Known secret formats (no label required)
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{15,}\b"),
    re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{10,}\b"),
    re.compile(r"\bpk_(?:live|test)_[A-Za-z0-9]{10,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bey[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]

KR_RRN_RE = re.compile(r"(?<!\d)(\d{6})[- ]?([1-4]\d{6})(?!\d)")
US_SSN_RE = re.compile(r"(?<!\d)(\d{3})-(\d{2})-(\d{4})(?!\d)")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")
CARD_RE = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")

# Fallback phone patterns (used only when `phonenumbers` is not installed)
PHONE_INTL_RE = re.compile(r"(?<![\w+])\+\d{1,3}[\s.-]?\(?\d{1,4}\)?(?:[\s.-]?\d{2,5}){2,4}")
PHONE_KR_RE = re.compile(r"(?<![\w-])0\d{1,2}[\s.-]?\d{3,4}[\s.-]?\d{4}(?![\w-])")

LABEL_SEPARATOR_RE = re.compile(r"^\s*(?P<label>[^:=|\uff1a]{1,45}?)\s*[:=|\uff1a]\s*(?P<value>.*?)\s*$")
VALUE_AFTER_LABEL_RE = re.compile(r"[A-Za-z0-9@+(]")

USERNAME_RE = re.compile(r"(?i)^\s*(?:username|user\s+name|user\s+id|login)\s*(?::|=|\|)")


# ============================================================
# SMALL HELPERS
# ============================================================

def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_value(value: Any) -> str:
    return _text(value).strip().strip(" \t\r\n\"'`,;").strip()


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def normalize_label_text(label: str) -> str:
    """Lower-case, keep letters/digits (any language), collapse the rest."""
    label = re.sub(r"['\u2019`]", "", _text(label).lower())
    label = re.sub(r"[^\w]+|_", " ", label, flags=re.UNICODE)
    return re.sub(r"\s+", " ", label).strip()


def is_username_field(text: str) -> bool:
    return bool(USERNAME_RE.match(_text(text)))


def luhn_ok(number: str) -> bool:
    digits = [int(ch) for ch in _digits(number)]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _card_issuer_ok(digits: str) -> bool:
    """Known card-network prefix + length (stops random 14-digit numbers passing Luhn)."""
    n = len(digits)
    if n < 13 or n > 19:
        return False
    if digits[0] == "4":
        return n in (13, 16, 19)
    if digits[:2] in {"51", "52", "53", "54", "55"} or 2221 <= int(digits[:4]) <= 2720:
        return n == 16
    if digits[:2] in {"34", "37"}:
        return n == 15
    if digits[:4] == "6011" or digits[:2] == "65" or digits[:3] in {"644", "645", "646", "647", "648", "649"}:
        return 16 <= n <= 19
    if digits[:2] in {"35", "62"}:
        return 16 <= n <= 19
    if digits[:2] in {"36", "38", "39"} or digits[:3] in {"300", "301", "302", "303", "304", "305"}:
        return 14 <= n <= 19
    return False


def _kr_rrn_ok(digits13: str) -> bool:
    if len(digits13) != 13 or not digits13.isdigit():
        return False
    month, day = int(digits13[2:4]), int(digits13[4:6])
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return False
    weights = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
    total = sum(int(d) * w for d, w in zip(digits13[:12], weights))
    return (11 - total % 11) % 10 == int(digits13[12])


def _iban_ok(value: str) -> bool:
    compact = re.sub(r"\s+", "", value).upper()
    if not 15 <= len(compact) <= 34 or not compact.isalnum():
        return False
    rearranged = compact[4:] + compact[:4]
    number = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(number) % 97 == 1


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts: Dict[str, int] = {}
    for ch in value:
        counts[ch] = counts.get(ch, 0) + 1
    return -sum((c / len(value)) * math.log2(c / len(value)) for c in counts.values())


def _looks_like_date_digits(digits: str) -> bool:
    if len(digits) != 8:
        return False
    year, month, day = int(digits[:4]), int(digits[4:6]), int(digits[6:])
    if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
        return True
    day, month, year = int(digits[:2]), int(digits[2:4]), int(digits[4:])
    return 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31


def _make_detection(
    detection_type: str,
    value: str,
    confidence: float = 0.95,
    box: Optional[Any] = None,
    source: str = "regex",
    box_kind: Optional[str] = None,
) -> Dict[str, Any]:
    item = {
        "type": detection_type,
        "value": _text(value).strip(),
        "confidence": max(0.0, min(1.0, float(confidence))),
        "box": box,
        "source": source,
    }
    if box_kind:
        item["box_kind"] = box_kind
    return item


# ============================================================
# LABEL CLASSIFICATION
# ============================================================

@lru_cache(maxsize=4096)
def _repair_token(token: str) -> str:
    """Fix OCR typos in a label word using a small vocabulary."""
    if token in LABEL_VOCAB or len(token) < 4 or not token.isascii():
        return token
    swapped = token.translate(str.maketrans("01$5", "olss"))
    if swapped in LABEL_VOCAB:
        return swapped
    match = get_close_matches(swapped, LABEL_VOCAB, n=1, cutoff=0.82)
    return match[0] if match else token


@lru_cache(maxsize=4096)
def classify_label(label: str) -> Optional[str]:
    """
    Return a detection type for a field label, "NEG" for labels that mean
    "not personal" (Order ID, Invoice No ...), or None if unknown.
    """
    normalized = normalize_label_text(label)
    if not normalized or len(normalized) > 45:
        return None

    if normalized in EXACT_LABELS:
        return EXACT_LABELS[normalized]

    tokens = [_repair_token(t) for t in normalized.split()]
    token_set = set(tokens)

    if not token_set:
        return None

    if token_set & {"dob", "birthdate", "birthday"} or {"birth", "date"} <= token_set or (
        "birth" in token_set and token_set <= {"date", "of", "birth", "day"}
    ):
        return "DATE_OF_BIRTH"

    if token_set & NEGATIVE_TOKENS and not token_set & {
        "password", "passcode", "passwd", "email", "phone", "mobile"
    }:
        return "NEG"

    if token_set & PASSWORD_TOKENS:
        # "Password", "New password", "WiFi password", "PIN code" are field labels.
        # "Password guessing", "Weak password policy" are ordinary text.
        if len(tokens) <= 3 and token_set <= PASSWORD_TOKENS | PASSWORD_LABEL_EXTRA:
            return "PASSWORD"
        return None

    if "private" in token_set and "key" in token_set or token_set & {"credential", "credentials"}:
        return "SECRET"

    if (
        token_set & {"api", "token", "secret", "apikey"}
        and token_set <= {"api", "key", "token", "secret", "access", "auth",
                          "authentication", "bearer", "client", "apikey"}
    ):
        return "API_KEY"

    if "email" in token_set or "mail" in token_set:
        if token_set <= {"email", "mail", "e", "address", "id", "personal", "work",
                         "university", "school", "company", "office"}:
            return "EMAIL"

    if token_set & {"phone", "mobile", "tel", "telephone", "cell", "whatsapp", "handphone"}:
        if token_set <= {"phone", "mobile", "tel", "telephone", "cell", "whatsapp",
                         "handphone", "no", "number", "num", "contact", "home",
                         "work", "office", "personal", "emergency"}:
            return "PHONE"
    if "contact" in token_set and token_set & NUMBER_TOKENS and token_set <= {"contact", "no", "number", "num"}:
        return "PHONE"

    if "address" in token_set and not token_set & {"ip", "mac", "web", "url", "website", "wallet", "email", "mail"}:
        return "ADDRESS"

    if (
        token_set & {"passport", "ssn", "rrn", "nid"}
        or {"social", "security"} <= token_set
        or {"resident", "registration"} <= token_set
        or ({"national", "government", "govt"} & token_set and token_set & ({"id", "identification"} | NUMBER_TOKENS))
        or (token_set & {"license", "licence", "driving", "driver", "drivers"} and token_set & ({"no", "number", "num", "id"} | {"license", "licence"})
            and token_set <= {"license", "licence", "driving", "driver", "drivers", "no", "number", "num", "id", "dl"})
        or ("tax" in token_set and token_set & {"id", "number", "no"})
    ):
        return "GOVERNMENT_ID"

    if token_set & {"card", "credit", "debit"} and token_set <= {
        "card", "credit", "debit", "bank", "number", "no", "num", "cc", "payment"
    }:
        return "BANK_CARD"

    if token_set & {"account", "acct", "iban"} or tokens == ["a", "c"]:
        if token_set <= {"account", "acct", "iban", "bank", "no", "number", "num",
                         "a", "c", "savings", "checking", "current", "id"}:
            return "ACCOUNT_NUMBER"
        if "holder" in token_set or "name" in token_set:
            return "NAME"

    if "name" in token_set and token_set <= PERSON_NAME_PREFIX and not token_set & {"user", "username", "file", "company", "business", "organization"}:
        return "NAME"
    if "holder" in token_set and token_set <= {"account", "card", "holder", "name"}:
        return "NAME"

    if token_set & NUMBER_TOKENS:
        extra = token_set - NUMBER_TOKENS
        if not extra:
            return "ID" if token_set & {"id", "identifier", "ident"} else None
        if extra <= PERSONAL_ID_PREFIX | {"of", "the", "personal", "identification"}:
            return "ID"
        if "identification" in token_set and not extra - {"identification", "personal"}:
            return "ID"

    return None


# ============================================================
# VALUE VALIDATION FOR A KNOWN LABEL
# ============================================================

def _find_phones(text: str) -> List[Tuple[str, int, int]]:
    """Return (raw, start, end) for every phone-like value in text."""
    found: List[Tuple[str, int, int]] = []

    def shape_ok(raw: str) -> bool:
        raw = raw.strip()
        digits = _digits(raw)
        if not 7 <= len(digits) <= 15:
            return False
        if DATE_RE.fullmatch(raw):
            return False
        # A bare digit string such as 2024891090 is more likely an ID than a phone.
        return (
            raw.startswith("+")
            or raw.startswith("0")
            or "(" in raw
            or bool(re.search(r"\d[\s.-]\d", raw))
        )

    if phonenumbers is not None and PhoneNumberMatcher is not None:
        for region in (*PHONE_REGIONS, None):
            try:
                matcher = PhoneNumberMatcher(text, region, leniency=Leniency.VALID)
                for match in matcher:
                    raw = match.raw_string
                    if shape_ok(raw):
                        found.append((raw, match.start, match.end))
            except Exception:
                continue
    else:
        for pattern in (PHONE_INTL_RE, PHONE_KR_RE):
            for match in pattern.finditer(text):
                if shape_ok(match.group(0)):
                    found.append((match.group(0), match.start(), match.end()))

    unique: List[Tuple[str, int, int]] = []
    for item in sorted(found, key=lambda x: (x[1], -(x[2] - x[1]))):
        if not any(_overlap((item[1], item[2]), (u[1], u[2])) for u in unique):
            unique.append(item)
    return unique


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _is_person_like(value: str) -> bool:
    value = re.sub(r"\([^)]*\)", " ", value).strip()
    words = value.split()
    if not 1 <= len(words) <= 5 or len(value) < 2 or len(value) > 60:
        return False
    if any(ch.isdigit() for ch in value) or "@" in value:
        return False
    if not all(all(ch.isalpha() or ch in ".'`-\u2019" for ch in word) for word in words):
        return False
    if value.lower() in {"n/a", "na", "none", "null", "unknown", "male", "female",
                         "student", "employee", "customer", "name", "your name"}:
        return False
    return True


def _looks_like_secret(token: str) -> bool:
    """API keys / tokens always contain digits or symbols (or are long and mixed-case)."""
    if len(token) < 8 or not re.fullmatch(r"[A-Za-z0-9_\-+/=.:@!#$%^&*~]+", token):
        return False
    if re.fullmatch(r"[A-Za-z]+[.,;:!?]?", token):
        has_upper = any(c.isupper() for c in token)
        has_lower = any(c.islower() for c in token)
        return has_upper and has_lower and len(token) >= 16  # e.g. ThisIsALongMixedKey
    return True


def _looks_like_password(value: str, strict: bool = False) -> bool:
    """Reject sentences. strict=True (value on the next line) also needs complexity."""
    words = value.split()
    if not 3 <= len(value) <= 128 or not words:
        return False
    if len(words) >= 4:
        return False  # prose
    if len(words) >= 2 and value.rstrip().endswith((".", "!", "?")):
        return False  # a sentence
    if re.fullmatch(r"[A-Za-z]+[.,;:]", value.strip()):
        return False  # a plain word with punctuation, e.g. "key."
    if strict:
        if len(words) != 1:
            return False
        has_digit = any(c.isdigit() for c in value)
        has_symbol = bool(re.search(r"[^A-Za-z0-9]", value))
        mixed = any(c.islower() for c in value) and any(c.isupper() for c in value)
        return has_digit or has_symbol or mixed
    return True


def _validate_for_type(
    detection_type: str,
    raw_value: str,
    strict: bool = False,
) -> List[Tuple[str, float, str]]:
    """
    Validate a value that appeared after a label of the given type.
    Returns a list of (type, confidence, cleaned_value).
    """
    value = _clean_value(raw_value)
    if not value:
        return []

    if detection_type == "PASSWORD":
        return [("PASSWORD", 0.99, value)] if _looks_like_password(value, strict) else []

    if detection_type in {"API_KEY", "SECRET"}:
        token = value.split()[0].strip(",;\"'")
        return [(detection_type, 0.98, token)] if _looks_like_secret(token) else []

    if detection_type == "EMAIL":
        return [("EMAIL", 0.99, _normalise_email(m.group(0))) for m in EMAIL_LOOSE_RE.finditer(value)]

    if detection_type == "PHONE":
        phones = _find_phones(value)
        if phones:
            return [("PHONE", 0.97, raw.strip()) for raw, _, _ in phones]
        out = []
        for part in re.split(r"\s*[/;,]\s*|\s+or\s+", value):
            part = part.strip()
            if re.fullmatch(r"[+(]?\d[\d\s().-]{5,}\d", part) and 7 <= len(_digits(part)) <= 15:
                out.append(("PHONE", 0.9, part))
        return out

    if detection_type == "NAME":
        value = re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()
        return [("NAME", 0.95, value)] if _is_person_like(value) else []

    if detection_type == "ADDRESS":
        if len(value) >= 5 and any(ch.isalpha() for ch in value):
            return [("ADDRESS", 0.94, value)]
        return []

    if detection_type == "DATE_OF_BIRTH":
        return [("DATE_OF_BIRTH", 0.96, m.group(0)) for m in DATE_RE.finditer(value)]

    if detection_type == "BANK_CARD":
        digits = _digits(value)
        if 13 <= len(digits) <= 19:
            return [("BANK_CARD", 0.99 if luhn_ok(digits) else 0.8, value)]
        return []

    if detection_type == "ACCOUNT_NUMBER":
        compact = re.sub(r"\s+", "", value)
        if re.fullmatch(r"[A-Za-z]{2}\d{2}[A-Za-z0-9]{10,30}", compact) and _iban_ok(compact):
            return [("ACCOUNT_NUMBER", 0.99, value)]
        if re.fullmatch(r"[0-9][0-9 -]{4,40}[0-9]", value) and 6 <= len(_digits(value)) <= 30:
            return [("ACCOUNT_NUMBER", 0.96, value)]
        return []

    if detection_type == "GOVERNMENT_ID":
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ./-]{3,30}", value) and any(c.isdigit() for c in value):
            return [("GOVERNMENT_ID", 0.97, value)]
        return []

    if detection_type == "ID":
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ./_-]{2,30}", value)
            and sum(c.isdigit() for c in value) >= 3
            and len(re.findall(r"[A-Za-z]{3,}", value)) <= 1
        ):
            return [("ID", 0.96, value)]
        return []

    return []


def _normalise_email(value: str) -> str:
    value = re.sub(r"\s+", "", value)
    local, _, domain = value.partition("@")
    return f"{local}@{domain.replace(',', '.')}"


# ============================================================
# STANDALONE (UNLABELED) VALUE DETECTION
# ============================================================

def _document_context(items: Sequence[Dict[str, Any]]) -> Dict[str, bool]:
    joined = " ".join(_text(item.get("text")) for item in items).lower()
    return {
        name: any(word in joined for word in words)
        for name, words in CONTEXT_WORDS.items()
    }


def _classify_standalone_token(
    token: str,
    context: Dict[str, bool],
) -> Optional[Tuple[str, float]]:
    """
    Classify a whole-line token that has no label. Returns (type, confidence)
    or None. Confidence is intentionally lower than for labeled values.
    """
    token = token.strip()
    digits = _digits(token)

    if re.fullmatch(r"\d{2,4}(?:[- ]\d{2,8}){2,4}", token) and 10 <= len(digits) <= 16:
        # e.g. 110-123-456789 : grouped digits look like a bank account
        if DATE_RE.fullmatch(token):
            return None
        return ("ACCOUNT_NUMBER", 0.72 if context["finance"] else 0.6)

    if re.fullmatch(r"\d{8,16}", token):
        if _looks_like_date_digits(token) or token.startswith("0"):
            return None
        if len(token) >= 12 and context["finance"]:
            return ("ACCOUNT_NUMBER", 0.7)
        if len(token) >= 10 and context["finance"] and not context["education"]:
            return ("ACCOUNT_NUMBER", 0.7)
        if context["education"] or context["employment"] or context["health"]:
            return ("ID", 0.72)
        if context["government"] and len(token) >= 9:
            return ("GOVERNMENT_ID", 0.6)
        return ("ID", 0.6)

    if re.fullmatch(r"\d{4}[- ]\d{3,4}[- ]\d{3,4}", token) and context["education"]:
        return ("ID", 0.65)

    # letter(s) + digits, e.g. M12345678 or S2024891
    if re.fullmatch(r"[A-Za-z]{1,3}\d{6,9}", token):
        if context["government"]:
            return ("GOVERNMENT_ID", 0.7)
        return ("ID", 0.55)

    # year-department-number student IDs, e.g. 2024-CS-0123
    if re.fullmatch(r"(?:19|20)\d{2}[- ]?[A-Za-z]{2,4}[- ]?\d{3,6}", token) and (
        context["education"] or context["employment"]
    ):
        return ("ID", 0.65)

    return None


def _high_entropy_secret(token: str) -> bool:
    if not 24 <= len(token) <= 200:
        return False
    if not re.fullmatch(r"[A-Za-z0-9_\-+/=.]+", token):
        return False
    classes = sum([
        bool(re.search(r"[a-z]", token)),
        bool(re.search(r"[A-Z]", token)),
        bool(re.search(r"\d", token)),
    ])
    return classes >= 3 and _shannon_entropy(token) >= 3.6


def _detect_value_patterns(
    line: str,
    consumed: List[Tuple[int, int]],
) -> List[Tuple[str, float, str, int, int, str]]:
    """
    Value-first detection inside one line. Returns tuples of
    (type, confidence, value, start, end, source).
    """
    results: List[Tuple[str, float, str, int, int, str]] = []

    def free(start: int, end: int) -> bool:
        return not any(_overlap((start, end), span) for span in consumed)

    def add(kind, conf, value, start, end, source):
        if free(start, end):
            results.append((kind, conf, value, start, end, source))
            consumed.append((start, end))

    # 1. e-mail
    for match in EMAIL_LOOSE_RE.finditer(line):
        value = _normalise_email(match.group(0))
        if EMAIL_RE.fullmatch(value):
            add("EMAIL", 0.99, value, match.start(), match.end(), "pattern")

    # 2. well-known secret formats
    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(line):
            add("API_KEY", 0.97, match.group(0), match.start(), match.end(), "pattern")

    # 3. national IDs with checksum
    for match in KR_RRN_RE.finditer(line):
        if _kr_rrn_ok(match.group(1) + match.group(2)):
            add("GOVERNMENT_ID", 0.97, match.group(0), match.start(), match.end(), "checksum")
    for match in US_SSN_RE.finditer(line):
        area, group, serial = match.groups()
        if area not in {"000", "666"} and not area.startswith("9") and group != "00" and serial != "0000":
            add("GOVERNMENT_ID", 0.85, match.group(0), match.start(), match.end(), "pattern")
    for match in IBAN_RE.finditer(line):
        if _iban_ok(match.group(0)):
            add("ACCOUNT_NUMBER", 0.98, match.group(0), match.start(), match.end(), "checksum")

    # 4. payment cards (Luhn)
    for match in CARD_RE.finditer(line):
        if luhn_ok(match.group(0)) and _card_issuer_ok(_digits(match.group(0))):
            add("BANK_CARD", 0.97, match.group(0).strip(), match.start(), match.end(), "luhn")

    # 5. phones
    for raw, start, end in _find_phones(line):
        add("PHONE", 0.95, raw.strip(), start, end, "phone")

    return results


# ============================================================
# LINE-LEVEL ENGINE (shared by text, PDF and OCR)
# ============================================================

def _split_label_value(line: str) -> Optional[Tuple[str, str, str]]:
    """Return (label, value, kind) for 'Label: value' or 'Label value'."""
    match = LABEL_SEPARATOR_RE.match(line)
    if match and match.group("value"):
        return match.group("label").strip(), match.group("value").strip(), "separator"

    # No separator: "Student ID 2024891090", "Passport M12345678"
    words = line.split()
    if 2 <= len(words) <= 7:
        # longest label first: "Passport No M123" -> label "Passport No", value "M123"
        for cut in range(min(len(words) - 1, 4), 0, -1):
            label = " ".join(words[:cut])
            value = " ".join(words[cut:])
            kind = classify_label(label)
            if kind and kind != "NEG" and VALUE_AFTER_LABEL_RE.match(value):
                if kind in {"NAME", "ADDRESS", "PASSWORD", "API_KEY", "SECRET"}:
                    continue  # too ambiguous in normal sentences without a ":" or "="
                if not any(ch.isdigit() for ch in value) and "@" not in value:
                    continue
                return label, value, "no-separator"
    return None


def _label_only(line: str) -> Optional[str]:
    """If the whole line is just a field label, return its type."""
    stripped = re.sub(r"[:=|\uff1a\s]+$", "", line.strip())
    if not stripped or len(stripped) > 40 or any(ch.isdigit() for ch in stripped) or "@" in stripped:
        return None
    kind = classify_label(stripped)
    return kind if kind and kind != "NEG" else None


def _box_for(item: Dict[str, Any], start: int, end: int, base: int) -> Tuple[Any, Optional[str]]:
    box = item.get("box")
    if box:
        return box, None
    return [base + start, 0, base + end, 1], "offset"


# ------------------------------------------------------------
# Unlabeled person names (no label, no AI model needed)
# ------------------------------------------------------------

NAME_STOPWORDS = set(NEGATIVE_TOKENS) | set(LABEL_VOCAB) | {
    # organisations / document headings
    "university", "college", "school", "institute", "academy", "bank", "hospital",
    "clinic", "company", "corporation", "corp", "inc", "ltd", "llc", "department",
    "faculty", "office", "center", "centre", "information", "student", "employee",
    "statement", "report", "form", "certificate", "services", "service", "system",
    "national", "republic", "ministry", "government", "confidential", "private",
    "public", "welcome", "dear", "profile", "education", "experience", "skills",
    "skill", "career", "interest", "project", "projects", "technical", "academic",
    "teamwork", "summary", "objective", "references", "languages", "certifications",
    "developer", "engineer", "manager", "intern", "software", "python", "aspiring",
    "capstone", "design", "application", "workflow", "detection", "risk", "analyst",
    "director", "assistant", "consultant", "specialist", "teacher", "professor",
    "undergraduate", "graduate", "degree", "bachelor", "master", "science",
    "computer", "engineering", "business", "management", "development", "research",
    "security", "privacy", "protection", "digital", "content", "data", "image",
    # generic words that start lines
    "total", "amount", "balance", "description", "details", "status", "type",
    "title", "subject", "sent", "received", "hello", "thanks", "thank", "regards",
    "sincerely", "signature", "approved", "pending", "active", "main", "home",
    "about", "terms", "conditions", "policy", "settings", "login", "logout",
    "sign", "submit", "cancel", "next", "back", "search", "menu", "help",
    "support", "news", "personal", "details", "resume", "curriculum", "vitae",
    # function words
    "and", "or", "of", "the", "for", "with", "in", "to", "a", "an", "at", "by",
    "is", "are", "on", "from", "not", "no", "yes",
    # places often written in Title Case
    "korea", "seoul", "busan", "usa", "nepal", "japan", "india", "china",
    "america", "kingdom", "york", "city", "street", "road", "south", "north",
    "east", "west", "united", "states", "republic",
    # security / IT headings and document structure
    "phishing", "smishing", "vishing", "malware", "ransomware", "spyware", "virus",
    "viruses", "trojan", "worm", "botnet", "attack", "attacks", "threat", "threats",
    "vulnerability", "vulnerabilities", "exploit", "social", "engineering", "brute",
    "force", "dictionary", "injection", "scripting", "cross", "site", "man", "middle",
    "denial", "firewall", "encryption", "authentication", "authorization", "network",
    "protocol", "cyber", "cybersecurity", "hacking", "hacker", "breach", "leak",
    "fraud", "scam", "spoofing", "sniffing", "keylogger", "backdoor", "rootkit",
    "zero", "day", "patch", "antivirus", "vpn", "proxy", "guessing", "reuse",
    "common", "types", "example", "examples", "introduction", "conclusion",
    "chapter", "section", "lecture", "notes", "slide", "figure", "table", "tips",
    "prevention", "protection", "defense", "defence", "risks", "overview",
    # buttons, menus and labels of apps / banking screenshots ("View All", "Send Money")
    "view", "all", "see", "more", "less", "show", "hide", "details", "detail",
    "history", "transactions", "transaction", "transfer", "transfers", "send",
    "receive", "pay", "payment", "payments", "paid", "deposit", "deposits",
    "withdraw", "withdrawal", "withdrawals", "balances", "accounts", "setting",
    "settings", "notifications", "notification", "filter", "filters", "sort",
    "today", "yesterday", "tomorrow", "recent", "recently", "activity",
    "activities", "cards", "savings", "checking", "current", "complete",
    "completed", "success", "successful", "failed", "failure", "error", "edit",
    "done", "next", "previous", "continue", "back", "close", "cancel", "ok",
    "okay", "confirm", "confirmation", "log", "signup", "add", "remove",
    "delete", "copy", "share", "download", "upload", "language", "new", "free",
    "buy", "sell", "trade", "invest", "investment", "loan", "loans", "credit",
    "debit", "reward", "rewards", "offers", "offer", "shop", "store", "cart",
    "order", "orders", "track", "tracking", "update", "updates", "version",
    "terms", "forgot", "reset", "verify", "verification", "code", "resend",
    "skip", "learn", "explore", "discover", "get", "started", "start", "finish",
    "review", "rate", "rates", "feedback", "alerts", "alert", "message",
    "messages", "inbox", "drafts", "spam", "trash", "archive", "select",
    "selected", "enable", "disable", "quick", "actions", "action", "overview",
    "dashboard", "statements", "bills", "top", "up", "down", "scan", "qr", "now",
    "money", "wallet", "mobile", "banking", "online", "secure", "safe", "easy",
    "fast", "instant", "daily", "weekly", "monthly", "yearly", "annual", "limit",
    "limits", "fee", "fees", "charge", "charges", "exchange", "currency", "via",
    "per", "each", "every", "any", "other", "others", "another", "your", "you",
    "my", "me", "we", "us", "our", "it", "its", "this", "that", "these", "those",
    "here", "there", "what", "when", "where", "why", "how", "who", "which",
    "please", "tap", "click", "press", "swipe", "enter", "choose", "pick", "set",
    "change", "manage", "create", "open", "refresh", "reload", "retry", "try",
    "again", "loading", "wait", "processing", "premium", "platinum", "standard",
    "basic", "default", "custom", "general", "advanced", "options", "option",
    "profile", "category", "categories", "income", "expense", "expenses",
    "budget", "spending", "insights", "analytics", "reports", "history",
    # calendar
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday",
}

PERSONAL_NEIGHBOUR_TYPES = {
    "EMAIL", "PHONE", "ID", "GOVERNMENT_ID", "ACCOUNT_NUMBER",
    "DATE_OF_BIRTH", "ADDRESS", "BANK_CARD",
}

TITLED_NAME_RE = re.compile(
    r"(?:\b(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof)\.?\s+|\bDear\s+|\b(?:Prepared|Submitted|Signed|Issued)\s+(?:by|to)\s+)"
    r"([A-Z][A-Za-z'\u2019-]+(?:\s+[A-Z][A-Za-z'\u2019-]+){0,2})"
)


def _name_candidate(text: str) -> bool:
    """True when a whole line looks like a person's name (2-4 Latin words)."""
    text = text.strip()
    if not 4 <= len(text) <= 40 or any(ch.isdigit() for ch in text):
        return False
    if re.search(r"[^\w\s.'\u2019-]", text) or "_" in text:
        return False
    words = text.split()
    if not 2 <= len(words) <= 4:
        return False
    for word in words:
        bare = re.sub(r"[.'\u2019-]", "", word)
        if not bare or not bare.isalpha() or not (word[0].isascii() and word[0].isalpha()):
            return False
        if len(bare) < 2 and not word.endswith("."):
            return False
    if {w.lower().strip(".") for w in words} & NAME_STOPWORDS:
        return False
    all_upper = all(w.replace(".", "").isupper() for w in words)
    title_case = all(w[0].isupper() and not w.isupper() for w in words)
    return all_upper or title_case


def _unlabeled_names(
    items: Sequence[Dict[str, Any]],
    used: Sequence[bool],
    detections: Sequence[Dict[str, Any]],
    det_indexes: Sequence[int],
) -> List[Tuple[int, str, float, str]]:
    """Return (item_index, name, confidence, source) for names without a label."""
    personal_at = [
        det_indexes[i] for i, det in enumerate(detections)
        if det.get("type") in PERSONAL_NEIGHBOUR_TYPES
    ]
    explained = set(det_indexes)
    found: List[Tuple[int, str, float, str]] = []
    nonempty = [i for i, item in enumerate(items) if _text(item.get("text")).strip()]

    for position, index in enumerate(nonempty):
        text = _text(items[index].get("text")).strip()
        if index in explained or used[index] or classify_label(text) or _label_only(text):
            continue

        # "Dr. Anish Kandel", "Dear Anish Kandel", "Prepared by Anish Kandel"
        titled = TITLED_NAME_RE.search(text)
        if titled:
            name = titled.group(1).strip()
            if not ({w.lower() for w in name.split()} & NAME_STOPWORDS):
                found.append((index, name, 0.85, "name_title"))
                continue

        if not _name_candidate(text):
            continue

        # a. one of the first lines of the page/document, and the document has personal data
        if position <= 2 and any(i <= index + 15 for i in personal_at):
            found.append((index, text, 0.72, "name_header"))
            continue

        # b. right next to an email / phone / ID / address
        if any(abs(i - index) <= 2 for i in personal_at):
            found.append((index, text, 0.62, "name_nearby"))

    return found


def detect_lines(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Core engine. `items` is a list of {"text", "confidence", "box"} in reading
    order (OCR boxes, or the lines of a text document with box=None).
    """
    items = [dict(item) for item in items if _text(item.get("text")).strip()]
    if not items:
        return []

    context = _document_context(items)
    detections: List[Dict[str, Any]] = []
    used = [False] * len(items)          # item fully explained by a detection
    negative_next = False                # previous line was a NEG label-only line

    # character offset of each line inside the joined text (for offset boxes)
    offsets: List[int] = []
    running = 0
    for item in items:
        offsets.append(running)
        running += len(_text(item.get("text"))) + 1

    def ocr_conf(item: Dict[str, Any]) -> Optional[float]:
        value = item.get("confidence")
        return None if value is None else max(0.5, min(1.0, _safe_float(value, 1.0)))

    det_indexes: List[int] = []          # item index of every detection (same order)

    def emit(kind, conf, value, item, start, end, source, index):
        cap = ocr_conf(item)
        if cap is not None:
            conf = min(conf, cap)
        box, box_kind = _box_for(item, start, end, offsets[index])
        detections.append(_make_detection(kind, value, conf, box, source, box_kind))
        det_indexes.append(index)

    for index, item in enumerate(items):
        text = _text(item.get("text")).strip()
        consumed: List[Tuple[int, int]] = []
        item_negative = negative_next
        negative_next = False

        if used[index] or not text:
            continue

        # ---- 1. label + value on the same line -------------------------
        split = _split_label_value(text)
        labeled_hit = False
        if split:
            label, value, how = split
            kind = classify_label(label)

            if kind == "NEG":
                # "Order ID: 12345678" -> deliberately ignore the value
                used[index] = True
                continue

            if kind and not (is_username_field(text) and kind == "NAME"):
                found = _validate_for_type(kind, value)
                for det_type, conf, det_value in found:
                    start = text.find(det_value)
                    start = max(start, 0)
                    emit(det_type, conf if how == "separator" else min(conf, 0.9),
                         det_value, item, start, start + len(det_value),
                         "label" if how == "separator" else "label_no_sep", index)
                    consumed.append((start, start + len(det_value)))
                    labeled_hit = True
                if kind == "ADDRESS" and labeled_hit:
                    _address_continuation(items, index, used, emit)
                if labeled_hit:
                    used[index] = True

        # ---- 2. a line that is only a label: value is on a following line
        pending_type = None if labeled_hit else _label_only(text)
        if pending_type:
            for offset in (1, 2):
                nxt = index + offset
                if nxt >= len(items) or used[nxt]:
                    continue
                next_text = _text(items[nxt].get("text")).strip()
                if classify_label(next_text) or _label_only(next_text):
                    break  # next line is another label
                found = _validate_for_type(pending_type, next_text, strict=True)
                if found:
                    for det_type, conf, det_value in found:
                        start = max(next_text.find(det_value), 0)
                        emit(det_type, min(conf, 0.93), det_value, items[nxt],
                             start, start + len(det_value), "label_next_line", nxt)
                    used[nxt] = True
                    if pending_type == "ADDRESS":
                        _address_continuation(items, nxt, used, emit)
                    break
            used[index] = True
            continue

        if labeled_hit:
            # also catch extra values on the same line that have their own shape
            pass

        # a NEG label-only line makes the following value non-personal
        stripped = re.sub(r"[:=|\uff1a\s]+$", "", text)
        if classify_label(stripped) == "NEG" and len(stripped) <= 40 and not any(ch.isdigit() for ch in stripped):
            negative_next = True
            continue

        # ---- 3. value-first patterns (email, phone, card, secrets ...)
        for det_type, conf, det_value, start, end, source in _detect_value_patterns(text, consumed):
            emit(det_type, conf, det_value, item, start, end, source, index)

        # ---- 4. standalone number / ID without any label -----------------
        if not item_negative and not labeled_hit and not consumed:
            token = text.strip()
            if " " not in token or re.fullmatch(r"[\d -]+", token):
                guess = _classify_standalone_token(token, context)
                if guess:
                    kind, conf = guess
                    emit(kind, conf, token, item, 0, len(token), "unlabeled", index)

        # ---- 5. high-entropy token (unlabeled key / token) ---------------
        if not consumed and not labeled_hit and not item_negative:
            for word in text.split():
                if _high_entropy_secret(word):
                    start = text.find(word)
                    emit("API_KEY", 0.7, word, item, start, start + len(word), "entropy", index)

    # ---- 6. names that have no label (header line, next to personal data, "Dr. X")
    for index, name, conf, source in _unlabeled_names(items, used, detections, det_indexes):
        item = items[index]
        start = max(_text(item.get("text")).find(name), 0)
        emit("NAME", conf, name, item, start, start + len(name), source, index)

    return detections


def _address_continuation(items, index, used, emit) -> None:
    """Add up to two following lines that look like the rest of an address."""
    for step in (1, 2):
        nxt = index + step
        if nxt >= len(items) or used[nxt]:
            return
        text = _text(items[nxt].get("text")).strip()
        if (
            len(text) < 4
            or "@" in text
            or classify_label(text)
            or _split_label_value(text)
            or _label_only(text)
            or not (any(ch.isdigit() for ch in text) or "," in text)
            or sum(ch.isalpha() for ch in text) < 3
        ):
            return
        emit("ADDRESS", 0.8, text, items[nxt], 0, len(text), "address_continuation", nxt)
        used[nxt] = True


# ============================================================
# PUBLIC DETECTION API
# ============================================================

def _text_to_items(text: str, box: Any = None) -> List[Dict[str, Any]]:
    if box is not None:
        # A single OCR line with its real box.
        return [{"text": text, "confidence": None, "box": box}]
    return [
        {"text": line, "confidence": None, "box": None}
        for line in text.splitlines()
        if line.strip()
    ]


def _json_detections(text: str) -> List[Dict[str, Any]]:
    """Handle JSON documents ({"email": "..."}) where keys act as labels."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return []
    try:
        parsed = json.loads(stripped)
    except (ValueError, TypeError):
        return []

    results: List[Dict[str, Any]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                    kind = classify_label(str(key))
                    if kind and kind != "NEG":
                        for det_type, conf, det_value in _validate_for_type(kind, str(value)):
                            results.append(_make_detection(det_type, det_value, conf, None, "json"))
                else:
                    walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(parsed)
    return results


def _table_detections(lines: Sequence[str]) -> List[Dict[str, Any]]:
    """
    CSV / TSV / semicolon / pipe tables: the header row names the columns,
    every following row of the same width is read column by column.
        Name,Email,Phone
        Anish Kandel,a@b.com,+82 10-1234-5678
    """
    import csv

    results: List[Dict[str, Any]] = []
    lines = [line for line in lines if line.strip()]

    for delimiter in (",", "\t", ";", "|"):
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(lines, delimiter=delimiter, skipinitialspace=True)
        ]
        index = 0
        while index < len(rows) - 1:
            header = rows[index]
            kinds = [classify_label(cell) for cell in header]
            known = sum(1 for kind in kinds if kind and kind != "NEG")
            if len(header) < 2 or known < 2:
                index += 1
                continue

            index += 1
            while index < len(rows) and len(rows[index]) == len(header):
                for kind, value in zip(kinds, rows[index]):
                    if kind and kind != "NEG":
                        for det_type, conf, det_value in _validate_for_type(kind, value):
                            results.append(_make_detection(det_type, det_value, conf, None, "table"))
                index += 1
    return results


def detect_text(text: str, box: Any = None) -> List[Dict[str, Any]]:
    """Detect sensitive values in plain text (or in one OCR line if box is given)."""
    text = _text(text)
    if not text.strip():
        return []

    detections = detect_lines(_text_to_items(text, box))

    if box is None:
        detections.extend(_json_detections(text))
        detections.extend(_table_detections(text.splitlines()))
    else:
        for det in detections:
            det["box"] = det.get("box") or box

    return deduplicate_detections(detections)


def detect_from_ocr_results(
    ocr_results: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run all rule-based detectors on OCR items (label-next-line aware)."""
    items = []
    for item in ocr_results or []:
        if isinstance(item, dict) and _text(item.get("text")).strip():
            items.append({
                "text": item.get("text"),
                "confidence": _safe_float(item.get("confidence", 0.9), 0.9),
                "box": item.get("box", item.get("bbox")),
            })

    detections = detect_lines(items)
    for det in detections:
        det["source"] = f"ocr_{det.get('source', 'rule')}"
    return deduplicate_detections(detections)


def detect_sensitive(
    text: Optional[str] = None,
    ocr_results: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    if text:
        detections.extend(detect_text(text))
    if ocr_results:
        detections.extend(detect_from_ocr_results(ocr_results))
    return deduplicate_detections(detections)


# ============================================================
# NORMALIZATION / DEDUPLICATION / CONFLICTS
# ============================================================

def normalize_detections(
    detections: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    normalized = []
    for item in detections or []:
        if not isinstance(item, dict):
            continue

        detection_type = item.get("type") or item.get("label") or item.get("category") or ""
        value = item.get("value") or item.get("text") or item.get("entity") or ""
        detection_type = str(detection_type).upper().strip()
        value = str(value).strip()
        if not detection_type or not value:
            continue

        confidence = _safe_float(item.get("confidence", item.get("score", 0.9)), 0.9)
        normalized.append({
            **item,
            "type": detection_type,
            "value": value,
            "confidence": max(0.0, min(1.0, confidence)),
            "box": item.get("box", item.get("bbox")),
        })
    return normalized


def _box_key(item: Dict[str, Any]) -> Optional[tuple]:
    if item.get("box_kind") == "offset":
        return None  # character offsets are not real locations
    box = item.get("box")
    if isinstance(box, (list, tuple)) and box:
        try:
            return tuple(int(round(float(v) / 4.0)) for v in box[:4])  # tolerant to 3px jitter
        except (TypeError, ValueError):
            return None
    return None


def _dedup_key(item: Dict[str, Any]) -> tuple:
    return (
        str(item.get("type", "")).upper(),
        re.sub(r"\s+", " ", str(item.get("value", "")).strip().lower()),
        _box_key(item),
    )


def deduplicate_detections(
    detections: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge duplicates, but keep the same value if it appears in a different place."""
    output: Dict[tuple, Dict[str, Any]] = {}
    for item in normalize_detections(detections):
        key = _dedup_key(item)
        current = output.get(key)
        if current is None or float(item.get("confidence", 0)) > float(current.get("confidence", 0)):
            output[key] = item
    return resolve_conflicts(list(output.values()))


def resolve_conflicts(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    When the same value at the same place is reported with two different types
    (e.g. an ID and a phone), keep the best one.
    """
    best: Dict[tuple, Dict[str, Any]] = {}
    order: List[tuple] = []
    for item in detections:
        raw_value = str(item.get("value", ""))
        if re.fullmatch(r"[\d\s+().\-/]+", raw_value):
            value_key = re.sub(r"\D", "", raw_value)          # numeric: ignore separators
        else:
            value_key = re.sub(r"\s+", " ", raw_value.strip().lower())  # keep the full text
        key = (value_key, _box_key(item))
        if key not in best:
            best[key] = item
            order.append(key)
            continue
        rank = (float(item.get("confidence", 0)), TYPE_PRIORITY.get(item.get("type"), 0))
        current = best[key]
        current_rank = (float(current.get("confidence", 0)), TYPE_PRIORITY.get(current.get("type"), 0))
        if rank > current_rank:
            best[key] = item
    return [best[key] for key in order]


# ============================================================
# BACKWARD-COMPATIBLE HELPERS
# ============================================================

def _only(types: Iterable[str], text: str, box: Any = None) -> List[Dict[str, Any]]:
    wanted = set(types)
    return [d for d in detect_text(text, box) if d["type"] in wanted]


def detect_emails(text: str, box: Any = None):
    return _only({"EMAIL"}, text, box)


def detect_international_phones(text: str, box: Any = None):
    return _only({"PHONE"}, text, box)


def detect_labeled_phone(text: str, box: Any = None):
    return _only({"PHONE"}, text, box)


def detect_name(text: str, box: Any = None):
    return _only({"NAME"}, text, box)


def detect_date_of_birth(text: str, box: Any = None):
    return _only({"DATE_OF_BIRTH"}, text, box)


def detect_address(text: str, box: Any = None):
    return _only({"ADDRESS"}, text, box)


def detect_id_number(text: str, box: Any = None):
    return _only({"ID"}, text, box)


def detect_government_id(text: str, box: Any = None):
    return _only({"GOVERNMENT_ID"}, text, box)


def detect_password(text: str, box: Any = None):
    return _only({"PASSWORD"}, text, box)


def detect_api_key(text: str, box: Any = None):
    return _only({"API_KEY", "SECRET"}, text, box)


def detect_bank_account(text: str, box: Any = None):
    return _only({"ACCOUNT_NUMBER"}, text, box)


detect_sensitive_data = detect_sensitive
detect_sensitive_text = detect_text


def score_detections(detections):
    from backend.risk.scoring import score_detections as _score
    return _score(detections)


def analyze_sensitive_data(
    text: Optional[str] = None,
    ocr_results: Optional[Iterable[Dict[str, Any]]] = None,
):
    return score_detections(detect_sensitive(text=text, ocr_results=ocr_results))


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