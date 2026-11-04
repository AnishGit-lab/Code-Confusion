"""
PriviScan - AI + rule detection for OCR results (faster version).

Pipeline for one image
----------------------
1. Rule engine (backend.detection.sensitive): labeled values, labels on the
   previous line, and VALUE-FIRST patterns (email, phone, card+Luhn, keys,
   national IDs, unlabeled numbers). This costs milliseconds.
2. GLiNER (NER) runs ONLY on the OCR lines that the rule engine did not
   explain, and only for what rules cannot do well: person names and
   addresses. All candidate lines are sent to the model in BATCHES instead of
   one model call per OCR line.
3. Merge, remove conflicts, validate, sort by position.

Environment variables
---------------------
PRIVISCAN_USE_GLINER      "1" (default) or "0" to switch the model off
PRIVISCAN_GLINER_MODEL    default "urchade/gliner_small-v2.1"
PRIVISCAN_GLINER_LABELS   "core" (default: person name + address) or "full"
PRIVISCAN_GLINER_BATCH    batch size, default 16
PRIVISCAN_GLINER_THRESHOLD  default 0.55
PRIVISCAN_DEBUG           "1" prints how many items each stage found
"""

from __future__ import annotations

import os
import re
import traceback
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional

from backend.detection import sensitive as rules


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.environ.get("PRIVISCAN_GLINER_MODEL", "urchade/gliner_small-v2.1")
USE_GLINER = os.environ.get("PRIVISCAN_USE_GLINER", "1") != "0"
LABEL_SET = os.environ.get("PRIVISCAN_GLINER_LABELS", "core").lower()
BATCH_SIZE = max(1, int(os.environ.get("PRIVISCAN_GLINER_BATCH", "16")))
THRESHOLD = float(os.environ.get("PRIVISCAN_GLINER_THRESHOLD", "0.55"))
DEBUG = os.environ.get("PRIVISCAN_DEBUG", "0") == "1"

_GLINER_MODEL = None
_GLINER_LOAD_LOCK = Lock()
_GLINER_FAILED = False

SUPPORTED_TYPES = {
    "NAME", "EMAIL", "PHONE", "ADDRESS", "ID", "GOVERNMENT_ID",
    "ACCOUNT_NUMBER", "BANK_CARD", "PASSWORD", "API_KEY", "SECRET",
    "DATE_OF_BIRTH",
}

CORE_LABELS = ["person name", "physical address"]
FULL_LABELS = CORE_LABELS + [
    "email address", "phone number", "identification number",
    "government identification number", "account number", "bank card number",
    "password", "API key",
]
AI_LABELS = FULL_LABELS if LABEL_SET == "full" else CORE_LABELS

LABEL_MAPPING = {
    "person name": "NAME",
    "physical address": "ADDRESS",
    "email address": "EMAIL",
    "phone number": "PHONE",
    "identification number": "ID",
    "government identification number": "GOVERNMENT_ID",
    "account number": "ACCOUNT_NUMBER",
    "bank card number": "BANK_CARD",
    "password": "PASSWORD",
    "api key": "API_KEY",
    "secret": "SECRET",
}

# Words that mean "this is an organisation / heading, not a person".
ORG_WORDS = {
    "university", "college", "school", "institute", "academy", "bank", "hospital",
    "clinic", "company", "corporation", "corp", "inc", "ltd", "llc", "department",
    "faculty", "office", "center", "centre", "information", "student", "employee",
    "account", "statement", "report", "form", "card", "certificate", "invoice",
    "receipt", "services", "service", "system", "national", "republic", "ministry",
    "government", "page", "confidential", "private", "public", "welcome", "dear",
    "profile", "education", "experience", "skills", "skill", "career", "interest",
    "project", "projects", "technical", "academic", "teamwork", "summary",
    "objective", "contact", "references", "languages", "certifications",
    "developer", "engineer", "manager", "intern", "software", "python", "aspiring",
    "capstone", "design", "application", "workflow", "detection", "risk",
}

STREET_WORDS = {
    "street", "st", "road", "rd", "avenue", "ave", "lane", "ln", "drive", "dr",
    "blvd", "boulevard", "gu", "dong", "ro", "gil", "si", "do", "apt", "apartment",
    "building", "floor", "district", "city", "province", "block", "unit", "suite",
}


# ============================================================
# COMPATIBILITY HELPERS (names used by older code)
# ============================================================

def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9 ]", "", normalize_text(value))


def is_username_field(text: Any) -> bool:
    return rules.is_username_field(str(text or ""))


def is_valid_email(value: Any) -> bool:
    return bool(value) and bool(rules.EMAIL_RE.search(str(value)))


def is_numeric_identifier(value: Any) -> bool:
    return bool(value) and len(re.sub(r"\D", "", str(value))) >= 5


def is_probable_bank_card(value: Any) -> bool:
    return bool(value) and 13 <= len(re.sub(r"\D", "", str(value))) <= 19


def is_probable_api_key(value: Any) -> bool:
    return bool(value) and any(p.search(str(value)) for p in rules.SECRET_PATTERNS)


def detect_field_type(label: Any) -> Optional[str]:
    kind = rules.classify_label(str(label or ""))
    return None if kind in (None, "NEG") else kind


# ============================================================
# GLiNER (loaded once, warmed up at server start)
# ============================================================

def get_gliner_model():
    """Load GLiNER once and reuse it. Returns None when unavailable."""
    global _GLINER_MODEL, _GLINER_FAILED

    if not USE_GLINER or _GLINER_FAILED:
        return None
    if _GLINER_MODEL is not None:
        return _GLINER_MODEL

    with _GLINER_LOAD_LOCK:
        if _GLINER_MODEL is not None:
            return _GLINER_MODEL
        try:
            from gliner import GLiNER
        except ImportError:
            print("WARNING: GLiNER is not installed; names/addresses use rules only.")
            _GLINER_FAILED = True
            return None
        try:
            print(f"Loading GLiNER model: {MODEL_NAME}")
            _GLINER_MODEL = GLiNER.from_pretrained(MODEL_NAME)
            print("GLiNER model loaded successfully.")
        except Exception as exc:  # network / cache problems
            print(f"WARNING: Could not load GLiNER model: {exc}")
            _GLINER_FAILED = True
            _GLINER_MODEL = None
    return _GLINER_MODEL


def warm_up() -> None:
    """Load the model and run one tiny prediction so the first request is fast."""
    model = get_gliner_model()
    if model is None:
        return
    try:
        model.predict_entities("Name: Anish Kandel, Busan", AI_LABELS, threshold=THRESHOLD)
    except Exception as exc:
        print(f"GLiNER warm-up warning: {exc}")


# ============================================================
# INPUT NORMALISATION
# ============================================================

def _normalize_items(ocr_results: Iterable[Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for entry in ocr_results or []:
        if isinstance(entry, str):
            entry = {"text": entry, "confidence": 0.9, "box": None}
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        try:
            confidence = float(entry.get("confidence", 0.9))
        except (TypeError, ValueError):
            confidence = 0.9
        items.append({
            "text": text,
            "confidence": max(0.0, min(1.0, confidence)),
            "box": entry.get("box", entry.get("bbox")),
            "native": bool(entry.get("native", False)),
        })
    return items


def _box_tuple(box: Any) -> Optional[tuple]:
    if isinstance(box, (list, tuple)) and box:
        try:
            return tuple(int(v) for v in box[:4])
        except (TypeError, ValueError):
            return None
    return None


# ============================================================
# WHICH LINES NEED THE MODEL?
# ============================================================

def _ner_candidates(
    items: List[Dict[str, Any]],
    rule_detections: List[Dict[str, Any]],
) -> List[int]:
    """Indexes of OCR lines worth sending to GLiNER."""
    by_box: Dict[tuple, List[Dict[str, Any]]] = {}
    for det in rule_detections:
        key = _box_tuple(det.get("box"))
        if key is not None:
            by_box.setdefault(key, []).append(det)

    candidates: List[int] = []
    for index, item in enumerate(items):
        text = item["text"]
        letters = sum(ch.isalpha() for ch in text)

        if letters < 3 or "@" in text or is_username_field(text):
            continue

        # "Label: value" where the label is known -> rules already handled it
        split = rules._split_label_value(text)
        if split and rules.classify_label(split[0]) is not None:
            continue
        if rules._label_only(text) or rules.classify_label(text) is not None:
            continue

        # a rule detection already explains most of this line
        key = _box_tuple(item.get("box"))
        if key:
            explained = sum(len(d["value"]) for d in by_box.get(key, []))
        else:  # plain text / PDF text: no boxes, compare by value
            explained = sum(
                len(d["value"]) for d in rule_detections
                if d.get("value") and d["value"] in text
            )
        if explained >= 0.5 * len(text):
            continue

        # long sentences are unlikely to be a bare name / address field
        if len(text) > 90:
            continue

        candidates.append(index)
    return candidates


# ============================================================
# NER RESULT VALIDATION
# ============================================================

def _valid_name(value: str) -> bool:
    if not rules._is_person_like(value):
        return False
    words = re.sub(r"\([^)]*\)", " ", value).split()
    lowered = {w.lower().strip(".,") for w in words}
    if lowered & ORG_WORDS:
        return False
    if rules.classify_label(value) is not None:
        return False
    for word in words:
        if re.search(r"[a-z][A-Z]", word) and not word.startswith(("Mc", "Mac")):
            return False  # PriviScan, OpenCV ... are product names, not people
    latin = [w for w in words if w[0].isascii() and w[0].isalpha()]
    if latin and not all(w[0].isupper() for w in latin):
        return False
    if all(len(w.strip(".")) <= 1 for w in words):
        return False
    return True


def _valid_address(value: str) -> bool:
    if len(value) < 6 or "@" in value or not any(ch.isalpha() for ch in value):
        return False
    words = {w.lower().strip(".,") for w in re.split(r"[\s,]+", value)}
    return (
        any(ch.isdigit() for ch in value)
        or "," in value
        or bool(words & STREET_WORDS)
    )


def _validate_entity(detection_type: str, value: str) -> bool:
    if detection_type == "NAME":
        return _valid_name(value)
    if detection_type == "ADDRESS":
        return _valid_address(value)
    if detection_type == "EMAIL":
        return rules.EMAIL_RE.fullmatch(value.strip()) is not None
    if detection_type == "PHONE":
        return 7 <= len(re.sub(r"\D", "", value)) <= 15
    if detection_type in {"ID", "ACCOUNT_NUMBER", "GOVERNMENT_ID"}:
        return len(re.sub(r"\D", "", value)) >= 5
    if detection_type == "BANK_CARD":
        return is_probable_bank_card(value)
    if detection_type in {"PASSWORD", "API_KEY", "SECRET"}:
        return len(value) >= 6
    return False


def _predict_batch(model, texts: List[str]) -> List[List[Dict[str, Any]]]:
    """One model call per batch (falls back to per-line calls if unsupported)."""
    results: List[List[Dict[str, Any]]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        chunk = texts[start:start + BATCH_SIZE]
        try:
            results.extend(model.batch_predict_entities(chunk, AI_LABELS, threshold=THRESHOLD))
        except Exception:
            for text in chunk:
                try:
                    results.append(model.predict_entities(text, AI_LABELS, threshold=THRESHOLD))
                except Exception as exc:
                    print(f"WARNING: GLiNER prediction failed for '{text}': {exc}")
                    results.append([])
    return results


def detect_with_gliner(
    ocr_results: Iterable[Any],
    rule_detections: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    items = _normalize_items(ocr_results)
    model = get_gliner_model()
    if model is None or not items:
        return []

    indexes = _ner_candidates(items, rule_detections or [])
    if not indexes:
        return []  # nothing for the model to do -> big time saving

    texts = [items[i]["text"] for i in indexes]
    predictions = _predict_batch(model, texts)

    detections: List[Dict[str, Any]] = []
    for index, entities in zip(indexes, predictions):
        item = items[index]
        for entity in entities or []:
            detection_type = LABEL_MAPPING.get(normalize_text(entity.get("label")))
            value = str(entity.get("text", "")).strip()
            if not detection_type or not value:
                continue
            if not _validate_entity(detection_type, value):
                continue

            score = float(entity.get("score", 0.0))
            if item.get("native"):
                confidence = max(0.0, min(0.99, score))
            else:
                confidence = max(0.0, min(0.99, (score + item["confidence"]) / 2))
            detections.append({
                "type": detection_type,
                "value": value,
                "confidence": round(confidence, 4),
                "box": item["box"],
                "source": "gliner",
            })
    return detections


# ============================================================
# MERGE / CLEAN / VALIDATE
# ============================================================

def deduplicate_detections(detections):
    return rules.deduplicate_detections(detections)


def clean_conflicting_detections(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    email_locals = {
        normalize_text(d["value"]).split("@", 1)[0]
        for d in detections if d.get("type") == "EMAIL" and "@" in d.get("value", "")
    }
    secret_boxes = {
        _box_tuple(d.get("box")) for d in detections
        if d.get("type") in {"PASSWORD", "API_KEY", "SECRET"} and _box_tuple(d.get("box"))
    }
    strong_boxes = {
        _box_tuple(d.get("box")) for d in detections
        if d.get("type") in {"EMAIL", "PHONE", "ID", "ACCOUNT_NUMBER", "BANK_CARD",
                             "GOVERNMENT_ID", "ADDRESS", "DATE_OF_BIRTH"}
        and _box_tuple(d.get("box"))
    }

    cleaned = []
    for det in detections:
        if det.get("type") == "NAME":
            value = det.get("value", "")
            box = _box_tuple(det.get("box"))
            if "@" in value or rules.classify_label(value) is not None:
                continue
            if normalize_text(value) in email_locals:
                continue
            if box and (box in secret_boxes or box in strong_boxes) and det.get("source") == "gliner":
                continue
        cleaned.append(det)
    return cleaned


def final_validate_detections(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    validated = []
    for det in detections:
        kind = det.get("type")
        value = str(det.get("value", "")).strip()
        if kind not in SUPPORTED_TYPES or not value:
            continue
        if float(det.get("confidence", 0.0)) < 0.45:
            continue
        digits = re.sub(r"\D", "", value)
        if kind == "EMAIL" and not is_valid_email(value):
            continue
        if kind == "PHONE" and not 7 <= len(digits) <= 15:
            continue
        if kind == "BANK_CARD" and not 13 <= len(digits) <= 19:
            continue
        if kind in {"ACCOUNT_NUMBER", "ID"} and len(digits) < 3:
            continue
        validated.append(det)
    return validated


def _sort_key(det: Dict[str, Any]):
    box = det.get("box")
    if isinstance(box, (list, tuple)) and len(box) >= 2 and det.get("box_kind") != "offset":
        return (box[1], box[0])
    return (0, 0)


# ============================================================
# MAIN ENTRY POINTS
# ============================================================

def detect_from_ocr(ocr_results) -> List[Dict[str, Any]]:
    items = _normalize_items(ocr_results)
    if not items:
        return []

    # 1) fast, deterministic rules (labeled, label-on-previous-line, value-first)
    try:
        rule_detections = rules.detect_from_ocr_results(items)
    except Exception:
        print("ERROR in rule detection (continuing with AI only):")
        traceback.print_exc()
        rule_detections = []

    # 2) model only where rules could not explain the line
    try:
        ai_detections = detect_with_gliner(items, rule_detections)
    except Exception:
        print("ERROR in GLiNER detection (continuing with rules only):")
        traceback.print_exc()
        ai_detections = []

    # 3) merge
    try:
        detections = rules.deduplicate_detections(rule_detections + ai_detections)
        detections = clean_conflicting_detections(detections)
        detections = final_validate_detections(detections)
        detections.sort(key=_sort_key)
    except Exception:
        print("ERROR while merging detections (returning unmerged results):")
        traceback.print_exc()
        detections = rule_detections + ai_detections

    if DEBUG:
        print(
            f"[PriviScan] OCR lines={len(items)} rules={len(rule_detections)} "
            f"gliner={len(ai_detections)} final={len(detections)}"
        )
    return detections


def detect_text_with_ai(text: str) -> List[Dict[str, Any]]:
    """
    Rules + GLiNER for plain text (selectable PDF text, TXT, DOCX ...).
    Use this instead of rules.detect_text() so unlabeled names/addresses
    are found in documents too, not only in images.
    """
    text = text or ""
    try:
        rule_detections = rules.detect_text(text)
    except Exception:
        print("ERROR in rule detection of text:")
        traceback.print_exc()
        rule_detections = []

    try:
        lines = [
            {"text": line.strip(), "confidence": 1.0, "box": None, "native": True}
            for line in text.splitlines()
            if line.strip()
        ]
        ai_detections = detect_with_gliner(lines, rule_detections)
    except Exception:
        print("ERROR in GLiNER detection of text:")
        traceback.print_exc()
        ai_detections = []

    detections = rules.deduplicate_detections(rule_detections + ai_detections)
    detections = clean_conflicting_detections(detections)
    detections = final_validate_detections(detections)

    if DEBUG:
        print(
            f"[PriviScan] text lines rules={len(rule_detections)} "
            f"gliner={len(ai_detections)} final={len(detections)}"
        )
    return detections


def detect_with_ai(input_data):
    """Accepts OCR results (list) or an image path."""
    if isinstance(input_data, list):
        return detect_from_ocr(input_data)

    if isinstance(input_data, (str, Path)):
        from backend.ocr.detector import extract_text  # lazy: avoids loading OCR at import
        return detect_from_ocr(extract_text(str(input_data)))

    raise TypeError("detect_with_ai() expects an image path or OCR result list.")


# ============================================================
# CLI
# ============================================================

def print_sensitive_data(detections: List[Dict[str, Any]]) -> None:
    if not detections:
        print("No sensitive information detected.")
        return
    print(f"\nDetected {len(detections)} sensitive item(s)\n")
    for index, det in enumerate(detections, start=1):
        print(f"[{index}] {det.get('type')}")
        print(f"Value: {det.get('value')}")
        print(f"Confidence: {det.get('confidence')}")
        print(f"Box: {det.get('box')}")
        print(f"Source: {det.get('source')}\n")


def main() -> None:
    import sys
    image_path = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset/mixed/info.png")
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return
    print(f"Analyzing image: {image_path}")
    print_sensitive_data(detect_with_ai(image_path))


if __name__ == "__main__":
    main()
