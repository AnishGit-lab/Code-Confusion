from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List

from backend.ocr.detector import extract_text
from backend.detection.ai_detector import detect_with_ai
from backend.detection.sensitive import normalize_detections
from backend.risk.scoring import score_detections

try:
    from backend.vision import detect_visual
except Exception:
    detect_visual = None


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if result == result else default
    except (TypeError, ValueError, OverflowError):
        return default


def _risk_level(score: float) -> str:
    """Convert a 0-100 risk score into a PriviScan risk level."""
    score = max(0.0, min(100.0, score))

    if score <= 20:
        return "LOW"
    if score <= 40:
        return "MEDIUM"
    if score <= 70:
        return "HIGH"
    return "CRITICAL"


def _normalize_ocr_result(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        text = item.get("text", "")
        confidence = item.get(
            "confidence",
            item.get("score", 0.0),
        )

        return {
            **item,
            "text": str(text),
            "confidence": _safe_float(confidence),
            "box": item.get("box", item.get("bbox")),
        }

    return {
        "text": str(item),
        "confidence": 0.0,
        "box": None,
    }


def _normalize_visual_result(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {
            "type": "UNKNOWN",
            "value": str(item),
            "confidence": 0.0,
            "box": None,
            "source": "vision",
        }

    return {
        **item,
        "type": str(
            item.get("type", item.get("label", "UNKNOWN"))
        ).upper(),
        "value": str(
            item.get("value", item.get("text", ""))
        ),
        "confidence": _safe_float(
            item.get("confidence", item.get("score", 0.0))
        ),
        "box": item.get("box", item.get("bbox")),
        "source": item.get("source", "vision"),
    }


def _detection_key(item: Dict[str, Any]) -> tuple:
    return (
        str(item.get("type", "")).upper(),
        str(item.get("value", "")).strip().lower(),
    )


def merge_detections(
    *detection_lists: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge duplicate detections while retaining the best confidence."""
    merged: Dict[tuple, Dict[str, Any]] = {}

    for detection_list in detection_lists:
        if detection_list is None:
            continue

        for item in normalize_detections(detection_list):
            if not isinstance(item, dict):
                continue

            key = _detection_key(item)

            if key not in merged:
                merged[key] = item
                continue

            current = merged[key]

            if _safe_float(item.get("confidence")) > _safe_float(
                current.get("confidence")
            ):
                replacement = dict(item)

                if not replacement.get("box"):
                    replacement["box"] = current.get("box")

                merged[key] = replacement
            elif not current.get("box") and item.get("box"):
                current["box"] = item["box"]

    return list(merged.values())


def _score_detections_compat(
    detections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Normalize score_detections() output.

    Supports either:
      1. A dictionary containing detections and document risk metrics.
      2. A list of scored detection dictionaries.

    This prevents AttributeError when the scoring function returns a list.
    """
    raw_result = score_detections(detections)

    if isinstance(raw_result, dict):
        scored_items = raw_result.get(
            "detections",
            raw_result.get("items", detections),
        )

        if not isinstance(scored_items, list):
            scored_items = detections

        scored_items = [
            dict(item)
            for item in scored_items
            if isinstance(item, dict)
        ]

        scores = [
            _safe_float(item.get("risk_score"))
            for item in scored_items
        ]

        calculated_average = (
            sum(scores) / len(scores) if scores else 0.0
        )

        calculated_final = min(100.0, max(scores)) if scores else 0.0

        average_value = raw_result.get("average_risk_score")
        final_value = raw_result.get(
            "final_risk_score",
            raw_result.get("risk_score"),
        )

        average_score = (
            _safe_float(average_value)
            if average_value is not None
            else calculated_average
        )

        final_score = (
            _safe_float(final_value)
            if final_value is not None
            else calculated_final
        )

        level = raw_result.get("risk_level") or _risk_level(final_score)

        return {
            "detections": scored_items,
            "total_detections": len(scored_items),
            "average_risk_score": round(average_score, 2),
            "final_risk_score": round(final_score, 2),
            "risk_score": round(final_score, 2),
            "risk_level": str(level).upper(),
        }

    if isinstance(raw_result, list):
        scored_items = [
            dict(item)
            for item in raw_result
            if isinstance(item, dict)
        ]
    else:
        # Preserve the original detections if the scorer returns
        # an unexpected type instead of losing detected information.
        scored_items = [
            dict(item)
            for item in detections
            if isinstance(item, dict)
        ]

    scores = [
        _safe_float(item.get("risk_score"))
        for item in scored_items
    ]

    average_score = sum(scores) / len(scores) if scores else 0.0
    final_score = min(100.0, max(scores)) if scores else 0.0

    return {
        "detections": scored_items,
        "total_detections": len(scored_items),
        "average_risk_score": round(average_score, 2),
        "final_risk_score": round(final_score, 2),
        "risk_score": round(final_score, 2),
        "risk_level": _risk_level(final_score),
    }


# ============================================================
# OCR, AI, AND VISUAL DETECTION
# ============================================================

def _run_ocr(image_path: str) -> List[Dict[str, Any]]:
    result = extract_text(image_path)

    if result is None:
        return []

    if isinstance(result, dict):
        result = result.get(
            "results",
            result.get("ocr_results", []),
        )

    if not isinstance(result, list):
        return []

    return [_normalize_ocr_result(item) for item in result]


def _run_ai(
    ocr_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not ocr_results:
        return []

    try:
        result = detect_with_ai(ocr_results)

        if result is None:
            return []

        return normalize_detections(result)

    except TypeError:
        try:
            texts = [
                item.get("text", "")
                for item in ocr_results
            ]

            result = detect_with_ai(texts)

            return normalize_detections(result)

        except Exception as exc:
            print(f"AI detection warning: {exc}")
            return []

    except Exception as exc:
        print(f"AI detection warning: {exc}")
        return []


def _run_visual(image_path: str) -> List[Dict[str, Any]]:
    if detect_visual is None:
        return []

    try:
        result = detect_visual(image_path)

        if result is None:
            return []

        if isinstance(result, dict):
            result = result.get(
                "detections",
                result.get("results", []),
            )

        if not isinstance(result, list):
            return []

        return [_normalize_visual_result(item) for item in result]

    except Exception as exc:
        print(f"Vision detection warning: {exc}")
        return []


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(
    image_path: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    image_path = str(image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if verbose:
        print("[1/5] Running OCR...")

    ocr_results = _run_ocr(image_path)

    if verbose:
        print(f"OCR detected {len(ocr_results)} text regions.")
        print("[2/5] Running AI text detection...")

    text_detections = _run_ai(ocr_results)

    if verbose:
        print(
            f"AI detected {len(text_detections)} sensitive text items."
        )
        print("[3/5] Running visual detection...")

    visual_detections = _run_visual(image_path)

    if verbose:
        print(
            f"Vision detected {len(visual_detections)} visual items."
        )
        print("[4/5] Merging detection results...")

    merged = merge_detections(
        text_detections,
        visual_detections,
    )

    if verbose:
        print("[5/5] Calculating risk...")

    scored = _score_detections_compat(merged)

    if verbose:
        print(f"Detections : {scored['total_detections']}")
        print(f"Risk Score : {scored['final_risk_score']:.2f}")
        print(f"Risk Level : {scored['risk_level']}")

    return {
        "image_path": image_path,
        "ocr_results": ocr_results,
        "text_detections": text_detections,
        "visual_detections": visual_detections,
        **scored,
    }


# ============================================================
# TEXT ANALYSIS
# ============================================================

def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyze extracted text using PriviScan's existing detector
    and risk-scoring pipeline.
    """
    from backend.detection.sensitive import detect_text

    detections = detect_text(text)

    if not isinstance(detections, list):
        detections = []

    scored = _score_detections_compat(detections)

    return {
        "text": text,
        **scored,
    }


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PriviScan image analysis"
    )

    parser.add_argument(
        "image",
        help="Path to image",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output",
    )

    args = parser.parse_args()

    result = analyze_image(
        args.image,
        verbose=not args.json,
    )

    if args.json:
        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
