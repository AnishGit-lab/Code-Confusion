
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import fitz  # PyMuPDF

from backend.ocr.detector import extract_text
from backend.detection.ai_detector import detect_with_ai
from backend.detection.sensitive import normalize_detections
from backend.risk.scoring import score_detections, calculate_document_risk
from backend.detection.sensitive import detect_text, deduplicate_detections
from backend.detection.context import analyze_context

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
# OPTIMIZED PDF ANALYSIS (HYBRID TEXT + SCANNED PAGES)
# ============================================================

def _pdf_rect_to_list(rect: Any) -> List[float]:
    return [round(float(rect.x0), 2), round(float(rect.y0), 2),
            round(float(rect.x1), 2), round(float(rect.y1), 2)]


def _pdf_image_coverage(page: Any) -> float:
    """Estimate how much of a page is covered by raster images."""
    page_area = max(float(page.rect.width * page.rect.height), 1.0)
    total_area = 0.0
    try:
        for image in page.get_images(full=True):
            xref = image[0]
            for rect in page.get_image_rects(xref):
                total_area += max(0.0, float(rect.width * rect.height))
        return min(1.0, total_area / page_area)
    except Exception:
        return 0.0


def _pdf_page_needs_ocr(page: Any, text: str) -> bool:
    """Use OCR only on pages likely to contain scanned or image-based text."""
    cleaned = (text or '').strip()
    word_count = len(cleaned.split())
    if len(cleaned) < 60 or word_count < 10:
        return True

    # A mixed page can have some selectable text plus a large scanned image.
    # OCR those pages only when native text is still relatively sparse.
    coverage = _pdf_image_coverage(page)
    return coverage >= 0.50 and len(cleaned) < 500 and word_count < 80


def _ocr_pdf_page(page: Any, page_number: int, image_dir: Path,
                  scale: float = 1.5) -> List[Dict[str, Any]]:
    """Render one scanned page at a modest resolution and OCR it once."""
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image_path = image_dir / f'priviscan_page_{page_number}.png'
    pixmap.save(str(image_path))

    ocr_items = _run_ocr(str(image_path))
    converted: List[Dict[str, Any]] = []
    for item in ocr_items:
        text = str(item.get('text', '')).strip()
        if not text:
            continue
        raw_box = item.get('box') or item.get('bbox')
        box = None
        if isinstance(raw_box, (list, tuple)) and len(raw_box) >= 4:
            try:
                # PaddleOCR often returns a 4-point polygon [[x,y], ...].
                # Convert either polygons or [x1,y1,x2,y2] into PDF points.
                if (
                    isinstance(raw_box[0], (list, tuple))
                    and len(raw_box[0]) >= 2
                ):
                    xs = [float(point[0]) for point in raw_box]
                    ys = [float(point[1]) for point in raw_box]
                    box = [
                        round(min(xs) / scale, 2),
                        round(min(ys) / scale, 2),
                        round(max(xs) / scale, 2),
                        round(max(ys) / scale, 2),
                    ]
                else:
                    x1, y1, x2, y2 = [float(value) for value in raw_box[:4]]
                    box = [
                        round(min(x1, x2) / scale, 2),
                        round(min(y1, y2) / scale, 2),
                        round(max(x1, x2) / scale, 2),
                        round(max(y1, y2) / scale, 2),
                    ]
            except (TypeError, ValueError, IndexError):
                box = None
        converted.append({
            'text': text,
            'confidence': _safe_float(item.get('confidence'), 0.8),
            'box': box,
            'page': page_number,
            'source': 'ocr',
        })
    return converted


def _boxes_for_pdf_text(page: Any, value: str) -> List[List[float]]:
    """Return every exact selectable-text occurrence as PDF-point rectangles."""
    value = str(value or '').strip()
    if not value:
        return []
    try:
        return [_pdf_rect_to_list(rect) for rect in page.search_for(value)]
    except Exception:
        return []


def _box_for_pdf_text(page: Any, value: str) -> List[float] | None:
    """Compatibility helper: return the first PDF-point rectangle, if found."""
    boxes = _boxes_for_pdf_text(page, value)
    return boxes[0] if boxes else None


def _deduplicate_pdf_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate values per page while preserving distinct page occurrences."""
    output: Dict[tuple, Dict[str, Any]] = {}
    for item in items:
        key = (str(item.get('type', '')).upper(),
               str(item.get('value', '')).strip().casefold(),
               int(item.get('page', 1) or 1))
        current = output.get(key)
        if current is None:
            output[key] = dict(item)
            continue
        if _safe_float(item.get('confidence')) > _safe_float(current.get('confidence')):
            replacement = dict(item)
            if not replacement.get('box'):
                replacement['box'] = current.get('box')
            if not replacement.get('boxes'):
                replacement['boxes'] = current.get('boxes', [])
            output[key] = replacement
        else:
            # Merge locations for repeated values on the same page.
            current_boxes = list(current.get('boxes') or [])
            for candidate_box in (item.get('boxes') or []):
                if candidate_box not in current_boxes:
                    current_boxes.append(candidate_box)
            if current_boxes:
                current['boxes'] = current_boxes
                if not current.get('box'):
                    current['box'] = current_boxes[0]
            elif not current.get('box') and item.get('box'):
                current['box'] = item['box']
    return list(output.values())


def analyze_pdf(pdf_path: str | Path, verbose: bool = False,
                render_scale: float = 1.5) -> Dict[str, Any]:
    """Analyze a mixed PDF quickly: extract selectable text first and OCR only
    pages that look scanned. Standalone image analysis remains unchanged.

    render_scale=1.5 is a speed/legibility compromise. Use 2.0 if small text is
    being missed, at the cost of slower processing.
    """
    path = Path(pdf_path)
    if path.suffix.lower() != '.pdf':
        raise ValueError('Expected a PDF file.')
    if not path.is_file():
        raise FileNotFoundError(f'PDF not found: {path}')
    if render_scale < 1.0 or render_scale > 3.0:
        raise ValueError('render_scale must be between 1.0 and 3.0.')

    start_time = time.perf_counter()
    all_text: List[str] = []
    detections: List[Dict[str, Any]] = []
    ocr_page_count = 0
    text_page_count = 0

    try:
        document = fitz.open(str(path))
    except Exception as exc:
        raise ValueError(f'Could not open PDF: {exc}') from exc

    try:
        if document.is_encrypted and not document.authenticate(''):
            raise ValueError('This PDF is password-protected. Upload an unlocked PDF.')

        page_count = len(document)
        with tempfile.TemporaryDirectory(prefix='priviscan_fast_pdf_') as temp_dir:
            image_dir = Path(temp_dir)
            total_pages = page_count
            if verbose:
                print(f'Analyzing {path.name}: {total_pages} pages')

            for page_index, page in enumerate(document):
                page_number = page_index + 1
                page_start = time.perf_counter()
                native_text = (page.get_text('text') or '').strip()
                page_text_parts: List[str] = []

                # Always inspect selectable text; do not rasterize text pages.
                if native_text:
                    page_text_parts.append(native_text)
                    for detection in detect_text(native_text):
                        item = dict(detection)
                        item['page'] = page_number
                        item['source'] = 'pdf_text'

                        # IMPORTANT: detect_text() may supply a regex character-offset
                        # box (e.g. [34, 0, 58, 1]), not a PDF coordinate rectangle.
                        # Always replace it with real page coordinates from PyMuPDF.
                        matched_boxes = _boxes_for_pdf_text(
                            page, item.get('value', '')
                        )
                        item['boxes'] = matched_boxes
                        item['box'] = matched_boxes[0] if matched_boxes else None
                        detections.append(item)

                needs_ocr = _pdf_page_needs_ocr(page, native_text)
                if needs_ocr:
                    ocr_page_count += 1
                    ocr_items = _ocr_pdf_page(page, page_number, image_dir, render_scale)
                    ocr_text = '\n'.join(item['text'] for item in ocr_items)
                    if ocr_text:
                        page_text_parts.append(ocr_text)

                    # Detect line-level values first so each hit can retain its OCR box.
                    for ocr_item in ocr_items:
                        for detection in detect_text(ocr_item['text'], ocr_item.get('box')):
                            item = dict(detection)
                            item['page'] = page_number
                            item['source'] = 'ocr'
                            item['confidence'] = min(
                                _safe_float(item.get('confidence'), 0.9),
                                _safe_float(ocr_item.get('confidence'), 0.8),
                            )
                            detections.append(item)

                    # Catch values that span OCR lines; only attach a box when matched.
                    if ocr_text:
                        for detection in detect_text(ocr_text):
                            item = dict(detection)
                            value = str(item.get('value', '')).strip().casefold()
                            matching = next((line for line in ocr_items
                                             if value and value in line['text'].casefold()), None)
                            item['box'] = (matching.get('box') if matching else item.get('box'))
                            item['page'] = page_number
                            item['source'] = 'ocr_combined'
                            if matching:
                                item['confidence'] = min(
                                    _safe_float(item.get('confidence'), 0.9),
                                    _safe_float(matching.get('confidence'), 0.8),
                                )
                            detections.append(item)
                else:
                    text_page_count += 1

                all_text.append('\n'.join(page_text_parts))
                if verbose:
                    elapsed = time.perf_counter() - page_start
                    mode = 'OCR + text' if needs_ocr else 'selectable text'
                    print(f'Page {page_number}/{total_pages}: {mode}, {elapsed:.1f}s')
    finally:
        document.close()

    detections = _deduplicate_pdf_items(detections)
    context = analyze_context(
        [{'text': '\n'.join(all_text), 'confidence': 1.0, 'box': None}],
        detections,
    )
    scored_detections = score_detections(detections)
    risk = calculate_document_risk(detections)

    # The scorer may return a new list; attach stable UI indices and page numbers.
    if not isinstance(scored_detections, list):
        scored_detections = risk.get('detections', detections)
    for index, item in enumerate(scored_detections):
        item['index'] = index
        item['page'] = int(item.get('page', 1) or 1)
        box = item.get('box') or item.get('bbox')
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            try:
                item['box'] = [round(float(value), 2) for value in box[:4]]
            except (TypeError, ValueError):
                item['box'] = None
        else:
            item['box'] = None

    elapsed_total = time.perf_counter() - start_time
    if verbose:
        print(f'Finished in {elapsed_total:.1f}s; OCR pages: {ocr_page_count}; text pages: {text_page_count}')
        print(f'Detections: {len(scored_detections)}; document risk: {risk.get("risk_score", 0)} ({risk.get("risk_level", "LOW")})')

    return {
        'file_name': path.name,
        'file_type': 'PDF',
        'page_count': page_count,
        'text_source': {'ocr_pages': ocr_page_count, 'text_pages': text_page_count},
        'analysis_seconds': round(elapsed_total, 2),
        'sector': context.get('sector', 'UNKNOWN'),
        'sector_confidence': context.get('confidence', 0.0),
        'sector_matched_keywords': context.get('matched_keywords', []),
        'risk_score': risk.get('risk_score', 0.0),
        'risk_level': risk.get('risk_level', 'LOW'),
        'total_detections': len(scored_detections),
        'average_risk_score': risk.get('average_risk_score', 0.0),
        'aggregate_item_risk': risk.get('aggregate_item_risk', 0.0),
        'linkability_adjustment': risk.get('linkability_adjustment', 0.0),
        'credential_override': risk.get('credential_override', False),
        'detections': scored_detections,
    }


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PriviScan image and PDF analysis"
    )

    parser.add_argument(
        "file",
        help="Path to an image or PDF",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output",
    )

    args = parser.parse_args()

    if Path(args.file).suffix.lower() == '.pdf':
        result = analyze_pdf(args.file, verbose=not args.json)
    else:
        result = analyze_image(args.file, verbose=not args.json)

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
