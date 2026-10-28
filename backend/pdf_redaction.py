"""Secure PDF redaction for PriviScan.

Uses PyMuPDF redaction annotations and apply_redactions() so covered text is
removed from the PDF content, not merely covered by a black drawing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import fitz


def _rect_from_box(value: Any) -> Optional[fitz.Rect]:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        # One rectangle: [x0, y0, x1, y1]
        if len(value) == 4 and not any(
            isinstance(v, (list, tuple, dict)) for v in value
        ):
            rect = fitz.Rect(*(float(v) for v in value))
        # Polygon: [[x,y], [x,y], ...]
        elif len(value) >= 4 and isinstance(value[0], (list, tuple)):
            points = [(float(p[0]), float(p[1])) for p in value if len(p) >= 2]
            if len(points) < 3:
                return None
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
        else:
            return None
    except (TypeError, ValueError, IndexError):
        return None

    if rect.is_empty or rect.is_infinite:
        return None
    return rect


def _detection_boxes(detection: Dict[str, Any]) -> List[fitz.Rect]:
    """Prefer every real text occurrence; otherwise use the single OCR box."""
    raw_boxes = detection.get("boxes")
    output: List[fitz.Rect] = []

    if isinstance(raw_boxes, list):
        for raw in raw_boxes:
            rect = _rect_from_box(raw)
            if rect is not None and rect not in output:
                output.append(rect)

    if not output:
        rect = _rect_from_box(detection.get("box") or detection.get("bbox"))
        if rect is not None:
            output.append(rect)

    return output


def redact_pdf(
    source_path: str | Path,
    output_path: str | Path,
    detections: Iterable[Dict[str, Any]],
    mode: str = "automatic",
    selected_indices: Optional[Iterable[int]] = None,
    method: str = "blackout",
) -> Dict[str, Any]:
    """Redact all valid detections (automatic) or selected detection indices.

    Indices are zero-based, matching the UI list. If each detection includes an
    'index', that stable index is used; otherwise its list position is used.
    """
    source = Path(source_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Source PDF not found: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("Output PDF must be different from the source PDF.")

    mode = str(mode or "automatic").strip().lower()
    if mode not in {"automatic", "manual"}:
        raise ValueError("mode must be 'automatic' or 'manual'.")

    method = str(method or "blackout").strip().lower()
    if method not in {"blackout", "white"}:
        raise ValueError("method must be 'blackout' or 'white'.")

    items = [item for item in (detections or []) if isinstance(item, dict)]
    selected = {int(i) for i in (selected_indices or [])}
    if mode == "manual" and not selected:
        raise ValueError("Manual mode requires at least one selected detection index.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    fill = (0, 0, 0) if method == "blackout" else (1, 1, 1)
    redactions_applied = 0
    detections_skipped = 0
    selected_detections = 0

    document = fitz.open(str(source))
    try:
        if document.is_encrypted and not document.authenticate(""):
            raise ValueError("This PDF is password-protected. Upload an unlocked PDF.")

        # First add all annotations; only then apply them to each page.
        for position, detection in enumerate(items):
            detection_index = int(detection.get("index", position))
            if mode == "manual" and detection_index not in selected:
                continue

            selected_detections += 1
            try:
                page_number = int(detection.get("page", 1))
            except (TypeError, ValueError):
                detections_skipped += 1
                continue

            if page_number < 1 or page_number > len(document):
                detections_skipped += 1
                continue

            page = document[page_number - 1]
            boxes = _detection_boxes(detection)
            if not boxes:
                detections_skipped += 1
                continue

            added_for_detection = 0
            for rect in boxes:
                # Slightly pad to cover glyph edges and anti-aliasing.
                rect = fitz.Rect(
                    rect.x0 - 1.0, rect.y0 - 1.0,
                    rect.x1 + 1.0, rect.y1 + 1.0,
                ) & page.rect
                if rect.is_empty:
                    continue
                page.add_redact_annot(rect, fill=fill, cross_out=False)
                redactions_applied += 1
                added_for_detection += 1

            if added_for_detection == 0:
                detections_skipped += 1

        if redactions_applied == 0:
            raise ValueError(
                "No redactions were added. Detection boxes are missing or outside the page."
            )

        for page in document:
            page.apply_redactions()

        # garbage=4 removes unreachable objects; deflate/clean compact the result.
        document.save(
            str(destination),
            garbage=4,
            deflate=True,
            clean=True,
        )
    finally:
        document.close()

    return {
        "output_path": str(destination),
        "redactions_applied": redactions_applied,
        "selected_detections": selected_detections,
        "detections_skipped": detections_skipped,
        "mode": mode,
        "method": method,
        "secure_redaction": True,
    }
