"""Secure PDF redaction for PriviScan.

Uses PyMuPDF redaction annotations and apply_redactions(), so covered text is
REMOVED from the PDF content, not merely hidden behind a drawing.

Protection methods (the same five as image protection)
------------------------------------------------------
blur             the area is replaced by a heavily blurred picture of itself
blackout         black box
star             black box with "****" (same length as the original value)
hashtag          black box with "####"
random_alphabet  black box with random letters
(white           white box; kept for older callers, not offered in the web page)

For every method the original text / image pixels under the box are deleted
first. The blurred picture is created from a rendering taken BEFORE the
deletion and is pixelated and blurred so the original can not be read back.
"""
from __future__ import annotations

import random
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import fitz

SUPPORTED_PDF_METHODS = {
    "blur",
    "blackout",
    "star",
    "hashtag",
    "random_alphabet",
    "white",
}
TEXT_REPLACEMENT_METHODS = {"star", "hashtag", "random_alphabet"}

BLUR_RENDER_ZOOM = 3.0  # render the area at 216 dpi before blurring


# ============================================================
# BOXES
# ============================================================

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


# ============================================================
# REPLACEMENT TEXT (same rule as protection.py for images)
# ============================================================

def _replacement_text(method: str, original: str) -> str:
    length = max(3, len(original))

    if method == "star":
        return "*" * length
    if method == "hashtag":
        return "#" * length
    if method == "random_alphabet":
        return "".join(random.choice(string.ascii_letters) for _ in range(length))

    return "*" * length


def _fit_fontsize(text: str, width: float, height: float) -> Optional[float]:
    """Largest Helvetica size (<= 24) that fits the text inside the box, or None."""
    if width <= 2 or height <= 2 or not text:
        return None

    size = min(24.0, height / 1.3)          # a text box needs ~1.2-1.3 x font size of height
    unit_width = fitz.get_text_length(text, fontname="helv", fontsize=1.0)
    if unit_width > 0:
        size = min(size, (width - 2.0) / unit_width)

    return size if size >= 4.0 else None


# ============================================================
# BLUR
# ============================================================

def _blur_pixels(rgb):
    """Pixelate, then blur. Pixelating first destroys the letter shapes."""
    import cv2

    height, width = rgb.shape[:2]
    block = max(4, min(height, width) // 4)

    small = cv2.resize(
        rgb,
        (max(1, width // block), max(1, height // block)),
        interpolation=cv2.INTER_AREA,
    )
    mosaic = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)

    kernel = max(15, ((min(width, height) // 5) * 2) + 1)  # same rule as image blur
    if kernel % 2 == 0:
        kernel += 1

    return cv2.GaussianBlur(mosaic, (kernel, kernel), 0)


def _render_blurred_png(page: Any, rect: fitz.Rect) -> Optional[bytes]:
    """Picture of the area, blurred, as PNG bytes. Taken BEFORE redaction."""
    try:
        import cv2
        import numpy as np

        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(BLUR_RENDER_ZOOM, BLUR_RENDER_ZOOM),
            clip=rect,
            alpha=False,
            annots=False,  # do not draw pending redaction outlines into the picture
        )
        pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        if pixmap.n == 1:
            rgb = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)
        else:
            rgb = np.ascontiguousarray(pixels[:, :, :3])

        blurred = _blur_pixels(rgb)
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))
        return encoded.tobytes() if ok else None
    except Exception:
        return None  # the area stays a plain filled box (still secure)


# ============================================================
# MAIN
# ============================================================

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
    if method not in SUPPORTED_PDF_METHODS:
        raise ValueError(
            "method must be one of: " + ", ".join(sorted(SUPPORTED_PDF_METHODS)) + "."
        )

    items = [item for item in (detections or []) if isinstance(item, dict)]
    selected = {int(i) for i in (selected_indices or [])}

    if mode == "manual" and not selected:
        raise ValueError("Manual mode requires at least one selected detection index.")

    destination.parent.mkdir(parents=True, exist_ok=True)

    box_fill = (1, 1, 1) if method == "white" else (0, 0, 0)

    redactions_applied = 0
    detections_skipped = 0
    selected_detections = 0
    blur_jobs: Dict[int, List[Tuple[fitz.Rect, bytes]]] = {}

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

            value = str(detection.get("value", detection.get("text", "")))
            added_for_detection = 0

            for rect in boxes:
                # Slightly pad to cover glyph edges and anti-aliasing.
                rect = fitz.Rect(
                    rect.x0 - 1.0,
                    rect.y0 - 1.0,
                    rect.x1 + 1.0,
                    rect.y1 + 1.0,
                ) & page.rect

                if rect.is_empty:
                    continue

                if method == "blur":
                    picture = _render_blurred_png(page, rect)
                    if picture:
                        blur_jobs.setdefault(page_number - 1, []).append((rect, picture))
                    page.add_redact_annot(rect, fill=(1, 1, 1), cross_out=False)

                elif method in TEXT_REPLACEMENT_METHODS:
                    text = _replacement_text(method, value)
                    size = _fit_fontsize(text, rect.width, rect.height)
                    if size is None:  # box too small for readable text
                        page.add_redact_annot(rect, fill=(0, 0, 0), cross_out=False)
                    else:
                        page.add_redact_annot(
                            rect,
                            text=text,
                            fontname="helv",
                            fontsize=size,
                            align=fitz.TEXT_ALIGN_LEFT,
                            fill=(0, 0, 0),
                            text_color=(1, 1, 1),
                            cross_out=False,
                        )

                else:  # blackout / white
                    page.add_redact_annot(rect, fill=box_fill, cross_out=False)

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

        # The blurred pictures go on AFTER the originals have been deleted.
        for page_index, jobs in blur_jobs.items():
            page = document[page_index]
            for rect, picture in jobs:
                page.insert_image(rect, stream=picture, keep_proportion=False, overlay=True)

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