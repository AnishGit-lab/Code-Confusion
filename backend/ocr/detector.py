"""
PriviScan OCR layer (faster version).

OCR only extracts text, confidence and bounding box. Classification of the
text is done in backend/detection/.

Why this is faster than the old version
---------------------------------------
* PaddleOCR 3.x runs extra models by default (document orientation,
  document unwarping, text-line orientation). Screenshots and normal
  documents do not need them, so they are switched OFF.
* Large images are shrunk before OCR (longest side <= OCR_MAX_SIDE) and the
  boxes are scaled back to the ORIGINAL image size, so redaction still lines
  up exactly.
* The model is created once (thread-safe) and can be loaded when the server
  starts with warm_up(), so the first user request is not slow.
* OCR items are returned in reading order (row by row, left to right), so a
  label box and the value box next to it are neighbours in the list.

Environment variables
---------------------
PRIVISCAN_OCR_MAX_SIDE   default 1600 (raise to 2000+ if small text is missed)
PRIVISCAN_OCR_MKLDNN     "1" to enable oneDNN (faster, but crashes on some
                         PaddlePaddle builds - default "0")
PRIVISCAN_OCR_THREADS    CPU threads for inference (default: all cores, max 8)
"""

from __future__ import annotations

import os
import statistics
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

OCR_MAX_SIDE = int(os.environ.get("PRIVISCAN_OCR_MAX_SIDE", "1600"))
USE_MKLDNN = os.environ.get("PRIVISCAN_OCR_MKLDNN", "0") == "1"
CPU_THREADS = int(os.environ.get("PRIVISCAN_OCR_THREADS", str(min(os.cpu_count() or 4, 8))))

_OCR = None
_OCR_LOCK = Lock()


# ============================================================
# MODEL (created once)
# ============================================================

def _create_ocr():
    from paddleocr import PaddleOCR

    base = dict(lang="en", device="cpu", enable_mkldnn=USE_MKLDNN)
    fast = dict(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    attempts = [
        {**base, **fast, "cpu_threads": CPU_THREADS},  # all speed options
        {**base, **fast},                              # older 3.x without cpu_threads
        base,                                          # the original configuration
    ]

    last_error: Optional[Exception] = None
    for options in attempts:
        try:
            return PaddleOCR(**options)
        except Exception as exc:  # unknown argument for this PaddleOCR version
            last_error = exc
    raise RuntimeError(f"Could not create PaddleOCR: {last_error}")


def get_ocr():
    global _OCR
    if _OCR is not None:
        return _OCR
    with _OCR_LOCK:
        if _OCR is None:
            _OCR = _create_ocr()
    return _OCR


def warm_up() -> None:
    """Load the model and run it once on a tiny blank image."""
    try:
        import numpy as np

        blank = np.full((64, 256, 3), 255, dtype="uint8")
        list(get_ocr().predict(blank))
    except Exception as exc:
        print(f"OCR warm-up warning: {exc}")


# ============================================================
# HELPERS
# ============================================================

def convert_box(box: Any) -> List[int]:
    """PaddleOCR box -> [left, top, right, bottom] as plain ints."""
    if hasattr(box, "tolist"):
        box = box.tolist()
    if not isinstance(box, (list, tuple)):
        return []
    result = []
    for value in box:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _load_image(image_path: str):
    """Read an image (works with non-ASCII Windows paths). Returns None on failure."""
    try:
        import cv2
        import numpy as np

        data = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


def _shrink(image) -> Tuple[Any, float]:
    """Shrink so the longest side is <= OCR_MAX_SIDE. Returns (image, scale<=1)."""
    import cv2

    height, width = image.shape[:2]
    longest = max(height, width)
    if OCR_MAX_SIDE <= 0 or longest <= OCR_MAX_SIDE:
        return image, 1.0
    scale = OCR_MAX_SIDE / float(longest)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _reading_order(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group boxes into rows (similar vertical centre), rows top->bottom, left->right."""
    usable = [i for i in items if len(i.get("box") or []) >= 4]
    if len(usable) != len(items) or not items:
        return items

    heights = [max(1, i["box"][3] - i["box"][1]) for i in items]
    row_gap = 0.6 * statistics.median(heights)

    ordered = sorted(items, key=lambda i: ((i["box"][1] + i["box"][3]) / 2, i["box"][0]))
    rows: List[List[Dict[str, Any]]] = []
    for item in ordered:
        centre = (item["box"][1] + item["box"][3]) / 2
        if rows:
            last = rows[-1]
            last_centre = sum((x["box"][1] + x["box"][3]) / 2 for x in last) / len(last)
            if abs(centre - last_centre) <= row_gap:
                last.append(item)
                continue
        rows.append([item])

    result: List[Dict[str, Any]] = []
    for row in rows:
        result.extend(sorted(row, key=lambda i: i["box"][0]))
    return result


# ============================================================
# OCR EXTRACTION
# ============================================================

def extract_text(image_path: str) -> List[Dict[str, Any]]:
    """
    Extract text, confidence and bounding boxes from an image.

    Returns:
        [{"text": "Email: test@example.com", "confidence": 0.998,
          "box": [left, top, right, bottom]}]   # coordinates of the ORIGINAL image
    """
    image_path = str(image_path)
    ocr = get_ocr()

    source: Any = image_path
    scale = 1.0
    image = _load_image(image_path)
    if image is not None:
        source, scale = _shrink(image)

    extracted: List[Dict[str, Any]] = []

    for result in ocr.predict(source):
        try:
            data = result.json
        except Exception:
            continue

        if isinstance(data, dict) and "res" in data:
            data = data["res"]
        if not isinstance(data, dict):
            continue

        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])
        boxes = data.get("rec_boxes", [])

        for index, text in enumerate(texts):
            text = str(text).strip()
            if not text:
                continue

            try:
                confidence = float(scores[index]) if index < len(scores) else 0.0
            except (TypeError, ValueError):
                confidence = 0.0

            box = convert_box(boxes[index]) if index < len(boxes) else []
            if box and scale != 1.0:
                box = [int(round(value / scale)) for value in box]

            extracted.append({
                "text": text,
                "confidence": round(confidence, 4),
                "box": box,
            })

    return _reading_order(extracted)


# ============================================================
# OCR TEXT ONLY / DEBUG
# ============================================================

def get_full_text(ocr_results: List[Dict[str, Any]]) -> str:
    return "\n".join(item["text"] for item in ocr_results if item.get("text"))


def print_ocr_results(ocr_results: List[Dict[str, Any]]) -> None:
    if not ocr_results:
        print("No text detected.")
        return
    print("\n===== OCR RESULTS =====")
    for index, item in enumerate(ocr_results, start=1):
        print(f"\n[{index}]")
        print(f"Text: {item['text']}")
        print(f"Confidence: {item['confidence']}")
        print(f"Box: {item['box']}")
    print("\n=======================\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("python -m backend.ocr.detector <image_path>")
        sys.exit(1)

    print_ocr_results(extract_text(sys.argv[1]))
