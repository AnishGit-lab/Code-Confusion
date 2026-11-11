"""
PriviScan diagnostic - finds WHICH stage is wrong.

Put this file in the project root (the folder that contains "backend") and run:

    python diagnose.py path/to/image.png
    python diagnose.py path/to/document.pdf

It prints what each stage produced (OCR -> rules -> GLiNER -> final) with
timings, and shows any hidden error. Copy the whole output when asking for help.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from importlib import metadata

os.environ.setdefault("PRIVISCAN_DEBUG", "1")


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except Exception:
        return "NOT INSTALLED"


def _show(detections, limit: int = 40) -> None:
    if not detections:
        print("   (none)")
        return
    for det in detections[:limit]:
        print(
            f"   {det.get('type'):<15} {str(det.get('value'))[:60]!r:<64} "
            f"conf={float(det.get('confidence', 0)):.2f} src={det.get('source')} box={det.get('box')}"
        )
    if len(detections) > limit:
        print(f"   ... {len(detections) - limit} more")


def _timed(label: str, function, *args):
    start = time.perf_counter()
    try:
        result = function(*args)
    except Exception:
        print(f"\n!! {label} FAILED:")
        traceback.print_exc()
        return None, time.perf_counter() - start
    elapsed = time.perf_counter() - start
    print(f"   -> {label}: {elapsed:.2f}s")
    return result, elapsed


def diagnose_image(path: str) -> None:
    from backend.ocr.detector import extract_text
    from backend.detection import sensitive as rules
    from backend.detection import ai_detector as ai

    print("\n[1] OCR")
    items, _ = _timed("OCR (first call also loads the model)", extract_text, path)
    if items is None:
        return
    if not items:
        print("   OCR returned NOTHING. Problem is in OCR (image, language, or model).")
        return

    print(f"   {len(items)} OCR items")
    for item in items[:60]:
        print(f"   {item['text'][:70]!r:<74} conf={item['confidence']:.2f} box={item['box']}")

    short = sum(1 for item in items if len(item["text"]) <= 2)
    if short >= max(5, len(items) // 2):
        print(
            f"\n   WARNING: {short} of {len(items)} OCR items are only 1-2 characters. "
            "The OCR is splitting text into single characters or reading it badly. "
            "Try:  set PRIVISCAN_OCR_MAX_SIDE=0  (no shrinking) and run again."
        )

    low = [item for item in items if item["confidence"] < 0.6]
    if low:
        print(f"   NOTE: {len(low)} OCR items have confidence below 0.60 (text may be misread).")

    print("\n[2] Rule detectors (labels, value patterns, unlabeled numbers)")
    rule_dets, _ = _timed("rules", rules.detect_from_ocr_results, items)
    _show(rule_dets or [])

    print("\n[3] GLiNER (names / addresses the rules could not explain)")
    ai_dets, _ = _timed("GLiNER", ai.detect_with_gliner, items, rule_dets or [])
    _show(ai_dets or [])

    print("\n[4] Final merged result")
    final, _ = _timed("merge", ai.detect_from_ocr, items)
    _show(final or [])

    print("\n[5] Full pipeline exactly as the web app runs it")
    from backend.analyze import analyze_image

    result, elapsed = _timed("analyze_image", analyze_image, path, False)
    if result:
        print(
            f"   detections={result.get('total_detections')} "
            f"risk={result.get('final_risk_score')} level={result.get('risk_level')} "
            f"visual={len(result.get('visual_detections', []))}"
        )
        print(f"   where the time went (seconds): {result.get('timings')}")


def diagnose_pdf(path: str) -> None:
    from backend.detection import sensitive as rules

    try:
        import fitz
    except ImportError:
        print("PyMuPDF (fitz) is not installed: pip install pymupdf")
        return

    document = fitz.open(path)
    print(f"\n[1] PDF has {len(document)} page(s)")
    for number, page in enumerate(document, start=1):
        text = (page.get_text("text") or "").strip()
        print(f"\n   Page {number}: {len(text)} characters of selectable text")
        print("   first 300 characters:", repr(text[:300]))
        if not text:
            print("   -> no selectable text: this page will go through OCR")
            continue
        dets = rules.detect_text(text)
        print(f"   rules only found {len(dets)}:")
        _show(dets)

        from backend.detection import ai_detector as ai
        both, _ = _timed("rules + GLiNER (what the web app uses for PDF text)", ai.detect_text_with_ai, text)
        print(f"   rules + GLiNER found {len(both or [])}:")
        _show(both or [])
    document.close()

    print("\n[2] Full pipeline exactly as the web app runs it")
    from backend.analyze import analyze_pdf

    result, _ = _timed("analyze_pdf", analyze_pdf, path, False)
    if result:
        print(
            f"   detections={result.get('total_detections')} "
            f"risk={result.get('risk_score')} level={result.get('risk_level')} "
            f"text_source={result.get('text_source')}"
        )
        _show(result.get("detections", []))


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print("Versions:")
    for package in ("paddleocr", "paddlepaddle", "gliner", "phonenumbers", "opencv-python", "pymupdf"):
        print(f"   {package:<14} {_version(package)}")
    print("Settings:", {k: v for k, v in os.environ.items() if k.startswith("PRIVISCAN_")})

    if path.lower().endswith(".pdf"):
        diagnose_pdf(path)
    else:
        diagnose_image(path)


if __name__ == "__main__":
    main()