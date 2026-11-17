r"""
PriviScan speed check - shows WHERE the time goes for one image.

Run from the project root (the folder that contains "backend"):

    python benchmark.py path\to\image.png
    python benchmark.py path\to\image.png 3        (3 runs; the first run also loads the models)

It prints the time of every stage and tells you which setting to change.
"""

from __future__ import annotations

import os
import sys
import time


def main() -> None:
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        print(__doc__)
        return

    path = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    print("Settings:", {k: v for k, v in os.environ.items() if k.startswith("PRIVISCAN_")} or "(defaults)")

    start = time.perf_counter()
    from backend.analyze import analyze_image
    from backend.ocr.detector import warm_up as warm_ocr
    from backend.detection.ai_detector import warm_up as warm_ai

    print(f"\nImporting the project: {time.perf_counter() - start:.1f}s")
    start = time.perf_counter()
    warm_ocr()
    print(f"Loading OCR model:     {time.perf_counter() - start:.1f}s   (once per server start)")
    start = time.perf_counter()
    warm_ai()
    print(f"Loading AI model:      {time.perf_counter() - start:.1f}s   (once per server start)")

    results = []
    for number in range(1, runs + 1):
        started = time.perf_counter()
        result = analyze_image(path, verbose=False)
        wall = time.perf_counter() - started
        results.append((wall, result.get("timings", {})))
        print(f"   run {number}: {wall:.1f}s")

    wall, timings = results[-1]
    ocr_detail = timings.get("ocr_detail", {})
    ai_detail = timings.get("gliner_detail", {})

    print("\n=== where the time went (last run) ===")
    rows = [
        ("OCR (reading the text)", timings.get("ocr", 0)),
        ("   of which the OCR model", ocr_detail.get("model", 0)),
        ("Detection (rules + AI model)", timings.get("text_detection", 0)),
        ("   of which GLiNER", ai_detail.get("gliner_seconds", 0)),
        ("Face / QR scan (runs next to OCR)", timings.get("vision", 0)),
        ("Risk score", timings.get("risk", 0)),
        ("TOTAL", timings.get("total", wall)),
    ]
    for label, seconds in rows:
        print(f"   {label:38s} {float(seconds):6.2f}s")

    print(f"\n   image size {ocr_detail.get('original_size')} -> sent to OCR {ocr_detail.get('size_sent_to_model')}, "
          f"{ocr_detail.get('lines')} text lines, {ai_detail.get('gliner_lines', 0)} lines sent to GLiNER")

    total = float(timings.get("total", wall)) or 1.0
    advice = []
    if float(timings.get("ocr", 0)) > 0.5 * total:
        advice.append('OCR is the main cost. Try:  $env:PRIVISCAN_OCR_MAX_SIDE = "1280"   (smaller image = faster)')
        if "retry_model" in ocr_detail:
            advice.append('The OCR ran twice (first pass found almost no text). Set  $env:PRIVISCAN_OCR_RETRY = "0"  to skip the second pass.')
    if float(ai_detail.get("gliner_seconds", 0)) > 0.25 * total:
        advice.append('The AI model is slow. Try:  $env:PRIVISCAN_GLINER_MAX_LINES = "8"   or switch it off:  $env:PRIVISCAN_USE_GLINER = "0"')
    if float(timings.get("vision", 0)) > 0.25 * total:
        advice.append('Face/QR scan is slow. Try:  $env:PRIVISCAN_VISION_MAX_SIDE = "1280"')
    if len(results) > 1 and results[0][0] > 1.5 * results[-1][0] + 2:
        advice.append("The first run is much slower than the next ones: the first image after a server start pays a warm-up cost.")

    print("\n=== suggestions ===")
    for line in advice or ["Nothing stands out. Send me this whole output if it is still slow."]:
        print("   -", line)


if __name__ == "__main__":
    main()
