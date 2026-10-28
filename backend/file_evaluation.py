"""
PriviScan File End-to-End Evaluation
====================================

Tests supported document formats through:

    File
      ↓
    Text extraction
      ↓
    PriviScan text detection
      ↓
    Risk scoring
      ↓
    Result

This test intentionally keeps document extraction separate
from the existing image evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


from backend.file_support import (
    SUPPORTED_EXTENSIONS,
    analyze_supported_file,
)


# ============================================================
# HELPERS
# ============================================================

def print_separator():
    print("-" * 70)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def print_detections(
    detections,
):
    if not detections:
        print("No sensitive data detected.")
        return

    for index, detection in enumerate(
        detections,
        start=1,
    ):
        dtype = detection.get(
            "type",
            "UNKNOWN",
        )

        value = detection.get(
            "value",
            "",
        )

        confidence = safe_float(
            detection.get(
                "confidence",
                0.0,
            )
        )

        risk_score = safe_float(
            detection.get(
                "risk_score",
                0.0,
            )
        )

        source = detection.get(
            "source",
            "unknown",
        )

        location = detection.get(
            "source_location",
            "",
        )

        print(
            f"[{index}] {dtype}"
        )

        print(
            f"    Value      : {value}"
        )

        print(
            f"    Confidence : {confidence:.4f}"
        )

        print(
            f"    Risk Score : {risk_score:.2f}"
        )

        print(
            f"    Source     : {source}"
        )

        if location:
            print(
                f"    Location   : {location}"
            )


# ============================================================
# SINGLE FILE
# ============================================================

def evaluate_file(
    file_path: Path,
) -> Dict[str, Any]:
    """
    Analyze one supported document.
    """

    from backend.analyze import (
        analyze_text,
        analyze_image,
    )

    print()
    print("=" * 70)
    print(
        f"Testing file: {file_path.name}"
    )
    print("=" * 70)

    print(
        f"File type: "
        f"{SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), 'UNKNOWN')}"
    )

    print()
    print("[1/4] Extracting document content...")

    try:
        result = analyze_supported_file(
            file_path,
            analyze_text_function=analyze_text,
            analyze_image_function=analyze_image,
            include_embedded_images=True,
            render_pdf=True,
        )

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}"
        )

        return {
            "file": str(file_path),
            "status": "ERROR",
            "error": str(exc),
        }

    print(
        f"Extracted characters: "
        f"{result.get('text_length', 0)}"
    )

    print()
    print("[2/4] Running sensitive-data detection...")

    detections = result.get(
        "detections",
        [],
    )

    print(
        f"Detections: "
        f"{len(detections)}"
    )

    print()
    print("[3/4] Calculating risk...")

    final_score = safe_float(
        result.get(
            "final_risk_score",
            result.get(
                "risk_score",
                0.0,
            ),
        )
    )

    average_score = safe_float(
        result.get(
            "average_risk_score",
            0.0,
        )
    )

    risk_level = result.get(
        "risk_level",
        "LOW",
    )

    print(
        f"Final Risk Score : {final_score:.2f}"
    )

    print(
        f"Average Risk     : {average_score:.2f}"
    )

    print(
        f"Risk Level       : {risk_level}"
    )

    print()
    print("[4/4] Detection details...")

    print_detections(
        detections
    )

    print()
    print_separator()

    status = "PASS"

    print(
        f"STATUS: {status}"
    )

    return {
        "file": str(file_path),
        "file_name": file_path.name,
        "file_type": result.get(
            "file_type"
        ),
        "status": status,
        "text_length": result.get(
            "text_length",
            0,
        ),
        "total_detections": len(
            detections
        ),
        "average_risk_score": average_score,
        "final_risk_score": final_score,
        "risk_level": risk_level,
        "detections": detections,
    }


# ============================================================
# DIRECTORY EVALUATION
# ============================================================

def evaluate_directory(
    directory: Path,
) -> Dict[str, Any]:
    """
    Evaluate all supported files in a directory.
    """

    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )

    if not files:
        print(
            f"No supported files found in: "
            f"{directory}"
        )

        return {
            "directory": str(directory),
            "total_files": 0,
            "results": [],
        }

    results = []

    for file_path in files:
        results.append(
            evaluate_file(
                file_path
            )
        )

    passed = sum(
        1
        for result in results
        if result.get("status") == "PASS"
    )

    failed = (
        len(results)
        - passed
    )

    print()
    print()
    print("=" * 70)
    print("PRIVISCAN FILE EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Files tested : {len(results)}"
    )

    print(
        f"Passed       : {passed}"
    )

    print(
        f"Failed       : {failed}"
    )

    print()

    for result in results:
        print(
            f"{result.get('file_name', result.get('file'))}"
            f" -> "
            f"{result.get('status')}"
        )

    return {
        "directory": str(directory),
        "total_files": len(results),
        "passed": passed,
        "failed": failed,
        "results": results,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "PriviScan document/file "
            "end-to-end evaluation"
        )
    )

    parser.add_argument(
        "path",
        help=(
            "Supported file or directory"
        ),
    )

    parser.add_argument(
        "--output",
        default="file_evaluation_results.json",
        help=(
            "Output JSON filename"
        ),
    )

    args = parser.parse_args()

    target = Path(
        args.path
    )

    if not target.exists():
        print(
            f"ERROR: Path does not exist: "
            f"{target}"
        )
        sys.exit(1)

    if target.is_file():

        if target.suffix.lower() not in (
            SUPPORTED_EXTENSIONS
        ):
            print(
                f"ERROR: Unsupported file type: "
                f"{target.suffix}"
            )
            print()
            print(
                "Supported:"
            )

            for extension in (
                SUPPORTED_EXTENSIONS
            ):
                print(
                    f"  {extension}"
                )

            sys.exit(1)

        report = evaluate_file(
            target
        )

    else:
        report = evaluate_directory(
            target
        )

    output_path = Path(
        args.output
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Results saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()