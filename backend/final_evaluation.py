from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from backend.analyze import analyze_image


DATASET_DIR = (
    Path(__file__).resolve().parent.parent
    / "dataset"
    / "week9_tests"
)

RESULTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED = {
    "01_safe.png": {
        "purpose": "Safe image without sensitive information",
        "minimum": 0,
        "types": set(),
        "risk": "LOW",
    },
    "02_basic_pii.png": {
        "purpose": "Basic personal information",
        "minimum": 2,
        "types": {
            "NAME",
            "EMAIL",
            "PHONE",
            "ADDRESS",
        },
        "risk": None,
    },
    "03_identity.png": {
        "purpose": "Identity information",
        "minimum": 2,
        "types": {
            "NAME",
            "ID",
        },
        "risk": None,
    },
    "04_financial.png": {
        "purpose": "Financial information",
        "minimum": 1,
        "types": {
            "ACCOUNT_NUMBER",
            "BANK_CARD",
            "GOVERNMENT_ID",
        },
        "risk": None,
    },
    "05_credentials.png": {
        "purpose": "Credentials and secrets",
        "minimum": 2,
        "types": {
            "PASSWORD",
            "API_KEY",
        },
        "risk": "CRITICAL",
    },
    "06_mixed_critical.png": {
        "purpose": "Mixed sensitive information",
        "minimum": 4,
        "types": {
            "NAME",
            "EMAIL",
            "PHONE",
            "ID",
            "PASSWORD",
            "API_KEY",
        },
        "risk": "CRITICAL",
    },
}


def evaluate_single_image(
    image_path: Path,
    specification: Dict[str, Any],
) -> Dict[str, Any]:

    start = time.perf_counter()

    result = analyze_image(
        str(image_path),
        verbose=True,
    )

    elapsed = time.perf_counter() - start

    detections = result.get(
        "detections",
        [],
    )

    actual_types = {
        str(item.get("type", "")).upper()
        for item in detections
    }

    count_ok = (
        len(detections)
        >= specification["minimum"]
    )

    expected_types = specification["types"]

    type_ok = (
        expected_types.issubset(actual_types)
        if expected_types
        else True
    )

    expected_risk = specification["risk"]

    risk_ok = (
        expected_risk is None
        or result.get("risk_level") == expected_risk
    )

    passed = (
        count_ok
        and type_ok
        and risk_ok
    )

    return {
        "image": image_path.name,
        "purpose": specification["purpose"],
        "detections": len(detections),
        "types": sorted(actual_types),
        "expected_types": sorted(expected_types),
        "risk_score": result.get(
            "final_risk_score",
            0.0,
        ),
        "risk_level": result.get(
            "risk_level",
            "LOW",
        ),
        "average_risk_score": result.get(
            "average_risk_score",
            0.0,
        ),
        "elapsed_seconds": round(
            elapsed,
            4,
        ),
        "count_ok": count_ok,
        "type_ok": type_ok,
        "risk_ok": risk_ok,
        "status": "PASS" if passed else "FAIL",
    }


def main() -> None:

    print("=" * 70)
    print("PRIVISCAN WEEK 9 FINAL EVALUATION")
    print("=" * 70)

    results: List[Dict[str, Any]] = []

    if not DATASET_DIR.exists():

        print(
            f"Dataset not found: {DATASET_DIR}"
        )

        return

    for filename, specification in EXPECTED.items():

        image_path = DATASET_DIR / filename

        if not image_path.exists():

            result = {
                "image": filename,
                "status": "MISSING",
            }

            results.append(result)

            print(
                f"{filename}: MISSING"
            )

            continue

        print("-" * 70)
        print(f"Testing: {filename}")
        print(
            f"Purpose: {specification['purpose']}"
        )

        try:

            result = evaluate_single_image(
                image_path,
                specification,
            )

        except Exception as exc:

            result = {
                "image": filename,
                "status": "ERROR",
                "error": str(exc),
            }

        results.append(result)

        print(
            f"Detections : "
            f"{result.get('detections', 0)}"
        )

        print(
            f"Risk Score : "
            f"{result.get('risk_score', 0.0)}"
        )

        print(
            f"Risk Level : "
            f"{result.get('risk_level', 'UNKNOWN')}"
        )

        print(
            f"STATUS     : "
            f"{result.get('status')}"
        )

    passed = sum(
        1
        for result in results
        if result.get("status") == "PASS"
    )

    total = len(results)

    overall = (
        "PASS"
        if total > 0 and passed == total
        else "FAIL"
    )

    output = {
        "project": "PriviScan",
        "algorithm": {
            "impact": (
                "Average of Identifiability, "
                "Uniqueness, Potential Harm, "
                "and Reusability"
            ),
            "likelihood": (
                "Detection Confidence, "
                "Linkability, and Context"
            ),
            "individual_risk": (
                "Impact × Likelihood normalized to 0–100"
            ),
            "aggregation": (
                "Aggregate item risks + linkability adjustment"
            ),
            "credential_override": (
                "PASSWORD/API_KEY/SECRET with "
                "confidence >= 0.85 -> CRITICAL"
            ),
            "sector": "Removed from core implementation",
        },
        "summary": {
            "passed": passed,
            "total": total,
            "pass_rate": (
                round(
                    passed / total * 100,
                    2,
                )
                if total
                else 0.0
            ),
            "overall": overall,
        },
        "results": results,
    }

    output_path = (
        RESULTS_DIR
        / "week9_final_evaluation.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 70)
    print(
        f"RESULT: {overall}"
    )
    print(
        f"Passed: {passed}/{total}"
    )
    print(
        f"Saved: {output_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()