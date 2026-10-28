
"""
PriviScan Week 9 Report Generator
Reads evaluation and performance JSON files and prints a report.
"""

from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FINAL_EVALUATION_FILE = PROJECT_ROOT / "week9_final_evaluation.json"
PERFORMANCE_FILE = PROJECT_ROOT / "week9_performance_results.json"


def load_json(path):
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR reading {path.name}: {exc}")
        return None


def separator():
    print("=" * 70)


def main():
    print()
    separator()
    print("PriviScan Week 9 Report")
    separator()
    print(f"Project directory: {PROJECT_ROOT}")

    evaluation = load_json(FINAL_EVALUATION_FILE)
    performance = load_json(PERFORMANCE_FILE)

    if evaluation is None:
        print("Cannot generate the evaluation report.")
        return

    summary = evaluation.get("summary", {})
    tests = evaluation.get("tests", [])

    print()
    separator()
    print(evaluation.get("evaluation", "Final Evaluation"))
    separator()

    print(f"Total Tests       : {summary.get('total_tests', len(tests))}")
    print(f"Passed            : {summary.get('passed', 0)}")
    print(f"Failed            : {summary.get('failed', 0)}")
    print(f"Pass Rate         : {summary.get('pass_rate_percent', 0):.2f}%")
    print(
        "Average Time      : "
        f"{summary.get('average_processing_seconds', 0):.4f} sec/image"
    )

    print()
    print("Individual Test Results")
    print("-" * 70)

    for test in tests:
        print(f"Image       : {test.get('file', 'Unknown')}")
        print(f"Status      : {test.get('status', 'UNKNOWN')}")
        print(f"Detections  : {test.get('detections', 0)}")
        print(f"Types       : {', '.join(test.get('detected_types', [])) or 'None'}")
        print(f"Risk Score  : {test.get('risk_score', 0)}")
        print(f"Risk Level  : {test.get('risk_level', 'UNKNOWN')}")
        print(
            "Time        : "
            f"{test.get('processing_seconds', 0):.4f} sec"
        )
        print("-" * 70)

    if performance is not None:
        print()
        separator()
        print("Performance Test Results")
        separator()

        for key, value in performance.items():
            if isinstance(value, (dict, list)):
                continue
            print(f"{key.replace('_', ' ').title():<30}: {value}")

    print()
    separator()

    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)

    if failed == 0 and passed == summary.get("total_tests", len(tests)):
        print("OVERALL RESULT: PASS")
    else:
        print("OVERALL RESULT: REVIEW REQUIRED")

    print("Week 9 report generation completed.")
    separator()


if __name__ == "__main__":
    main()
