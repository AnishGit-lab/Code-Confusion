"""
PriviScan Week 7 Vision Evaluation

Evaluates:
    FACE
    QR

Metrics:
    Precision
    Recall
    F1
    Average latency
"""

from __future__ import annotations

import argparse
import json
import time

from pathlib import Path

from backend.vision import detect_vision


# ============================================================
# METRICS
# ============================================================

def metrics(
    tp: int,
    fp: int,
    fn: int,
) -> dict[str, float]:

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "precision": round(
            precision,
            4,
        ),
        "recall": round(
            recall,
            4,
        ),
        "f1": round(
            f1,
            4,
        ),
    }


# ============================================================
# DATASET LABEL
# ============================================================

def expected_label(
    path: Path,
) -> str | None:

    name = path.name.lower()

    if name.startswith(
        "face_real_"
    ):
        return "FACE"

    if name.startswith(
        "qr_valid_"
    ):
        return "QR"

    return None


# ============================================================
# EVALUATION
# ============================================================

def run(
    root: str | Path,
) -> dict:

    root = Path(root)

    if not root.exists():

        raise FileNotFoundError(
            f"Dataset not found: {root}"
        )

    counts = {
        "FACE": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        },

        "QR": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        },
    }

    rows = []

    latencies = []

    image_paths = sorted(
        [
            path
            for path in root.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".bmp",
                    ".webp",
                }
            )
        ]
    )

    for path in image_paths:

        expected = expected_label(
            path
        )

        if expected is None:
            continue

        start = time.perf_counter()

        detections = detect_vision(
            str(path)
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        latencies.append(
            elapsed
        )

        predicted = {
            str(
                detection.get(
                    "type",
                    "",
                )
            ).upper()
            for detection in detections
        }

        # ----------------------------------------------------
        # FACE
        # ----------------------------------------------------

        for label in [
            "FACE",
            "QR",
        ]:

            actual = (
                label == expected
            )

            found = (
                label in predicted
            )

            if actual and found:

                counts[label]["tp"] += 1

            elif not actual and found:

                counts[label]["fp"] += 1

            elif actual and not found:

                counts[label]["fn"] += 1

        rows.append({

            "image": path.name,

            "expected": expected,

            "predicted": sorted(
                predicted
            ),

            "latency_seconds": round(
                elapsed,
                4,
            ),

            "detections": detections,
        })

    result_metrics = {}

    for label, values in counts.items():

        result_metrics[label] = {
            **values,
            **metrics(
                values["tp"],
                values["fp"],
                values["fn"],
            ),
        }

    average_latency = (
        sum(latencies)
        / len(latencies)
        if latencies
        else 0.0
    )

    return {

        "images": len(rows),

        "metrics": result_metrics,

        "average_latency_seconds": round(
            average_latency,
            4,
        ),

        "images_detail": rows,
    }


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PriviScan FACE and QR detectors"
        )
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        default="dataset/week7_vision",
    )

    parser.add_argument(
        "--output",
        default="vision_evaluation_results.json",
    )

    args = parser.parse_args()

    report = run(
        args.dataset
    )

    Path(
        args.output
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=" * 55
    )

    print(
        "PRIVISCAN WEEK 7 VISION EVALUATION"
    )

    print(
        "=" * 55
    )

    for label, values in report[
        "metrics"
    ].items():

        print(
            f"{label}: "
            f"P={values['precision']:.4f} "
            f"R={values['recall']:.4f} "
            f"F1={values['f1']:.4f}"
        )

    print(
        f"Average latency: "
        f"{report['average_latency_seconds']:.4f}s"
    )

    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()