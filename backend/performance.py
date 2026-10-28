from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from backend.analyze import analyze_image


def benchmark(
    image_path: str,
    runs: int = 1,
) -> Dict[str, Any]:

    if runs < 1:
        raise ValueError(
            "runs must be at least 1"
        )

    timings: List[float] = []
    results: List[Dict[str, Any]] = []

    for index in range(runs):

        print(
            f"Run {index + 1}/{runs}"
        )

        start = time.perf_counter()

        result = analyze_image(
            image_path,
            verbose=True,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        timings.append(elapsed)

        results.append(
            {
                "run": index + 1,
                "elapsed_seconds": elapsed,
                "detections": result.get(
                    "total_detections",
                    0,
                ),
                "average_risk_score": result.get(
                    "average_risk_score",
                    0.0,
                ),
                "final_risk_score": result.get(
                    "final_risk_score",
                    0.0,
                ),
                "risk_level": result.get(
                    "risk_level",
                    "LOW",
                ),
            }
        )

    average = statistics.mean(
        timings
    )

    median = statistics.median(
        timings
    )

    minimum = min(timings)
    maximum = max(timings)

    standard_deviation = (
        statistics.stdev(timings)
        if len(timings) > 1
        else 0.0
    )

    return {
        "project": "PriviScan",
        "image": image_path,
        "runs": runs,
        "average_latency_seconds": average,
        "median_latency_seconds": median,
        "min_latency_seconds": minimum,
        "max_latency_seconds": maximum,
        "stddev_latency_seconds":
            standard_deviation,
        "results": results,
    }


def main() -> None:

    parser = argparse.ArgumentParser(
        description="PriviScan performance benchmark"
    )

    parser.add_argument(
        "image",
        help="Image to benchmark",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output",
        default="performance_results.json",
    )

    args = parser.parse_args()

    result = benchmark(
        args.image,
        args.runs,
    )

    print()
    print("=" * 60)
    print("PRIVISCAN PERFORMANCE")
    print("=" * 60)

    print(
        f"Average : "
        f"{result['average_latency_seconds']:.4f} sec"
    )

    print(
        f"Median  : "
        f"{result['median_latency_seconds']:.4f} sec"
    )

    print(
        f"Minimum : "
        f"{result['min_latency_seconds']:.4f} sec"
    )

    print(
        f"Maximum : "
        f"{result['max_latency_seconds']:.4f} sec"
    )

    print(
        f"Std Dev : "
        f"{result['stddev_latency_seconds']:.4f} sec"
    )

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()