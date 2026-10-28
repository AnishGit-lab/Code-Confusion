from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from backend.analyze import analyze_image


TYPE_ALIASES = {
    "STUDENT_ID": "ID",
    "ID_NUMBER": "ID",
    "IDENTIFICATION_NUMBER": "ID",
    "PERSON_NAME": "NAME",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "PHYSICAL_ADDRESS": "ADDRESS",
    "GOV_ID": "GOVERNMENT_ID",
    "ACCOUNT": "ACCOUNT_NUMBER",
    "API_TOKEN": "API_KEY",
}


def normalize_type(value: Any) -> str:
    value = str(value or "").upper().strip()

    return TYPE_ALIASES.get(
        value,
        value,
    )


def load_ground_truth(
    image_path: Path,
) -> Set[str]:

    candidates = [
        image_path.with_suffix(".json"),
        image_path.with_suffix(".txt"),
    ]

    for path in candidates:

        if not path.exists():
            continue

        try:

            if path.suffix.lower() == ".json":

                data = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )

                if isinstance(data, dict):
                    values = (
                        data.get("types")
                        or data.get("labels")
                        or data.get("detections")
                        or []
                    )

                else:
                    values = data

                if isinstance(values, list):

                    labels = set()

                    for value in values:

                        if isinstance(value, dict):
                            value = value.get(
                                "type",
                                value.get("label", ""),
                            )

                        labels.add(
                            normalize_type(value)
                        )

                    return {
                        x
                        for x in labels
                        if x
                    }

            else:

                return {
                    normalize_type(line.strip())
                    for line in path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                }

        except Exception:
            continue

    return set()


def precision_recall_f1(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> Tuple[float, float, float]:

    precision = (
        true_positive
        / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1


def expected_calibration_error(
    probabilities: List[float],
    correctness: List[int],
    bins: int = 10,
) -> float:

    if not probabilities:
        return 0.0

    total = len(probabilities)
    error = 0.0

    for index in range(bins):

        lower = index / bins
        upper = (index + 1) / bins

        selected = [
            i
            for i, probability
            in enumerate(probabilities)
            if (
                lower <= probability < upper
                or (
                    index == bins - 1
                    and probability == 1.0
                )
            )
        ]

        if not selected:
            continue

        confidence = sum(
            probabilities[i]
            for i in selected
        ) / len(selected)

        accuracy = sum(
            correctness[i]
            for i in selected
        ) / len(selected)

        error += (
            len(selected)
            / total
        ) * abs(
            accuracy - confidence
        )

    return error


def brier_score(
    probabilities: List[float],
    correctness: List[int],
) -> float:

    if not probabilities:
        return 0.0

    return sum(
        (
            probability - actual
        ) ** 2
        for probability, actual
        in zip(probabilities, correctness)
    ) / len(probabilities)


def evaluate_dataset(
    dataset_path: str,
) -> Dict[str, Any]:

    dataset = Path(dataset_path)

    images = sorted(
        path
        for path in dataset.iterdir()
        if path.suffix.lower()
        in {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".webp",
        }
    )

    true_positive = 0
    false_positive = 0
    false_negative = 0

    per_class: Dict[str, Dict[str, int]] = {}

    probabilities: List[float] = []
    correctness: List[int] = []

    labelled_images = 0
    excluded_images = 0

    image_results = []

    for image_path in images:

        ground_truth = load_ground_truth(
            image_path
        )

        if not ground_truth:

            excluded_images += 1

            continue

        labelled_images += 1

        result = analyze_image(
            str(image_path),
            verbose=False,
        )

        detected = {
            normalize_type(
                item.get("type")
            )
            for item in result.get(
                "detections",
                [],
            )
            if item.get("type")
        }

        tp = len(
            detected & ground_truth
        )

        fp = len(
            detected - ground_truth
        )

        fn = len(
            ground_truth - detected
        )

        true_positive += tp
        false_positive += fp
        false_negative += fn

        for label in (
            detected
            | ground_truth
        ):

            stats = per_class.setdefault(
                label,
                {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                },
            )

            if label in detected and label in ground_truth:
                stats["tp"] += 1

            elif label in detected:
                stats["fp"] += 1

            else:
                stats["fn"] += 1

        for detection in result.get(
            "detections",
            [],
        ):

            probability = float(
                detection.get(
                    "confidence",
                    0.0,
                )
                or 0.0
            )

            label = normalize_type(
                detection.get("type")
            )

            probabilities.append(
                max(
                    0.0,
                    min(
                        1.0,
                        probability,
                    ),
                )
            )

            correctness.append(
                1
                if label in ground_truth
                else 0
            )

        image_results.append(
            {
                "image": image_path.name,
                "ground_truth": sorted(
                    ground_truth
                ),
                "detected": sorted(
                    detected
                ),
                "risk_score": result.get(
                    "final_risk_score",
                    0.0,
                ),
                "risk_level": result.get(
                    "risk_level",
                    "LOW",
                ),
            }
        )

    precision, recall, f1 = precision_recall_f1(
        true_positive,
        false_positive,
        false_negative,
    )

    class_metrics = {}

    f1_values = []

    for label, stats in per_class.items():

        p, r, f = precision_recall_f1(
            stats["tp"],
            stats["fp"],
            stats["fn"],
        )

        class_metrics[label] = {
            **stats,
            "precision": p,
            "recall": r,
            "f1": f,
        }

        f1_values.append(f)

    macro_f1 = (
        sum(f1_values)
        / len(f1_values)
        if f1_values
        else 0.0
    )

    ece = expected_calibration_error(
        probabilities,
        correctness,
    )

    brier = brier_score(
        probabilities,
        correctness,
    )

    return {
        "dataset": str(dataset),
        "images_total": len(images),
        "images_labelled": labelled_images,
        "images_excluded": excluded_images,
        "micro": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "macro_f1": macro_f1,
        "ece": ece,
        "brier_score": brier,
        "per_class": class_metrics,
        "images": image_results,
    }


def main() -> None:

    parser = argparse.ArgumentParser(
        description="PriviScan dataset evaluation"
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        default="dataset",
    )

    parser.add_argument(
        "--output",
        default="evaluation_results.json",
    )

    args = parser.parse_args()

    result = evaluate_dataset(
        args.dataset
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
            ensure_ascii=False,
        )

    print(
        json.dumps(
            {
                "Images total":
                    result["images_total"],
                "Images labelled":
                    result["images_labelled"],
                "Images excluded":
                    result["images_excluded"],
                "Micro F1":
                    result["micro"]["f1"],
                "Macro F1":
                    result["macro_f1"],
                "ECE":
                    result["ece"],
                "Brier score":
                    result["brier_score"],
            },
            indent=2,
        )
    )

    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()