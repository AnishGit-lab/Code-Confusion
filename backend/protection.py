from __future__ import annotations

import argparse
import os
import random
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import cv2
import numpy as np


SUPPORTED_METHODS = {
    "blur",
    "blackout",
    "star",
    "random_alphabet",
    "hashtag",
}

SUPPORTED_MODES = {
    "automatic",
    "manual",
}


def _normalize_box(
    box: Any,
    image_shape: Sequence[int],
) -> Optional[tuple]:
    if box is None:
        return None

    height, width = image_shape[:2]

    try:
        if isinstance(box, np.ndarray):
            box = box.tolist()

        if len(box) == 4 and all(
            isinstance(x, (int, float))
            for x in box
        ):
            x1, y1, x2, y2 = map(float, box)

        else:
            points = np.asarray(
                box,
                dtype=float,
            )

            if points.ndim == 2 and points.shape[1] >= 2:
                x1 = float(points[:, 0].min())
                y1 = float(points[:, 1].min())
                x2 = float(points[:, 0].max())
                y2 = float(points[:, 1].max())
            else:
                return None

        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(0, min(width, int(x2)))
        y2 = max(0, min(height, int(y2)))

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    except Exception:
        return None


def apply_blur(
    image: np.ndarray,
    box: tuple,
) -> np.ndarray:
    x1, y1, x2, y2 = box

    region = image[y1:y2, x1:x2]

    if region.size == 0:
        return image

    width = region.shape[1]
    height = region.shape[0]

    kernel = max(
        15,
        ((min(width, height) // 5) * 2) + 1,
    )

    if kernel % 2 == 0:
        kernel += 1

    image[y1:y2, x1:x2] = cv2.GaussianBlur(
        region,
        (kernel, kernel),
        0,
    )

    return image


def apply_blackout(
    image: np.ndarray,
    box: tuple,
) -> np.ndarray:
    x1, y1, x2, y2 = box

    image[y1:y2, x1:x2] = 0

    return image


def _replacement_text(
    method: str,
    original: str,
) -> str:
    length = max(3, len(original))

    if method == "star":
        return "*" * length

    if method == "hashtag":
        return "#" * length

    if method == "random_alphabet":
        alphabet = string.ascii_letters
        return "".join(
            random.choice(alphabet)
            for _ in range(length)
        )

    return "*" * length


def apply_text_replacement(
    image: np.ndarray,
    box: tuple,
    method: str,
    value: str,
) -> np.ndarray:
    x1, y1, x2, y2 = box

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        -1,
    )

    replacement = _replacement_text(
        method,
        value,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    max_width = max(
        10,
        x2 - x1 - 4,
    )

    font_scale = 0.55

    while font_scale > 0.15:
        size, _ = cv2.getTextSize(
            replacement,
            font,
            font_scale,
            1,
        )

        if size[0] <= max_width:
            break

        font_scale -= 0.05

    cv2.putText(
        image,
        replacement,
        (x1 + 2, y1 + max(15, (y2 - y1) // 2)),
        font,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return image


def protect_detection(
    image: np.ndarray,
    detection: Dict[str, Any],
    method: str = "blur",
) -> np.ndarray:

    method = method.lower().strip()

    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported protection method: {method}"
        )

    box = detection.get(
        "box",
        detection.get("bbox"),
    )

    normalized = _normalize_box(
        box,
        image.shape,
    )

    if normalized is None:
        return image

    if method == "blur":
        return apply_blur(
            image,
            normalized,
        )

    if method == "blackout":
        return apply_blackout(
            image,
            normalized,
        )

    return apply_text_replacement(
        image,
        normalized,
        method,
        str(
            detection.get(
                "value",
                detection.get("text", ""),
            )
        ),
    )


def _select_manual(
    detections: List[Dict[str, Any]],
    selected_indices: Optional[Iterable[int]] = None,
    selected_values: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:

    selected = []

    indices = {
        int(index)
        for index in (selected_indices or [])
    }

    values = {
        str(value).strip().lower()
        for value in (selected_values or [])
    }

    for index, detection in enumerate(detections):
        value = str(
            detection.get(
                "value",
                detection.get("text", ""),
            )
        ).strip().lower()

        if index in indices or value in values:
            selected.append(detection)

    return selected


def protect_image(
    image_path: str,
    detections: Optional[List[Dict[str, Any]]] = None,
    mode: str = "automatic",
    method: str = "blur",
    selected_indices: Optional[Iterable[int]] = None,
    selected_values: Optional[Iterable[str]] = None,
    output_path: Optional[str] = None,
) -> str:

    mode = mode.lower().strip()

    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported mode: {mode}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    detections = detections or []

    if mode == "automatic":
        selected = detections

    else:
        selected = _select_manual(
            detections,
            selected_indices,
            selected_values,
        )

    for detection in selected:
        protect_detection(
            image,
            detection,
            method,
        )

    if output_path is None:
        source = Path(image_path)

        output_path = str(
            source.with_name(
                f"{source.stem}_protected{source.suffix}"
            )
        )

    output_parent = Path(output_path).parent
    output_parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if not success:
        raise IOError(
            f"Failed to write protected image: {output_path}"
        )

    return str(output_path)


def redact_image(*args, **kwargs):
    return protect_image(*args, **kwargs)


def apply_protection(*args, **kwargs):
    return protect_image(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PriviScan image protection"
    )

    parser.add_argument(
        "image",
        help="Image path",
    )

    parser.add_argument(
        "--method",
        default="blur",
        choices=sorted(SUPPORTED_METHODS),
    )

    parser.add_argument(
        "--mode",
        default="automatic",
        choices=sorted(SUPPORTED_MODES),
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    result = protect_image(
        args.image,
        detections=[],
        mode=args.mode,
        method=args.method,
        output_path=args.output,
    )

    print(f"Protected image: {result}")


if __name__ == "__main__":
    main()