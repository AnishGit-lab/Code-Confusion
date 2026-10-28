"""
PriviScan Visual Privacy Detection

Visual detectors:
    QR   -> OpenCV QR detector
    FACE -> OpenCV Haar Cascade

These detectors work independently from OCR.

The result follows the same detection structure used by
the rest of PriviScan:

{
    "type": "...",
    "value": "...",
    "confidence": 0.0,
    "box": [x1, y1, x2, y2],
    "source": "VISION"
}
"""

from __future__ import annotations

from typing import Any

import cv2


# ============================================================
# BOX HELPERS
# ============================================================

def _box_from_points(
    points: Any,
) -> list[int] | None:

    try:

        pts = points.reshape(-1, 2)

        return [
            int(pts[:, 0].min()),
            int(pts[:, 1].min()),
            int(pts[:, 0].max()),
            int(pts[:, 1].max()),
        ]

    except Exception:
        return None


def _iou(
    box_a: list[int],
    box_b: list[int],
) -> float:

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0,
        ix2 - ix1,
    )

    ih = max(
        0,
        iy2 - iy1,
    )

    intersection = iw * ih

    if intersection == 0:
        return 0.0

    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# QR DETECTION
# ============================================================

def _qr_detection(
    value: str,
    box: list[int],
    confidence: float = 0.99,
) -> dict:

    return {
        "type": "QR",
        "value": value or "QR Code",
        "confidence": round(
            float(confidence),
            4,
        ),
        "box": box,
        "source": "VISION",
    }


def detect_qr_codes(
    image,
) -> list[dict]:

    detections = []

    detector = cv2.QRCodeDetector()

    # --------------------------------------------------------
    # Multiple QR codes
    # --------------------------------------------------------

    try:

        result = detector.detectAndDecodeMulti(
            image
        )

        if len(result) == 4:

            success, decoded_info, points, _ = result

            if success and points is not None:

                for index, point_set in enumerate(points):

                    box = _box_from_points(
                        point_set
                    )

                    if box is None:
                        continue

                    value = ""

                    if (
                        decoded_info is not None
                        and index < len(decoded_info)
                    ):
                        value = str(
                            decoded_info[index] or ""
                        ).strip()

                    # Only accept a QR if a payload
                    # can actually be decoded.
                    if value:

                        detections.append(
                            _qr_detection(
                                value,
                                box,
                                0.99,
                            )
                        )

    except cv2.error:
        pass

    except Exception:
        pass

    # --------------------------------------------------------
    # Single QR fallback
    # --------------------------------------------------------

    if not detections:

        try:

            value, points, _ = (
                detector.detectAndDecode(
                    image
                )
            )

            if points is not None:

                box = _box_from_points(
                    points
                )

                value = str(
                    value or ""
                ).strip()

                if (
                    box is not None
                    and value
                ):

                    detections.append(
                        _qr_detection(
                            value,
                            box,
                            0.99,
                        )
                    )

        except cv2.error:
            pass

        except Exception:
            pass

    return detections


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(
    image,
) -> list[dict]:

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    cascade = cv2.CascadeClassifier(
        cascade_path
    )

    if cascade.empty():
        return []

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    detections = []

    for x, y, width, height in faces:

        detections.append({

            "type": "FACE",

            "value": "Face detected",

            # Haar Cascade does not provide
            # calibrated probabilities.
            "confidence": 0.90,

            "box": [
                int(x),
                int(y),
                int(x + width),
                int(y + height),
            ],

            "source": "VISION",
        })

    return detections


# ============================================================
# VISION DEDUPLICATION
# ============================================================

def deduplicate_vision_detections(
    detections: list[dict],
) -> list[dict]:

    kept = []

    for detection in detections:

        duplicate = False

        for existing in kept:

            if (
                detection.get("type")
                != existing.get("type")
            ):
                continue

            box_a = detection.get("box")
            box_b = existing.get("box")

            if (
                box_a
                and box_b
                and _iou(
                    box_a,
                    box_b,
                ) >= 0.70
            ):

                duplicate = True

                if float(
                    detection.get(
                        "confidence",
                        0,
                    )
                ) > float(
                    existing.get(
                        "confidence",
                        0,
                    )
                ):

                    existing.update(
                        detection
                    )

                break

        if not duplicate:
            kept.append(
                detection
            )

    return kept


# ============================================================
# MAIN VISION FUNCTION
# ============================================================

def detect_vision(
    image_path: str,
) -> list[dict]:

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            f"Could not read image: {image_path}"
        )

    detections = []

    # QR
    detections.extend(
        detect_qr_codes(
            image
        )
    )

    # Face
    detections.extend(
        detect_faces(
            image
        )
    )

    return deduplicate_vision_detections(
        detections
    )