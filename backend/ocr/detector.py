"""
PriviScan OCR Detection Layer

OCR is responsible only for extracting:
    - text
    - OCR confidence
    - bounding box

OCR does NOT decide whether the text is:
    NAME
    EMAIL
    PHONE
    ID
    ACCOUNT_NUMBER
    etc.

That classification is handled by ai_detector.py.
"""

from typing import Any, Dict, List

from paddleocr import PaddleOCR


# ============================================================
# OCR MODEL
# ============================================================

ocr = PaddleOCR(
    lang="en",
    device="cpu",
    enable_mkldnn=False,
)


# ============================================================
# HELPER
# ============================================================

def convert_box(box: Any) -> List[int]:
    """
    Convert PaddleOCR bounding box into a normal Python list.

    Expected format:

        [left, top, right, bottom]
    """

    if hasattr(box, "tolist"):
        box = box.tolist()

    # Make sure the result is a list.
    if not isinstance(box, (list, tuple)):
        return []

    # Convert numeric values safely.
    result = []

    for value in box:

        try:
            result.append(int(value))

        except (TypeError, ValueError):
            continue

    return result


# ============================================================
# OCR EXTRACTION
# ============================================================

def extract_text(
    image_path: str,
) -> List[Dict[str, Any]]:
    """
    Extract text, confidence and bounding boxes from an image.

    Returns:

        [
            {
                "text": "Email: test@example.com",
                "confidence": 0.998,
                "box": [0, 125, 390, 167]
            }
        ]
    """

    extracted_text: List[Dict[str, Any]] = []

    results = ocr.predict(
        image_path
    )

    for result in results:

        # ----------------------------------------------------
        # PaddleOCR result handling
        # ----------------------------------------------------

        try:
            data = result.json
        except Exception:
            continue

        # Some PaddleOCR versions return:
        #
        # {
        #     "res": {...}
        # }
        #
        # while others expose the result directly.

        if isinstance(data, dict) and "res" in data:

            data = data["res"]

        if not isinstance(data, dict):
            continue

        # ----------------------------------------------------
        # Extract OCR arrays
        # ----------------------------------------------------

        texts = data.get(
            "rec_texts",
            []
        )

        scores = data.get(
            "rec_scores",
            []
        )

        boxes = data.get(
            "rec_boxes",
            []
        )

        # ----------------------------------------------------
        # Process each OCR result
        # ----------------------------------------------------

        for index, text in enumerate(texts):

            text = str(text).strip()

            if not text:
                continue

            # -----------------------------------------------
            # Confidence
            # -----------------------------------------------

            if index < len(scores):

                try:
                    confidence = float(
                        scores[index]
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    confidence = 0.0

            else:

                confidence = 0.0

            # -----------------------------------------------
            # Bounding box
            # -----------------------------------------------

            if index < len(boxes):

                box = convert_box(
                    boxes[index]
                )

            else:

                box = []

            # -----------------------------------------------
            # Store result
            # -----------------------------------------------

            extracted_text.append(
                {
                    "text": text,

                    "confidence": round(
                        confidence,
                        4,
                    ),

                    "box": box,
                }
            )

    return extracted_text


# ============================================================
# OCR TEXT ONLY
# ============================================================

def get_full_text(
    ocr_results: List[Dict[str, Any]]
) -> str:
    """
    Combine OCR results into one text string.

    This is useful for context/sector analysis.
    """

    return "\n".join(
        item["text"]
        for item in ocr_results
        if item.get("text")
    )


# ============================================================
# OCR DEBUG OUTPUT
# ============================================================

def print_ocr_results(
    ocr_results: List[Dict[str, Any]]
) -> None:
    """
    Print OCR results in a readable format.
    """

    if not ocr_results:

        print("No text detected.")

        return

    print("\n===== OCR RESULTS =====")

    for index, item in enumerate(
        ocr_results,
        start=1,
    ):

        print(
            f"\n[{index}]"
        )

        print(
            f"Text: {item['text']}"
        )

        print(
            f"Confidence: "
            f"{item['confidence']}"
        )

        print(
            f"Box: {item['box']}"
        )

    print(
        "\n=======================\n"
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python -m backend.ocr.detector "
            "<image_path>"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    results = extract_text(
        image_path
    )

    print_ocr_results(
        results
    )