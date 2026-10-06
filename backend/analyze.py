from backend.ocr.detector import extract_text
from backend.detection.sensitive import detect_sensitive_information
from backend.vision import detect_vision
from backend.risk.scoring import calculate_average_risk


def analyze_image(image_path):
    """
    Analyze an image using:
    1. OCR
    2. Sensitive text detection
    3. Vision detection (QR / Face)
    """

    all_detections = []

    # ========================================================
    # 1. OCR
    # ========================================================

    ocr_results = extract_text(image_path)

    # ========================================================
    # 2. TEXT-BASED SENSITIVE DATA DETECTION
    # ========================================================

    for item in ocr_results:

        text = item["text"]

        detections = detect_sensitive_information(text)

        for detection in detections:

            detection["confidence"] = (
                item["confidence"]
            )

            detection["box"] = (
                item["box"]
            )

            detection["source"] = "OCR"

        all_detections.extend(
            detections
        )

    # ========================================================
    # 3. IMAGE-BASED DETECTION
    #    QR CODE + FACE
    # ========================================================

    try:

        vision_detections = detect_vision(
            image_path
        )

        all_detections.extend(
            vision_detections
        )

    except Exception as error:

        print(
            "Vision detection warning:",
            error
        )

    # ========================================================
    # 4. REMOVE EXACT DUPLICATES
    # ========================================================

    unique_detections = []

    seen = set()

    for detection in all_detections:

        box = detection.get(
            "box",
            []
        )

        if isinstance(box, list):

            box_key = tuple(
                box
            )

        else:

            box_key = str(
                box
            )

        key = (
            detection.get(
                "type"
            ),
            detection.get(
                "value"
            ),
            box_key
        )

        if key not in seen:

            seen.add(
                key
            )

            unique_detections.append(
                detection
            )

    return unique_detections


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    image_path = (
        "dataset/mixed/mixed_05.png"
    )

    results = analyze_image(
        image_path
    )

    print(
        "\n========== PRIVISCAN ANALYSIS ==========\n"
    )

    if not results:

        print(
            "No sensitive information detected."
        )

    else:

        for result in results:

            print(
                "Type:",
                result["type"]
            )

            print(
                "Value:",
                result["value"]
            )

            print(
                "Risk Score:",
                result["risk"]
            )

            if result.get(
                "confidence"
            ) is not None:

                print(
                    "Confidence:",
                    round(
                        result["confidence"],
                        4
                    )
                )

            print(
                "Box:",
                result.get(
                    "box"
                )
            )

            print(
                "Source:",
                result.get(
                    "source",
                    "Unknown"
                )
            )

            print(
                "--------------------------------"
            )

    average_score = (
        calculate_average_risk(
            results
        )
    )

    print(
        "\nAverage Risk Score:",
        average_score
    )

    print(
        "Total Detections:",
        len(results)
    )