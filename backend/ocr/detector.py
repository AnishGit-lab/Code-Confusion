from paddleocr import PaddleOCR


ocr = PaddleOCR(
    lang="en",
    device="cpu",
    enable_mkldnn=False
)


def extract_text(image_path):

    results = ocr.predict(
        image_path
    )

    extracted_text = []

    for result in results:

        data = result.json

        # Current PaddleOCR result structure
        if "res" in data:
            data = data["res"]

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


        for text, score, box in zip(
            texts,
            scores,
            boxes
        ):

            # PaddleOCR may return lists or
            # numpy arrays depending on version.
            if hasattr(box, "tolist"):
                box = box.tolist()

            extracted_text.append({

                "text": str(text),

                "confidence": float(
                    score
                ),

                "box": box
            })


    return extracted_text


if __name__ == "__main__":

    image_path = (
        "dataset/mixed/mixed_05.png"
    )

    results = extract_text(
        image_path
    )

    print(
        "\n========== OCR RESULTS ==========\n"
    )

    for result in results:

        print(
            "Text:",
            result["text"]
        )

        print(
            "Confidence:",
            round(
                result["confidence"],
                4
            )
        )

        print(
            "Box:",
            result["box"]
        )

        print("--------------------------------")