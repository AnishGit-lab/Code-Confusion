from paddleocr import PaddleOCR


ocr = PaddleOCR(
    lang="en"
)


def extract_text(image_path):

    result = ocr.predict(image_path)

    return result