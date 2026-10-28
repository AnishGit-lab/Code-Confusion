"""
PDF extraction helpers for PriviScan.

Text PDFs are read with PyMuPDF. Pages without usable embedded text are
rendered to images and passed to the project's existing PaddleOCR detector.
All returned bounding boxes use PDF page coordinates (points), not pixels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import tempfile

import fitz  # PyMuPDF


def _rect_list(rect: Any) -> List[float]:
    return [
        round(float(rect.x0), 2),
        round(float(rect.y0), 2),
        round(float(rect.x1), 2),
        round(float(rect.y1), 2),
    ]


def _ocr_page(page: Any, page_number: int, scale: float = 2.0) -> Dict[str, Any]:
    """OCR a rendered PDF page and convert image boxes back to PDF points."""
    from backend.ocr.detector import extract_text

    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)

    with tempfile.TemporaryDirectory(prefix="priviscan_pdf_ocr_") as tmp:
        image_path = Path(tmp) / f"page_{page_number}.png"
        pix.save(str(image_path))
        ocr_items = extract_text(str(image_path))

    cleaned = []

    for item in ocr_items or []:
        text = str(item.get("text", "")).strip()
        if not text:
            continue

        raw_box = item.get("box") or item.get("bbox")
        box = None

        if isinstance(raw_box, (list, tuple)) and len(raw_box) >= 4:
            try:
                box = [
                    round(float(value) / scale, 2)
                    for value in raw_box[:4]
                ]
            except (TypeError, ValueError):
                box = None

        try:
            confidence = float(item.get("confidence", 0.8) or 0.8)
        except (TypeError, ValueError):
            confidence = 0.8

        cleaned.append({
            "text": text,
            "confidence": max(0.0, min(1.0, confidence)),
            "box": box,
            "page": page_number,
            "source": "ocr",
        })

    return {
        "page": page_number,
        "text": "\n".join(item["text"] for item in cleaned),
        "source": "ocr",
        "ocr_items": cleaned,
    }


def extract_pdf_pages(
    pdf_path: str | Path,
    ocr: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extract page text from a PDF.

    Returns page number, text, source, and OCR line boxes when applicable.
    The source PDF is not modified.
    """
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("PriviScan currently accepts PDF files only.")

    pages: List[Dict[str, Any]] = []

    try:
        document = fitz.open(str(path))
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc

    try:
        if document.is_encrypted and not document.authenticate(""):
            raise ValueError(
                "This PDF is password-protected. Please upload an unlocked PDF."
            )

        for page_index, page in enumerate(document):
            page_number = page_index + 1
            text = (page.get_text("text") or "").strip()

            # Pages with very little embedded text may be scanned pages.
            if ocr and len(text) < 30:
                record = _ocr_page(page, page_number)

                if record["text"].strip():
                    pages.append(record)
                    continue

            pages.append({
                "page": page_number,
                "text": text,
                "source": "embedded_text",
                "ocr_items": [],
            })

    finally:
        document.close()

    return pages


def locate_text_boxes(
    pdf_path: str | Path,
    page_number: int,
    value: str,
) -> List[List[float]]:
    """Locate an exact detected value on a text-based PDF page."""
    if not value:
        return []

    boxes: List[List[float]] = []

    with fitz.open(str(pdf_path)) as document:
        if page_number < 1 or page_number > len(document):
            return []

        page = document[page_number - 1]

        try:
            matches = page.search_for(value, quads=False)
        except Exception:
            matches = []

        for rect in matches:
            boxes.append(_rect_list(rect))

    return boxes