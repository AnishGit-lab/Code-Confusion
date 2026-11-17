"""
PriviScan regression tests for the rule detector (no OCR / GLiNER needed).

Run from the project root:
    python test_detection.py
Every line prints PASS or FAIL. Add your own cases at the bottom of CASES.
"""

from backend.detection.sensitive import detect_text, detect_from_ocr_results


def kinds(text):
    return sorted({d["type"] for d in detect_text(text)})


def values(text, kind):
    return [d["value"] for d in detect_text(text) if d["type"] == kind]


# (name, text, expected types that MUST be found, types that must NOT be found)
CASES = [
    ("labeled form",
     "Name: Anish Kandel\nEmail: a@b.com\nPhone: +82 10-1234-5678\nStudent ID: 2024891090",
     {"NAME", "EMAIL", "PHONE", "ID"}, set()),
    ("every field type in one document (digits shared between values)",
     "Name: Anish Kandel\nAddress: 123 Example Street, Seoul\nGovernment ID: P123456789\n"
     "Password: MySecretPassword123\nAPI Key: sk-test-123456789\nAccount Number: 123456789012",
     {"NAME", "ADDRESS", "GOVERNMENT_ID", "PASSWORD", "API_KEY", "ACCOUNT_NUMBER"}, set()),
    ("unlabeled email", "please write to anish@example.com today", {"EMAIL"}, set()),
    ("unlabeled phone", "010-1234-5678", {"PHONE"}, set()),
    ("unlabeled international phone", "+977 9841234567", {"PHONE"}, set()),
    ("unlabeled card (Luhn)", "4111 1111 1111 1111", {"BANK_CARD"}, set()),
    ("unlabeled api key", "sk-live-abcdef1234567890abcdef", {"API_KEY"}, set()),
    ("unlabeled student-style number", "2024891090", {"ID"}, set()),
    ("label without colon", "Student ID 2024891090", {"ID"}, set()),
    ("label on previous line", "Student ID\n2024891090", {"ID"}, set()),
    ("order number is NOT personal", "Order ID: 12345678", set(), {"ID", "ACCOUNT_NUMBER"}),
    ("invoice number is NOT personal", "Invoice ID: INV-2024-0099", set(), {"ID"}),
    ("date is NOT an id", "2024-05-12", set(), {"ID", "PHONE", "ACCOUNT_NUMBER"}),
    ("heading is NOT account data", "Account Information", set(), {"ACCOUNT_NUMBER"}),
    ("normal sentence", "The meeting is at 10:30 in room 204.", set(), {"ID", "PHONE", "NAME"}),
    ("CV: name with no label at the top of a document",
     "KANDEL ANISH\nIT STUDENT | ASPIRING SOFTWARE DEVELOPER\nStudent ID: 2024891090",
     {"NAME", "ID"}, set()),
    ("name after a title", "Patient seen by Dr. Anish Kandel yesterday.", {"NAME"}, set()),
    ("CV headings are NOT names",
     "PROFILE\nEDUCATION\nTECHNICAL SKILLS\nPROJECT EXPERIENCE\nStudent ID: 2024891090",
     {"ID"}, {"NAME"}),
    ("job titles are NOT names", "Software Engineer\nPython Developer\nStudent ID: 2024891090",
     {"ID"}, {"NAME"}),
    ("security notes are NOT passwords or keys",
     "Password guessing\nTrying likely passwords. Use strong passwords and attempt limits.\n"
     "Secret key\nNever share your secret key.\nWeak password policy: allow short passwords and never expire them.\n"
     "Enter your PIN code to continue.\nAPI key leak\nNever commit your API key.",
     set(), {"PASSWORD", "API_KEY", "SECRET", "NAME"}),
    ("real credentials are still found",
     "Password: MySecretPassword123\nWi-Fi password: Home1234\nPIN: 4821\nAPI Key: sk-test-123456789",
     {"PASSWORD", "API_KEY"}, set()),
    ("password on the next line is still found", "Password\nHunter2!x", {"PASSWORD"}, set()),
    ("CSV columns", "Name,Email,Phone,Student ID\nAnish Kandel,a@b.com,+82 10-1234-5678,2024891090",
     {"NAME", "EMAIL", "PHONE", "ID"}, set()),
]

passed = 0
for name, text, must, must_not in CASES:
    found = set(kinds(text))
    ok = must <= found and not (must_not & found)
    passed += ok
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   found={sorted(found)}"))

# The same email in two places of an image must stay two detections (two boxes to blur).
ocr = [
    {"text": "Email: a@b.com", "confidence": 0.98, "box": [0, 0, 300, 30]},
    {"text": "Contact: a@b.com", "confidence": 0.98, "box": [0, 40, 300, 70]},
]
two = len([d for d in detect_from_ocr_results(ocr) if d["type"] == "EMAIL"]) == 2
passed += two
print(("PASS  " if two else "FAIL  ") + "same email in two places keeps two boxes")

# ---- PDF / text path WITH the AI model (a fake model, so no download needed) ----
import io, contextlib


def model_tests():
    import backend.detection.ai_detector as ai

    class FakeModel:
        def batch_predict_entities(self, texts, labels, threshold=0.5, **kw):
            out = []
            for text in texts:
                found = []
                if text == "Smishing":
                    found.append({"label": "person name", "text": "Smishing", "score": 0.67})
                if text.startswith("Maria Garcia"):
                    found.append({"label": "person name", "text": "Maria Garcia", "score": 0.93})
                if text.startswith("OCR / Computer Vision"):
                    found.append({"label": "person name", "text": "Pillow", "score": 0.90})
                out.append(found)
            return out

    ai._GLINER_MODEL = FakeModel()
    ai.get_gliner_model = lambda: ai._GLINER_MODEL
    text = "Smishing\nOCR / Computer Vision: PaddleOCR, OpenCV, Pillow\nMaria Garcia is a student here.\nStudent ID: 2024891090"
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        found = ai.detect_text_with_ai(text)
    names = [d["value"] for d in found if d["type"] == "NAME"]
    results = [
        ("model path ran without an error", "ERROR" not in log.getvalue() and "Traceback" not in log.getvalue()),
        ("confident two-word name from the model is found", "Maria Garcia" in names),
        ("one-word heading 'Smishing' is NOT a name", "Smishing" not in names),
        ("tool name 'Pillow' is NOT a name", "Pillow" not in names),
    ]
    return results


model_results = model_tests()
for name, ok in model_results:
    passed += ok
    print(("PASS  " if ok else "FAIL  ") + name)

total = len(CASES) + 1 + len(model_results)
print(f"\n{passed}/{total} passed")