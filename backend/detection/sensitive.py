import re


EMAIL_PATTERN = r"""
\b[A-Za-z0-9._%+-]+
@
[A-Za-z0-9.-]+
\.[A-Za-z]{2,}
\b
"""

PHONE_PATTERN = r"""
\b
(?:010[- ]?\d{4}[- ]?\d{4})
\b
"""


def detect_email(text):

    return re.findall(
        EMAIL_PATTERN,
        text,
        re.VERBOSE
    )


def detect_phone(text):

    return re.findall(
        PHONE_PATTERN,
        text,
        re.VERBOSE
    )


def detect_sensitive_information(text):

    results = []

    emails = detect_email(text)

    for email in emails:
        results.append({
            "type": "EMAIL",
            "value": email,
            "risk": 40
        })

    phones = detect_phone(text)

    for phone in phones:
        results.append({
            "type": "PHONE",
            "value": phone,
            "risk": 60
        })

    return results