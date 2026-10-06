import re

import phonenumbers
from phonenumbers import PhoneNumberMatcher

from backend.risk.scoring import RISK_SCORES


# ============================================================
# DETECTION PATTERNS
# ============================================================

# ------------------------------------------------------------
# EMAIL
# ------------------------------------------------------------
EMAIL_PATTERN = (
    r'\b[A-Za-z0-9._%+-]+@'
    r'[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
)


# ------------------------------------------------------------
# PHONE
# ------------------------------------------------------------
# International examples:
# +82 10 1234 5678
# +1 202-555-0123
# +44 20 7946 0958
# +91 98765 43210
#
# Local labeled examples:
# Phone: 010-7777-8888
# Mobile: 020 1234 5678
# Tel: 123-456-7890

PHONE_LABEL_PATTERN = (
    r'(?im)^\s*'
    r'(?:phone|mobile|telephone|tel|contact\s*number)'
    r'\s*[:\-]\s*'
    r'([0-9+ ()\-.]{7,25})'
    r'\s*$'
)


# ------------------------------------------------------------
# ID NUMBER
# ------------------------------------------------------------
# General identifier detection.
#
# Examples:
# Student ID: STU-2026-A123
# Staff ID: STAFF-7788
# Employee Number: EMP-00125
# Faculty ID: FAC-AB2048
# Member ID: MBR-A1029

ID_PATTERN = (
    r'(?im)^\s*'
    r'(?:student\s*(?:id|number)'
    r'|staff\s*(?:id|number)'
    r'|employee\s*(?:id|number)'
    r'|faculty\s*(?:id|number)'
    r'|member\s*(?:id|number)'
    r'|user\s*(?:id|number)'
    r'|registration\s*(?:id|number)'
    r'|id\s*(?:number|no))'
    r'\s*[:#-]?\s*'
    r'([A-Za-z0-9][A-Za-z0-9._/-]{1,39})'
    r'\s*$'
)


# ------------------------------------------------------------
# NAME
# ------------------------------------------------------------
NAME_PATTERN = (
    r'(?im)^\s*'
    r'(?:name|full\s*name|first\s*name|last\s*name)'
    r'\s*[:\-]\s*'
    r'([^\d\W_]+(?:[ \t]+[^\d\W_]+){0,3})'
    r'\s*$'
)


# ------------------------------------------------------------
# DATE OF BIRTH
# ------------------------------------------------------------
DOB_PATTERN = (
    r'(?im)^\s*'
    r'(?:date\s*of\s*birth|dob|birth\s*date)'
    r'\s*[:\-]\s*'
    r'('
    r'(?:\d{1,4}[\/\-.]\d{1,2}[\/\-.]\d{1,4})'
    r'|'
    r'(?:'
    r'(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)'
    r'\s+\d{1,2},?\s+\d{4}'
    r')'
    r'|'
    r'(?:'
    r'\d{1,2}\s+'
    r'(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)'
    r'\s+\d{4}'
    r')'
    r')'
    r'\s*$'
)


# ------------------------------------------------------------
# ADDRESS
# ------------------------------------------------------------
ADDRESS_PATTERN = (
    r'(?im)^\s*'
    r'(?:address|home\s*address|residential\s*address)'
    r'\s*[:\-]\s*'
    r'([^\r\n]{5,150})'
    r'\s*$'
)


# ------------------------------------------------------------
# GOVERNMENT ID
# ------------------------------------------------------------
GOVERNMENT_ID_PATTERN = (
    r'(?im)^\s*'
    r'(?:government\s*id'
    r'|national\s*id'
    r'|identity\s*number'
    r'|passport\s*(?:number|no)'
    r'|driving\s*license\s*(?:number|no)'
    r'|driver\s*license\s*(?:number|no))'
    r'\s*[:#-]?\s*'
    r'([A-Za-z0-9][A-Za-z0-9./-]{2,39})'
    r'\s*$'
)


# ------------------------------------------------------------
# PASSWORD
# ------------------------------------------------------------
PASSWORD_PATTERN = (
    r'(?im)^\s*'
    r'(?:password|passcode|passwd)'
    r'\s*[:=]\s*'
    r'(\S+)'
    r'\s*$'
)


# ------------------------------------------------------------
# API KEY / SECRET
# ------------------------------------------------------------
API_KEY_PATTERN = (
    r'(?im)^\s*'
    r'(?:api\s*key|api\s*token|secret\s*key|access\s*token)'
    r'\s*[:=]\s*'
    r'([A-Za-z0-9_\-./+=]{8,})'
    r'\s*$'
)


# ------------------------------------------------------------
# BANK / ACCOUNT NUMBER
# ------------------------------------------------------------
BANK_ACCOUNT_PATTERN = (
    r'(?im)^\s*'
    r'(?:bank\s*account|account\s*number|account\s*no)'
    r'\s*[:#-]?\s*'
    r'([A-Za-z0-9][A-Za-z0-9 -]{3,33})'
    r'\s*$'
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def add_detection(
    detections,
    detection_type,
    value
):
    detections.append({
        "type": detection_type,
        "value": value.strip(),
        "risk": RISK_SCORES[detection_type]
    })


# ============================================================
# MAIN DETECTION FUNCTION
# ============================================================

def detect_sensitive_information(text):

    detections = []


    # ========================================================
    # EMAIL
    # ========================================================

    emails = re.findall(
        EMAIL_PATTERN,
        text
    )

    for email in emails:

        add_detection(
            detections,
            "EMAIL",
            email
        )


    # ========================================================
    # PHONE
    # ========================================================

    phone_values = []


    # ----------------------------------------
    # International / recognizable numbers
    # ----------------------------------------

    try:

        for match in PhoneNumberMatcher(
            text,
            None
        ):

            phone_number = match.number

            if phonenumbers.is_possible_number(
                phone_number
            ):

                phone_values.append(
                    match.raw_string.strip()
                )

    except Exception:

        pass


    # ----------------------------------------
    # Labeled local numbers
    # ----------------------------------------

    labeled_phones = re.findall(
        PHONE_LABEL_PATTERN,
        text
    )

    for phone in labeled_phones:

        phone = phone.strip()

        digits_only = re.sub(
            r'\D',
            '',
            phone
        )

        if len(digits_only) >= 7:

            phone_values.append(
                phone
            )


    # ----------------------------------------
    # Remove duplicate phone numbers
    # ----------------------------------------

    seen_phones = set()

    for phone in phone_values:

        normalized = re.sub(
            r'\D',
            '',
            phone
        )

        if normalized in seen_phones:
            continue

        seen_phones.add(normalized)

        add_detection(
            detections,
            "PHONE",
            phone
        )


    # ========================================================
    # ID NUMBER
    # ========================================================

    id_matches = re.findall(
        ID_PATTERN,
        text
    )

    for identifier in id_matches:

        add_detection(
            detections,
            "ID_NUMBER",
            identifier
        )


    # ========================================================
    # NAME
    # ========================================================

    names = re.findall(
        NAME_PATTERN,
        text
    )

    for name in names:

        add_detection(
            detections,
            "NAME",
            name
        )


    # ========================================================
    # DATE OF BIRTH
    # ========================================================

    dates_of_birth = re.findall(
        DOB_PATTERN,
        text
    )

    for date_of_birth in dates_of_birth:

        add_detection(
            detections,
            "DATE_OF_BIRTH",
            date_of_birth
        )


    # ========================================================
    # ADDRESS
    # ========================================================

    addresses = re.findall(
        ADDRESS_PATTERN,
        text
    )

    for address in addresses:

        add_detection(
            detections,
            "ADDRESS",
            address
        )


    # ========================================================
    # GOVERNMENT ID
    # ========================================================

    government_ids = re.findall(
        GOVERNMENT_ID_PATTERN,
        text
    )

    for government_id in government_ids:

        add_detection(
            detections,
            "GOVERNMENT_ID",
            government_id
        )


    # ========================================================
    # PASSWORD
    # ========================================================

    passwords = re.findall(
        PASSWORD_PATTERN,
        text
    )

    for password in passwords:

        add_detection(
            detections,
            "PASSWORD",
            password
        )


    # ========================================================
    # API KEY
    # ========================================================

    api_keys = re.findall(
        API_KEY_PATTERN,
        text
    )

    for api_key in api_keys:

        add_detection(
            detections,
            "API_KEY",
            api_key
        )


    # ========================================================
    # BANK / ACCOUNT NUMBER
    # ========================================================

    bank_accounts = re.findall(
        BANK_ACCOUNT_PATTERN,
        text
    )

    for account in bank_accounts:

        add_detection(
            detections,
            "BANK_ACCOUNT",
            account
        )


    return detections


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_text = """
Name: Anish Kandel

Email: test@example.com

Phone: +82 10 1234 5678
Phone: +1 202-555-0123
Phone: +44 20 7946 0958
Phone: +91 98765 43210
Phone: +81 90 1234 5678
Phone: 010-7777-8888

Student ID: STU-2026-A123
Staff ID: STAFF-7788
Employee Number: EMP-00125
Faculty ID: FAC-AB2048
Member ID: MBR-A1029

Date of Birth: 2001-01-15

Address: 123 Main Street, London

Government ID: GOV-AB12345

Password: TestPassword123!

API Key: sk_test_123456789

Bank Account: 123456789012
"""


    results = detect_sensitive_information(
        test_text
    )


    print(
        "\n========== PRIVISCAN DETECTION ==========\n"
    )


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

        print("--------------------------------")


    print(
        "\nTotal Detections:",
        len(results)
    )


    if results:

        total_risk = sum(
            result["risk"]
            for result in results
        )

        average_risk = (
            total_risk / len(results)
        )

        print(
            "Average Risk Score:",
            round(
                average_risk,
                2
            )
        )