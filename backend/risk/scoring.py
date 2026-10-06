RISK_SCORES = {
    "EMAIL": 40,
    "PHONE": 60,
    "ID_NUMBER": 30,
    "QR": 40,
    "FACE": 50,
    "GOVERNMENT_ID": 45,
    "PASSWORD": 100,
    "API_KEY": 100,
    "ADDRESS": 60,
    "NAME": 20,
    "DATE_OF_BIRTH": 45,
    "BANK_ACCOUNT": 60
}


def calculate_average_risk(detections):

    if not detections:
        return 0.0

    total = sum(
        detection["risk"]
        for detection in detections
    )

    return round(
        total / len(detections),
        2
    )