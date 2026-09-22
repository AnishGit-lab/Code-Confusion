def calculate_risk(detections):

    if not detections:
        return {
            "average_score": 0,
            "detections": []
        }

    scores = []

    for detection in detections:
        score = detection["risk"]
        scores.append(score)

        print(
            detection["type"],
            "→ Risk Score:",
            score
        )

    average_score = sum(scores) / len(scores)

    return {
        "average_score": average_score,
        "detections": detections
    }


detections = [
    {
        "type": "EMAIL",
        "value": "test@example.com",
        "risk": 40
    },
    {
        "type": "PHONE",
        "value": "010-1234-5678",
        "risk": 60
    },
    {
        "type": "STUDENT_ID",
        "value": "2024891090",
        "risk": 70
    }
]

result = calculate_risk(detections)

print("Average Risk Score:", round(result["average_score"], 2))