import cv2

from backend.risk.scoring import RISK_SCORES


# ============================================================
# QR DETECTION
# ============================================================

def detect_qr_codes(image):

    detections = []

    detector = cv2.QRCodeDetector()


    try:

        (
            success,
            decoded_info,
            points,
            _
        ) = detector.detectAndDecodeMulti(
            image
        )


        if success and points is not None:

            for index, point_set in enumerate(
                points
            ):

                points_array = point_set.astype(
                    int
                )


                x1 = int(
                    points_array[:, 0].min()
                )

                y1 = int(
                    points_array[:, 1].min()
                )

                x2 = int(
                    points_array[:, 0].max()
                )

                y2 = int(
                    points_array[:, 1].max()
                )


                value = "QR Code"


                if (
                    decoded_info
                    and index < len(decoded_info)
                ):

                    if decoded_info[index]:

                        value = decoded_info[index]


                detections.append({

                    "type": "QR",

                    "value": value,

                    "risk": RISK_SCORES["QR"],

                    "confidence": 1.0,

                    "box": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],

                    "source": "VISION"

                })


    except cv2.error:

        pass


    return detections


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(image):

    detections = []


    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )


    face_cascade = (
        cv2.CascadeClassifier(
            cascade_path
        )
    )


    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )


    for x, y, width, height in faces:

        detections.append({

            "type": "FACE",

            "value": "Face detected",

            "risk": RISK_SCORES["FACE"],

            "confidence": None,

            "box": [
                int(x),
                int(y),
                int(x + width),
                int(y + height)
            ],

            "source": "VISION"

        })


    return detections


# ============================================================
# MAIN VISION DETECTOR
# ============================================================

def detect_vision(image_path):

    image = cv2.imread(
        image_path
    )


    if image is None:

        raise ValueError(
            "Could not read image."
        )


    detections = []


    detections.extend(
        detect_qr_codes(
            image
        )
    )


    detections.extend(
        detect_faces(
            image
        )
    )


    return detections