from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path("dataset")

CATEGORIES = [
    "safe",
    "email",
    "phone",
    "student_id",
    "qr",
    "face",
    "mixed"
]

for category in CATEGORIES:
    (BASE_DIR / category).mkdir(parents=True, exist_ok=True)


# Try to use Arial. If unavailable, use Pillow's default font.
try:
    FONT = ImageFont.truetype("arial.ttf", 30)
    SMALL_FONT = ImageFont.truetype("arial.ttf", 24)
except OSError:
    FONT = ImageFont.load_default()
    SMALL_FONT = ImageFont.load_default()


def create_text_image(category, filename, lines):
    image = Image.new("RGB", (1000, 600), "white")
    draw = ImageDraw.Draw(image)

    y = 60

    for line in lines:
        draw.text(
            (60, y),
            line,
            fill="black",
            font=FONT
        )
        y += 80

    file_path = BASE_DIR / category / filename
    image.save(file_path)

    print(f"Created: {file_path}")


# -------------------------
# SAFE — 5 images
# -------------------------

safe_images = [
    ["Welcome to PriviScan", "Normal project document"],
    ["Weekly Meeting", "Meeting starts at 10:00 AM"],
    ["Computer Science", "Capstone Design Project"],
    ["Weekly Report", "Week 3 discussion"],
    ["Normal Document", "No sensitive information"]
]

for i, lines in enumerate(safe_images, 1):
    create_text_image(
        "safe",
        f"safe_{i:02d}.png",
        lines
    )


# -------------------------
# EMAIL — 4 images
# -------------------------

email_images = [
    ["Contact Information", "Email: test@example.com"],
    ["Student Contact", "Email: student01@example.com"],
    ["Project Team", "Email: team@example.com"],
    ["Support Contact", "Email: support@example.com"]
]

for i, lines in enumerate(email_images, 1):
    create_text_image(
        "email",
        f"email_{i:02d}.png",
        lines
    )


# -------------------------
# PHONE — 4 images
# -------------------------

phone_images = [
    ["Contact Information", "Phone: 010-1234-5678"],
    ["Emergency Contact", "Phone: 010-5555-1111"],
    ["Student Contact", "Phone: 010-2222-3333"],
    ["Office Contact", "Phone: 010-9876-5432"]
]

for i, lines in enumerate(phone_images, 1):
    create_text_image(
        "phone",
        f"phone_{i:02d}.png",
        lines
    )


# -------------------------
# STUDENT ID — 4 images
# -------------------------

student_id_images = [
    ["Student Information", "Student ID: 2024001001"],
    ["University Record", "Student ID: 2024001002"],
    ["Registration", "Student ID: 2024001003"],
    ["Student Card", "Student ID: 2024001004"]
]

for i, lines in enumerate(student_id_images, 1):
    create_text_image(
        "student_id",
        f"student_id_{i:02d}.png",
        lines
    )


# -------------------------
# QR — 4 test images
# -------------------------

qr_images = [
    ["QR CODE TEST 01", "https://priviscan.test/01"],
    ["QR CODE TEST 02", "https://priviscan.test/02"],
    ["QR CODE TEST 03", "https://priviscan.test/03"],
    ["QR CODE TEST 04", "https://priviscan.test/04"]
]

for i, lines in enumerate(qr_images, 1):
    create_text_image(
        "qr",
        f"qr_{i:02d}.png",
        lines
    )


# -------------------------
# FACE — 4 synthetic images
# -------------------------

def create_face_image(filename):
    image = Image.new("RGB", (1000, 600), "white")
    draw = ImageDraw.Draw(image)

    # Head
    draw.ellipse(
        (350, 80, 650, 380),
        outline="black",
        width=5
    )

    # Eyes
    draw.ellipse(
        (420, 170, 450, 200),
        fill="black"
    )

    draw.ellipse(
        (550, 170, 580, 200),
        fill="black"
    )

    # Nose
    draw.line(
        (500, 210, 480, 270, 510, 275),
        fill="black",
        width=4
    )

    # Mouth
    draw.arc(
        (450, 260, 550, 340),
        0,
        180,
        fill="black",
        width=4
    )

    draw.text(
        (390, 450),
        "Synthetic Face Sample",
        fill="black",
        font=FONT
    )

    file_path = BASE_DIR / "face" / filename
    image.save(file_path)

    print(f"Created: {file_path}")


for i in range(1, 5):
    create_face_image(f"face_{i:02d}.png")


# -------------------------
# MIXED — 5 images
# -------------------------

mixed_images = [
    [
        "Student Information",
        "Email: mixed01@example.com",
        "Phone: 010-1111-2222"
    ],
    [
        "Project Member",
        "Student ID: 2024002001",
        "Email: mixed02@example.com"
    ],
    [
        "Contact Record",
        "Phone: 010-3333-4444",
        "Student ID: 2024002002"
    ],
    [
        "User Information",
        "Email: mixed04@example.com",
        "Phone: 010-5555-6666",
        "Student ID: 2024002003"
    ],
    [
        "Privacy Test",
        "Email: mixed05@example.com",
        "Phone: 010-7777-8888",
        "Student ID: 2024002004"
    ]
]

for i, lines in enumerate(mixed_images, 1):
    create_text_image(
        "mixed",
        f"mixed_{i:02d}.png",
        lines
    )


print()
print("====================================")
print("30 synthetic dataset images created")
print("====================================")