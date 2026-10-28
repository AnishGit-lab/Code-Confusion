"""
PriviScan Week 9 Test Dataset Generator

Creates a small controlled test dataset for final system validation.

The generated images are synthetic and contain no real personal information.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "week9_tests"


def get_font(size=28):
    """Try to load a Windows font, otherwise use PIL default."""
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ]

    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except Exception:
                pass

    return ImageFont.load_default()


def create_image(filename, title, lines):
    """Create one synthetic test image."""
    width = 1200
    height = max(500, 180 + len(lines) * 70)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    title_font = get_font(34)
    body_font = get_font(28)

    y = 40

    draw.text(
        (40, y),
        title,
        fill="black",
        font=title_font,
    )

    y += 80

    for line in lines:
        draw.text(
            (40, y),
            line,
            fill="black",
            font=body_font,
        )
        y += 65

    output_path = OUTPUT_DIR / filename
    image.save(output_path)

    print(f"Created: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PRIVISCAN WEEK 9 TEST DATASET GENERATOR")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. SAFE IMAGE
    # ---------------------------------------------------------
    create_image(
        "01_safe.png",
        "PriviScan Safe Test",
        [
            "Welcome to PriviScan.",
            "This document contains general public information.",
            "No personal information is included.",
        ],
    )

    # ---------------------------------------------------------
    # 2. BASIC PERSONAL INFORMATION
    # ---------------------------------------------------------
    create_image(
        "02_basic_pii.png",
        "Personal Information Test",
        [
            "Name: Anish Kandel",
            "Email: anish@example.com",
            "Phone: +82 10-7777-8888",
            "Address: 123 Main Street",
        ],
    )

    # ---------------------------------------------------------
    # 3. ID INFORMATION
    # ---------------------------------------------------------
    create_image(
        "03_identity.png",
        "Identity Information Test",
        [
            "Name: Anish Kandel",
            "Student ID: 2024891090",
            "Employee ID: EMP202489",
        ],
    )

    # ---------------------------------------------------------
    # 4. FINANCIAL INFORMATION
    # ---------------------------------------------------------
    create_image(
        "04_financial.png",
        "Financial Information Test",
        [
            "Name: Anish Kandel",
            "Account Number: 26107010002015",
            "Card Number: 4111 1111 1111 1111",
        ],
    )

    # ---------------------------------------------------------
    # 5. CREDENTIALS
    # ---------------------------------------------------------
    create_image(
        "05_credentials.png",
        "Credential Security Test",
        [
            "Username: anish",
            "Password: MySecretPassword123",
            "API Key: sk-test-123456789",
        ],
    )

    # ---------------------------------------------------------
    # 6. MIXED HIGH-RISK INFORMATION
    # ---------------------------------------------------------
    create_image(
        "06_mixed_critical.png",
        "Mixed Sensitive Information Test",
        [
            "Name: Anish Kandel",
            "Email: anish@example.com",
            "Phone: +82 10-7777-8888",
            "Address: 123 Main Street",
            "Student ID: 2024891090",
            "Account Number: 26107010002015",
            "Card Number: 4111 1111 1111 1111",
            "Password: MySecretPassword123",
            "API Key: sk-test-123456789",
        ],
    )

    print()
    print("=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Location: {OUTPUT_DIR}")
    print()
    print("Generated test images:")

    for image_path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {image_path.name}")


if __name__ == "__main__":
    main()