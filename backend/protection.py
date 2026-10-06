import cv2
import random
import string


# ============================================================
# GET BOX
# ============================================================

def get_box(detection):

    box = detection.get(
        "box"
    )

    if not box:
        return None

    if len(box) != 4:
        return None


    x1, y1, x2, y2 = box


    return (
        max(0, int(x1)),
        max(0, int(y1)),
        max(0, int(x2)),
        max(0, int(y2))
    )


# ============================================================
# BLUR
# ============================================================

def apply_blur(
    image,
    box
):

    x1, y1, x2, y2 = box


    if x2 <= x1 or y2 <= y1:
        return


    region = image[
        y1:y2,
        x1:x2
    ]


    if region.size == 0:
        return


    blurred = cv2.GaussianBlur(
        region,
        (31, 31),
        0
    )


    image[
        y1:y2,
        x1:x2
    ] = blurred


# ============================================================
# BLACKOUT
# ============================================================

def apply_blackout(
    image,
    box
):

    x1, y1, x2, y2 = box


    if x2 <= x1 or y2 <= y1:
        return


    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        -1
    )


# ============================================================
# BACKGROUND COLOR
# ============================================================

def get_background_color(
    image,
    box
):

    x1, y1, x2, y2 = box


    height, width = (
        image.shape[:2]
    )


    samples = []


    # Above
    if y1 > 2:

        samples.append(
            image[
                y1 - 2:y1,
                x1:x2
            ]
        )


    # Below
    if y2 < height - 2:

        samples.append(
            image[
                y2:y2 + 2,
                x1:x2
            ]
        )


    # Left
    if x1 > 2:

        samples.append(
            image[
                y1:y2,
                x1 - 2:x1
            ]
        )


    # Right
    if x2 < width - 2:

        samples.append(
            image[
                y1:y2,
                x2:x2 + 2
            ]
        )


    if not samples:
        return (
            255,
            255,
            255
        )


    pixels = []


    for sample in samples:

        if sample.size > 0:

            pixels.append(
                sample.reshape(
                    -1,
                    3
                )
            )


    if not pixels:

        return (
            255,
            255,
            255
        )


    pixels = cv2.vconcat(
        pixels
    )


    average = pixels.mean(
        axis=0
    )


    return tuple(
        int(value)
        for value in average
    )


# ============================================================
# GET REPLACEMENT LENGTH
# ============================================================

def get_mask_length(
    original_text
):

    if not original_text:
        return 8


    return max(
        4,
        len(original_text)
    )


# ============================================================
# GENERATE RANDOM ALPHABET
# ============================================================

def generate_random_text(
    length
):

    return "".join(
        random.choice(
            string.ascii_uppercase
        )
        for _ in range(length)
    )


# ============================================================
# DRAW REPLACEMENT TEXT
# ============================================================

def draw_replacement_text(
    image,
    box,
    replacement_text
):

    x1, y1, x2, y2 = box


    if x2 <= x1 or y2 <= y1:
        return


    box_width = x2 - x1
    box_height = y2 - y1


    font = cv2.FONT_HERSHEY_SIMPLEX


    font_scale = max(
        0.35,
        min(
            1.2,
            box_height / 35
        )
    )


    thickness = max(
        1,
        int(
            box_height / 15
        )
    )


    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        replacement_text,
        font,
        font_scale,
        thickness
    )


    # Reduce size until text fits
    while (
        text_width > box_width - 6
        and font_scale > 0.2
    ):

        font_scale -= 0.05


        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            replacement_text,
            font,
            font_scale,
            thickness
        )


    # If still too large, shorten it
    if text_width > box_width - 6:

        max_chars = max(
            1,
            int(
                len(replacement_text)
                * (box_width / text_width)
            )
        )


        replacement_text = (
            replacement_text[:max_chars]
        )


        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            replacement_text,
            font,
            font_scale,
            thickness
        )


    text_x = x1 + 3


    text_y = (
        y1
        + (
            (
                box_height
                + text_height
            ) // 2
        )
    )


    cv2.putText(
        image,
        replacement_text,
        (
            text_x,
            text_y
        ),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# APPLY TEXT REPLACEMENT
# ============================================================

def apply_text_replacement(
    image,
    box,
    original_text,
    action
):

    x1, y1, x2, y2 = box


    background = get_background_color(
        image,
        box
    )


    # Cover original text
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        background,
        -1
    )


    length = get_mask_length(
        original_text
    )


    # STAR
    if action == "star":

        replacement = (
            "*" * length
        )


    # HASHTAG
    elif action == "hashtag":

        replacement = (
            "#" * length
        )


    # RANDOM ALPHABET
    elif action == "random":

        replacement = (
            generate_random_text(
                length
            )
        )


    else:

        replacement = (
            "*" * length
        )


    draw_replacement_text(
        image,
        box,
        replacement
    )


# ============================================================
# MAIN PROTECTION FUNCTION
# ============================================================

def protect_image(
    image_path,
    output_path,
    detections,
    mode="automatic",
    action="blur",
    selected_ids=None
):

    image = cv2.imread(
        image_path
    )


    if image is None:

        raise ValueError(
            "Could not open image."
        )


    selected_ids = (
        selected_ids
        if selected_ids
        else []
    )


    for index, detection in enumerate(
        detections
    ):

        # ====================================================
        # AUTOMATIC MODE
        # ====================================================

        if mode == "automatic":

            should_protect = True


        # ====================================================
        # MANUAL MODE
        # ====================================================

        elif mode == "manual":

            should_protect = (
                index in selected_ids
            )


        else:

            should_protect = False


        if not should_protect:
            continue


        box = get_box(
            detection
        )


        # No box = cannot protect image region
        if box is None:
            continue


        original_text = str(
            detection.get(
                "value",
                ""
            )
        )


        # ====================================================
        # BLUR
        # ====================================================

        if action == "blur":

            apply_blur(
                image,
                box
            )


        # ====================================================
        # BLACKOUT
        # ====================================================

        elif action == "blackout":

            apply_blackout(
                image,
                box
            )


        # ====================================================
        # STAR
        # ====================================================

        elif action == "star":

            apply_text_replacement(
                image,
                box,
                original_text,
                "star"
            )


        # ====================================================
        # RANDOM ALPHABET
        # ====================================================

        elif action == "random":

            apply_text_replacement(
                image,
                box,
                original_text,
                "random"
            )


        # ====================================================
        # HASHTAG
        # ====================================================

        elif action == "hashtag":

            apply_text_replacement(
                image,
                box,
                original_text,
                "hashtag"
            )


    # ========================================================
    # SAVE
    # ========================================================

    success = cv2.imwrite(
        output_path,
        image
    )


    if not success:

        raise ValueError(
            "Could not save protected image."
        )


    return output_path