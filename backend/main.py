from pathlib import Path
from uuid import uuid4
import html
import shutil

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form
)

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.analyze import analyze_image
from backend.risk.scoring import (
    calculate_average_risk
)
from backend.protection import (
    protect_image
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="PriviScan"
)


# ============================================================
# UPLOAD DIRECTORY
# ============================================================

UPLOAD_DIR = Path(
    "uploads"
)

UPLOAD_DIR.mkdir(
    exist_ok=True
)


app.mount(
    "/uploads",
    StaticFiles(
        directory=str(
            UPLOAD_DIR
        )
    ),
    name="uploads"
)


# ============================================================
# HOME
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <title>PriviScan</title>

        <style>

            body {
                font-family: Arial, sans-serif;
                max-width: 850px;
                margin: 60px auto;
                padding: 20px;
                text-align: center;
            }

            .box {
                border: 2px dashed #888;
                padding: 45px;
                border-radius: 15px;
            }

            input {
                margin: 20px;
                font-size: 16px;
            }

            button {
                padding: 12px 25px;
                font-size: 16px;
                cursor: pointer;
            }

        </style>

    </head>


    <body>

        <h1>
            PriviScan
        </h1>

        <p>
            Intelligent Privacy Risk Assessment
        </p>


        <div class="box">

            <form
                action="/analyze"
                method="post"
                enctype="multipart/form-data"
            >

                <input
                    type="file"
                    name="file"
                    accept="image/*"
                    required
                >

                <br>

                <button type="submit">
                    Analyze Image
                </button>

            </form>

        </div>

    </body>

    </html>
    """


# ============================================================
# ANALYZE
# ============================================================

@app.post(
    "/analyze",
    response_class=HTMLResponse
)
async def analyze(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------------

    if (
        not file.content_type
        or not file.content_type.startswith(
            "image/"
        )
    ):

        return """

        <h1>
            PriviScan
        </h1>

        <p>
            Please upload a JPG or PNG image.
        </p>

        <a href="/">
            Go Back
        </a>

        """


    # --------------------------------------------------------
    # FILE NAME
    # --------------------------------------------------------

    original_name = Path(
        file.filename
    ).name


    stored_name = (
        uuid4().hex
        + "_"
        + original_name
    )


    file_path = (
        UPLOAD_DIR
        / stored_name
    )


    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    with file_path.open(
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )


    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    try:

        results = analyze_image(
            str(file_path)
        )


        average_score = (
            calculate_average_risk(
                results
            )
        )


    except Exception as error:

        return f"""

        <h1>
            PriviScan Error
        </h1>

        <p>
            {html.escape(
                str(error)
            )}
        </p>

        <a href="/">
            Try Again
        </a>

        """


    # ========================================================
    # RESULT PAGE
    # ========================================================

    page = f"""

    <!DOCTYPE html>

    <html>

    <head>

        <title>
            PriviScan Result
        </title>


        <style>

            body {{
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 40px auto;
                padding: 20px;
            }}

            h1 {{
                text-align: center;
            }}

            .summary {{
                background: #f2f2f2;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                margin: 25px 0;
            }}

            .detection {{
                border: 1px solid #ddd;
                padding: 18px;
                margin: 12px 0;
                border-radius: 10px;
            }}

            .protection {{
                margin-top: 30px;
                border: 2px solid #888;
                padding: 25px;
                border-radius: 10px;
            }}

            .method {{
                margin-top: 20px;
            }}

            button {{
                padding: 12px 22px;
                cursor: pointer;
                margin-top: 15px;
                font-size: 16px;
            }}

            select {{
                padding: 8px;
                font-size: 16px;
            }}

        </style>

    </head>


    <body>

        <h1>
            PriviScan Analysis
        </h1>


        <p>

            <strong>
                File:
            </strong>

            {html.escape(
                original_name
            )}

        </p>


        <div class="summary">

            <h2>
                Average Risk Score:
                {average_score:.2f}
            </h2>


            <p>

                Total Detections:
                {len(results)}

            </p>

        </div>

    """


    # ========================================================
    # DETECTIONS
    # ========================================================

    if results:

        page += """

        <h2>
            Detected Sensitive Information
        </h2>


        <form
            action="/protect"
            method="post"
        >


        """


        page += f"""

        <input
            type="hidden"
            name="filename"
            value="{html.escape(
                stored_name
            )}"
        >

        """


        for index, result in enumerate(
            results
        ):

            confidence_html = ""


            if (
                result.get(
                    "confidence"
                ) is not None
            ):

                confidence_html = f"""

                <p>

                    <strong>
                        Confidence:
                    </strong>

                    {result["confidence"]:.4f}

                </p>

                """


            page += f"""

            <div class="detection">

                <label>

                    <input
                        type="checkbox"
                        name="selected_ids"
                        value="{index}"
                    >

                    <strong>
                        Select
                    </strong>

                </label>


                <p>

                    <strong>
                        Type:
                    </strong>

                    {html.escape(
                        str(
                            result["type"]
                        )
                    )}

                </p>


                <p>

                    <strong>
                        Value:
                    </strong>

                    {html.escape(
                        str(
                            result["value"]
                        )
                    )}

                </p>


                <p>

                    <strong>
                        Risk Score:
                    </strong>

                    {result["risk"]}

                </p>


                {confidence_html}


                <p>

                    <strong>
                        Box:
                    </strong>

                    {html.escape(
                        str(
                            result["box"]
                        )
                    )}

                </p>

            </div>

            """


        # ====================================================
        # PROTECTION
        # ====================================================

        page += """

        <div class="protection">

            <h2>
                Protection Options
            </h2>


            <p>
                Choose how PriviScan should
                protect the detected information.
            </p>


            <h3>
                Protection Mode
            </h3>


            <label>

                <input
                    type="radio"
                    name="mode"
                    value="automatic"
                    checked
                >

                Automatic

            </label>


            <br><br>


            <label>

                <input
                    type="radio"
                    name="mode"
                    value="manual"
                >

                Manual

            </label>


            <div class="method">

                <h3>
                    Protection Method
                </h3>


                <select
                    name="action"
                >

                    <option value="blur">
                        Blur
                    </option>


                    <option value="blackout">
                        Blackout
                    </option>


                    <option value="star">
                        Star (*)
                    </option>


                    <option value="random">
                        Random Alphabet
                    </option>


                    <option value="hashtag">
                        Hashtag (#)
                    </option>

                </select>

            </div>


            <br>


            <button
                type="submit"
            >

                Protect Image

            </button>


        </div>


        </form>

        """


    else:

        page += """

        <div class="summary">

            <h2>
                No sensitive information detected.
            </h2>

        </div>

        """


    page += """

        <br>

        <p style="text-align:center;">

            <a href="/">
                Analyze Another Image
            </a>

        </p>


    </body>

    </html>

    """


    return page


# ============================================================
# PROTECT IMAGE
# ============================================================

@app.post(
    "/protect",
    response_class=HTMLResponse
)
async def protect(
    filename: str = Form(...),
    mode: str = Form(...),
    action: str = Form(...),
    selected_ids: list[int] | None = Form(None)
):

    filename = Path(
        filename
    ).name


    input_path = (
        UPLOAD_DIR
        / filename
    )


    if not input_path.exists():

        return """

        <h1>
            PriviScan
        </h1>

        <p>
            Original image was not found.
        </p>

        <a href="/">
            Go Back
        </a>

        """


    try:

        # Analyze image again
        detections = analyze_image(
            str(input_path)
        )


        protected_name = (
            "protected_"
            + uuid4().hex[:8]
            + "_"
            + Path(filename).name
        )


        output_path = (
            UPLOAD_DIR
            / protected_name
        )


        protect_image(
            str(input_path),
            str(output_path),
            detections,
            mode=mode,
            action=action,
            selected_ids=selected_ids
        )


    except Exception as error:

        return f"""

        <h1>
            PriviScan Protection Error
        </h1>

        <p>
            {html.escape(
                str(error)
            )}
        </p>

        <a href="/">
            Go Back
        </a>

        """


    return f"""

    <!DOCTYPE html>

    <html>

    <head>

        <title>
            PriviScan Protected Image
        </title>


        <style>

            body {{
                font-family: Arial, sans-serif;
                max-width: 950px;
                margin: 40px auto;
                padding: 20px;
                text-align: center;
            }}

            img {{
                max-width: 100%;
                border: 1px solid #ddd;
                margin: 20px 0;
            }}

        </style>

    </head>


    <body>

        <h1>
            Protected Image
        </h1>


        <p>

            <strong>
                Mode:
            </strong>

            {html.escape(
                mode
            )}

        </p>


        <p>

            <strong>
                Protection Method:
            </strong>

            {html.escape(
                action
            )}

        </p>


        <img
            src="/uploads/{html.escape(
                protected_name
            )}"
            alt="Protected Image"
        >


        <br>


        <a
            href="/uploads/{html.escape(
                protected_name
            )}"
            target="_blank"
        >

            Open Protected Image

        </a>


        <br><br>


        <a href="/">
            Analyze Another Image
        </a>


    </body>

    </html>

    """