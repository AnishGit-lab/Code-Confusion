from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from backend.analyze import analyze_image, analyze_text
from backend.protection import protect_image
from backend.risk.scoring import calculate_average_risk


BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
PROTECTED_DIR = BASE_DIR / "protected"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROTECTED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


app = FastAPI(
    title="PriviScan",
    description="Privacy risk assessment and sensitive information protection system",
    version="1.0.0",
)


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PriviScan</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1100px;
            margin: 40px auto;
            padding: 20px;
        }

        h1 {
            margin-bottom: 5px;
        }

        .card {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }

        button {
            padding: 10px 18px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

        th, td {
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
        }

        pre {
            background: #f5f5f5;
            padding: 15px;
            overflow-x: auto;
        }
    </style>
</head>

<body>

<h1>PriviScan</h1>
<p>Privacy Risk Assessment and Sensitive Information Protection</p>

<div class="card">
    <h2>Upload Image</h2>

    <form id="uploadForm">
        <input
            type="file"
            id="file"
            name="file"
            accept="image/*"
            required
        >

        <button type="submit">
            Analyze
        </button>
    </form>
</div>

<div id="result"></div>

<script>

const form = document.getElementById("uploadForm");

form.addEventListener("submit", async function(event) {

    event.preventDefault();

    const file = document.getElementById("file").files[0];

    const data = new FormData();

    data.append("file", file);

    const response = await fetch("/analyze", {
        method: "POST",
        body: data
    });

    const result = await response.json();

    renderResult(result);
});


function renderResult(result) {

    const container = document.getElementById("result");

    let html = `
        <div class="card">
            <h2>Analysis Report</h2>

            <p>
                <strong>Final Risk:</strong>
                ${result.final_risk_score}
            </p>

            <p>
                <strong>Risk Level:</strong>
                ${result.risk_level}
            </p>

            <p>
                <strong>Total Detections:</strong>
                ${result.total_detections}
            </p>

            <p>
                <strong>Average Risk:</strong>
                ${result.average_risk_score}
            </p>
    `;

    if (!result.detections || result.detections.length === 0) {

        html += `
            <p>No sensitive information detected.</p>
        `;

    } else {

        html += `
            <table>
                <tr>
                    <th>Type</th>
                    <th>Value</th>
                    <th>Risk Score</th>
                    <th>Confidence</th>
                    <th>Box</th>
                </tr>
        `;

        for (const detection of result.detections) {

            html += `
                <tr>
                    <td>${escapeHtml(detection.type)}</td>
                    <td>${escapeHtml(detection.value)}</td>
                    <td>${detection.risk_score}</td>
                    <td>${detection.confidence}</td>
                    <td>${escapeHtml(
                        JSON.stringify(detection.box)
                    )}</td>
                </tr>
            `;
        }

        html += "</table>";
    }

    html += `
        </div>
    `;

    container.innerHTML = html;
}


function escapeHtml(value) {

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

</script>

</body>
</html>
"""


@app.get(
    "/",
    response_class=HTMLResponse,
)
def home():
    return HTML_PAGE


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "PriviScan",
    }


@app.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
):
    extension = Path(
        file.filename or ""
    ).suffix.lower()

    allowed = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
    }

    if extension not in allowed:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Unsupported image format"
            },
        )

    filename = (
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    destination = UPLOAD_DIR / filename

    with destination.open("wb") as output:
        shutil.copyfileobj(
            file.file,
            output,
        )

    try:
        result = analyze_image(
            str(destination),
            verbose=False,
        )

        return result

    except Exception as exc:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc)
            },
        )


@app.post("/analyze-text")
async def analyze_text_endpoint(
    text: str = Form(...),
):
    try:
        return analyze_text(text)

    except Exception as exc:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc)
            },
        )


@app.post("/protect")
async def protect_endpoint(
    image_path: str = Form(...),
    detections: str = Form("[]"),
    mode: str = Form("automatic"),
    method: str = Form("blur"),
    selected_indices: str = Form(""),
):
    try:

        parsed_detections = json.loads(
            detections
        )

        if not isinstance(
            parsed_detections,
            list,
        ):
            parsed_detections = []

        indices = []

        if selected_indices.strip():

            indices = [
                int(x.strip())
                for x in selected_indices.split(",")
                if x.strip()
            ]

        source_path = Path(image_path)

        output_path = PROTECTED_DIR / (
            f"{source_path.stem}_protected"
            f"{source_path.suffix}"
        )

        protected = protect_image(
            image_path,
            detections=parsed_detections,
            mode=mode,
            method=method,
            selected_indices=indices,
            output_path=str(output_path),
        )

        return {
            "status": "success",
            "protected_image": protected,
            "mode": mode,
            "method": method,
        }

    except Exception as exc:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc)
            },
        )


@app.post("/redact")
async def redact_endpoint(
    image_path: str = Form(...),
    detections: str = Form("[]"),
    mode: str = Form("automatic"),
    method: str = Form("blur"),
):
    return await protect_endpoint(
        image_path=image_path,
        detections=detections,
        mode=mode,
        method=method,
        selected_indices="",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )