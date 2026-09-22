from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pathlib import Path
import shutil

app = FastAPI(title="PriviScan")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTpiYPE html>
    <html>
    <head>
        <title>PriviScan</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 700px;
                margin: 80px auto;
                padding: 20px;
                text-align: center;
            }

            .box {
                border: 2px dashed #888;
                padding: 50px;
                border-radius: 15px;
            }

            button {
                margin-top: 20px;
                padding: 12px 25px;
                font-size: 16px;
                cursor: pointer;
            }
        </style>
    </head>

    <body>

        <h1>PriviScan</h1>

        <p>
            Intelligent Privacy Risk Assessment
        </p>

        <div class="box">

            <form action="/analyze" method="post"
                  enctype="multipart/form-data">

                <input type="file"
                       name="file"
                       accept="image/*"
                       required>

                <br>

                <button type="submit">
                    Analyze Image
                </button>

            </form>

        </div>

    </body>
    </html>
    """


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(file: UploadFile = File(...)):

    file_path = UPLOAD_DIR / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return f"""
    <!DOCTYPE html>
    <html>

    <head>
        <title>PriviScan Result</title>
    </head>

    <body>

        <h1>PriviScan Analysis</h1>

        <p>File received successfully.</p>

        <p>
            <strong>File:</strong>
            {file.filename}
        </p>

        <p>
            AI detection will be added next.
        </p>

        <a href="/">
            Analyze another image
        </a>

    </body>

    </html>
    """