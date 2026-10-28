from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from backend.analyze import analyze_image, analyze_pdf
from backend.protection import protect_image, SUPPORTED_METHODS
from backend.pdf_redaction import redact_pdf

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROTECTED_DIR = BASE_DIR / "protected"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROTECTED_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

app = FastAPI(
    title="PriviScan",
    description="Image and PDF privacy risk assessment",
    version="2.0.0",
)


HTML_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PriviScan</title>
<style>
body{font-family:system-ui,Arial,sans-serif;max-width:1100px;margin:30px auto;padding:0 16px;background:#f4f6fa;color:#172033}
.panel{background:white;border:1px solid #dbe1ea;border-radius:12px;padding:18px;margin:16px 0}
button,a.button{border:0;border-radius:8px;padding:10px 14px;background:#2457c5;color:white;font-weight:600;cursor:pointer;text-decoration:none}
input,select{margin:6px 4px;padding:8px}
table{border-collapse:collapse;width:100%;margin-top:12px}
th,td{border-bottom:1px solid #dbe1ea;text-align:left;padding:9px;vertical-align:top;overflow-wrap:anywhere}
th{font-size:12px;text-transform:uppercase;color:#5d6678}
.metric{display:inline-block;padding:12px;margin:4px;background:#f5f7fb;border:1px solid #dbe1ea;border-radius:8px}
.muted{color:#5d6678}
.warning{background:#fff8e6;padding:12px;border-radius:8px}
#status,#protectStatus{white-space:pre-wrap}
</style>
</head>
<body>
<h1>PriviScan</h1>
<p class="muted">Sensitive information detection, risk assessment, and protection for images and PDFs.</p>

<section class="panel">
<h2>1. Upload a file</h2>
<input id="file" type="file" accept=".png,.jpg,.jpeg,.bmp,.webp,.pdf" />
<button id="analyzeButton">Analyze</button>
<p class="muted">Supported: PNG, JPG, JPEG, BMP, WEBP, PDF. Maximum upload size: 50 MB.</p>
<div id="status" role="status"></div>
</section>

<section id="reportPanel" class="panel" hidden>
<h2>2. Analysis report</h2>
<div id="metrics"></div>
<p id="fileInfo" class="muted"></p>
<p class="warning">Review the detections carefully. OCR and AI can miss information or identify false positives. Verify the protected output before sharing.</p>
<div style="overflow-x:auto">
<table>
<thead><tr><th>Select</th><th>Type</th><th>Value</th><th>Risk</th><th>Confidence</th><th>Page</th><th>Box</th></tr></thead>
<tbody id="rows"></tbody>
</table>
</div>
</section>

<section id="protectPanel" class="panel" hidden>
<h2>3. Protect the file</h2>
<label><input type="radio" name="mode" value="automatic" checked> Automatic</label>
<label><input type="radio" name="mode" value="manual"> Manual selection</label>
<label>Method:
<select id="method"></select>
</label>
<button id="protectButton">Create protected file</button>
<div id="protectStatus" role="status"></div>
</section>

<script>
let current = null;
let selectedFileType = null;

const $ = id => document.getElementById(id);

function addMetric(label,value){
  const el=document.createElement("span");
  el.className="metric";
  el.textContent=`${label}: ${value}`;
  $("metrics").append(el);
}

function renderResult(result){
  $("reportPanel").hidden=false;
  $("protectPanel").hidden=false;
  $("metrics").replaceChildren();

  addMetric("Risk",`${result.final_risk_score ?? result.risk_score ?? 0}/100`);
  addMetric("Level",result.risk_level || "UNKNOWN");
  addMetric("Detections",result.total_detections ?? 0);
  addMetric("Average item risk",result.average_risk_score ?? 0);

  $("fileInfo").textContent =
    `${result.file_name || ""} | ${result.file_type || ""}` +
    (result.page_count ? ` | ${result.page_count} pages` : "");

  const body=$("rows");
  body.replaceChildren();

  (result.detections || []).forEach((d,i)=>{
    const tr=document.createElement("tr");
    const selectCell=document.createElement("td");
    const checkbox=document.createElement("input");
    checkbox.type="checkbox";
    checkbox.dataset.index=String(i);
    checkbox.setAttribute("aria-label",`Select detection ${i+1}`);

    const box=d.box;
    const validBox=Array.isArray(box) && box.length>=4;
    const hasPage=result.file_type==="PDF";
    checkbox.disabled=!validBox || (hasPage && !d.page);
    selectCell.append(checkbox);
    tr.append(selectCell);

    const values=[
      d.type || "UNKNOWN",
      d.value || "",
      d.risk_score ?? "—",
      d.confidence == null ? "—" : `${Math.round(Number(d.confidence)*100)}%`,
      d.page ?? "—",
      Array.isArray(d.box) ? d.box.map(n=>Math.round(Number(n))).join(", ") : "Unavailable"
    ];

    values.forEach(v=>{
      const td=document.createElement("td");
      td.textContent=String(v);
      tr.append(td);
    });
    body.append(tr);
  });

  if(!(result.detections || []).length){
    const tr=document.createElement("tr");
    const td=document.createElement("td");
    td.colSpan=7;
    td.textContent="No sensitive information detected.";
    tr.append(td);
    body.append(tr);
  }
}

function setMethods(fileType){
  const select=$("method");
  select.replaceChildren();
  const methods=fileType==="PDF"
    ? [["blackout","Black box"],["white","White box"]]
    : [["blur","Blur"],["blackout","Blackout"],["star","Stars"],["hashtag","Hashtags"],["random_alphabet","Random letters"]];

  methods.forEach(([value,label])=>{
    const option=document.createElement("option");
    option.value=value;
    option.textContent=label;
    select.append(option);
  });
}

$("analyzeButton").addEventListener("click",async()=>{
  const file=$("file").files[0];
  if(!file){$("status").textContent="Choose an image or PDF first.";return;}

  if(file.size>50*1024*1024){$("status").textContent="File exceeds 50 MB.";return;}

  $("analyzeButton").disabled=true;
  $("status").textContent="Analyzing. Scanned PDFs and large images may take time.";
  $("reportPanel").hidden=true;
  $("protectPanel").hidden=true;

  try{
    const form=new FormData();
    form.append("file",file);
    const response=await fetch("/analyze",{method:"POST",body:form});
    const data=await response.json();
    if(!response.ok) throw new Error(data.detail || data.error || "Analysis failed.");

    current=data;
    selectedFileType=data.file_type;
    setMethods(data.file_type);
    renderResult(data);
    $("status").textContent="Analysis complete.";
  }catch(error){
    $("status").textContent=error.message || String(error);
  }finally{
    $("analyzeButton").disabled=false;
  }
});

$("protectButton").addEventListener("click",async()=>{
  if(!current)return;

  const mode=document.querySelector('input[name="mode"]:checked').value;
  const selected=[...document.querySelectorAll('#rows input[type="checkbox"]:checked')]
    .map(el=>Number(el.dataset.index));

  if(mode==="manual" && !selected.length){
    $("protectStatus").textContent="Select at least one detection for manual protection.";
    return;
  }

  $("protectButton").disabled=true;
  $("protectStatus").textContent="Creating protected output...";

  try{
    const form=new FormData();
    form.append("source_filename",current.source_filename);
    form.append("detections",JSON.stringify(current.detections || []));
    form.append("mode",mode);
    form.append("method",$("method").value);
    form.append("selected_indices",JSON.stringify(selected));

    const response=await fetch("/protect",{method:"POST",body:form});
    const data=await response.json();
    if(!response.ok) throw new Error(data.detail || data.error || "Protection failed.");

    const link=document.createElement("a");
    link.href=data.download_url;
    link.textContent="Download protected file";
    link.className="button";

    $("protectStatus").replaceChildren(
      document.createTextNode(`Protected detections: ${data.redactions_applied}. `),
      link
    );
  }catch(error){
    $("protectStatus").textContent=error.message || String(error);
  }finally{
    $("protectButton").disabled=false;
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


@app.get("/health")
def health():
    return {"status": "ok", "service": "PriviScan"}


@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()

    if extension not in IMAGE_EXTENSIONS | {".pdf"}:
        raise HTTPException(
            status_code=400,
            detail="Supported formats: PNG, JPG, JPEG, BMP, WEBP, PDF.",
        )

    stored_name = f"{uuid.uuid4().hex}{extension}"
    destination = UPLOAD_DIR / stored_name
    total = 0

    try:
        with destination.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the 50 MB upload limit.",
                    )
                output.write(chunk)

        if extension == ".pdf":
            with destination.open("rb") as source:
                if source.read(5) != b"%PDF-":
                    raise HTTPException(
                        status_code=400,
                        detail="The uploaded file is not a valid PDF.",
                    )
            result = analyze_pdf(str(destination), verbose=False)
        else:
            # Verify that the uploaded file is a readable image.
            import cv2
            image = cv2.imread(str(destination))
            if image is None:
                raise HTTPException(
                    status_code=400,
                    detail="Could not read the uploaded image.",
                )
            result = analyze_image(str(destination), verbose=False)

        result["source_filename"] = stored_name
        result["original_name"] = original_name
        result["file_type"] = "PDF" if extension == ".pdf" else "IMAGE"
        return JSONResponse(content=result)

    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc
    finally:
        await file.close()


@app.post("/protect")
async def protect_endpoint(
    source_filename: str = Form(...),
    detections: str = Form("[]"),
    mode: str = Form("automatic"),
    method: str = Form("blur"),
    selected_indices: str = Form("[]"),
):
    if Path(source_filename).name != source_filename:
        raise HTTPException(status_code=400, detail="Invalid source filename.")

    source_path = UPLOAD_DIR / source_filename
    if not source_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Source file not found. Upload it again.",
        )

    try:
        parsed = json.loads(detections)
        selected = json.loads(selected_indices)

        if not isinstance(parsed, list) or not isinstance(selected, list):
            raise ValueError("Detections and selected indices must be lists.")

        selected = [int(index) for index in selected]
        extension = source_path.suffix.lower()
        output_name = (
            f"{source_path.stem}_protected_{uuid.uuid4().hex[:8]}"
            f"{extension}"
        )
        output_path = PROTECTED_DIR / output_name

        if extension == ".pdf":
            # PDF redaction supports only blackout or white fill.
            if method not in {"blackout", "white"}:
                raise ValueError(
                    "For PDF files, choose Black box or White box."
                )

            result = redact_pdf(
                source_path,
                output_path,
                parsed,
                mode=mode,
                selected_indices=selected,
                method=method,
            )
            applied = result["redactions_applied"]
        elif extension in IMAGE_EXTENSIONS:
            if method not in SUPPORTED_METHODS:
                raise ValueError(f"Unsupported image protection method: {method}")

            protect_image(
                str(source_path),
                detections=parsed,
                mode=mode,
                method=method,
                selected_indices=selected,
                output_path=str(output_path),
            )
            applied = (
                len(parsed)
                if mode == "automatic"
                else len(selected)
            )
        else:
            raise ValueError("Unsupported file type.")

        return {
            "status": "success",
            "redactions_applied": applied,
            "mode": mode,
            "method": method,
            "download_url": f"/download/{output_name}",
        }

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Protection failed: {exc}",
        ) from exc


@app.get("/download/{filename}")
def download_file(filename: str):
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    path = PROTECTED_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Protected file not found.")

    media_type = (
        "application/pdf"
        if path.suffix.lower() == ".pdf"
        else "application/octet-stream"
    )

    return FileResponse(
        path,
        media_type=media_type,
        filename=path.name,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )