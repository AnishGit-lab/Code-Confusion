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
<meta name="theme-color" content="#0b1220">
<title>PriviScan | Privacy Intelligence</title>
<style>
:root{--bg:#0b1220;--panel:#111c2e;--panel2:#17243a;--line:#293951;--text:#edf4ff;--muted:#9aabc2;--cyan:#57d6d0;--blue:#739cff;--pink:#ff8eaa;--green:#6fe0ad;--amber:#ffd27a;--shadow:0 18px 50px #02081766}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(ellipse at 15% 0%,#193455 0,transparent 35%),radial-gradient(ellipse at 95% 10%,#26345a 0,transparent 28%),var(--bg);color:var(--text);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
button,input,select{font:inherit}button{cursor:pointer}.wrap{width:min(1160px,calc(100% - 36px));margin:auto}
header{position:sticky;top:0;z-index:20;background:#0b1220d9;backdrop-filter:blur(16px);border-bottom:1px solid #ffffff12}
.nav{height:76px;display:flex;align-items:center;justify-content:space-between;gap:20px}.brand{display:flex;align-items:center;gap:12px;font-size:21px;font-weight:800;letter-spacing:-.6px}.brandmark{width:38px;height:38px;border-radius:13px;background:linear-gradient(135deg,var(--cyan),var(--blue));display:grid;place-items:center;color:#071323;font-weight:900;box-shadow:0 8px 25px #57d6d044}.brand span small{display:block;color:var(--muted);font-size:10px;letter-spacing:1.7px;font-weight:700}
.navlinks{display:flex;gap:10px;align-items:center}.navlinks a{color:var(--muted);text-decoration:none;padding:9px 13px;border-radius:10px;font-weight:650}.navlinks a:hover{background:#ffffff0d;color:var(--text)}
.hero{padding:64px 0 35px;display:grid;grid-template-columns:1.35fr .65fr;gap:35px;align-items:center}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:2px;text-transform:uppercase;font-weight:800}.hero h1{font-size:clamp(36px,5.5vw,62px);line-height:1.04;letter-spacing:-2.5px;margin:15px 0}.gradient{background:linear-gradient(90deg,#fff,var(--cyan),#9bb6ff);background-clip:text;color:transparent}.hero p{max-width:650px;color:var(--muted);font-size:17px}.hero-card{padding:25px;border:1px solid #ffffff19;border-radius:24px;background:linear-gradient(145deg,#1a2c45cc,#111c2ecc);box-shadow:var(--shadow)}.shield{font-size:50px}.hero-stat{display:flex;justify-content:space-between;border-top:1px solid #ffffff15;padding:13px 0;color:var(--muted)}.hero-stat b{color:var(--text)}
.section-title{display:flex;justify-content:space-between;align-items:end;gap:15px;margin:22px 0 14px}.section-title h2{font-size:23px;margin:0;letter-spacing:-.6px}.section-title p{margin:4px 0 0;color:var(--muted)}
.panel{background:linear-gradient(145deg,#142238f2,#101a2bf2);border:1px solid #ffffff17;border-radius:20px;padding:24px;margin:15px 0;box-shadow:var(--shadow)}
.uploadbox{border:1.5px dashed #506782;border-radius:17px;padding:30px 18px;text-align:center;background:#0c1728;transition:.2s}.uploadbox.drag{border-color:var(--cyan);background:#163047}.upload-icon{width:56px;height:56px;margin:0 auto 12px;border-radius:17px;display:grid;place-items:center;background:#57d6d018;color:var(--cyan);font-size:27px}.uploadbox h3{margin:4px 0 3px}.muted{color:var(--muted)}.small{font-size:12px}.file-row{display:flex;justify-content:center;align-items:center;gap:12px;flex-wrap:wrap;margin-top:18px}.file-name{color:var(--cyan);font-size:13px;overflow-wrap:anywhere}
input[type=file]{max-width:100%;color:var(--muted)}.btn{border:0;border-radius:12px;padding:12px 18px;font-weight:800;transition:transform .18s,filter .18s;display:inline-flex;align-items:center;justify-content:center;gap:8px}.btn:hover{transform:translateY(-1px);filter:brightness(1.08)}.btn:disabled{opacity:.45;cursor:not-allowed;transform:none}.primary{color:#071323;background:linear-gradient(100deg,var(--cyan),#8bb2ff);box-shadow:0 8px 24px #57d6d033}.danger{background:#4b1f32;color:#ffdce5;border:1px solid #a84a68}.secondary{background:#22334c;color:var(--text);border:1px solid #40536d}.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}
#status{margin-top:16px;color:var(--muted);white-space:pre-wrap}.progress-wrap{margin-top:15px;display:none}.progress-meta{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-bottom:7px}.progress{height:7px;background:#26364c;border-radius:9px;overflow:hidden}.progress span{display:block;width:35%;height:100%;background:linear-gradient(90deg,var(--cyan),var(--blue));border-radius:9px;animation:progress 1.4s ease-in-out infinite alternate}@keyframes progress{from{transform:translateX(-40%)}to{transform:translateX(190%)}}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:15px 0}.metric{background:#0c1728;border:1px solid #ffffff12;border-radius:15px;padding:15px}.metric span{display:block;color:var(--muted);font-size:12px}.metric b{font-size:24px;display:block;margin-top:4px}.table-wrap{overflow:auto;border:1px solid #ffffff15;border-radius:13px}table{width:100%;border-collapse:collapse;min-width:760px}th,td{text-align:left;padding:12px;border-bottom:1px solid #ffffff12;vertical-align:top;overflow-wrap:anywhere}th{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;background:#0d192a}td{font-size:13px}tbody tr:hover{background:#ffffff08}.risk{font-weight:800}.warning{background:#3b301c55;border:1px solid #80652e77;border-radius:12px;padding:12px;color:#e8d6aa;font-size:12px;margin:14px 0}
.mode-row{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}.mode-option{padding:12px 15px;border:1px solid #344861;border-radius:12px;background:#0d192a;color:var(--muted)}.mode-option:has(input:checked){border-color:var(--cyan);color:var(--text);background:#1b3a47}.mode-option input{accent-color:var(--cyan)}select{background:#0c1728;color:var(--text);border:1px solid #40536d;border-radius:9px;padding:9px;margin-left:7px}
.about-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}.about-card{background:#0c1728;border:1px solid #ffffff13;border-radius:15px;padding:18px}.about-card h3{margin:0 0 8px;color:var(--cyan);font-size:16px}.about-card p,.about-card li{color:var(--muted);font-size:13px}.formula{font-family:ui-monospace,Consolas,monospace;background:#07111f;border:1px solid #31445c;border-radius:10px;padding:13px;color:#c7e9ff;overflow-wrap:anywhere;margin:8px 0}.sector-list{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.sector{padding:10px;border-radius:10px;background:#17283e;color:#c6d7ec;font-size:12px}
footer{padding:35px 0 45px;color:var(--muted);font-size:12px;text-align:center}
[hidden]{display:none!important}
@media(max-width:760px){.hero{grid-template-columns:1fr;padding-top:40px}.hero-card{display:none}.nav{height:auto;min-height:68px}.navlinks a{padding:8px;font-size:13px}.metrics{grid-template-columns:repeat(2,1fr)}.about-grid{grid-template-columns:1fr}.sector-list{grid-template-columns:repeat(2,1fr)}.panel{padding:17px}.section-title{align-items:start;flex-direction:column}}
</style>
</head>
<body>
<header><div class="wrap nav">
  <div class="brand"><div class="brandmark">P</div><span>PriviScan<small>PRIVACY INTELLIGENCE</small></span></div>
  <nav class="navlinks"><a href="#scanner">Scanner</a><a href="#about">About us</a></nav>
</div></header>
<main class="wrap">
<section class="hero">
 <div><div class="eyebrow">Privacy starts before sharing</div><h1>Share with confidence.<br><span class="gradient">Protect what matters.</span></h1>
 <p>PriviScan analyzes images and PDFs, finds potentially sensitive information, explains privacy risk, and lets you choose how to protect it.</p>
 <a class="btn secondary" href="#scanner">Explore the scanner ↓</a></div>
 <aside class="hero-card"><div class="shield">◈</div><h2>One scan. Clearer decisions.</h2><p class="muted">Detect · Assess · Explain · Protect</p>
 <div class="hero-stat"><span>Risk scale</span><b>0–100</b></div><div class="hero-stat"><span>Protection modes</span><b>Auto / Manual</b></div><div class="hero-stat"><span>Current upload formats</span><b>Image / PDF</b></div></aside>
</section>

<section id="scanner">
 <div class="section-title"><div><div class="eyebrow">01 / Analyze content</div><h2>Privacy scanner</h2><p>Upload an image or PDF to inspect its contents.</p></div></div>
 <div class="panel">
  <div class="uploadbox" id="dropZone"><div class="upload-icon">⇧</div><h3>Choose a file to analyze</h3><p class="muted small">Drag and drop here, or select a file from your device.</p>
   <div class="file-row"><input id="file" type="file" accept=".png,.jpg,.jpeg,.bmp,.webp,.pdf"><span id="fileName" class="file-name">No file selected</span></div>
  </div>
  <div class="actions"><button class="btn primary" id="analyzeButton">↗ Analyze</button><button class="btn danger" id="stopButton" hidden>■ Stop analysis</button></div>
  <div id="status" role="status" aria-live="polite">Ready when you are. Maximum upload size: 50 MB.</div>
  <div class="progress-wrap" id="progressWrap"><div class="progress-meta"><span id="progressLabel">Analyzing file…</span><span id="elapsed">Elapsed: 0s</span></div><div class="progress"><span></span></div><p class="muted small">Processing time depends on file size, page count, image quality, and OCR/AI processing. A reliable remaining-time estimate is not available yet.</p></div>
 </div>
</section>

<section id="reportPanel" class="panel" hidden>
 <div class="section-title"><div><div class="eyebrow">02 / Findings</div><h2>Analysis report</h2><p id="fileInfo" class="muted"></p></div></div>
 <div class="metrics" id="metrics"></div>
 <div class="warning">Review detections before sharing. OCR and AI may miss information or produce false positives. Always verify the protected output.</div>
 <div class="table-wrap"><table><thead><tr><th>Select</th><th>Type</th><th>Value</th><th>Risk</th><th>Confidence</th><th>Page</th><th>Location</th></tr></thead><tbody id="rows"></tbody></table></div>
 <p class="muted small" id="selectionHint">No detections selected: Automatic protection is the default.</p>
</section>

<section id="protectPanel" class="panel" hidden>
 <div class="eyebrow">03 / Protect content</div><h2>Choose your protection mode</h2>
 <div class="mode-row">
  <label class="mode-option"><input type="radio" name="mode" value="automatic" checked> Automatic protection</label>
  <label class="mode-option"><input type="radio" name="mode" value="manual"> Manual selection</label>
 </div>
 <p class="muted small">Selecting one or more detections in the report switches this to Manual automatically. Clear all selections to return to Automatic.</p>
 <label>Protection method: <select id="method"></select></label>
 <div class="actions"><button class="btn primary" id="protectButton">✦ Create protected file</button></div><div id="protectStatus" role="status"></div>
</section>

<section id="about" class="panel">
 <div class="section-title"><div><div class="eyebrow">About the project</div><h2>About PriviScan</h2><p>What it does, how it works, and how privacy risk is calculated.</p></div></div>
 <div class="about-grid">
  <article class="about-card"><h3>Project overview</h3><p>PriviScan is a privacy-risk assessment and sensitive-information protection system. It checks digital content before sharing, identifies potentially private information, estimates exposure risk, and gives the user control over protection.</p></article>
  <article class="about-card"><h3>Detection pipeline</h3><p>Upload → OCR / text extraction → rule and AI-assisted detection → risk assessment → report → automatic or manual protection.</p><p>Current website upload support in this code: PNG, JPG, JPEG, BMP, WEBP, and PDF.</p></article>
  <article class="about-card"><h3>Information categories</h3><p>Names, emails, international phone numbers, addresses, identifiers (such as student/staff/government IDs), financial/account data, faces, QR codes, and credentials such as passwords/API keys, depending on the detector and file content.</p></article>
  <article class="about-card"><h3>Technologies</h3><p>Python and FastAPI power the backend. The project uses PaddleOCR/PaddlePaddle for OCR, AI/NLP-assisted entity detection, and OpenCV/Pillow for image processing and protection.</p></article>
 </div>
 <h3 style="margin:22px 0 10px">Risk score system used in the current code</h3>
 <p class="muted">The current scoring implementation in <code>backend/risk/scoring.py</code> is factor-based. It does not assign one permanent final score to every information category.</p>
 <div class="formula">Impact = (Identifiability + Uniqueness + Potential Harm + Reusability) / 4</div>
 <p class="muted small">Each impact factor is rated from 1 to 5. The actual detected value can adjust factors slightly (for example, a full name or a long identifier).</p>
 <div class="formula">Confidence factor = 1 + (Detection confidence × 4)</div>
 <div class="formula">Likelihood = (Confidence factor + Linkability) / 2</div>
 <div class="formula">Individual risk = (Impact × Likelihood / 25) × 100</div>
 <p class="muted small">Likelihood uses a 1–5 scale. Linkability considers how the detected value can connect to other information. The document-level algorithm adds a linkability adjustment for complementary combinations, such as identity plus contact information or financial information combined with identity/contact information. Final risk is capped at 100.</p>
 <div class="formula">Credential override: PASSWORD / API_KEY / SECRET and related credential types with confidence ≥ 0.85 → 100 / CRITICAL</div>
 <div class="about-grid">
  <article class="about-card"><h3>Risk levels</h3><ul><li>0–20: LOW</li><li>21–40: MEDIUM</li><li>41–70: HIGH</li><li>71–100: CRITICAL</li></ul></article>
  <article class="about-card"><h3>Important distinction</h3><p>Confidence tells how certain the detector is. Risk estimates the possible privacy harm if the information is exposed. A high-confidence detection is not automatically high-risk, except for the defined credential override.</p></article>
 </div>
 <h3 style="margin:22px 0 10px">Context and sectors</h3>
 <div class="sector-list"><div class="sector">Healthcare / Hospital</div><div class="sector">Finance / Bank</div><div class="sector">Education / University</div><div class="sector">Business & Employment</div><div class="sector">Government & Legal</div><div class="sector">Consumer & Services</div></div>
 <p class="muted small">These are the project’s intended context areas. Important implementation note: the current <code>scoring.py</code> explicitly states that sector/context classification is not used in its present risk formula. Do not describe sector-based score adjustment as already implemented until that module is updated.</p>
 <h3 style="margin:22px 0 10px">Research references and algorithm source</h3>
 <p class="muted">The uploaded repository contains the scoring implementation and its design explanation, but it does not include bibliographic references, DOI links, or source URLs for the algorithm. To avoid inventing citations, this page identifies the current algorithm as the PriviScan implementation and does not claim that its exact numerical weights come from a specific paper.</p>
</section>
</main>
<footer><div class="wrap">PriviScan · Privacy risk assessment and user-controlled protection</div></footer>

<script>
let current=null;
let analysisController=null;
let timerId=null;
let startedAt=0;
const $=id=>document.getElementById(id);
function setMode(mode){document.querySelector(`input[name="mode"][value="${mode}"]`).checked=true;}
function selectedIndices(){return [...document.querySelectorAll('#rows input[type="checkbox"]:checked')].map(el=>Number(el.dataset.index));}
function updateSelectionMode(){
 const selected=selectedIndices();
 setMode(selected.length?'manual':'automatic');
 $('selectionHint').textContent=selected.length
  ? `${selected.length} detection(s) selected: Manual protection is active.`
  : 'No detections selected: Automatic protection is active.';
}
function metric(label,value){const el=document.createElement('div');el.className='metric';const l=document.createElement('span');l.textContent=label;const v=document.createElement('b');v.textContent=value;el.append(l,v);$('metrics').append(el);}
function renderResult(result){
 $('reportPanel').hidden=false;$('protectPanel').hidden=false;$('metrics').replaceChildren();
 metric('Overall risk',`${result.final_risk_score??result.risk_score??0}/100`);
 metric('Risk level',result.risk_level||'UNKNOWN');
 metric('Detections',String(result.total_detections??0));
 metric('Average item risk',String(result.average_risk_score??0));
 $('fileInfo').textContent=`${result.original_name||result.file_name||''} · ${result.file_type||''}`+(result.page_count?` · ${result.page_count} pages`:'');
 const body=$('rows');body.replaceChildren();
 (result.detections||[]).forEach((d,i)=>{
  const tr=document.createElement('tr'),sc=document.createElement('td'),cb=document.createElement('input');
  cb.type='checkbox';cb.dataset.index=String(i);cb.setAttribute('aria-label',`Select ${d.type||'detection'} ${i+1}`);
  const box=d.box;const valid=Array.isArray(box)&&box.length>=4;
  cb.disabled=!valid||(result.file_type==='PDF'&&!d.page);
  cb.addEventListener('change',updateSelectionMode);sc.append(cb);tr.append(sc);
  [d.type||'UNKNOWN',d.value||'',d.risk_score??'—',d.confidence==null?'—':`${Math.round(Number(d.confidence)*100)}%`,d.page??'—',valid?box.map(n=>Math.round(Number(n))).join(', '):'Unavailable'].forEach(v=>{const td=document.createElement('td');td.textContent=String(v);tr.append(td);});
  body.append(tr);
 });
 if(!(result.detections||[]).length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=7;td.textContent='No sensitive information detected.';tr.append(td);body.append(tr);}
 updateSelectionMode();
}
function setMethods(fileType){
 const s=$('method');s.replaceChildren();
 const list=fileType==='PDF'?[['blackout','Black box'],['white','White box']]:[['blur','Blur'],['blackout','Blackout'],['star','Stars'],['hashtag','Hashtags'],['random_alphabet','Random letters']];
 list.forEach(([v,l])=>{const o=document.createElement('option');o.value=v;o.textContent=l;s.append(o);});
}
function stopTimer(){if(timerId){clearInterval(timerId);timerId=null;}}
function updateElapsed(){$('elapsed').textContent=`Elapsed: ${Math.floor((Date.now()-startedAt)/1000)}s`;}
$('file').addEventListener('change',()=>{$('fileName').textContent=$('file').files[0]?.name||'No file selected';});
$('dropZone').addEventListener('dragover',e=>{e.preventDefault();$('dropZone').classList.add('drag');});
$('dropZone').addEventListener('dragleave',()=>$('dropZone').classList.remove('drag'));
$('dropZone').addEventListener('drop',e=>{e.preventDefault();$('dropZone').classList.remove('drag');if(e.dataTransfer.files.length){$('file').files=e.dataTransfer.files;$('fileName').textContent=e.dataTransfer.files[0].name;}});
$('analyzeButton').addEventListener('click',async()=>{
 const file=$('file').files[0];if(!file){$('status').textContent='Choose an image or PDF first.';return;}
 if(file.size>50*1024*1024){$('status').textContent='File exceeds 50 MB.';return;}
 analysisController=new AbortController();
 $('analyzeButton').disabled=true;$('analyzeButton').textContent='◌ Analyzing…';$('stopButton').hidden=false;
 $('status').textContent='Analysis is running. You can stop waiting for the result at any time.';
 $('progressWrap').style.display='block';$('reportPanel').hidden=true;$('protectPanel').hidden=true;
 startedAt=Date.now();updateElapsed();timerId=setInterval(updateElapsed,1000);
 try{
  const form=new FormData();form.append('file',file);
  const response=await fetch('/analyze',{method:'POST',body:form,signal:analysisController.signal});
  const data=await response.json();if(!response.ok)throw new Error(data.detail||data.error||'Analysis failed.');
  current=data;setMethods(data.file_type);renderResult(data);$('status').textContent='Analysis complete.';
 }catch(error){
  $('status').textContent=error.name==='AbortError'?'Analysis request stopped. Note: the server may continue processing an already-started task.':(error.message||String(error));
 }finally{
  stopTimer();analysisController=null;$('analyzeButton').disabled=false;$('analyzeButton').textContent='↗ Analyze';$('stopButton').hidden=true;$('progressWrap').style.display='none';
 }
});
$('stopButton').addEventListener('click',()=>{if(analysisController)analysisController.abort();});
$('protectButton').addEventListener('click',async()=>{
 if(!current)return;
 const selected=selectedIndices();
 const mode=selected.length?'manual':'automatic';
 setMode(mode);
 $('protectButton').disabled=true;$('protectStatus').textContent='Creating protected output…';
 try{
  const form=new FormData();form.append('source_filename',current.source_filename);form.append('detections',JSON.stringify(current.detections||[]));form.append('mode',mode);form.append('method',$('method').value);form.append('selected_indices',JSON.stringify(selected));
  const response=await fetch('/protect',{method:'POST',body:form});const data=await response.json();if(!response.ok)throw new Error(data.detail||data.error||'Protection failed.');
  const link=document.createElement('a');link.href=data.download_url;link.textContent='Download protected file';link.className='btn primary';
  $('protectStatus').replaceChildren(document.createTextNode(`Protected detections: ${data.redactions_applied}. `),link);
 }catch(error){$('protectStatus').textContent=error.message||String(error);}
 finally{$('protectButton').disabled=false;}
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