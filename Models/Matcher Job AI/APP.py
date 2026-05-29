"""
app.py — Pure FastAPI app (no Gradio)
• GET  /          → HTML UI
• POST /recommend → JSON API
• GET  /health    → status check
• GET  /specialities → job categories
• GET  /docs      → Swagger UI
"""

import os
import tempfile
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from model import load_data, build_vectorizer, recommend, recommend_from_text

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# STARTUP — load model once, store in _state
# ─────────────────────────────────────────────────────────────

_state    = {}
DATA_PATH = os.getenv("DATA_PATH", "Data.csv")
MAX_PDF_MB = int(os.getenv("MAX_PDF_MB", "10"))   # reject CVs > 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("⏳ Loading model …")
    try:
        df = load_data(DATA_PATH)
        vectorizer, job_vectors, centroids, intra_avg = build_vectorizer(df)
        _state["df"]         = df
        _state["vectorizer"] = vectorizer
        _state["job_vectors"]= job_vectors
        _state["centroids"]  = centroids
        _state["intra_avg"]  = intra_avg
        log.info(f"✅ Model ready — {df.shape[0]} jobs | {df['Speciality'].nunique()} specialities")
    except Exception as e:
        log.error(f"❌ Model failed to load: {e}")
        raise
    yield
    _state.clear()


app = FastAPI(
    title="CV Job Recommender API",
    description=(
        "Upload a CV (PDF) and receive ranked job recommendations "
        "with skill detection and gap analysis.\n\n"
        "**One endpoint you need:** `POST /recommend`"
    ),
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# HTML UI
# ─────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>CV Job Recommender</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;padding:2rem 1rem}
.wrap{max-width:900px;margin:0 auto}
header{text-align:center;margin-bottom:2.5rem}
header h1{font-size:2rem;font-weight:700;color:#fff}
header h1 span{color:#60a5fa}
header p{margin-top:.5rem;color:#94a3b8;font-size:.95rem}
.card{background:#1e293b;border:1px solid #334155;border-radius:1rem;padding:2rem;margin-bottom:1.5rem}
.drop{border:2px dashed #334155;border-radius:.75rem;padding:2.5rem 1rem;text-align:center;cursor:pointer;transition:border-color .2s,background .2s;position:relative}
.drop:hover,.drop.over{border-color:#60a5fa;background:#1e3a5f22}
.drop input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.drop-icon{font-size:2.5rem;margin-bottom:.75rem}
.drop p{color:#94a3b8;font-size:.9rem}
.fname{color:#60a5fa;font-weight:600;margin-top:.4rem;font-size:.95rem;min-height:1.4rem}
.controls{display:flex;align-items:center;gap:1.5rem;margin-top:1.5rem;flex-wrap:wrap}
.sl-group{display:flex;align-items:center;gap:.75rem;flex:1}
.sl-group label{white-space:nowrap;color:#94a3b8;font-size:.9rem}
.sl-group input[type=range]{flex:1;accent-color:#60a5fa}
.sl-group span{min-width:1.5rem;text-align:center;font-weight:600;color:#60a5fa}
.btn{padding:.75rem 2rem;background:#2563eb;color:#fff;border:none;border-radius:.6rem;font-size:1rem;font-weight:600;cursor:pointer;transition:background .2s,transform .1s;white-space:nowrap}
.btn:hover{background:#1d4ed8}
.btn:active{transform:scale(.97)}
.btn:disabled{background:#1e3a5f;color:#4a6a8a;cursor:not-allowed}
/* Loading overlay */
.loader{display:none;text-align:center;padding:2.5rem;color:#60a5fa}
.loader.show{display:block}
.spin{width:2.5rem;height:2.5rem;border:3px solid #1e3a5f;border-top-color:#60a5fa;border-radius:50%;animation:spin .7s linear infinite;margin:0 auto 1rem}
@keyframes spin{to{transform:rotate(360deg)}}
.loading-steps{color:#64748b;font-size:.85rem;margin-top:.5rem}
/* Error */
.err{display:none;background:#450a0a;border:1px solid #991b1b;border-radius:.75rem;padding:1rem 1.5rem;color:#fca5a5;margin-bottom:1.5rem;font-size:.9rem}
.err.show{display:block}
/* Summary */
.sum{display:none;background:#1e293b;border:1px solid #334155;border-radius:1rem;padding:1.75rem;margin-bottom:1.5rem}
.sum.show{display:block}
.cbadge{display:inline-block;background:#1e3a5f;color:#60a5fa;border:1px solid #2563eb;border-radius:2rem;padding:.35rem 1.1rem;font-weight:700;font-size:1.1rem;margin-bottom:1rem}
.cbar-wrap{display:flex;align-items:center;gap:.75rem;margin-bottom:1.25rem}
.cbar-bg{flex:1;height:8px;background:#0f172a;border-radius:4px;overflow:hidden}
.cbar-fill{height:100%;background:#2563eb;border-radius:4px;transition:width .8s ease}
.cbar-lbl{font-size:.85rem;color:#94a3b8;white-space:nowrap}
.sec{font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:#64748b;margin-bottom:.6rem}
.runners{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1.25rem}
.rchip{background:#0f172a;border:1px solid #334155;border-radius:2rem;padding:.25rem .85rem;font-size:.8rem;color:#94a3b8}
.pills{display:flex;flex-wrap:wrap;gap:.4rem}
.pill{background:#0f172a;border:1px solid #334155;color:#94a3b8;border-radius:.4rem;padding:.2rem .6rem;font-size:.78rem}
/* Jobs */
.jobs{display:flex;flex-direction:column;gap:1rem}
.jcard{background:#1e293b;border:1px solid #334155;border-radius:1rem;padding:1.5rem;transition:border-color .2s}
.jcard:hover{border-color:#2563eb}
.jhead{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap}
.jrank{font-size:.75rem;color:#64748b;font-weight:600;margin-bottom:.2rem}
.jtitle{font-size:1.05rem;font-weight:700;color:#f1f5f9}
.jco{font-size:.9rem;color:#94a3b8;margin-top:.2rem}
.badges{display:flex;gap:.5rem;flex-wrap:wrap;align-items:center}
.badge{padding:.3rem .75rem;border-radius:2rem;font-size:.78rem;font-weight:600;white-space:nowrap}
.bb{background:#1e3a5f;color:#60a5fa;border:1px solid #2563eb}
.bg{background:#052e16;color:#4ade80;border:1px solid #166534}
.meta{display:flex;gap:.75rem;flex-wrap:wrap;margin:.85rem 0;font-size:.82rem;color:#64748b}
.gap-sec{margin-top:.85rem}
.grow{display:flex;gap:.5rem;align-items:flex-start;margin-bottom:.4rem;flex-wrap:wrap}
.glbl{font-size:.75rem;font-weight:600;white-space:nowrap;padding-top:.15rem}
.ghave{color:#4ade80}
.gmiss{color:#f87171}
.tag{border-radius:.35rem;padding:.15rem .55rem;font-size:.75rem}
.tg{background:#052e16;color:#4ade80;border:1px solid #166534}
.tr{background:#450a0a;color:#f87171;border:1px solid #991b1b}
.abtn{display:inline-block;margin-top:1rem;padding:.5rem 1.25rem;background:#1e3a5f;color:#60a5fa;border:1px solid #2563eb;border-radius:.5rem;text-decoration:none;font-size:.85rem;font-weight:600;transition:background .2s}
.abtn:hover{background:#2563eb;color:#fff}
footer{text-align:center;margin-top:3rem;color:#334155;font-size:.8rem}
footer a{color:#475569;text-decoration:none}
footer a:hover{color:#60a5fa}
/* Tabs */
.tab-bar{display:flex;gap:.5rem;margin-bottom:0;border-bottom:1px solid #334155;padding-bottom:0}
.tab{padding:.65rem 1.4rem;background:none;border:none;border-bottom:2px solid transparent;color:#64748b;font-size:.92rem;font-weight:600;cursor:pointer;transition:color .2s,border-color .2s;margin-bottom:-1px}
.tab:hover{color:#94a3b8}
.tab.active{color:#60a5fa;border-bottom-color:#2563eb}
/* Text input */
textarea#msg{width:100%;background:#0f172a;border:1px solid #334155;border-radius:.6rem;color:#e2e8f0;font-size:.92rem;padding:.85rem 1rem;resize:vertical;outline:none;transition:border-color .2s;font-family:inherit;margin-top:.25rem}
textarea#msg:focus{border-color:#2563eb}
.examples{display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:.75rem;align-items:center}
.ex-label{font-size:.75rem;color:#64748b;white-space:nowrap}
.ex-chip{background:#0f172a;border:1px solid #334155;border-radius:2rem;color:#94a3b8;font-size:.75rem;padding:.2rem .75rem;cursor:pointer;transition:border-color .2s,color .2s;text-align:left}
.ex-chip:hover{border-color:#2563eb;color:#60a5fa}
@media(max-width:600px){header h1{font-size:1.5rem}.controls{flex-direction:column;align-items:stretch}.btn{width:100%}.tab{padding:.5rem .85rem;font-size:.82rem}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>📄 CV <span>Job Recommender</span></h1>
    <p>Upload your CV and get ranked job matches with skill gap analysis</p>
  </header>

  <!-- Tab switcher -->
  <div class="tab-bar">
    <button class="tab active" id="tab-pdf" onclick="switchTab('pdf')">📎 Upload CV (PDF)</button>
    <button class="tab" id="tab-txt" onclick="switchTab('txt')">💬 Describe Your Skills</button>
  </div>

  <!-- PDF tab -->
  <div class="card" id="panel-pdf">
    <div class="drop" id="dz">
      <input type="file" id="fi" accept=".pdf"/>
      <div class="drop-icon">📎</div>
      <p>Drag &amp; drop your CV here or <strong style="color:#60a5fa">click to browse</strong></p>
      <p style="font-size:.8rem;margin-top:.3rem">PDF only · max 10 MB</p>
      <p class="fname" id="fn"></p>
    </div>
    <div class="controls">
      <div class="sl-group">
        <label>Results</label>
        <input type="range" id="tn" min="1" max="20" value="5"/>
        <span id="tnv">5</span>
      </div>
      <button class="btn" id="sb" disabled>🔍 Find Jobs</button>
    </div>
  </div>

  <!-- Text tab -->
  <div class="card" id="panel-txt" style="display:none">
    <p style="color:#94a3b8;font-size:.9rem;margin-bottom:1rem">
      Describe your role and skills in plain language — no CV needed.
    </p>
    <div class="examples">
      <span class="ex-label">Try:</span>
      <button class="ex-chip" onclick="fillExample(this)">I'm a frontend developer, my skills are react, typescript, angular, figma and git</button>
      <button class="ex-chip" onclick="fillExample(this)">python machine learning tensorflow pytorch deep learning nlp computer vision</button>
      <button class="ex-chip" onclick="fillExample(this)">mobile developer flutter dart ios swift swiftui firebase react native</button>
      <button class="ex-chip" onclick="fillExample(this)">odoo development laravel php backend sql postgresql docker erp</button>
      <button class="ex-chip" onclick="fillExample(this)">penetration testing ethical hacking kali linux siem cybersecurity vulnerability scanning</button>
    </div>
    <textarea id="msg" rows="4"
      placeholder="e.g. I am a full stack developer with 3 years experience. My skills are react.js, node.js, python, django, postgresql and docker."
    ></textarea>
    <div style="font-size:.78rem;color:#475569;margin-top:.4rem;text-align:right">
      <span id="charcount">0</span>/2000
    </div>
    <div class="controls" style="margin-top:1rem">
      <div class="sl-group">
        <label>Results</label>
        <input type="range" id="tn2" min="1" max="20" value="5"/>
        <span id="tnv2">5</span>
      </div>
      <button class="btn" id="sb2" disabled>🔍 Find Jobs</button>
    </div>
  </div>

  <div class="err" id="eb"></div>

  <div class="loader" id="ld">
    <div class="spin"></div>
    <p>Analysing your CV…</p>
    <p class="loading-steps" id="lstep">Extracting text from PDF</p>
  </div>

  <div class="sum" id="sc"></div>
  <div class="jobs" id="jg"></div>

  <footer>
    <p>REST API &nbsp;·&nbsp; <a href="/docs">/docs</a> &nbsp;·&nbsp; <a href="/redoc">/redoc</a> &nbsp;·&nbsp; <a href="/health">/health</a> &nbsp;·&nbsp; <a href="/specialities">/specialities</a></p>
  </footer>
</div>

<script>
const fi=document.getElementById('fi'),dz=document.getElementById('dz'),fn=document.getElementById('fn');
const tn=document.getElementById('tn'),tnv=document.getElementById('tnv');
const sb=document.getElementById('sb'),ld=document.getElementById('ld'),lstep=document.getElementById('lstep');
const eb=document.getElementById('eb'),sc=document.getElementById('sc'),jg=document.getElementById('jg');
const MAX_MB = 10;

tn.addEventListener('input',()=>tnv.textContent=tn.value);

function pick(f){
  if(!f) return;
  if(!f.name.toLowerCase().endsWith('.pdf')){ showErr('❌ Please upload a PDF file.'); return; }
  if(f.size > MAX_MB*1024*1024){ showErr(`❌ File too large (${(f.size/1024/1024).toFixed(1)} MB). Max ${MAX_MB} MB.`); return; }
  fn.textContent='✅ '+f.name+' ('+( f.size/1024).toFixed(0)+' KB)';
  sb.disabled=false; hideErr();
}
fi.addEventListener('change',()=>pick(fi.files[0]));
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('over');});
dz.addEventListener('dragleave',()=>dz.classList.remove('over'));
dz.addEventListener('drop',e=>{
  e.preventDefault(); dz.classList.remove('over');
  if(e.dataTransfer.files[0]) { fi.files=e.dataTransfer.files; pick(e.dataTransfer.files[0]); }
});

// Animated loading steps
const STEPS=['Extracting text from PDF','Detecting skills…','Identifying career path…','Ranking job matches…','Calculating skill gaps…'];
let stepTimer=null;
function startSteps(){
  let i=0; lstep.textContent=STEPS[0];
  stepTimer=setInterval(()=>{ i=(i+1)%STEPS.length; lstep.textContent=STEPS[i]; },1800);
}
function stopSteps(){ clearInterval(stepTimer); }

sb.addEventListener('click',async()=>{
  const f=fi.files[0]; if(!f) return;
  load(true); hideErr(); sc.classList.remove('show'); sc.innerHTML=''; jg.innerHTML='';
  const fd=new FormData(); fd.append('cv',f);
  try{
    const r=await fetch('/recommend?top_n='+tn.value,{method:'POST',body:fd});
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||'Server error ('+r.status+')');
    render(d);
  } catch(e){ showErr('❌ '+e.message); }
  finally{ load(false); }
});

function render(d){
  // Summary card
  const runners=d.top3_careers.slice(1).map(c=>`<span class="rchip">${c.career} &nbsp;${c.score}%</span>`).join('');
  const pills=d.cv_skills.length
    ? d.cv_skills.map(s=>`<span class="pill">${s}</span>`).join('')
    : '<span style="color:#64748b;font-size:.85rem">No known skills detected — check PDF is not scanned/image-only</span>';
  sc.innerHTML=`
    <div class="cbadge">🎯 ${d.detected_career}</div>
    <div class="cbar-wrap">
      <div class="cbar-bg"><div class="cbar-fill" style="width:${d.confidence}%"></div></div>
      <span class="cbar-lbl">${d.confidence}% confidence</span>
    </div>
    ${runners?`<div class="sec">Runner-up careers</div><div class="runners">${runners}</div>`:''}
    <div class="sec">Skills detected in CV &nbsp;<span style="color:#475569">(${d.total_cv_skills} found)</span></div>
    <div class="pills">${pills}</div>
    ${d.fallback_used?'<p style="margin-top:1rem;color:#f59e0b;font-size:.85rem">⚠️ Not enough jobs in detected category — showing best matches from full dataset.</p>':''}
  `;
  sc.classList.add('show');

  // Job cards
  jg.innerHTML=d.jobs.map((j,i)=>{
    const hh=j.matched_skills.slice(0,8).map(s=>`<span class="tag tg">${s}</span>`).join('')
            ||'<span style="color:#64748b;font-size:.78rem">—</span>';
    const mh=j.missing_skills.slice(0,8).map(s=>`<span class="tag tr">${s}</span>`).join('')
            ||'<span style="color:#64748b;font-size:.78rem">—</span>';
    return `<div class="jcard">
      <div class="jhead">
        <div>
          <div class="jrank">#${i+1}</div>
          <div class="jtitle">${j.title}</div>
          <div class="jco">${j.company}</div>
        </div>
        <div class="badges">
          <span class="badge bb">⚡ ${j.similarity}% match</span>
          <span class="badge bg">🎯 ${j.skill_match_rate}% skills</span>
        </div>
      </div>
      <div class="meta">
        <span>📍 ${j.location}</span>
        <span>🌐 ${j.job_location_type}</span>
        <span>⏱️ ${j.job_type}</span>
        <span>📂 ${j.speciality}</span>
        <span>🔧 ${j.matched_skills.length}/${j.total_required} skills matched</span>
      </div>
      <div class="gap-sec">
        <div class="grow"><span class="glbl ghave">✅ Have</span>${hh}</div>
        <div class="grow"><span class="glbl gmiss">❌ Missing</span>${mh}</div>
      </div>
      <a class="abtn" href="${j.url}" target="_blank" rel="noopener">Apply →</a>
    </div>`;
  }).join('');
}

function load(on){
  ld.classList.toggle('show',on);
  sb.disabled=on;
  if(activeTab==='txt') sb2.disabled=on;
  if(on) startSteps(); else stopSteps();
}
function showErr(m){ eb.textContent=m; eb.classList.add('show'); }
function hideErr(){ eb.classList.remove('show'); }

// ── Text tab ───────────────────────────────────────────────
const msg=document.getElementById('msg');
const sb2=document.getElementById('sb2');
const tn2=document.getElementById('tn2');
const tnv2=document.getElementById('tnv2');
const charcount=document.getElementById('charcount');
let activeTab='pdf';

tn2.addEventListener('input',()=>tnv2.textContent=tn2.value);

msg.addEventListener('input',()=>{
  const len=msg.value.length;
  charcount.textContent=len;
  charcount.style.color = len>1800?'#f87171': len>1200?'#f59e0b':'#475569';
  sb2.disabled = len===0 || len>2000;
});

function fillExample(btn){
  msg.value=btn.textContent.trim();
  msg.dispatchEvent(new Event('input'));
  msg.focus();
}

sb2.addEventListener('click',async()=>{
  const text=msg.value.trim();
  if(!text) return;
  load(true); hideErr(); sc.classList.remove('show'); sc.innerHTML=''; jg.innerHTML='';
  try{
    const url='/recommend/text?top_n='+tn2.value+'&message='+encodeURIComponent(text);
    const r=await fetch(url,{method:'POST'});
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||'Server error ('+r.status+')');
    render(d);
  } catch(e){ showErr('❌ '+e.message); }
  finally{ load(false); }
});

// ── Tab switching ───────────────────────────────────────────
function switchTab(tab){
  activeTab=tab;
  document.getElementById('panel-pdf').style.display = tab==='pdf'?'':'none';
  document.getElementById('panel-txt').style.display = tab==='txt'?'':'none';
  document.getElementById('tab-pdf').classList.toggle('active', tab==='pdf');
  document.getElementById('tab-txt').classList.toggle('active', tab==='txt');
  hideErr();
  sc.classList.remove('show'); sc.innerHTML=''; jg.innerHTML='';
}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return HTML_PAGE


@app.get("/health", tags=["Info"], summary="Model health check")
def health():
    ready = "df" in _state
    return {
        "status":       "ready" if ready else "loading",
        "jobs_loaded":  int(_state["df"].shape[0]) if ready else 0,
        "specialities": int(_state["df"]["Speciality"].nunique()) if ready else 0,
        "version":      app.version,
    }


@app.get("/specialities", tags=["Info"], summary="List all job categories")
def specialities():
    if "df" not in _state:
        raise HTTPException(503, "Model not ready — try again in a moment.")
    counts = _state["df"]["Speciality"].value_counts().to_dict()
    return {
        "total": len(counts),
        "specialities": [
            {"name": k, "job_count": v}
            for k, v in sorted(counts.items())
        ],
    }


@app.post(
    "/recommend",
    tags=["Recommend"],
    summary="Analyse CV and return job matches",
    response_description="Detected career, CV skills, and ranked job recommendations with skill gap analysis",
)
async def recommend_jobs(
    cv: UploadFile = File(..., description="CV file in PDF format (max 10 MB)"),
    top_n: int = Query(5, ge=1, le=20, description="Number of job recommendations (1–20)"),
):
    """
    Upload a CV PDF and receive:
    - **Detected career** with confidence score
    - **Skills extracted** from the CV
    - **Top-N ranked jobs** with similarity score
    - **Skill gap** per job — what you have vs what's missing
    """
    # Validate file type
    if not cv.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    # Validate file size
    content = await cv.read()
    if len(content) > MAX_PDF_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Maximum size is {MAX_PDF_MB} MB.")

    if "df" not in _state:
        raise HTTPException(503, "Model is still loading. Try again in a moment.")

    # Write to temp file for pdfplumber
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = recommend(
            cv_filepath  = tmp_path,
            df           = _state["df"],
            vectorizer   = _state["vectorizer"],
            job_vectors  = _state["job_vectors"],
            centroids    = _state["centroids"],
            intra_avg    = _state["intra_avg"],
            top_n        = top_n,
        )
    except Exception as e:
        log.error(f"Processing error: {e}")
        raise HTTPException(500, f"CV processing error: {str(e)}")
    finally:
        os.unlink(tmp_path)

    return result


@app.post(
    "/recommend/text",
    tags=["Recommend"],
    summary="Analyse a text message and return job matches",
    response_description="Detected career and ranked job recommendations from free text",
)
async def recommend_from_message(
    message: str = Query(..., description='e.g. "I am a frontend developer, my skills are react typescript angular figma"'),
    top_n:   int = Query(5, ge=1, le=20, description="Number of results (1–20)"),
):
    """
    Send a plain-text message describing your role and skills.
    No file upload needed.

    **Examples:**
    - `I am a frontend developer, my skills are react, typescript, angular and figma`
    - `python machine learning tensorflow deep learning nlp computer vision`
    - `mobile developer flutter dart ios swift firebase`
    - `i work in cyber security penetration testing kali linux siem`
    """
    if not message or not message.strip():
        raise HTTPException(400, "Message cannot be empty.")
    if len(message) > 2000:
        raise HTTPException(400, "Message too long. Maximum 2000 characters.")
    if "df" not in _state:
        raise HTTPException(503, "Model is still loading. Try again in a moment.")

    try:
        result = recommend_from_text(
            user_message = message.strip(),
            df           = _state["df"],
            vectorizer   = _state["vectorizer"],
            job_vectors  = _state["job_vectors"],
            centroids    = _state["centroids"],
            intra_avg    = _state["intra_avg"],
            top_n        = top_n,
        )
    except Exception as e:
        log.error(f"Text processing error: {e}")
        raise HTTPException(500, f"Processing error: {str(e)}")

    return result


# ─────────────────────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
