from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, HttpUrl
import subprocess

from app.infer_core import analyze_youtube_url, ART

app = FastAPI(title="Genre Mapper", version="1.0")

class AnalyzeBody(BaseModel):
    url: HttpUrl
    cookies: str | None = "none"
    ipv4: bool = False

@app.get("/health")
def health() -> dict:
    ok = all((ART / p).exists() for p in [
        "feature_order.json","label_map.json","scaler.pkl","model.pkl","test_features_scaled.npz"
    ])
    return {"status": "ok" if ok else "missing_artifacts"}

@app.get("/debug/formats", response_class=PlainTextResponse)
def debug_formats(url: str = Query(..., description="YouTube URL")):
    try:
        # mimic your working interactive test
        proc = subprocess.run(["yt-dlp","-F", url, "-4", "-v"], capture_output=True, text=True, timeout=60)
        return (proc.stdout or proc.stderr or "").strip()
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/analyze")
def analyze(body: AnalyzeBody):
    try:
        result = analyze_youtube_url(str(body.url), cookies_from=body.cookies or "none", force_ipv4=body.ipv4)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse("""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Genre Mapper</title>
<style>
  :root{
    --bg:#0b0d10; --bg2:#0f1318;
    --card:#11161dCC; --muted:#a7b1c2;
    --text:#e6edf3; --accent:#6ae3ff; --accent2:#7c4dff;
    --border:#1f2a37; --err:#ef4444;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{
    margin:0; color:var(--text); background:
      radial-gradient(1200px 600px at 10% -10%, #0b1b28 0%, transparent 50%),
      radial-gradient(1000px 600px at 110% 10%, #1b1235 0%, transparent 55%),
      var(--bg);
    font:14px/1.4 Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  }
  .container{max-width:900px;margin:48px auto;padding:0 20px}
  h1{margin:0 0 8px;font-weight:700;letter-spacing:.2px}
  .sub{color:var(--muted);margin-bottom:18px}
  .card{
    background:linear-gradient(145deg, rgba(17,22,29,.9), rgba(12,16,22,.8));
    border:1px solid var(--border); border-radius:14px; padding:16px;
    box-shadow:0 10px 30px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.03);
    backdrop-filter: blur(8px);
  }
  .row{display:flex;gap:10px;flex-wrap:wrap}
  input[type=text]{
    flex:1; min-width:260px; padding:12px 14px;
    color:var(--text); background:#0c1117; border:1px solid var(--border);
    border-radius:10px; outline:none;
  }
  input[type=text]:focus{border-color:#223446; box-shadow:0 0 0 3px rgba(106,227,255,.15)}
  button{
    padding:12px 16px; border-radius:10px; border:1px solid #2a3340;
    background:linear-gradient(180deg,#15202b,#121821);
    color:var(--text); cursor:pointer; transition:.15s transform ease;
  }
  button:hover{transform:translateY(-1px); border-color:#364454}
  .youre{display:flex; align-items:center; gap:12px; margin:2px 0 12px}
  /* Gradient text for genre (no pill/outline) */
  .badge{
    font-weight:900; letter-spacing:.3px;
    background:linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip:text; background-clip:text;
    color:transparent; padding:0; border:0;
  }
  .meta{margin-left:auto; color:var(--muted); font-size:12px}
  .bar{height:12px; background:#0c1117; border:1px solid var(--border); border-radius:999px; overflow:hidden}
  .bar > div{
    height:100%; width:0%;
    background:linear-gradient(90deg, var(--accent), var(--accent2));
    transition:width .45s cubic-bezier(.2,.8,.2,1);
  }
  .list{display:flex; flex-direction:column; gap:10px; margin-top:8px}
  .rowi{display:flex; align-items:center; gap:12px}
  .lbl{width:110px; color:var(--muted)}
  .pct{width:70px; text-align:right; font-variant-numeric:tabular-nums}
  .err{color:var(--err)}
  .loading{color:var(--muted)}
  .dots:after{content:""; display:inline-block; width:1ch; text-align:left; animation:dots 1.2s steps(4,end) infinite}
  @keyframes dots{0%{content:""}25%{content:"."}50%{content:".."}75%{content:"..."}100%{content:""}}
  .stats{display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:10px; margin-top:12px}
  .kv{display:flex; justify-content:space-between; gap:8px; padding:10px 12px; border:1px solid var(--border); border-radius:10px; background:#0c1117}
</style>
</head>
<body>
  <div class="container">
    <h1>Genre Mapper</h1>
    <div class="sub">Paste a YouTube link. We analyze the middle 30 seconds and show Top-1 and Top-3 confidences.</div>

    <div class="card">
      <div class="row">
        <input id="url" type="text" placeholder="https://www.youtube.com/watch?v=..." />
        <button id="go">Analyze</button>
      </div>
    </div>

    <div id="out" class="card" style="display:none; margin-top:16px;"></div>
  </div>

<script>
const out = document.getElementById('out');
const btn = document.getElementById('go');

function renderLoading(){
  out.style.display='block';
  out.innerHTML = '<div class="loading dots">Processing</div>';
}

function renderResult(data){
  const t1 = data.top1, t3 = data.top3, s = data.stats || {};

  let rows = t3.map((t,i) => {
    const pct = (t.conf*100).toFixed(1);
    const id = 'b'+i;
    return `
      <div class="rowi">
        <div class="lbl">${t.label}</div>
        <div class="bar" style="flex:1"><div id="${id}" style="width:0%"></div></div>
        <div class="pct">${pct}%</div>
      </div>
    `;
  }).join('');

  const stats = `
    <div class="stats">
      <div class="kv"><span>Tempo</span><b>${(s.tempo_bpm ?? 0).toFixed(0)} BPM</b></div>
      <div class="kv"><span>Brightness</span><b>${(s.brightness_hz ?? 0).toFixed(0)} Hz</b></div>
      <div class="kv"><span>Energy (RMS)</span><b>${(s.energy_rms ?? 0).toFixed(3)}</b></div>
      <div class="kv"><span>Noisiness (ZCR)</span><b>${(s.noisiness_zcr ?? 0).toFixed(3)}</b></div>
      <div class="kv"><span>Bandwidth</span><b>${(s.bandwidth_hz ?? 0).toFixed(0)} Hz</b></div>
      <div class="kv"><span>Rolloff</span><b>${(s.rolloff_hz ?? 0).toFixed(0)} Hz</b></div>
    </div>
  `;

  out.innerHTML = `
    <div class="youre">
      <div><b>You're listening to:</b> <span class="badge">${t1.label}</span></div>
      <div class="meta">${(t1.conf*100).toFixed(1)}% • ${data.model_version || "v?"}</div>
    </div>
    <div class="list">${rows}</div>
    ${stats}
  `;

  requestAnimationFrame(() => {
    t3.forEach((t,i)=>{
      const el = document.getElementById('b'+i);
      if (el) el.style.width = (t.conf*100).toFixed(1) + '%';
    });
  });
}

btn.onclick = async () => {
  const url = document.getElementById('url').value.trim();
  if(!url){ alert('Paste a YouTube URL'); return; }
  renderLoading();
  try{
    const res = await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url})});
    const data = await res.json();
    if(!res.ok){ throw new Error(data.detail || 'Request failed'); }
    renderResult(data);
  }catch(e){
    out.style.display='block';
    out.innerHTML = '<div class="err">Error: '+e.message+'</div>';
  }
};
</script>
</body></html>
""")
