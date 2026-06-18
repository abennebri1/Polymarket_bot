#!/usr/bin/env python3
"""
Polymarket screener — web control panel
=======================================

A small password-protected web app that lets you START / STOP the wallet
screener from any device and watch results stream in live. It runs the existing
`wallet_screener.py` as a subprocess (which already writes an auto-refreshing
HTML dashboard + index into the reports/ folder) and serves everything behind a
login.

Run locally:
    pip install -r requirements.txt
    set APP_PASSWORD=your-password           (Windows)   /  export on Linux
    set SECRET_KEY=some-long-random-string
    uvicorn server:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000  (deploy guide: DEPLOY.md).

Environment:
    APP_PASSWORD   the single login password (REQUIRED in production)
    SECRET_KEY     random string used to sign the session cookie (REQUIRED)
    PYTHON_BIN     python executable to run the screener (defaults to this one)
    PORT           port to bind (most hosts set this automatically)
"""

from __future__ import annotations

import hmac
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse, RedirectResponse)
from starlette.middleware.sessions import SessionMiddleware

# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent
REPORTS = BASE / "reports"
REPORTS.mkdir(exist_ok=True)
LOG = REPORTS / "run.log"
SCRIPT = BASE / "wallet_screener.py"
MANIFEST = REPORTS / "index.json"

APP_PASSWORD = os.environ.get("APP_PASSWORD", "changeme")
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-insecure-change-me")
PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)

app = FastAPI(title="Polymarket Screener")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=60 * 60 * 24 * 30)

_state = {"proc": None, "started": None, "mode": None, "logf": None}
_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _running() -> bool:
    p = _state["proc"]
    return p is not None and p.poll() is None


def _authed(request: Request) -> bool:
    return bool(request.session.get("auth"))


def _load_manifest():
    if not MANIFEST.exists():
        return []
    try:
        recs = json.loads(MANIFEST.read_text())
        return sorted(recs, key=lambda r: r.get("ts", ""), reverse=True)
    except (OSError, ValueError):
        return []


def _log_tail(n=60):
    if not LOG.exists():
        return ""
    try:
        return "".join(LOG.read_text(errors="ignore").splitlines(keepends=True)[-n:])
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# auth
# --------------------------------------------------------------------------- #
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, bad: int = 0):
    if _authed(request):
        return RedirectResponse("/", status_code=302)
    err = "<div class='err'>Wrong password.</div>" if bad else ""
    return HTMLResponse(LOGIN_HTML.replace("<!--ERR-->", err))


@app.post("/login")
def login_submit(request: Request, password: str = Form("")):
    if hmac.compare_digest(password, APP_PASSWORD):
        request.session["auth"] = True
        return RedirectResponse("/", status_code=302)
    return RedirectResponse("/login?bad=1", status_code=302)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


# --------------------------------------------------------------------------- #
# app + API
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    if not _authed(request):
        return RedirectResponse("/login", status_code=302)
    return HTMLResponse(APP_HTML)


@app.post("/api/run")
async def api_run(request: Request):
    if not _authed(request):
        raise HTTPException(401)
    if _running():
        return JSONResponse({"ok": False, "error": "A run is already in progress."},
                            status_code=409)
    body = await request.json()
    mode = (body.get("mode") or "hunt").lower()
    cmd = [PYTHON_BIN, "-u", str(SCRIPT),
           "--reports-dir", str(REPORTS), "-o", str(REPORTS / "latest.csv")]
    if mode == "auto":
        cmd += ["--auto"]
        if body.get("expand_rounds") is not None:
            cmd += ["--expand-rounds", str(int(body["expand_rounds"]))]
    else:
        cmd += ["--hunt", str(int(body.get("target", 5))),
                "--hunt-max-rounds", str(int(body.get("max_rounds", 40)))]
    if body.get("min_score") is not None:
        cmd += ["--min-score", str(float(body["min_score"]))]
    if body.get("low_mem"):
        cmd += ["--low-mem"]
    if body.get("no_cash"):
        cmd += ["--no-cash-balance"]

    logf = open(LOG, "w", encoding="utf-8")
    popen_kwargs = dict(cwd=str(BASE), stdout=logf, stderr=subprocess.STDOUT)
    if os.name == "posix":
        popen_kwargs["preexec_fn"] = os.setsid          # own group -> clean SIGINT
    else:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    with _lock:
        proc = subprocess.Popen(cmd, **popen_kwargs)
        _state.update(proc=proc, started=time.time(), mode=mode, logf=logf)
    return {"ok": True, "mode": mode}


@app.post("/api/stop")
async def api_stop(request: Request):
    if not _authed(request):
        raise HTTPException(401)
    p = _state["proc"]
    if not _running():
        return {"ok": True, "note": "nothing running"}
    try:
        # SIGINT triggers the screener's graceful save of partial results.
        if os.name == "posix":
            os.killpg(os.getpgid(p.pid), signal.SIGINT)
        else:
            p.send_signal(signal.CTRL_BREAK_EVENT)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True}


@app.get("/api/status")
def api_status(request: Request):
    if not _authed(request):
        raise HTTPException(401)
    runs = _load_manifest()
    return {
        "running": _running(),
        "started": _state["started"],
        "mode": _state["mode"],
        "latest": runs[0] if runs else None,
        "runs": runs[:50],
        "log": _log_tail(),
    }


@app.get("/reports/{path:path}")
def serve_report(path: str, request: Request):
    if not _authed(request):
        return RedirectResponse("/login", status_code=302)
    target = (REPORTS / path).resolve()
    if not str(target).startswith(str(REPORTS.resolve())) or not target.exists():
        raise HTTPException(404)
    if target.is_dir():
        target = target / "index.html"
        if not target.exists():
            raise HTTPException(404)
    return FileResponse(target)


@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"


# --------------------------------------------------------------------------- #
# inline frontend
# --------------------------------------------------------------------------- #
_CSS = """
:root{--bg:#0b0e14;--panel:#11161f;--panel2:#161c27;--line:#222b39;--txt:#e6edf3;
--muted:#8b949e;--accent:#58a6ff;--green:#3fb950;--red:#f85149;--amber:#d29922;}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1100px 560px at 72% -12%,#16243a55,transparent),var(--bg);
color:var(--txt);font-family:'Inter',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;}
a{color:var(--accent);text-decoration:none}
.wrap{max-width:1200px;margin:0 auto;padding:26px 22px 80px}
.top{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);
padding-bottom:18px;margin-bottom:22px;flex-wrap:wrap;gap:12px}
.title{font-size:22px;font-weight:800;letter-spacing:-.02em;margin:0}
.title span{color:var(--accent)}
.grid{display:grid;grid-template-columns:340px 1fr;gap:18px}
@media(max-width:840px){.grid{grid-template-columns:1fr}}
.card{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);
border-radius:14px;padding:18px}
.card h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 14px}
label{display:block;font-size:12px;color:var(--muted);margin:12px 0 5px}
select,input[type=number]{width:100%;background:#0e131c;border:1px solid var(--line);color:var(--txt);
border-radius:9px;padding:9px 11px;font-size:14px}
.row{display:flex;gap:12px}.row>div{flex:1}
.chkrow{display:flex;align-items:center;gap:8px;margin-top:12px;color:var(--txt);font-size:13px}
.btns{display:flex;gap:10px;margin-top:18px}
button{flex:1;border:none;border-radius:10px;padding:11px;font-weight:700;font-size:14px;cursor:pointer}
.start{background:var(--green);color:#04240f}.start:disabled{opacity:.4;cursor:not-allowed}
.stop{background:#2a1416;color:#ff9a93;border:1px solid #5a2a2a}.stop:disabled{opacity:.4;cursor:not-allowed}
.statline{display:flex;align-items:center;gap:10px;font-size:14px;margin-bottom:10px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--muted)}
.dot.on{background:var(--green);box-shadow:0 0 0 4px #3fb95022}
pre{background:#0e131c;border:1px solid var(--line);border-radius:10px;padding:12px;font-size:11.5px;
color:var(--muted);max-height:230px;overflow:auto;white-space:pre-wrap;line-height:1.5;margin:0}
iframe{width:100%;height:560px;border:1px solid var(--line);border-radius:14px;background:var(--panel)}
.hist a{display:flex;gap:14px;align-items:center;padding:11px 12px;border:1px solid var(--line);
border-radius:10px;margin-top:8px;color:var(--txt)}
.hist a:hover{border-color:var(--accent)}
.hist .m{font-weight:700;text-transform:capitalize;width:90px}
.hist .t{color:var(--muted);font-size:12px;flex:1}
.hist b{color:var(--txt)}
.muted{color:var(--muted)}
.err{background:#2a1416;border:1px solid #5a2a2a;color:#ff9a93;padding:9px 12px;border-radius:9px;
margin-bottom:14px;font-size:13px}
.login{max-width:360px;margin:14vh auto 0;text-align:center}
.login input{width:100%;margin:14px 0}
.login button{width:100%}
"""

LOGIN_HTML = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Sign in</title>
<style>{_CSS}</style></head><body><div class=wrap><div class=login>
<h1 class=title>Polymarket <span>Screener</span></h1>
<p class=muted>Enter your password to continue.</p>
<!--ERR-->
<form method=post action=/login>
<input type=password name=password placeholder=Password autofocus>
<button class=start type=submit>Sign in</button>
</form></div></div></body></html>"""

APP_HTML = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Polymarket Screener</title>
<style>{_CSS}</style></head><body><div class=wrap>
<div class=top>
  <h1 class=title>Polymarket <span>Screener</span></h1>
  <div><span id=hb class=muted>—</span> &nbsp; <a href=/logout>Log out</a></div>
</div>
<div class=grid>
  <div>
    <div class=card>
      <h2>Run the bot</h2>
      <label>Mode</label>
      <select id=mode onchange=syncMode()>
        <option value=hunt>Hunt — until N qualify</option>
        <option value=auto>Auto — screen the whole pool</option>
      </select>
      <div class=row>
        <div id=targetBox><label>Target (qualified)</label><input id=target type=number value=5 min=1></div>
        <div><label>Min score</label><input id=minscore type=number value=55 min=0 max=100></div>
      </div>
      <div class=row>
        <div id=roundsBox><label>Max expansions</label><input id=maxrounds type=number value=40 min=0></div>
        <div id=expandBox style=display:none><label>Snowball rounds</label><input id=expand type=number value=1 min=0></div>
      </div>
      <div class=chkrow><input type=checkbox id=lowmem><label style=margin:0>Low-memory mode</label></div>
      <div class=chkrow><input type=checkbox id=nocash><label style=margin:0>Skip cash-balance lookup (faster)</label></div>
      <div class=btns>
        <button class=start id=startBtn onclick=startRun()>Start run</button>
        <button class=stop id=stopBtn onclick=stopRun() disabled>Stop</button>
      </div>
    </div>
    <div class=card style=margin-top:18px>
      <h2>Status</h2>
      <div class=statline><span class=dot id=dot></span><span id=statetxt>Idle</span></div>
      <div class=muted id=substat style=font-size:12px;margin-bottom:10px></div>
      <pre id=log>No run yet.</pre>
    </div>
  </div>
  <div>
    <div class=card style=margin-bottom:18px>
      <h2>Live dashboard</h2>
      <iframe id=frame src=about:blank></iframe>
    </div>
    <div class=card>
      <h2>Run history</h2>
      <div class=hist id=hist><div class=muted>No runs yet.</div></div>
    </div>
  </div>
</div>
</div>
<script>
function syncMode(){{
  var m=document.getElementById('mode').value;
  document.getElementById('targetBox').style.display = m=='hunt'?'block':'none';
  document.getElementById('roundsBox').style.display = m=='hunt'?'block':'none';
  document.getElementById('expandBox').style.display = m=='auto'?'block':'none';
}}
async function startRun(){{
  var b={{mode:document.getElementById('mode').value,
    target:+document.getElementById('target').value,
    max_rounds:+document.getElementById('maxrounds').value,
    expand_rounds:+document.getElementById('expand').value,
    min_score:+document.getElementById('minscore').value,
    low_mem:document.getElementById('lowmem').checked,
    no_cash:document.getElementById('nocash').checked}};
  var r=await fetch('/api/run',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(b)}});
  if(r.status==409){{alert('A run is already in progress.');return;}}
  poll();
}}
async function stopRun(){{ await fetch('/api/stop',{{method:'POST'}}); poll(); }}
var curFile=null;
async function poll(){{
  let s; try{{ s=await (await fetch('/api/status')).json(); }}catch(e){{ return; }}
  var on=s.running;
  document.getElementById('dot').className='dot'+(on?' on':'');
  document.getElementById('statetxt').textContent = on ? ('Running · '+(s.mode||'')) : 'Idle';
  document.getElementById('startBtn').disabled=on;
  document.getElementById('stopBtn').disabled=!on;
  document.getElementById('hb').textContent = on?'● live':'○ idle';
  document.getElementById('log').textContent = s.log || 'No output yet.';
  var L=s.latest;
  if(L){{
    document.getElementById('substat').textContent =
      L.qualified+' qualified / '+L.screened+' screened'+(L.live?' · updating…':' · done');
    if(L.file && L.file!=curFile){{ curFile=L.file; document.getElementById('frame').src='/reports/'+L.file; }}
  }}
  if(s.runs){{
    document.getElementById('hist').innerHTML = s.runs.map(function(r){{
      return '<a target=_blank href="/reports/'+r.file+'"><span class=m>'+r.mode+'</span>'+
        '<span class=t>'+r.human+'</span><span><b>'+r.qualified+'</b> qual · avg <b>'+r.avg_score+
        '</b></span></a>';}}).join('') || '<div class=muted>No runs yet.</div>';
  }}
}}
syncMode(); poll(); setInterval(poll, 4000);
</script>
</body></html>"""
