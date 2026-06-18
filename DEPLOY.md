# Deploying the Polymarket Screener web app

This turns the screener into a password-protected website you can open on any
device and use to start/stop runs and watch results stream in.

You'll deploy three files — `server.py`, `wallet_screener.py`, `requirements.txt`
(plus `Dockerfile`) — to an always-on cloud host. Below is the easy path
(Railway) and a couple of alternatives.

---

## What you need

- A **strong password** (you'll set it as `APP_PASSWORD`).
- A **random secret** for signing the login cookie (`SECRET_KEY`) — e.g. run
  `python -c "import secrets; print(secrets.token_hex(32))"` and copy the output.
- A GitHub repo containing these files (you already have Git / GitHub Desktop).

---

## Option A — Railway (easiest, ~$5/mo after trial credit)

1. **Put the project on GitHub.** Create a repo (e.g. `polymarket-screener-app`)
   and push `server.py`, `wallet_screener.py`, `requirements.txt`, `Dockerfile`,
   `.dockerignore`. (GitHub Desktop: add the folder, commit, publish.)
2. Go to **railway.app → sign in with GitHub → New Project → Deploy from GitHub
   repo** and pick the repo. Railway detects the `Dockerfile` and builds it.
3. Open the service → **Variables** → add:
   - `APP_PASSWORD` = your password
   - `SECRET_KEY` = the random secret you generated
4. **Add a Volume** (service → *Variables/Settings → Volumes*) mounted at
   **`/app/reports`**. This keeps your run history + dashboards across restarts
   and redeploys. (Without it, history resets when the app restarts.)
5. Under **Settings → Networking**, click **Generate Domain**. That public URL is
   your site — open it, log in, and run the bot from anywhere.

Railway terminates HTTPS for you, so the connection is encrypted.

---

## Option B — Fly.io (also cheap, CLI-based)

```bash
# one-time
curl -L https://fly.io/install.sh | sh
fly auth login
fly launch --no-deploy            # accept the Dockerfile; pick a name/region
fly volumes create reports --size 1
# mount it: add to fly.toml ->  [mounts]\n  source="reports"\n  destination="/app/reports"
fly secrets set APP_PASSWORD='your-password' SECRET_KEY='your-random-secret'
fly deploy
```

`fly open` launches the site. HTTPS is automatic.

---

## Option C — A plain VPS (e.g. Hetzner ~€4/mo)

```bash
# on the server (Docker installed)
git clone <your repo> app && cd app
docker build -t pmscreener .
docker run -d --restart unless-stopped \
  -e APP_PASSWORD='your-password' \
  -e SECRET_KEY='your-random-secret' \
  -p 80:8000 \
  -v /opt/pmreports:/app/reports \
  --name pmscreener pmscreener
```

For HTTPS, put **Caddy** in front (it auto-issues certificates):

```
yourdomain.com {
    reverse_proxy localhost:8000
}
```

---

## Using it

- Open the URL → enter your password.
- **Run the bot**: pick Hunt (until N qualify) or Auto, set min-score etc., press
  **Start run**. The status panel streams the log; the **Live dashboard** updates
  itself as wallets qualify.
- **Stop** saves whatever's been found so far (graceful interrupt).
- **Run history** lists every past run; click one to open its dashboard.

## Notes & limits

- **One run at a time** — the app refuses to start a second concurrent run.
- **Persistence** depends on the mounted volume (Options A/B/C all use one). The
  API cache (`.wallet_cache`) lives in the container and resets on redeploy —
  that only affects speed, not results.
- **Cost**: you enter card details directly with the host; this app itself is
  free. A single small instance (512MB–1GB RAM) is enough; use the in-app
  **Low-memory mode** toggle if you pick the smallest tier.
- **Scheduling in the cloud**: the daily scheduled task you set up runs on your
  PC. If you want the cloud app to auto-run each morning instead, most hosts have
  a built-in cron — point it at the container running
  `python wallet_screener.py --hunt 5 --min-score 55 --reports-dir reports`.
