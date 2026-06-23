# Polymarket Copy-Trade Wallet Screener

Finds Polymarket wallets that are genuinely worth copy-trading — and now runs as a
password-protected website you can drive from any device.

There are two parts:

- **`wallet_screener.py`** — the engine. Discovers candidate wallets on its own,
  screens them on skill/consistency/copyability, and writes a styled, auto-updating
  HTML dashboard into `reports/`.
- **`server.py`** — a small FastAPI web app (login + control panel) that runs the
  screener on demand and serves the dashboards. Deployed on Railway.

---

## What the screener actually checks

A wallet has to clear every gate to qualify (all tunable near the top of
`wallet_screener.py` in `FILTERS`):

- **Real, still-held profit** — total P/L = realized **+** unrealized, so a wallet
  that booked gains but is bagholding losers (negative on its Polymarket profile)
  is dropped. Needs meaningful absolute profit, ROI ≥ 2%, profit factor ≥ 1.3, and
  to still hold ≥ 50% of its realized gains.
- **Demonstrated skill** — positive confidence-adjusted edge, or positive CLV
  (closing-line value) measured over ≥ 5 markets. Single-sample CLV is ignored.
- **Copyable sizing** — drops wallets whose stake range (p95/p5) is too wide to
  mirror above Polymarket's $1 minimum.
- **Active enough** — minimum bets per week.
- **Composite score floor** — must clear `min_score` overall.
- Plus the originals: market-maker detection, drawdown, loss-chasing, sample size,
  account age, recency.

It also reports each wallet's **balance** (open-position value + on-chain USDC cash)
and a category (Politics / Sports / Crypto / …) classified from market titles.

### How it finds wallets

`--auto` casts a wide net (full leaderboard sweep + trending-market holders +
curated tracker sites), then **snowballs**: every qualified wallet leads to the
other big holders of the markets it holds. `--hunt N` keeps discovering and
snowballing until N wallets qualify (or it exhausts leads / hits the round cap),
publishing results live as it goes.

---

## Run it locally

```bash
pip install -r requirements.txt

# CLI
python wallet_screener.py --hunt 5 --min-score 55      # hunt until 5 qualify
python wallet_screener.py --auto                       # screen a broad pool once
python wallet_screener.py --help                       # all flags

# Web app (control panel)
set APP_PASSWORD=your-password        # Windows;  export on macOS/Linux
set SECRET_KEY=some-long-random-string
uvicorn server:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

Output (HTML dashboards + CSVs + a browsable `index.html`) lands in `reports/`.

Handy flags: `--low-mem` (single worker, smaller CLV sample — for tiny machines),
`--no-cash-balance` (skip the on-chain lookup, faster), `--reports-dir`, `--out`.

---

## Deploy (Railway)

The web app is containerised (`Dockerfile`). Pushing to GitHub auto-deploys.

1. Railway → New Project → Deploy from GitHub repo → this repo.
2. Variables: `APP_PASSWORD`, `SECRET_KEY`.
3. Attach a **Volume at `/app/reports`** so run history persists.
4. Settings → Networking → **Generate Domain** → that's your public URL.

Full details and alternatives (Fly.io, a VPS) are in **`DEPLOY.md`**.

To run without low-memory mode, upgrade the Railway plan (Hobby ≈ 8 GB RAM ceiling).

---

## Working across machines

**GitHub is the source of truth.** Code is versioned; run output is not (`reports/`,
CSVs, and `.wallet_cache/` are git-ignored — run data lives on the Railway volume /
the website).

- Switching machines: **pull first** (GitHub Desktop → Fetch/Pull origin), edit,
  then **commit + push**. Railway auto-deploys on push.
- A fresh laptop: clone the repo, install Python + deps, point Cowork at the folder.

---

## File map

| File | Purpose |
| --- | --- |
| `wallet_screener.py` | The screening engine + CLI (auto / hunt modes, scoring, reports) |
| `server.py` | FastAPI web app: login, start/stop runs, live status, serves dashboards |
| `requirements.txt` | Python dependencies |
| `Dockerfile` / `.dockerignore` | Container build for cloud deploy |
| `DEPLOY.md` | Step-by-step cloud deployment guide |
| `reports/` | Generated dashboards, CSVs, `index.json` history (git-ignored) |

---

*This tool produces a shortlist to investigate, not financial advice. Copy-trading
is never passive — past performance is survivorship-biased, and CLV is noisy on
short-lived markets.*
