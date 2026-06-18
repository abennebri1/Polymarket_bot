#!/usr/bin/env python3
"""
Polymarket copy-trade wallet screener — v2 (research-driven)
============================================================

Scores candidate wallets for copy-trading suitability from public on-chain data.
v2 folds in what the prediction-market and copy-trading literature actually says
separates durable skill from luck and from un-copyable wallets.

RESEARCH -> FEATURE MAP
-----------------------
* CLOSING-LINE VALUE (CLV) is the gold-standard skill metric in betting: it is
  ~variance-free and converges far faster than win-rate or ROI (a wallet with
  consistent +CLV is mathematically +EV; results catch up over time). We compute,
  for a sample of a wallet's bets, whether the market price moved TOWARD their
  position after they entered. This is the single strongest signal here.
      -> metrics.clv_entry  (--clv)

* "CAPTURABLE" EDGE. Sharp money enters early and the price moves right after
  them, so a copier who follows the trade is chasing a worse price — and a lot of
  the edge can already be gone. We also compute the edge that SURVIVES if you
  entered a short lag after them (clv_copier). A wallet can be sharp yet
  un-copyable; this number tells them apart.
      -> metrics.clv_copier (--clv)

* CATEGORY SPECIALISTS beat generalists ("smart money concentrates in one
  niche"). We measure how concentrated a wallet is in a single category and
  reward focused specialists (distinct from one-and-done insider wallets, which
  fail the sample-size / account-age filters anyway).
      -> metrics.top_category, category_concentration (--gamma)

* MAXIMUM DRAWDOWN + EQUITY-CURVE SMOOTHNESS is the top copy-trading risk metric
  (<20% manageable, >40-50% red flag). Spiky up-down curves = gambling/martingale.
      -> metrics.max_drawdown, sharpe

* MARTINGALE / LOSS-CHASING (bigger stakes after losses) is a hard red flag for
  copiers — one tail loss wipes a string of small wins. We detect stake
  escalation following losing bets.
      -> metrics.loss_chasing_ratio  (disqualifier)

* PROFIT FACTOR (>1.5) and risk-adjusted return, not raw PnL/win-rate.
      -> metrics.profit_factor

* MARKET-MAKER DETECTION stays (un-copyable): churn, both-sides, near-zero holds,
  rarely to resolution. Plus COPY-STYLE classification — hold-to-resolution
  wallets are far easier to copy than scalpers you'd have to exit in real time.
      -> metrics.maker_score, copy_style

* CONVERGENCE SCANNER (separate --convergence mode): when 3+ already-qualified
  wallets independently hold the same market/side right now, that's a strong
  live consensus signal worth acting on.

Everything operates on a normalised trades DataFrame, so you can also feed it
Polygonscan CSV exports via an adapter into normalise_activity().

Data sources (all public, no auth):
  data-api.polymarket.com  /v1/leaderboard, /activity, /positions
  gamma-api.polymarket.com /markets            (liquidity + category)   [--gamma]
  clob.polymarket.com      /prices-history      (CLV price trajectory)  [--clv]

LIMITATIONS (read these):
  - Past-performance selection is survivorship-biased. Output is a shortlist to
    investigate, not a buy-list. CLV mitigates but does not eliminate this.
  - CLV uses CLOB /prices-history at 12h fidelity (finer granularities return
    EMPTY for resolved markets — Polymarket CLOB limitation). It therefore only
    works on longer-lived markets (politics/macro/crypto); same-day sports
    markets are skipped and fall back to the edge metric. Run
    `--check-clv <wallet-or-token>` once against live data to confirm.
  - "Liquidity at trade time" is approximated by current market volume.
  - None of this is financial or betting advice; copy-trading is never passive.
  - Note: front-running suspected insiders sits in a legal/ethical grey area and
    venues have begun enforcing against it. This tool targets *repeatable skill*,
    not one-and-done insider hits (which it deliberately filters out).

Usage:
    python wallet_screener.py --auto                                   # autonomous discovery (NEW)
    python wallet_screener.py --auto --expand-rounds 2 --seed-url <url>
    python wallet_screener.py --hunt 5                                 # keep hunting until 5 qualify (NEW)
    python wallet_screener.py --leaderboard
    python wallet_screener.py --leaderboard --lb-count 50 --gamma --clv
    python wallet_screener.py --from-market <condition-id|slug|url> --clv   # holders seed
    python wallet_screener.py --file wallets.txt --clv -o ranked.csv --top 25
    python wallet_screener.py --file qualified.txt --convergence   # live consensus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------- #
# CONFIG — tune to your strategy.
# --------------------------------------------------------------------------- #

DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

# On-chain idle-cash lookup: a wallet's USDC.e balance (Polymarket's collateral)
# read from a Polygon RPC. Adds one eth_call per enriched wallet; best-effort.
POLYGON_RPC = "https://polygon-rpc.com"
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"   # USDC.e on Polygon
FETCH_CASH = True            # read on-chain USDC cash balance (toggle via --no-cash-balance)
RPC_URL = POLYGON_RPC        # overridable via --rpc-url

REQUEST_PAUSE = 0.2           # seconds between API calls (be polite / avoid 429s)
PAGE_SIZE = 500
MAX_PAGES = 40
SETTLE_DAYS = 14          # no activity for this long => treat market as resolved

# Closed-positions endpoint (Polymarket's own per-market realized PnL accounting).
CLOSED_PAGE = 50          # max rows per /closed-positions call (API cap)
CLOSED_MAX_PAGES = 50     # cap pages per wallet (recent-first); 50*50 = 2500 markets

# Batch price-history (POST /batch-prices-history): up to 20 asset ids per call.
BATCH_PRICE_MAX = 20

# Disk cache (re-runs reuse fetched data instead of re-downloading).
CACHE_DIR = ".wallet_cache"
CACHE_TTL_HOURS = 24      # responses older than this are refetched

# CLV (closing-line value) settings.
#
# Hard constraint from Polymarket's CLOB (py-clob-client issue #216): /prices-
# history returns EMPTY for resolved/closed markets at any fidelity finer than
# 12 hours. Since CLV is computed on a wallet's *settled* (resolved) markets, we
# MUST request 12h bars (fidelity=720) — finer values silently return nothing.
#
# At 12h resolution the metric measures a MULTI-DAY forward move: how far the
# price drifted toward the wallet's position over the days after entry, while the
# market was still live. We bound the horizon to a fraction of the market's life
# so a market that resolves inside the window can't collapse CLV into win/loss,
# and we skip short-lived markets entirely. Consequence: CLV is meaningful for
# longer markets (politics, macro, crypto) and NOT for same-day sports markets,
# which resolve before a 12h bar lands — those fall back to the edge metric.
CLV_SAMPLE = 40           # how many of a wallet's bets to price-trace (cost control)
CLV_FIDELITY = 720        # price-history resolution in minutes (12h; required for resolved mkts)
CLV_HORIZON_HOURS = 72    # measure the forward price move this long after entry
CLV_DELAY_HOURS = 12      # a realistic delayed fill for a copier at this resolution
CLV_MIN_LIFESPAN_HOURS = 36  # skip markets that resolved sooner than this after entry
CLV_GUARD_FRAC = 0.8      # never measure past this fraction of entry->last sample
CLV_MIN_SAMPLE = 5        # need this many priced markets before CLV is trusted/credited
CLV_SKILL_MIN = 0.01      # CLV must clear this to count as a "demonstrated skill" signal

# Convergence scanner
CONVERGE_MIN_WALLETS = 3 # >= this many qualified wallets same market+side = signal

# --------------------------------------------------------------------------- #
# AUTONOMOUS DISCOVERY (--auto): how the screener finds candidates ON ITS OWN.
# Instead of one fixed leaderboard slice or markets you name by hand, --auto
# casts a wide net across several public sources, screens the pool, then lets the
# wallets that qualify lead it to more candidates (snowball).
# --------------------------------------------------------------------------- #
AUTO_LB_PERIODS = ["DAY", "WEEK", "MONTH", "ALL"]  # leaderboard windows to sweep
AUTO_LB_ORDERS = ["PNL", "VOL"]                    # rank by profit AND by volume
AUTO_LB_CATEGORIES = ["OVERALL"]                   # add category slugs to widen the sweep
AUTO_TRENDING_MARKETS = 30    # how many hot live markets to mine holders from
AUTO_EXPAND_ROUNDS = 1        # snowball rounds off qualified wallets (0 = off)
AUTO_EXPAND_MARKETS = 60      # cap markets inspected per snowball round
AUTO_MAX_CANDIDATES = 1500    # safety cap on pool size handed to the screener per round
# Hunt mode (--hunt N): keep discovering & snowballing until N wallets QUALIFY.
HUNT_TARGET = 5               # default number of good-enough wallets to find
HUNT_BATCH = 80              # screen this many at a time, checking the target after each
HUNT_MAX_ROUNDS = 15         # safety cap on snowball expansions
WEB_ADDR_RE = re.compile(r"0x[a-fA-F0-9]{40}")     # full 0x wallet addresses on a web page
# Truncated form many tracker sites display, e.g. "0x0197...9f3c" or "0x4924…3782".
# We can't use these directly, but we CAN match them back to full addresses we've
# already gathered from Polymarket's on-chain data (prefix + suffix is ~unique).
WEB_TRUNC_RE = re.compile(r"0x([a-fA-F0-9]{3,8})\s*(?:\.{2,3}|…)\s*([a-fA-F0-9]{3,6})")

# Curated "places that track good Polymarket wallets". Scanned by --auto unless
# --no-web. NOTE: most of these truncate addresses or render client-side (JS), so
# a plain fetch often yields little — the scanner extracts what it can and matches
# truncated handles against on-chain data. For full-address pages (gists, CSVs, or
# a JS dashboard you've SAVED to a local .html file), add them with --seed-url
# (a URL or a local file path both work).
KNOWN_WALLET_SOURCES = [
    "https://polymonit.com/leaderboard/top-profitable-polymarket-wallets",
    "https://polymonit.com/leaderboard/best-polymarket-wallets-to-follow",
    "https://polymonit.com/leaderboard/april-2026",
    "https://polymonit.com/leaderboard/polymarket-whales",
    "https://polymarketanalytics.com/traders",
    "https://www.walletmaster.tools/polymarket-wallet-tracker/",
]

# Hard disqualifiers — fail any and the wallet is dropped before scoring.
FILTERS = {
    "min_settled_markets": 30,
    "min_account_age_days": 60,
    "max_days_since_active": 21,
    "min_realized_pnl": 0.0,
    # --- consistency / real-skill gates (need accurate PnL + CLV; applied post-enrich) ---
    "min_total_pnl": 1000.0,        # realized + unrealized net profit, in $ (matches Polymarket)
    "min_total_roi": 0.02,          # that net profit must be >=2% of capital deployed
    "min_profit_factor": 1.30,      # gross wins / gross losses
    "min_kept_pnl_frac": 0.50,      # must still HOLD >=50% of realized gains (anti-bagholding)
    # --- behavioural / copyability gates (cheap; applied pre-enrich) ---
    "max_maker_score": 0.55,        # drop likely market-makers (un-copyable)
    "max_stake_cv": 3.0,
    "max_stake_range_ratio": 400,   # p95/p5 stake spread; wider = un-copyable at the $1 floor
    "min_trades_per_week": 5,       # must bet often enough to be worth copying week-to-week
    "max_pnl_top1_share": 0.60,     # drop one-lucky-hit wallets
    "max_drawdown": 0.60,           # worst drop > 60% of all capital ever staked = catastrophic
    "max_loss_chasing": 1.60,       # drop martingale/recovery bettors
    "max_trades_per_week": 800,     # drop un-copyable HFT bots (fees would eat any edge)
    # --- final quality bar (applied after the composite score is computed) ---
    "min_score": 55.0,              # composite score floor; below this isn't "good enough"
}

# Composite score weights (normalised internally).
WEIGHTS = {
    "skill": 0.30,          # edge-over-price, upgraded by CLV when available
    "profitability": 0.16,  # ROI + profit factor + monthly consistency
    "risk": 0.16,           # low drawdown + Sharpe + no loss-chasing
    "breadth": 0.10,        # profit spread, not concentrated in one market
    "copyability": 0.16,    # low MM-likeness + liquidity + copy-style + specialism
    "robustness": 0.12,     # sample size + account age + recency
}

LB_DEFAULTS = {"category": "OVERALL", "timePeriod": "MONTH", "orderBy": "PNL", "count": 100}
LB_PAGE = 50
LB_MAX_OFFSET = 1000


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #

class PolymarketClient:
    def __init__(self, use_gamma=False, use_clv=False, cache=True,
                 cache_dir=CACHE_DIR, cache_ttl_hours=CACHE_TTL_HOURS):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "wallet-screener/2.0"})
        self.use_gamma = use_gamma
        self.use_clv = use_clv
        self._market_cache: dict[str, dict] = {}
        self._price_cache: dict[str, list] = {}
        self.cache = cache
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl_hours * 3600
        if self.cache:
            os.makedirs(self.cache_dir, exist_ok=True)

    # ---- disk cache: re-runs reuse responses instead of refetching --------- #
    def _cache_path(self, method, url, params, body):
        raw = f"{method}|{url}|{json.dumps(params, sort_keys=True, default=str)}" \
              f"|{json.dumps(body, sort_keys=True, default=str)}"
        return os.path.join(self.cache_dir, hashlib.sha1(raw.encode()).hexdigest() + ".json")

    def _cache_read(self, path):
        try:
            if time.time() - os.path.getmtime(path) > self.cache_ttl:
                return None                       # stale
            with open(path) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None                           # missing / corrupt

    def _cache_write(self, path, value):
        try:
            tmp = path + f".{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "w") as fh:
                json.dump(value, fh)
            os.replace(tmp, path)                 # atomic
        except (OSError, TypeError):
            pass                                  # caching is best-effort

    def _get(self, url, params=None, tries=4):
        if self.cache:
            path = self._cache_path("GET", url, params, None)
            hit = self._cache_read(path)
            if hit is not None:
                return hit
        for attempt in range(tries):
            try:
                r = self.s.get(url, params=params, timeout=25)
                if r.status_code == 429:
                    time.sleep(1.5 * (attempt + 1)); continue
                r.raise_for_status()
                time.sleep(REQUEST_PAUSE)
                data = r.json()
                if self.cache and data is not None:
                    self._cache_write(path, data)
                return data
            except requests.RequestException as e:
                if attempt == tries - 1:
                    print(f"  ! request failed ({e})", file=sys.stderr)
                    return None
                time.sleep(1.0 * (attempt + 1))
        return None

    def _post(self, url, body, tries=4):
        if self.cache:
            path = self._cache_path("POST", url, None, body)
            hit = self._cache_read(path)
            if hit is not None:
                return hit
        for attempt in range(tries):
            try:
                r = self.s.post(url, json=body, timeout=25)
                if r.status_code == 429:
                    time.sleep(1.5 * (attempt + 1)); continue
                r.raise_for_status()
                time.sleep(REQUEST_PAUSE)
                data = r.json()
                if self.cache and data is not None:
                    self._cache_write(path, data)
                return data
            except requests.RequestException as e:
                if attempt == tries - 1:
                    print(f"  ! request failed ({e})", file=sys.stderr)
                    return None
                time.sleep(1.0 * (attempt + 1))
        return None

    def fetch_leaderboard(self, category="OVERALL", time_period="MONTH",
                          order_by="PNL", count=100):
        out, offset = [], 0
        while len(out) < count and offset <= LB_MAX_OFFSET:
            batch = self._get(f"{DATA_API}/v1/leaderboard", {
                "category": category, "timePeriod": time_period, "orderBy": order_by,
                "limit": min(LB_PAGE, count - len(out)), "offset": offset})
            if not batch:
                break
            out.extend(batch)
            if len(batch) < LB_PAGE:
                break
            offset += LB_PAGE
        return out[:count]

    def fetch_activity(self, wallet):
        """Full TRADE/REDEEM history, paged by descending timestamp.

        The Data API caps `offset` (high offsets 400), and offset paging from the
        oldest end silently truncates active wallets to their earliest events.
        We page newest-first using the `end` timestamp filter instead — no offset
        cap, full history. Dedupe + a no-progress guard protect against `end`
        being ignored.
        """
        rows, end_ts, prev_oldest, seen = [], None, None, set()
        for _ in range(MAX_PAGES):
            params = {"user": wallet, "type": "TRADE,REDEEM", "limit": PAGE_SIZE,
                      "sortBy": "TIMESTAMP", "sortDirection": "DESC"}
            if end_ts is not None:
                params["end"] = end_ts
            batch = self._get(f"{DATA_API}/activity", params)
            if not batch:
                break
            new = 0
            for r in batch:
                key = (r.get("transactionHash"), r.get("asset"), r.get("side"),
                       r.get("size"), r.get("timestamp"))
                if key in seen:
                    continue
                seen.add(key); rows.append(r); new += 1
            if len(batch) < PAGE_SIZE or new == 0:
                break
            oldest = min(int(_to_float(r.get("timestamp"), 0)) for r in batch)
            if oldest <= 0 or (prev_oldest is not None and oldest >= prev_oldest):
                break
            prev_oldest = oldest
            end_ts = oldest - 1
        return rows

    def fetch_positions(self, wallet):
        """Current OPEN positions (for the convergence scanner)."""
        data = self._get(f"{DATA_API}/positions", {"user": wallet, "limit": 500})
        return data if isinstance(data, list) else []

    def fetch_recent_ts(self, wallet):
        """Timestamp of a wallet's single most recent event — one tiny call, used
        to skip dormant wallets before downloading their whole history."""
        data = self._get(f"{DATA_API}/activity", {
            "user": wallet, "type": "TRADE,REDEEM", "limit": 1,
            "sortBy": "TIMESTAMP", "sortDirection": "DESC"})
        if isinstance(data, list) and data:
            return int(_to_float(data[0].get("timestamp"), 0))
        return None

    def fetch_first_ts(self, wallet):
        """Timestamp of a wallet's OLDEST event — one tiny call. Gives the TRUE
        account age regardless of the history-fetch cap, so very active wallets
        aren't mislabelled 'too new' just because we only pulled their recent
        events."""
        data = self._get(f"{DATA_API}/activity", {
            "user": wallet, "type": "TRADE,REDEEM", "limit": 1,
            "sortBy": "TIMESTAMP", "sortDirection": "ASC"})
        if isinstance(data, list) and data:
            return int(_to_float(data[0].get("timestamp"), 0))
        return None

    def fetch_markets_meta(self, condition_ids):
        """Batched Gamma metadata for many markets at once -> {cid: meta}.

        Replaces one-call-per-market (a wallet with 300 markets was 300 calls);
        batches uncached ids ~25 at a time. Misses are cached as {} so we never
        refetch them within a run.
        """
        if not self.use_gamma:
            return {}
        uniq = [c for c in dict.fromkeys(condition_ids) if c and c not in self._market_cache]
        for i in range(0, len(uniq), 25):
            chunk = uniq[i:i + 25]
            data = self._get(f"{GAMMA_API}/markets",
                             {"condition_ids": chunk, "limit": len(chunk)})
            got = {}
            if isinstance(data, list):
                for mm in data:
                    cid = mm.get("conditionId")
                    if cid:
                        got[cid] = {
                            "volume": _to_float(mm.get("volumeNum", mm.get("volume"))),
                            "liquidity": _to_float(mm.get("liquidityNum", mm.get("liquidity"))),
                            "category": mm.get("category") or "unknown",
                        }
            for cid in chunk:
                self._market_cache[cid] = got.get(cid, {})
        return {c: self._market_cache.get(c, {}) for c in dict.fromkeys(condition_ids)}

    def fetch_active_markets(self, limit=AUTO_TRENDING_MARKETS, min_volume=0):
        """Highest-volume currently-LIVE markets, for autonomous holder mining.

        Pulls active/open markets from Gamma sorted by volume, then sorts locally
        as a backstop in case the server ignores `order`. Returns a deduped list
        of condition ids — the 'go where the action is' discovery source.
        """
        data = self._get(f"{GAMMA_API}/markets", {
            "active": "true", "closed": "false", "archived": "false",
            "order": "volumeNum", "ascending": "false",
            "limit": max(limit * 2, limit)})
        rows = []
        if isinstance(data, list):
            for m in data:
                cid = m.get("conditionId")
                vol = _to_float(m.get("volumeNum", m.get("volume")))
                if cid and not np.isnan(vol) and vol >= min_volume:
                    rows.append((vol, cid))
        rows.sort(key=lambda x: -x[0])
        out, seen = [], set()
        for _, cid in rows:
            if cid not in seen:
                seen.add(cid); out.append(cid)
            if len(out) >= limit:
                break
        return out

    def fetch_top_holders(self, condition_ids, limit=20, min_balance=1):
        """Top holders of one or more markets (by condition id).

        /holders caps limit at 20 holders PER outcome token, and `market` is a
        comma-separated list of condition ids. Returns a flat list of dicts:
        {wallet, name, outcome_index, amount, token}.
        """
        out = []
        data = self._get(f"{DATA_API}/holders", {
            "market": ",".join(condition_ids),
            "limit": min(max(limit, 1), 20),
            "minBalance": min_balance})
        if isinstance(data, list):
            for meta in data:
                token = meta.get("token", "")
                for h in meta.get("holders", []):
                    w = (h.get("proxyWallet") or "").lower()
                    if w.startswith("0x") and len(w) == 42:
                        out.append({
                            "wallet": w,
                            "name": (h.get("name") or h.get("pseudonym") or "")[:20],
                            "outcome_index": h.get("outcomeIndex"),
                            "amount": h.get("amount"),
                            "token": token,
                        })
        return out

    def resolve_market_slug(self, slug):
        """Resolve a market OR event slug to condition id(s) via Gamma."""
        cids = []
        m = self._get(f"{GAMMA_API}/markets", {"slug": slug})
        if isinstance(m, list):
            cids += [mk.get("conditionId") for mk in m if mk.get("conditionId")]
        if not cids:                          # maybe it's an event slug
            e = self._get(f"{GAMMA_API}/events", {"slug": slug})
            if isinstance(e, list):
                for ev in e:
                    cids += [mk.get("conditionId") for mk in ev.get("markets", [])
                             if mk.get("conditionId")]
        return [c for c in cids if c]

    def market_meta(self, condition_id):
        """Gamma liquidity + category (cached)."""
        if not self.use_gamma or not condition_id:
            return {}
        if condition_id in self._market_cache:
            return self._market_cache[condition_id]
        data = self._get(f"{GAMMA_API}/markets", {"condition_ids": condition_id})
        meta = {}
        if isinstance(data, list) and data:
            m = data[0]
            cat = m.get("category")
            if not cat:
                tags = m.get("tags") or m.get("events", [{}])[0].get("tags") if m.get("events") else None
                if isinstance(tags, list) and tags:
                    cat = tags[0] if isinstance(tags[0], str) else tags[0].get("label")
            meta = {
                "volume": _to_float(m.get("volumeNum", m.get("volume"))),
                "liquidity": _to_float(m.get("liquidityNum", m.get("liquidity"))),
                "category": cat or "unknown",
            }
        self._market_cache[condition_id] = meta
        return meta

    def price_history(self, token_id, start_ts=None, end_ts=None, interval=None):
        """CLOB price trajectory for an outcome token (cached). For CLV.

        /prices-history REQUIRES `market` (the asset/token id) plus EITHER a
        [startTs, endTs] window OR an `interval`. Passing only `market` returns
        nothing — that was the original bug. We default to a windowed fetch so we
        pull just the slice around the wallet's entry. Response: {'history':
        [{'t':unix,'p':price}]}.
        """
        if not self.use_clv or not token_id:
            return []
        ckey = f"{token_id}:{start_ts}:{end_ts}:{interval}"
        if ckey in self._price_cache:
            return self._price_cache[ckey]
        params = {"market": token_id, "fidelity": CLV_FIDELITY}
        if interval:
            params["interval"] = interval
        else:
            if start_ts is not None:
                params["startTs"] = int(start_ts)
            if end_ts is not None:
                params["endTs"] = int(end_ts)
            if start_ts is None and end_ts is None:
                params["interval"] = "max"      # safe fallback: full history
        data = self._get(f"{CLOB_API}/prices-history", params)
        hist = []
        if isinstance(data, dict):
            for pt in data.get("history", []):
                t = pt.get("t") or pt.get("timestamp")
                p = pt.get("p") or pt.get("price")
                if t is not None and p is not None:
                    hist.append((int(_to_float(t)), _to_float(p)))
        hist.sort()
        self._price_cache[ckey] = hist
        return hist

    def batch_price_history(self, tokens, interval="max", fidelity=CLV_FIDELITY):
        """Fetch many tokens' price histories in one POST instead of one GET each
        (POST /batch-prices-history, max 20 ids/call). Populates the same cache
        price_history() reads, so CLV computation can pull from it for free.
        Returns {token: [(t, p), ...]}."""
        if not self.use_clv:
            return {}
        want = [t for t in dict.fromkeys(tokens) if t]
        out = {}
        for i in range(0, len(want), BATCH_PRICE_MAX):
            chunk = want[i:i + BATCH_PRICE_MAX]
            data = self._post(f"{CLOB_API}/batch-prices-history",
                              {"markets": chunk, "interval": interval, "fidelity": fidelity})
            if not isinstance(data, dict):
                continue            # batch failed -> leave uncached so price_history() retries singly
            hist_map = data.get("history", {}) or {}
            for tok in chunk:
                series = []
                for pt in (hist_map.get(tok) or []):
                    t = pt.get("t") or pt.get("timestamp")
                    p = pt.get("p") or pt.get("price")
                    if t is not None and p is not None:
                        series.append((int(_to_float(t)), _to_float(p)))
                series.sort()
                out[tok] = series
                self._price_cache[f"{tok}:None:None:{interval}"] = series   # share with price_history()
        return out

    def fetch_closed_positions(self, wallet):
        """Polymarket's own per-market accounting for fully-closed positions:
        realizedPnl, avgPrice, totalBought, timestamp. More accurate than
        reconstructing PnL from raw cashflows. Paged newest-first, capped."""
        rows = []
        for page in range(CLOSED_MAX_PAGES):
            batch = self._get(f"{DATA_API}/closed-positions", {
                "user": wallet, "limit": CLOSED_PAGE, "offset": page * CLOSED_PAGE,
                "sortBy": "TIMESTAMP", "sortDirection": "DESC"})
            if not isinstance(batch, list) or not batch:
                break
            rows.extend(batch)
            if len(batch) < CLOSED_PAGE:
                break
        return rows


# --------------------------------------------------------------------------- #
# Normalisation — raw activity -> clean trades DataFrame.
# Adapt THIS for Polygonscan CSV input.
# --------------------------------------------------------------------------- #

def _to_float(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def fetch_usdc_balance(wallet, session=None, rpc_url=None):
    """Idle USDC.e cash balance of a wallet, read on-chain via a Polygon RPC
    eth_call to balanceOf(address). Best-effort: returns nan if the RPC is down or
    rate-limited (USDC.e has 6 decimals). This is the cash NOT currently in any
    position; add it to portfolio_value for a true total balance."""
    rpc = rpc_url or RPC_URL
    addr = wallet.lower().replace("0x", "")
    if len(addr) != 40:
        return np.nan
    data = "0x70a08231" + addr.rjust(64, "0")          # balanceOf(address) selector
    body = {"jsonrpc": "2.0", "id": 1, "method": "eth_call",
            "params": [{"to": USDC_ADDRESS, "data": data}, "latest"]}
    try:
        poster = session or requests
        r = poster.post(rpc, json=body, timeout=15)
        r.raise_for_status()
        res = r.json().get("result")
        if res and res not in ("0x", "0x0"):
            return int(res, 16) / 1e6
        if res == "0x0":
            return 0.0
    except (requests.RequestException, ValueError, TypeError):
        return np.nan
    return np.nan


def _wilson_lower(wins, n, z=1.96):
    """Lower bound of the Wilson score interval for a win rate. Discounts small
    samples: 24/40 and 2400/4000 are both 60%, but this returns a much lower,
    more honest floor for the 40-bet case. Used so edge isn't over-credited on
    thin samples."""
    if n <= 0:
        return 0.0
    phat = wins / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt(max(phat * (1 - phat) + z * z / (4 * n), 0) / n)
    return max(0.0, (centre - margin) / denom)


def _est_fee(price):
    """Estimated Polymarket per-side fee at a given price: max(0.003,
    0.07*p*(1-p)). Highest near 0.5, lowest at the extremes."""
    p = min(max(price, 0.01), 0.99)
    return max(0.003, 0.07 * p * (1 - p))


# Category classification straight from the market TITLE. Gamma's /markets no
# longer returns a `category` (or `tags`) field, so the old per-market lookup
# always came back "unknown". Titles are already in the activity feed, so this
# needs no extra API calls. First matching bucket wins; order matters (more
# specific buckets first).
_CATEGORY_KEYWORDS = [
    ("Crypto",   ["bitcoin", "btc", "ethereum", " eth ", "solana", " sol ", "crypto",
                  "dogecoin", "xrp", "ripple", "altcoin", "stablecoin", "memecoin",
                  "binance", "coinbase", "satoshi", "nft", "$btc", "$eth"]),
    ("Sports",   ["nba", "nfl", "mlb", "nhl", "ufc", "fifa", "premier league", "la liga",
                  "champions league", "super bowl", "world cup", "playoff", "vs.",
                  " vs ", "match", "tournament", "open ", "grand prix", "f1 ",
                  "boxing", "soccer", "football", "basketball", "baseball", "tennis",
                  "golf", "cricket", "olympics", "wimbledon"]),
    ("Politics", ["election", "president", "senate", "congress", "governor", "primary",
                  "republican", "democrat", "gop", "parliament", "prime minister",
                  "trump", "biden", "putin", "vote", "ballot", "nominee", "cabinet",
                  "supreme court", "impeach", "referendum", "poll"]),
    ("Economy",  ["fed", "interest rate", "inflation", "cpi", "gdp", "recession",
                  "unemployment", "jobs report", "rate cut", "rate hike", "tariff",
                  "stock", "s&p", "nasdaq", "dow ", "earnings", "ipo"]),
    ("Geopolitics", ["war", "ceasefire", "peace deal", "invade", "invasion", "nato",
                     "nuclear", "sanction", "hostage", "missile", "treaty", "border",
                     "ukraine", "russia", "israel", "gaza", "iran", "china", "taiwan"]),
    ("Tech & AI", ["ai ", "openai", "chatgpt", "gpt-", "llm", "google", "apple",
                   "microsoft", "tesla", "spacex", "nvidia", "meta ", "twitter", " x ",
                   "tiktok", "iphone", "launch", "release", "agi"]),
    ("Pop Culture", ["movie", "box office", "oscar", "grammy", "album", "song",
                     "spotify", "netflix", "celebrity", "kardashian", "taylor swift",
                     "tv ", "show", "rotten tomatoes", "award", "billboard"]),
]


def _classify_category(title):
    """Bucket a market into a coarse category from its title. Returns 'other' when
    nothing matches. Cheap, deterministic, no network."""
    if not title:
        return "other"
    t = f" {str(title).lower()} "
    for cat, kws in _CATEGORY_KEYWORDS:
        for kw in kws:
            if kw in t:
                return cat
    return "other"


def normalise_activity(rows):
    recs = []
    for r in rows:
        usdc = _to_float(r.get("usdcSize"))
        size = _to_float(r.get("size"))
        price = _to_float(r.get("price"))
        if np.isnan(usdc) and not np.isnan(size) and not np.isnan(price):
            usdc = size * price
        recs.append({
            "condition_id": r.get("conditionId", ""),
            "asset": r.get("asset", ""),                 # outcome token id (for CLV)
            "type": (r.get("type") or "").upper(),
            "side": (r.get("side") or "").upper(),
            "usdc": usdc, "shares": size, "price": price,
            "ts": int(_to_float(r.get("timestamp"), 0)),
            "outcome": r.get("outcome", ""),
            "title": r.get("title", ""),
        })
    df = pd.DataFrame(recs)
    if not df.empty:
        df = df[df["condition_id"] != ""].copy()
    return df


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

@dataclass
class WalletMetrics:
    wallet: str
    n_trades: int = 0
    n_markets: int = 0
    n_settled: int = 0
    n_closed: int = 0                # closed positions from Polymarket accounting
    account_age_days: float = 0.0
    days_since_active: float = 1e9
    # profitability
    realized_pnl: float = 0.0
    open_unrealized_pnl: float = np.nan   # mark-to-market on CURRENT open positions
    total_pnl: float = np.nan             # realized + unrealized (matches Polymarket profile)
    portfolio_value: float = np.nan       # current market value of open positions ("balance")
    cash_balance: float = np.nan          # idle USDC.e cash, read on-chain
    total_balance: float = np.nan         # portfolio_value + cash_balance
    n_open: int = 0
    total_staked: float = 0.0
    roi: float = 0.0
    profit_factor: float = np.nan
    profitable_month_frac: float = 0.0
    # skill
    win_rate: float = np.nan
    avg_entry_price: float = np.nan
    edge: float = np.nan
    edge_lb: float = np.nan          # confidence-adjusted edge (Wilson lower bound)
    net_edge: float = np.nan         # edge after estimated round-trip fees
    pnl_source: str = "estimate"     # "closed_positions" once Polymarket accounting is used
    clv_entry: float = np.nan       # their closing-line value (skill)
    clv_copier: float = np.nan      # edge left for a copier after lag (capturable)
    n_clv: int = 0
    # risk
    max_drawdown: float = np.nan
    sharpe: float = np.nan
    loss_chasing_ratio: float = np.nan
    worst_market_pnl: float = 0.0
    # breadth
    pnl_top1_share: float = 1.0
    pnl_top5_share: float = 1.0
    # sizing / behaviour
    stake_median: float = 0.0
    stake_cv: float = np.nan
    stake_p5: float = np.nan
    stake_p95: float = np.nan
    stake_range_ratio: float = np.nan   # p95/p5 — how hard the sizing is to mirror
    maker_score: float = np.nan
    median_hold_hours: float = np.nan
    redeem_rate: float = np.nan
    both_sides_frac: float = np.nan
    trades_per_market: float = np.nan
    trades_per_week: float = np.nan
    copy_style: str = "?"
    # specialisation / liquidity
    top_category: str = ""
    category_concentration: float = np.nan
    median_market_liquidity: float = np.nan
    trade_to_liquidity: float = np.nan
    # output
    score: float = 0.0
    components: dict = field(default_factory=dict)
    disqualified: list = field(default_factory=list)


def _clv_for_market(g, client):
    """Closing-line value for one market's buys, at the 12h resolution that
    resolved markets support.

    Measures the MULTI-DAY forward move of the token price toward the wallet's
    position. Works for Yes or No buys identically (`asset` is the token they
    bought, so a price rise after their buy is favourable either way). Skips
    short-lived markets (e.g. same-day sports) where a 12h bar can't separate the
    forward move from resolution.

    Returns (clv_entry, clv_copier) in probability points:
      clv_entry  = price ~horizon after entry  - their entry price   (sharpness)
      clv_copier = price ~horizon after entry  - price ~12h after     (capturable
                   edge left for a copier who can't fill instantly)
    or (nan, nan) when unavailable.
    """
    buys = g[(g["type"] == "TRADE") & (g["side"] == "BUY")]
    if buys.empty:
        return np.nan, np.nan
    token = buys["asset"].iloc[0]
    if not token:
        return np.nan, np.nan
    entry_ts = int(buys["ts"].mean())
    entry_px = float(buys["price"].mean())
    if np.isnan(entry_px):
        return np.nan, np.nan

    # one call per token: full history at 12h bars (finer fidelity => empty for
    # resolved markets). We interpolate the points we need from it.
    hist = client.price_history(token, interval="max")
    if len(hist) < 3:
        return np.nan, np.nan
    ts = np.array([h[0] for h in hist]); px = np.array([h[1] for h in hist])
    last_ts = ts[-1]
    if (last_ts - entry_ts) / 3600.0 < CLV_MIN_LIFESPAN_HOURS:
        return np.nan, np.nan      # too short-lived (sports-like) to measure

    horizon_ts = min(entry_ts + CLV_HORIZON_HOURS * 3600,
                     entry_ts + CLV_GUARD_FRAC * (last_ts - entry_ts))
    if horizon_ts <= entry_ts:
        return np.nan, np.nan
    close_px = float(np.interp(horizon_ts, ts, px))
    delay_px = float(np.interp(min(entry_ts + CLV_DELAY_HOURS * 3600, horizon_ts), ts, px))
    return close_px - entry_px, close_px - delay_px


def compute_metrics(wallet, df, account_age_days=None, days_since_active=None):
    """CHEAP metrics from activity only — no CLV/Gamma calls. Returns (m, mk),
    where mk is the per-market summary reused later for enrichment. The hard
    disqualifiers all read from these cheap metrics, so we can drop bad wallets
    before spending any price-history / Gamma calls on them.

    account_age_days / days_since_active can be passed in from the cheap
    oldest/newest-event calls so they reflect TRUE account age even when the
    history fetch is capped on a very active wallet (otherwise a truncated
    history makes an old wallet look 'too new')."""
    m = WalletMetrics(wallet=wallet)
    if df.empty:
        m.disqualified.append("no_activity")
        return m, pd.DataFrame()

    now = time.time()
    trades = df[df["type"] == "TRADE"].copy()
    m.n_trades = len(trades)
    m.n_markets = df["condition_id"].nunique()
    first_ts, last_ts = df["ts"].min(), df["ts"].max()
    window_days = (last_ts - first_ts) / 86400 if last_ts > first_ts else 0.0  # fetched span
    m.account_age_days = (account_age_days if account_age_days is not None
                          else window_days)
    m.days_since_active = (days_since_active if days_since_active is not None
                           else (now - last_ts) / 86400)

    # per-market reconstruction (no API calls)
    rows = []
    for cid, g in df.groupby("condition_id"):
        buys = g[(g["type"] == "TRADE") & (g["side"] == "BUY")]
        sells = g[(g["type"] == "TRADE") & (g["side"] == "SELL")]
        redeems = g[g["type"] == "REDEEM"]
        spent = buys["usdc"].sum()
        received = sells["usdc"].sum() + redeems["usdc"].sum()
        shares_bought = buys["shares"].sum()
        m_last = g["ts"].max()
        settled = (len(redeems) > 0) or ((now - m_last) / 86400 > SETTLE_DAYS)
        rows.append({
            "cid": cid, "spent": spent, "received": received, "pnl": received - spent,
            "avg_entry": spent / shares_bought if shares_bought > 0 else np.nan,
            "won": received > spent, "settled": settled, "n_trades": len(g),
            "has_buy": len(buys) > 0, "has_sell": len(sells) > 0,
            "redeemed": len(redeems) > 0,
            "first_ts": g["ts"].min(), "last_ts": m_last,
        })
    mk = pd.DataFrame(rows)
    settled = mk[mk["settled"]].copy()
    m.n_settled = len(settled)

    # ---- profitability ---------------------------------------------------- #
    m.realized_pnl = float(settled["pnl"].sum()) if not settled.empty else 0.0
    m.total_staked = float(trades[trades["side"] == "BUY"]["usdc"].sum())
    m.roi = m.realized_pnl / m.total_staked if m.total_staked > 0 else 0.0
    if not settled.empty:
        gw = settled.loc[settled["pnl"] > 0, "pnl"].sum()
        gl = -settled.loc[settled["pnl"] < 0, "pnl"].sum()
        m.profit_factor = float(gw / gl) if gl > 0 else float("inf")
        s = settled.copy()
        s["month"] = pd.to_datetime(s["last_ts"], unit="s").dt.to_period("M")
        monthly = s.groupby("month")["pnl"].sum()
        m.profitable_month_frac = float((monthly > 0).mean()) if len(monthly) else 0.0

    # ---- skill: edge over price (CLV is added later, in enrich_metrics) ---- #
    if not settled.empty:
        m.win_rate = float(settled["won"].mean())
        ve = settled["avg_entry"].dropna()
        if len(ve):
            m.avg_entry_price = float(ve.mean())
            m.edge = m.win_rate - m.avg_entry_price
            # confidence-adjusted: use the Wilson lower bound of the win rate so
            # thin samples get a conservative edge, not the optimistic estimate.
            wr_lb = _wilson_lower(int(settled["won"].sum()), len(settled))
            m.edge_lb = wr_lb - m.avg_entry_price

    # ---- risk: drawdown, Sharpe, loss-chasing ----------------------------- #
    if not settled.empty:
        chrono = settled.sort_values("last_ts")
        equity = chrono["pnl"].cumsum().values         # cumulative PnL over time
        peak = np.maximum.accumulate(equity)
        worst_drop = float((peak - equity).max())       # largest peak-to-trough fall, in $
        # Normalise by total capital deployed, NOT peak equity (peak-equity
        # normalisation explodes when an early running-peak is tiny just before a
        # large loss; it once produced 261 = 26,100%). Bounded ~[0,1]: "worst
        # peak-to-trough loss as a fraction of all capital ever risked."
        m.max_drawdown = worst_drop / m.total_staked if m.total_staked > 0 else np.nan
        rets = (chrono["pnl"] / chrono["spent"].replace(0, np.nan)).dropna()
        if len(rets) > 2 and rets.std() > 0:
            m.sharpe = float(rets.mean() / rets.std())
        m.worst_market_pnl = float(settled["pnl"].min())
        # loss-chasing: avg stake after a loss vs after a win (by entry order)
        seq = settled.sort_values("first_ts").reset_index(drop=True)
        after_loss, after_win = [], []
        for i in range(1, len(seq)):
            stake = seq.loc[i, "spent"]
            (after_loss if not seq.loc[i - 1, "won"] else after_win).append(stake)
        if after_loss and after_win and np.mean(after_win) > 0:
            m.loss_chasing_ratio = float(np.mean(after_loss) / np.mean(after_win))

    # ---- breadth ---------------------------------------------------------- #
    if not settled.empty:
        pos = settled[settled["pnl"] > 0]["pnl"].sort_values(ascending=False)
        tot = pos.sum()
        if tot > 0:
            m.pnl_top1_share = float(pos.iloc[0] / tot)
            m.pnl_top5_share = float(pos.head(5).sum() / tot)

    # ---- sizing / behaviour ----------------------------------------------- #
    stakes = trades[trades["side"] == "BUY"]["usdc"].dropna()
    stakes = stakes[stakes > 0]
    if len(stakes):
        m.stake_median = float(stakes.median())
        mean = stakes.mean()
        m.stake_cv = float(stakes.std() / mean) if mean > 0 else np.nan
        # stake-range ratio: p95 / p5 (robust to single outliers). A wide spread —
        # e.g. $2 bets alongside $35k bets — can't be mirrored: anchor the small
        # ones at the $1 minimum and the big ones need an enormous bankroll; anchor
        # the big ones sensibly and the small ones fall below $1 and get skipped.
        if len(stakes) >= 5:
            p5 = float(np.percentile(stakes, 5))
            p95 = float(np.percentile(stakes, 95))
            m.stake_p5, m.stake_p95 = p5, p95
            m.stake_range_ratio = (p95 / p5) if p5 > 0 else np.inf
    if not mk.empty:
        m.trades_per_market = float(mk["n_trades"].mean())
        m.both_sides_frac = float((mk["has_buy"] & mk["has_sell"]).mean())
        m.redeem_rate = float(mk["redeemed"].mean())
        m.median_hold_hours = float(((mk["last_ts"] - mk["first_ts"]) / 3600.0).median())
    # recent trading frequency: trades over the FETCHED window (not true account
    # age) so it measures how often they trade NOW and isn't distorted when an
    # old wallet's history is capped to its recent events.
    if window_days > 0:
        m.trades_per_week = m.n_trades / (window_days / 7.0)

    # ---- specialism: category from market TITLES (no API calls) ----------- #
    # Gamma no longer exposes a category/tags field, so we classify each market
    # from its title and weight by capital deployed (BUY usdc).
    buys_df = trades[trades["side"] == "BUY"]
    if not buys_df.empty:
        cat_spend = {}
        # one title per market, weighted by that market's BUY $
        for cid, g in buys_df.groupby("condition_id"):
            title = next((t for t in g["title"] if t), "")
            cat = _classify_category(title)
            cat_spend[cat] = cat_spend.get(cat, 0.0) + float(g["usdc"].sum())
        cat_spend.pop("other", None) if len(cat_spend) > 1 else None
        if cat_spend:
            total_cat = sum(cat_spend.values())
            if total_cat > 0:
                shares = np.array([v / total_cat for v in cat_spend.values()])
                m.category_concentration = float((shares ** 2).sum())  # 1 = single category
                m.top_category = max(cat_spend, key=cat_spend.get)

    # maker likelihood
    sig = []
    if not np.isnan(m.trades_per_market):
        sig.append(np.clip((m.trades_per_market - 1) / 5.0, 0, 1))
    if not np.isnan(m.both_sides_frac):
        sig.append(m.both_sides_frac)
    if not np.isnan(m.median_hold_hours):
        sig.append(np.clip((6.0 - m.median_hold_hours) / 6.0, 0, 1))
    if not np.isnan(m.redeem_rate):
        sig.append(1.0 - m.redeem_rate)
    m.maker_score = float(np.mean(sig)) if sig else np.nan

    # copy-style: how hard is this wallet to mirror?
    h = m.median_hold_hours
    if not np.isnan(h):
        if h < 1:
            m.copy_style = "scalp"          # must mirror fast exits — hard
        elif h < 48:
            m.copy_style = "swing"          # days — medium
        else:
            m.copy_style = "resolution" if (m.redeem_rate or 0) > 0.4 else "swing"

    # fee-adjusted net edge: subtract estimated fees. One fee on entry, plus an
    # exit fee on the fraction they sold out of rather than held to resolution.
    if not np.isnan(m.edge) and not np.isnan(m.avg_entry_price):
        exits = 1.0 + (1.0 - (m.redeem_rate if not np.isnan(m.redeem_rate) else 1.0))
        m.net_edge = m.edge - _est_fee(m.avg_entry_price) * exits

    return m, mk


def _apply_closed_positions(m, client):
    """Override PnL / edge / risk with Polymarket's own per-market accounting
    (realizedPnl, avgPrice, totalBought, close timestamp) — more accurate than
    reconstructing cashflows from raw activity, which undercounts."""
    closed = client.fetch_closed_positions(m.wallet)
    if not closed:
        return
    cp = pd.DataFrame(closed)
    rp = pd.to_numeric(cp.get("realizedPnl"), errors="coerce")
    ap = pd.to_numeric(cp.get("avgPrice"), errors="coerce")
    tb = pd.to_numeric(cp.get("totalBought"), errors="coerce")
    ts = pd.to_numeric(cp.get("timestamp"), errors="coerce")
    keep = rp.notna()
    if keep.sum() < 5:                              # too few to trust; keep estimate
        return
    rp, ap, tb, ts = rp[keep], ap[keep], tb[keep], ts[keep]
    n, wins = len(rp), int((rp > 0).sum())

    m.pnl_source = "closed_positions"
    m.n_closed = n
    m.realized_pnl = float(rp.sum())
    staked = float(tb[tb > 0].sum())
    if staked > 0:
        m.total_staked = staked
        m.roi = m.realized_pnl / staked
    m.win_rate = wins / n
    if ap.notna().any():
        m.avg_entry_price = float(ap.mean())
        m.edge = m.win_rate - m.avg_entry_price
        m.edge_lb = _wilson_lower(wins, n) - m.avg_entry_price
        exits = 1.0 + (1.0 - (m.redeem_rate if not np.isnan(m.redeem_rate) else 1.0))
        m.net_edge = m.edge - _est_fee(m.avg_entry_price) * exits
    gw, gl = rp[rp > 0].sum(), -rp[rp < 0].sum()
    m.profit_factor = float(gw / gl) if gl > 0 else (float("inf") if gw > 0 else np.nan)
    m.worst_market_pnl = float(rp.min())
    # drawdown over the accurate equity curve (ordered by close time)
    chrono = pd.DataFrame({"pnl": rp.values, "ts": ts.fillna(0).values}).sort_values("ts")
    equity = chrono["pnl"].cumsum().values
    peak = np.maximum.accumulate(equity)
    worst = float((peak - equity).max())
    if staked > 0:
        m.max_drawdown = worst / staked


def enrich_metrics(m, df, mk, client):
    """EXPENSIVE metrics (accurate PnL via closed-positions, CLV price-history,
    Gamma metadata). Only called on wallets that survived the cheap filters, so
    the slow API work never touches a wallet that's going to be dropped anyway."""
    if mk.empty:
        return
    deep = client.use_clv or client.use_gamma
    settled = mk[mk["settled"]]

    # (a) accurate PnL / edge from Polymarket's closed-position accounting
    if deep:
        _apply_closed_positions(m, client)

    # (b) CLV on a sample of settled markets — batch-fetch the histories in one
    #     POST per 20 tokens, then compute from cache (no per-market GETs).
    if client.use_clv and not settled.empty:
        cids = list(settled["cid"])
        rng = np.random.default_rng(42)
        pick = rng.choice(len(cids), size=min(CLV_SAMPLE, len(cids)), replace=False)
        sample_cids = [cids[i] for i in pick]
        tokens = []
        for cid in sample_cids:
            g = df[df["condition_id"] == cid]
            buys = g[(g["type"] == "TRADE") & (g["side"] == "BUY")]
            if not buys.empty and buys["asset"].iloc[0]:
                tokens.append(buys["asset"].iloc[0])
        client.batch_price_history(tokens)          # 1 POST / 20 tokens -> cache
        clv_e, clv_c = [], []
        for cid in sample_cids:
            ce, cc = _clv_for_market(df[df["condition_id"] == cid], client)
            if not np.isnan(ce):
                clv_e.append(ce); clv_c.append(cc)
        if clv_e:
            m.n_clv = len(clv_e)
            # Only trust/credit CLV once we have enough priced markets — a CLV from
            # one or two markets is noise, not skill, and was inflating scores.
            if len(clv_e) >= CLV_MIN_SAMPLE:
                m.clv_entry = float(np.mean(clv_e))
                m.clv_copier = float(np.mean(clv_c))

    # (c) Gamma liquidity / footprint, fetched in batches (~25 markets/call).
    #     (Category now comes from titles in compute_metrics — Gamma no longer
    #     returns a category field — so we only use Gamma for liquidity here.)
    if client.use_gamma and not mk.empty:
        metas = client.fetch_markets_meta(list(mk["cid"]))
        liqs, foot = [], []
        for _, r in settled.iterrows():
            vol = metas.get(r["cid"], {}).get("volume", np.nan)
            if not np.isnan(vol) and vol > 0:
                liqs.append(vol); foot.append(r["spent"] / vol)
        if liqs:
            m.median_market_liquidity = float(np.median(liqs))
            m.trade_to_liquidity = float(np.median(foot))

    # (d) CURRENT open positions -> unrealized PnL. Closed-position accounting
    #     alone ignores what a wallet is still holding, so a trader who banked
    #     profits but is bagholding losers looks great here yet shows NEGATIVE on
    #     their Polymarket profile (which displays realized + unrealized). Folding
    #     in cashPnl makes total_pnl line up with the profile and lets us drop
    #     wallets that are underwater once you count open bets.
    if deep:
        open_pos = client.fetch_positions(m.wallet)
        if isinstance(open_pos, list) and open_pos:
            unreal = sum(_to_float(p.get("cashPnl"), 0.0) for p in open_pos)
            open_real = sum(_to_float(p.get("realizedPnl"), 0.0) for p in open_pos)
            m.n_open = len(open_pos)
            m.open_unrealized_pnl = float(unreal + open_real)
            # current portfolio value = market value of all open positions (what
            # Polymarket shows as "Portfolio"). It's the capital they have IN PLAY;
            # idle USDC cash isn't on this endpoint and needs an on-chain call.
            m.portfolio_value = float(sum(_to_float(p.get("currentValue"), 0.0)
                                          for p in open_pos))
        else:
            m.open_unrealized_pnl = 0.0
            m.portfolio_value = 0.0
        m.total_pnl = float(m.realized_pnl + (m.open_unrealized_pnl
                                              if not np.isnan(m.open_unrealized_pnl) else 0.0))
        # idle USDC cash on-chain -> true total balance (positions + cash)
        if FETCH_CASH:
            m.cash_balance = fetch_usdc_balance(m.wallet, session=client.s)
            if not np.isnan(m.cash_balance):
                pv = m.portfolio_value if not np.isnan(m.portfolio_value) else 0.0
                m.total_balance = float(pv + m.cash_balance)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _norm(x, lo, hi):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.5
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def apply_filters(m):
    f = FILTERS
    if m.n_settled < f["min_settled_markets"]:
        m.disqualified.append(f"too_few_settled({m.n_settled})")
    if m.account_age_days < f["min_account_age_days"]:
        m.disqualified.append(f"too_new({m.account_age_days:.0f}d)")
    if m.days_since_active > f["max_days_since_active"]:
        m.disqualified.append(f"inactive({m.days_since_active:.0f}d)")
    if m.realized_pnl < f["min_realized_pnl"]:
        m.disqualified.append(f"unprofitable(${m.realized_pnl:.0f})")
    if not np.isnan(m.maker_score) and m.maker_score > f["max_maker_score"]:
        m.disqualified.append(f"likely_market_maker({m.maker_score:.2f})")
    if not np.isnan(m.stake_cv) and m.stake_cv > f["max_stake_cv"]:
        m.disqualified.append(f"erratic_sizing(cv={m.stake_cv:.1f})")
    if not np.isnan(m.stake_range_ratio) and m.stake_range_ratio > f["max_stake_range_ratio"]:
        m.disqualified.append(f"wide_stake_range({m.stake_range_ratio:.0f}x)")
    if m.pnl_top1_share > f["max_pnl_top1_share"]:
        m.disqualified.append(f"one_lucky_hit({m.pnl_top1_share:.0%})")
    if not np.isnan(m.max_drawdown) and m.max_drawdown > f["max_drawdown"]:
        m.disqualified.append(f"deep_drawdown({m.max_drawdown:.0%})")
    if not np.isnan(m.loss_chasing_ratio) and m.loss_chasing_ratio > f["max_loss_chasing"]:
        m.disqualified.append(f"loss_chasing({m.loss_chasing_ratio:.2f}x)")
    if not np.isnan(m.trades_per_week) and m.trades_per_week > f["max_trades_per_week"]:
        m.disqualified.append(f"too_frequent({m.trades_per_week:.0f}/wk)")
    if not np.isnan(m.trades_per_week) and m.trades_per_week < f["min_trades_per_week"]:
        m.disqualified.append(f"too_inactive({m.trades_per_week:.1f}/wk)")


def _apply_quality_gates(m):
    """Stricter gates that need the EXPENSIVE metrics (accurate PnL, open-position
    mark-to-market, CLV). These enforce that a wallet's profit is real, still held,
    consistent, and skill-backed — not realized gains it handed back on open bets
    or won by sheer volume/variance. Most leaderboard whales fail here, by design.
    """
    f = FILTERS
    # realized accounting can flip a borderline wallet to net-negative
    if m.pnl_source == "closed_positions" and m.realized_pnl < f["min_realized_pnl"]:
        m.disqualified.append(f"unprofitable(${m.realized_pnl:.0f})")

    # total = realized + unrealized (their actual Polymarket profile number).
    if not np.isnan(m.total_pnl):
        if m.total_pnl < f["min_total_pnl"]:
            m.disqualified.append(f"thin_or_underwater(total ${m.total_pnl:,.0f})")
        if m.total_staked > 0 and (m.total_pnl / m.total_staked) < f["min_total_roi"]:
            m.disqualified.append(f"thin_margin(roi {m.total_pnl / m.total_staked:.1%})")

    # anti-bagholding: how much realized profit is still intact once open bets are
    # marked to market. Keeping <50% means they repeatedly give gains back.
    if m.realized_pnl > 0 and not np.isnan(m.total_pnl):
        kept = m.total_pnl / m.realized_pnl
        if kept < f["min_kept_pnl_frac"]:
            m.disqualified.append(f"bagholding(kept {kept:.0%})")

    # profit-factor floor (gross wins / gross losses)
    if np.isfinite(m.profit_factor) and m.profit_factor < f["min_profit_factor"]:
        m.disqualified.append(f"weak_profit_factor({m.profit_factor:.2f})")

    # demonstrated skill: require at least ONE positive skill signal — a confidence-
    # adjusted edge OR a measured closing-line value. Huge PnL with negative edge
    # AND negative CLV is variance/volume, not a copyable edge.
    edge_ok = (not np.isnan(m.edge_lb)) and m.edge_lb > 0
    clv_ok = (not np.isnan(m.clv_entry)) and m.clv_entry > CLV_SKILL_MIN
    if not (edge_ok or clv_ok):
        m.disqualified.append("no_demonstrated_skill")


def score(m):
    # apply_filters is run by the caller (screen) before enrichment, so the
    # expensive metrics only run on survivors. Here we just trust the verdict.
    if m.disqualified:
        m.score = 0.0
        return

    c = {}
    # SKILL: confidence-adjusted edge (Wilson lower bound) over price, upgraded by
    # CLV when present. Using edge_lb means a thin-sample edge can't dominate. Net
    # (after-fee) edge is blended in so fee-fragile edges score lower.
    #
    # IMPORTANT: edge = win_rate - avg_entry only equals true predictive skill for
    # wallets that HOLD TO RESOLUTION. A scalp/swing wallet that buys at 0.30 and
    # sells at 0.32 books a "win" (realizedPnl>0) at a 0.30 entry, which inflates
    # edge toward ~0.4 even though they captured a 2-cent move, not a 40-point one.
    # So we discount the edge-derived parts by how often the wallet actually carries
    # bets to resolution (redeem_rate). CLV is a *measured* forward price move and
    # stays at full weight — it's the honest skill signal for these traders.
    rel = m.redeem_rate if not np.isnan(m.redeem_rate) else 0.5
    rel = float(np.clip(rel, 0.25, 1.0))            # resolution-reliability of edge
    base_edge = m.edge_lb if not np.isnan(m.edge_lb) else m.edge
    skill_parts = [_norm(base_edge, -0.10, 0.15) * rel]
    if not np.isnan(m.net_edge):
        skill_parts.append(_norm(m.net_edge, -0.05, 0.12) * rel)   # survives fees?
    if not np.isnan(m.clv_entry):
        skill_parts.append(_norm(m.clv_entry, -0.02, 0.10))     # their sharpness (measured)
    if not np.isnan(m.clv_copier):
        skill_parts.append(_norm(m.clv_copier, -0.02, 0.06))    # what you can capture
    c["skill"] = float(np.mean(skill_parts))

    # PROFITABILITY: ROI + profit factor + monthly consistency
    pf = m.profit_factor if np.isfinite(m.profit_factor) else 3.0
    c["profitability"] = (0.45 * _norm(m.roi, 0.0, 0.5)
                          + 0.30 * _norm(pf, 1.0, 2.5)
                          + 0.25 * _norm(m.profitable_month_frac, 0.4, 0.9))

    # RISK: low drawdown + decent Sharpe + no loss-chasing
    dd_score = _norm(0.40 - (m.max_drawdown if not np.isnan(m.max_drawdown) else 0.20), 0.0, 0.40)
    sharpe_score = _norm(m.sharpe, 0.0, 1.5)
    lc = m.loss_chasing_ratio if not np.isnan(m.loss_chasing_ratio) else 1.0
    lc_score = _norm(1.6 - lc, 0.0, 0.8)
    c["risk"] = 0.5 * dd_score + 0.3 * sharpe_score + 0.2 * lc_score

    # BREADTH: profit spread across many markets
    c["breadth"] = _norm(1.0 - m.pnl_top5_share, 0.2, 0.8)

    # COPYABILITY: not a maker + adequate liquidity + easy copy-style + specialism
    cop = _norm(0.55 - (m.maker_score if not np.isnan(m.maker_score) else 0.3), 0.0, 0.55)
    if not np.isnan(m.trade_to_liquidity):
        cop = 0.6 * cop + 0.4 * _norm(0.10 - m.trade_to_liquidity, 0.0, 0.10)
    style_mult = {"resolution": 1.0, "swing": 0.85, "scalp": 0.5, "?": 0.8}[m.copy_style]
    cop *= style_mult
    if not np.isnan(m.category_concentration):                  # reward specialists
        cop = 0.8 * cop + 0.2 * _norm(m.category_concentration, 0.3, 0.8)
    # prefer tight, mirror-able stake sizing (low p95/p5). A tight spread scores
    # full; near the max_stake_range_ratio disqualifier it's ~0.
    if not np.isnan(m.stake_range_ratio):
        lim = FILTERS["max_stake_range_ratio"]
        tight = _norm(lim - m.stake_range_ratio, 0, lim * 0.93)
        cop = 0.8 * cop + 0.2 * tight
    c["copyability"] = cop

    # ROBUSTNESS: sample + account age + recency
    c["robustness"] = (0.5 * _norm(m.n_settled, 30, 300)
                       + 0.25 * _norm(m.account_age_days, 60, 365)
                       + 0.25 * _norm(21 - m.days_since_active, 0, 21))

    tw = sum(WEIGHTS.values())
    m.score = round(100 * sum(WEIGHTS[k] * c[k] for k in WEIGHTS) / tw, 1)
    m.components = {k: round(v, 3) for k, v in c.items()}
    # final bar: a wallet can clear every gate yet still be mediocre overall. Hold
    # candidates to a minimum composite score so only genuinely strong ones qualify.
    if m.score < FILTERS["min_score"]:
        m.disqualified.append(f"low_score({m.score})")


# --------------------------------------------------------------------------- #
# Convergence scanner — live consensus among already-qualified wallets
# --------------------------------------------------------------------------- #

def scan_convergence(wallets, client, min_wallets=CONVERGE_MIN_WALLETS):
    """Markets/sides currently held by >= min_wallets of the given wallets."""
    holders = {}   # (title, outcome) -> set(wallet)
    for w in wallets:
        for p in client.fetch_positions(w):
            sz = _to_float(p.get("size"))
            if np.isnan(sz) or sz <= 0:
                continue
            key = (p.get("title", "?"), p.get("outcome", "?"))
            holders.setdefault(key, set()).add(w)
    out = [(t, o, len(ws)) for (t, o), ws in holders.items() if len(ws) >= min_wallets]
    return sorted(out, key=lambda x: -x[2])


# --------------------------------------------------------------------------- #
# CLV live validation — run this BEFORE trusting the CLV column.
# --------------------------------------------------------------------------- #

def check_clv(target):
    """Validate /prices-history against live data.

    `target` may be an outcome-token/asset id, or a wallet address (0x...), in
    which case we pull its MOST RECENT buy's asset (newest-first, so no offset
    cap) and test that. Tests at 12h fidelity, the only granularity resolved
    markets support.
    """
    client = PolymarketClient(use_clv=True)
    token, market_title = target, ""
    if target.lower().startswith("0x") and len(target) == 42:
        print(f"Resolving a recent token from wallet {target} ...", file=sys.stderr)
        data = client._get(f"{DATA_API}/activity", {
            "user": target.lower(), "type": "TRADE", "limit": 50,
            "sortBy": "TIMESTAMP", "sortDirection": "DESC"})
        rows = data if isinstance(data, list) else []
        token = ""
        for r in rows:
            if r.get("asset"):
                token = r["asset"]; market_title = (r.get("title") or "")[:60]; break
        if not token:
            print("No recent buy with an asset id found.", file=sys.stderr); return
        print(f"  using asset {token}\n  market: {market_title}", file=sys.stderr)

    print(f"\n[prices-history] interval=max, fidelity={CLV_FIDELITY} (12h bars — "
          f"the granularity resolved markets support):")
    hist = client.price_history(token, interval="max")
    print(f"    -> {len(hist)} points")
    sample = (hist[:3] + hist[-2:]) if len(hist) > 5 else hist
    for t, p in sample:
        print(f"       {time.strftime('%Y-%m-%d %H:%M', time.gmtime(t))}   p={p:.4f}")
    if hist:
        ps = [p for _, p in hist]
        print(f"    price range {min(ps):.4f} .. {max(ps):.4f}")
    ok = len(hist) >= 3
    if ok:
        print("\nRESULT: OK — endpoint works at 12h fidelity, CLV column is trustworthy.")
    else:
        print("\nRESULT: EMPTY — even 12h fidelity returned nothing. The token may have "
              "no retained history (very old / illiquid). Try another wallet or token.")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

REPORT_COLS = [
    "wallet", "label", "score", "edge", "edge_lb", "net_edge", "skill_clv", "n_clv",
    "roi", "profit_factor", "max_drawdown", "copy_style", "trades_per_week",
    "stake_median", "stake_range_ratio",
    "top_category", "n_settled", "portfolio_value", "cash_balance", "total_balance",
    "realized_pnl", "open_unrealized_pnl", "total_pnl", "pnl_source", "disqualified",
]


_thread_local = threading.local()


def _client_for(use_gamma, use_clv, cache=True, cache_ttl=CACHE_TTL_HOURS):
    """One PolymarketClient per worker thread — requests.Session and the caches
    aren't safe to share across threads, so each thread gets its own."""
    c = getattr(_thread_local, "client", None)
    if c is None:
        c = PolymarketClient(use_gamma=use_gamma, use_clv=use_clv,
                             cache=cache, cache_ttl_hours=cache_ttl)
        _thread_local.client = c
    return c


def _process_wallet(w, use_gamma, use_clv, cache=True, cache_ttl=CACHE_TTL_HOURS):
    """Full pipeline for one wallet: cheap gates -> filters -> (survivors only)
    accurate-PnL + CLV/Gamma enrichment -> score. Returns WalletMetrics or None."""
    w = w.strip().lower()
    if not w.startswith("0x") or len(w) != 42:
        return None
    client = _client_for(use_gamma, use_clv, cache, cache_ttl)
    now = time.time()

    # (1) recency gate — skip dormant wallets before any full download
    recent = client.fetch_recent_ts(w)
    if recent and (now - recent) / 86400 > FILTERS["max_days_since_active"]:
        m = WalletMetrics(wallet=w)
        m.days_since_active = (now - recent) / 86400
        m.disqualified = [f"inactive({m.days_since_active:.0f}d)"]
        return m

    # (1b) account-age gate — TRUE first trade, robust to a capped history
    oldest = client.fetch_first_ts(w)
    age_days = (recent - oldest) / 86400 if (recent and oldest) else None
    recency_days = (now - recent) / 86400 if recent else None
    if age_days is not None and age_days < FILTERS["min_account_age_days"]:
        m = WalletMetrics(wallet=w)
        m.account_age_days = age_days
        if recency_days is not None:
            m.days_since_active = recency_days
        m.disqualified = [f"too_new({age_days:.0f}d)"]
        return m

    # (2) full history + cheap metrics + hard filters (no expensive calls yet)
    df = normalise_activity(client.fetch_activity(w))
    m, mk = compute_metrics(w, df, account_age_days=age_days, days_since_active=recency_days)
    apply_filters(m)

    # (3) expensive enrichment (accurate PnL + CLV + Gamma) ONLY on survivors
    if not m.disqualified:
        enrich_metrics(m, df, mk, client)
        _apply_quality_gates(m)
    score(m)
    # free this wallet's big in-memory buffers so memory stays flat across a long
    # hunt (the disk cache still backs re-reads). Without this the per-thread price
    # cache grows unbounded and can exhaust a low-memory / low-disk machine.
    del df, mk
    client._price_cache.clear()
    if len(client._market_cache) > 4000:
        client._market_cache.clear()
    return m


def screen(wallets, use_gamma, use_clv, labels=None, workers=5, cache=True,
           cache_ttl=CACHE_TTL_HOURS):
    wallets = [w.strip().lower() for w in wallets if w and w.strip()]
    total = len(wallets)
    mode = "screen + auto-analyse" if (use_gamma or use_clv) else "fast screen"
    cnote = "" if cache else " (cache off)"
    print(f"{mode}: {total} wallets, {workers} worker(s){cnote}\n", file=sys.stderr)

    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(_process_wallet, w, use_gamma, use_clv, cache, cache_ttl): w
                for w in wallets}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                m = fut.result()
            except Exception as e:                      # never let one wallet kill the run
                print(f"[{done}/{total}] {futs[fut][:10]}.. -> ERROR: {e}", file=sys.stderr)
                continue
            if m is None:
                print(f"[{done}/{total}] skip invalid: {futs[fut]}", file=sys.stderr)
                continue
            results.append(m)
            tag = f"{m.wallet[:10]}.."
            if m.disqualified:
                print(f"[{done}/{total}] {tag} -> DQ: {m.disqualified[0]}", file=sys.stderr)
            else:
                clv = f", clv={m.clv_entry:.3f}" if not np.isnan(m.clv_entry) else ""
                ne = f", net={m.net_edge:+.3f}" if not np.isnan(m.net_edge) else ""
                print(f"[{done}/{total}] {tag} -> {m.score:>5} "
                      f"(edge={m.edge:+.3f}{ne}{clv}, {m.n_settled} settled)", file=sys.stderr)

    rows = []
    for m in results:
        d = asdict(m)
        d["label"] = (labels or {}).get(m.wallet, "")
        d["skill_clv"] = m.clv_entry
        d["disqualified"] = "; ".join(m.disqualified)
        rows.append(d)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    q = out["disqualified"] == ""
    return pd.concat([out[q].sort_values("score", ascending=False),
                      out[~q].sort_values("realized_pnl", ascending=False)])


def seed_from_leaderboard(client, category, time_period, order_by, count):
    rows = client.fetch_leaderboard(category, time_period, order_by, count)
    wallets, labels = [], {}
    for r in rows:
        w = (r.get("proxyWallet") or "").lower()
        if w.startswith("0x") and len(w) == 42 and w not in labels:
            wallets.append(w); labels[w] = (r.get("userName") or "")[:20]
    print(f"Leaderboard seed: {len(wallets)} wallets [{order_by}/{time_period}/{category}]",
          file=sys.stderr)
    return wallets, labels


# --------------------------------------------------------------------------- #
# AUTONOMOUS DISCOVERY SOURCES (--auto)
# Each returns (wallets, labels). They decide WHERE to look on their own.
# --------------------------------------------------------------------------- #

def seed_from_leaderboard_sweep(client, count_per_slice,
                                periods=AUTO_LB_PERIODS,
                                orders=AUTO_LB_ORDERS,
                                categories=AUTO_LB_CATEGORIES):
    """Sweep EVERY public top-trader board (category x period x order-by) and pool
    everyone, instead of one fixed slice. The 'check all the leaderboards' source."""
    wallets, labels, slices = [], {}, 0
    for cat in categories:
        for per in periods:
            for order in orders:
                rows = client.fetch_leaderboard(cat, per, order, count_per_slice)
                slices += 1
                for r in rows:
                    w = (r.get("proxyWallet") or "").lower()
                    if w.startswith("0x") and len(w) == 42:
                        if w not in labels:
                            wallets.append(w)
                        nm = (r.get("userName") or "")[:20]
                        if nm:
                            labels.setdefault(w, nm)
    print(f"Leaderboard sweep: {len(wallets)} unique wallets from {slices} board slices",
          file=sys.stderr)
    return wallets, labels


def seed_from_trending_markets(client, n_markets, holders_limit, min_balance):
    """'Go where the action is': find the highest-volume LIVE markets and pull
    their biggest holders."""
    cids = client.fetch_active_markets(limit=n_markets)
    if not cids:
        print("Trending markets: none found.", file=sys.stderr)
        return [], {}
    holders = client.fetch_top_holders(cids, holders_limit, min_balance)
    wallets, labels = [], {}
    for h in holders:
        w = h["wallet"]
        if w not in labels:
            wallets.append(w)
        if h["name"]:
            labels.setdefault(w, h["name"])
    print(f"Trending-market seed: {len(wallets)} wallets from {len(cids)} hot live markets",
          file=sys.stderr)
    return wallets, labels


def _fetch_source_text(src):
    """Return the text of a web URL OR a local file path (so a JS-rendered page you
    SAVED to disk can be scanned, sidestepping the no-JavaScript problem)."""
    if os.path.exists(src):                       # local saved page / list / csv
        try:
            with open(src, encoding="utf-8", errors="ignore") as fh:
                return fh.read()
        except OSError as e:
            print(f"  ! could not read file {src}: {e}", file=sys.stderr)
            return None
    sess = requests.Session()
    sess.headers.update({"User-Agent":
                         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) wallet-screener/2.0"})
    try:
        r = sess.get(src, timeout=20)
        r.raise_for_status()
        return r.text
    except requests.RequestException as e:
        print(f"  ! web seed failed for {src}: {e}", file=sys.stderr)
        return None


def seed_from_web(sources):
    """Scan 'places that track good wallets' on the internet. Each source is a URL
    or a local file path (a saved page works for JS-rendered dashboards). Extracts:
      * FULL 0x addresses -> used directly.
      * TRUNCATED handles (0x1234...abcd) -> returned for later matching against
        on-chain data, since most tracker sites only show truncated addresses.
    Returns (wallets, labels, truncated_fragments). Fragile by design — it reports
    exactly what each source yielded so dead/JS sources are obvious, not silent."""
    wallets, labels, frags = [], {}, []
    for src in sources:
        text = _fetch_source_text(src)
        if text is None:
            continue
        full = {m.group(0).lower() for m in WEB_ADDR_RE.finditer(text)}
        full = {w for w in full if len(w) == 42}
        new = [w for w in full if w not in labels]
        for w in new:
            wallets.append(w); labels.setdefault(w, "web")
        trunc = [(p.lower(), s.lower()) for p, s in WEB_TRUNC_RE.findall(text)]
        frags.extend(trunc)
        where = "file" if os.path.exists(src) else "url"
        print(f"  [{where}] {src} -> {len(new)} full, {len(trunc)} truncated",
              file=sys.stderr)
    print(f"Web scan: {len(wallets)} full addresses + {len(frags)} truncated handles "
          f"from {len(sources)} source(s)", file=sys.stderr)
    return wallets, labels, frags


def resolve_truncated(frags, known_wallets, exclude=None):
    """Match truncated handles (prefix, suffix) from tracker sites against FULL
    addresses we already gathered on-chain. A 0x + >=3 hex prefix and >=3 hex suffix
    is effectively unique, so this safely recovers which of our candidates a curated
    site is pointing at (and confirms/labels them). Returns newly-confirmed wallets."""
    exclude = exclude or set()
    hits = []
    for pre, suf in frags:
        for w in known_wallets:
            body = w[2:]  # strip 0x
            if body.startswith(pre) and body.endswith(suf) and w not in exclude:
                hits.append(w)
    out = list(dict.fromkeys(hits))
    if out:
        print(f"  matched {len(out)} truncated handles to on-chain wallets",
              file=sys.stderr)
    return out


def expand_from_qualified(qualified_wallets, client, holders_limit, min_balance,
                          exclude, max_markets=AUTO_EXPAND_MARKETS):
    """Snowball: look at what the already-qualified wallets are holding RIGHT NOW,
    then pull the OTHER top holders of those same markets. Lets proven smart money
    lead the screener to more smart money. Returns NEW candidates not in `exclude`."""
    market_ids, seen_m = [], set()
    for w in qualified_wallets:
        for p in client.fetch_positions(w):
            sz = _to_float(p.get("size"))
            cid = p.get("conditionId") or p.get("condition_id")
            if cid and (np.isnan(sz) or sz > 0) and cid not in seen_m:
                seen_m.add(cid); market_ids.append(cid)
    if not market_ids:
        return [], {}
    market_ids = market_ids[:max_markets]
    holders = client.fetch_top_holders(market_ids, holders_limit, min_balance)
    wallets, labels = [], {}
    for h in holders:
        w = h["wallet"]
        if w in exclude or w in labels:
            continue
        wallets.append(w)
        if h["name"]:
            labels.setdefault(w, h["name"])
    print(f"Snowball: {len(wallets)} new wallets from {len(market_ids)} markets "
          f"held by {len(qualified_wallets)} qualified wallets", file=sys.stderr)
    return wallets, labels


def run_auto(args, deep, use_cache):
    """Autonomous discovery driver. Assembles a candidate pool from several public
    sources, screens it, then snowballs off the winners — no manual seeds needed.

    Sources (each can be turned off):
      1. leaderboard sweep      (every category x period x order-by)
      2. trending live markets  (biggest holders of the hottest markets)
      3. curated internet trackers (KNOWN_WALLET_SOURCES + --seed-url; --no-web off)
      4. snowball expansion     (other holders of markets the winners hold now)
    """
    seed_client = PolymarketClient(use_gamma=True, use_clv=False, cache=use_cache,
                                   cache_ttl_hours=args.cache_ttl)
    print("AUTO DISCOVERY — assembling candidate pool from public sources\n",
          file=sys.stderr)

    wallets, labels = [], {}

    def _add(pair):
        ws, ls = pair
        for w in ws:
            if w not in labels and w not in wallets:
                wallets.append(w)
        for k, v in ls.items():
            labels.setdefault(k, v)

    if not args.no_leaderboard:
        _add(seed_from_leaderboard_sweep(seed_client, args.lb_count))
    if not args.no_markets:
        _add(seed_from_trending_markets(seed_client, args.auto_markets,
                                        args.holders_limit, args.holders_min_balance))

    # source 3: curated internet trackers + any --seed-url (URLs or local files).
    web_frags = []
    sources = ([] if args.no_web else list(KNOWN_WALLET_SOURCES)) + list(args.seed_url or [])
    if sources:
        print("Scanning curated 'known-good-wallet' sources on the internet ...",
              file=sys.stderr)
        ws, ls, web_frags = seed_from_web(sources)
        _add((ws, ls))

    # Most trackers only show truncated addresses; match those back to full
    # addresses we already have on-chain and star them as tracker-endorsed.
    if web_frags:
        endorsed = resolve_truncated(web_frags, list(dict.fromkeys(wallets)))
        for w in endorsed:
            base = labels.get(w, "")
            if not base.startswith("★"):
                labels[w] = ("★" + base)[:20]   # ★ = highlighted by a tracker site
        print(f"Tracker-endorsed wallets already in pool: {len(endorsed)}", file=sys.stderr)

    wallets = list(dict.fromkeys(wallets))[:AUTO_MAX_CANDIDATES]
    if not wallets:
        print("Auto discovery found no candidates.", file=sys.stderr)
        return
    print(f"\nInitial candidate pool: {len(wallets)} wallets -> screening\n",
          file=sys.stderr)

    all_frames, seen = [], set(wallets)
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_human = time.strftime("%Y-%m-%d %H:%M")
    bsize = max(args.hunt_batch, 1)          # screen + publish in chunks this big
    qualified, interrupted = [], False
    try:
        # initial pool, screened in publishable chunks (so the report updates live)
        pending = list(wallets)
        while pending:
            batch, pending = pending[:bsize], pending[bsize:]
            print(f"\nScreening {len(batch)} wallets ({len(pending)} left in pool)\n",
                  file=sys.stderr)
            dfb = screen(batch, use_gamma=deep, use_clv=deep, labels=labels,
                         workers=args.workers, cache=use_cache, cache_ttl=args.cache_ttl)
            if not dfb.empty:
                all_frames.append(dfb)
                qualified.extend(list(dfb[dfb["disqualified"] == ""]["wallet"]))
            publish_progress(all_frames, args, run_ts, run_human)

        # snowball rounds off the wallets that qualified
        for rnd in range(max(args.expand_rounds, 0)):
            if not qualified:
                break
            new_w, new_l = expand_from_qualified(
                qualified, seed_client, args.holders_limit, args.holders_min_balance,
                exclude=seen)
            new_w = [w for w in new_w if w not in seen][:AUTO_MAX_CANDIDATES]
            if not new_w:
                print(f"Snowball round {rnd + 1}: no new wallets, stopping.", file=sys.stderr)
                break
            seen.update(new_w)
            labels.update(new_l)
            print(f"\nSnowball round {rnd + 1}: screening {len(new_w)} new wallets\n",
                  file=sys.stderr)
            qualified, pend2 = [], list(new_w)
            while pend2:
                b, pend2 = pend2[:bsize], pend2[bsize:]
                dfr = screen(b, use_gamma=deep, use_clv=deep, labels=labels,
                             workers=args.workers, cache=use_cache, cache_ttl=args.cache_ttl)
                if not dfr.empty:
                    all_frames.append(dfr)
                    qualified.extend(list(dfr[dfr["disqualified"] == ""]["wallet"]))
                publish_progress(all_frames, args, run_ts, run_human)
    except KeyboardInterrupt:
        interrupted = True
        print("\n\n[Ctrl+C] stopping — keeping everything found so far …", file=sys.stderr)

    _finalize_results(all_frames, args, ts=run_ts, human=run_human, interrupted=interrupted)


def _run_mode_str(args):
    if getattr(args, "hunt", None) is not None:
        return f"hunt {args.hunt}"
    if getattr(args, "auto", False):
        return "auto"
    return "screen"


def _combine_frames(all_frames):
    """Concatenate every screened batch, dedupe by wallet, rank qualified first."""
    final = pd.concat(all_frames, ignore_index=True)
    final = final.drop_duplicates(subset=["wallet"], keep="first")
    q = final["disqualified"] == ""
    return pd.concat([final[q].sort_values("score", ascending=False),
                      final[~q].sort_values("realized_pnl", ascending=False)])


def publish_progress(all_frames, args, ts, human):
    """Write the results-so-far to the live CSV + dashboard MID-RUN, so the report
    updates as the screen progresses instead of only at the end. Best-effort."""
    if not all_frames:
        return
    try:
        final = _combine_frames(all_frames)
        final.to_csv(args.out, index=False)
        archive_report(final, args, ts=ts, human=human, live=True)
        nq = int((final["disqualified"] == "").sum())
        print(f"  …published {nq} qualified so far -> "
              f"{os.path.join(args.reports_dir, 'report_' + ts + '.html')}", file=sys.stderr)
        if getattr(args, "publish_git", False):
            git_publish(args.reports_dir, args.git_remote, message="live update")
    except Exception as e:
        print(f"  ! live publish failed: {e}", file=sys.stderr)


def _finalize_results(all_frames, args, ts=None, human=None, interrupted=False):
    """Final write of the CSV + dashboard and the console report. Reuses the run's
    timestamp so the live file is finalised in place (not duplicated)."""
    if not all_frames:
        print("No usable results.", file=sys.stderr)
        return
    final = _combine_frames(all_frames)
    final.to_csv(args.out, index=False)
    if interrupted:
        print("\n[stopped early — saving everything found so far]", file=sys.stderr)
    _print_report(final, args.top)
    print(f"\nFull results -> {args.out}")
    try:
        html_path = archive_report(final, args, ts=ts, human=human, live=False)
        print(f"Dashboard   -> {html_path}", file=sys.stderr)
        print(f"Browse all  -> {os.path.join(args.reports_dir, 'index.html')}",
              file=sys.stderr)
        if getattr(args, "publish_git", False):
            git_publish(args.reports_dir, args.git_remote, message="run complete", force=True)
    except Exception as e:                              # never let reporting break a run
        print(f"  ! report archive failed: {e}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Styled HTML dashboard + run archive (every run lands in reports/ with an index)
# --------------------------------------------------------------------------- #

REPORT_CSS = """
:root{--bg:#0b0e14;--panel:#11161f;--panel2:#161c27;--line:#222b39;--txt:#e6edf3;
--muted:#8b949e;--accent:#58a6ff;--green:#3fb950;--red:#f85149;--amber:#d29922;}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1100px 560px at 72% -12%,#16243a55,transparent),var(--bg);
color:var(--txt);font-family:'Inter',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
-webkit-font-smoothing:antialiased;}
.wrap{max-width:1300px;margin:0 auto;padding:38px 28px 90px;}
.head{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;flex-wrap:wrap;
border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:26px;}
.title{font-size:27px;font-weight:800;letter-spacing:-.025em;margin:0;}
.title span{color:var(--accent);}
.sub{color:var(--muted);font-size:13px;margin-top:7px;}
.sub a{color:var(--accent);text-decoration:none;}
.badge{display:inline-block;padding:6px 13px;border:1px solid var(--line);border-radius:999px;
font-size:12.5px;color:var(--txt);background:var(--panel);font-weight:600;}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:28px;}
.card{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);
border-radius:14px;padding:15px 18px;}
.card .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.09em;}
.card .v{font-size:25px;font-weight:800;margin-top:7px;letter-spacing:-.02em;}
.tablewrap{border:1px solid var(--line);border-radius:14px;overflow:hidden;background:var(--panel);}
table{width:100%;border-collapse:separate;border-spacing:0;font-size:13px;}
thead th{background:#0e131c;color:var(--muted);text-align:left;font-weight:600;font-size:10.5px;
text-transform:uppercase;letter-spacing:.07em;padding:13px 14px;border-bottom:1px solid var(--line);}
tbody td{padding:13px 14px;border-bottom:1px solid var(--line);white-space:nowrap;}
tbody tr:last-child td{border-bottom:none;}
tbody tr:hover{background:#0f1724;}
.mono{font-family:'SF Mono',ui-monospace,Menlo,Consolas,monospace;font-size:11.5px;color:var(--muted);}
.name{font-weight:700;color:var(--txt);}
.name a,a.prof{color:var(--txt);text-decoration:none;}
a.prof:hover .name{color:var(--accent);}
.star{color:var(--amber);}
.scorecell{display:flex;align-items:center;gap:10px;min-width:120px;}
.scoreval{font-weight:800;width:30px;font-size:14px;}
.track{flex:1;height:6px;border-radius:4px;background:#1b2330;overflow:hidden;min-width:64px;}
.track>i{display:block;height:100%;border-radius:4px;}
.pos{color:var(--green);font-weight:600;} .neg{color:var(--red);font-weight:600;}
.chip{display:inline-block;padding:2px 9px;border-radius:6px;font-size:11px;font-weight:700;
border:1px solid var(--line);text-transform:capitalize;}
.chip.res{color:#7ee787;background:#0f2417;border-color:#1c3a26;}
.chip.swing{color:#a5d6ff;background:#0d1f33;border-color:#1b3350;}
.chip.scalp{color:#ffa657;background:#2a1a0d;border-color:#46301a;}
.cat{color:var(--muted);}
.section{margin-top:34px;font-size:14px;font-weight:800;}
.dqbar{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px;}
.dq{font-size:12px;color:var(--muted);background:var(--panel);border:1px solid var(--line);
border-radius:8px;padding:6px 11px;}
.dq b{color:var(--txt);}
.empty{color:var(--muted);padding:34px;text-align:center;border:1px dashed var(--line);border-radius:14px;}
.foot{margin-top:40px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);
padding-top:18px;line-height:1.65;}
"""

INDEX_CSS = """
.runs{display:flex;flex-direction:column;gap:10px;}
.runrow{display:grid;grid-template-columns:210px 1fr 120px 1.2fr 24px;align-items:center;gap:18px;
background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);
border-radius:12px;padding:16px 20px;color:var(--txt);text-decoration:none;transition:border-color .15s,transform .15s;}
.runrow:hover{border-color:var(--accent);transform:translateX(2px);}
.rl{display:flex;flex-direction:column;gap:3px;}
.rmode{font-weight:800;text-transform:capitalize;letter-spacing:-.01em;}
.rtime{color:var(--muted);font-size:12px;}
.rstat{font-size:14px;} .rstat b{color:var(--txt);font-weight:800;}
.muted{color:var(--muted);}
.rtop{font-size:13px;color:var(--muted);}
.go{color:var(--accent);text-align:right;font-size:18px;}
"""


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _h_money(v):
    return f"${v:,.0f}" if pd.notna(v) else "—"


def _h_money_c(v):
    if pd.isna(v):
        return "—"
    return f'<span class="{"pos" if v >= 0 else "neg"}">${v:,.0f}</span>'


def _h_pct(v):
    return f"{v * 100:.1f}%" if pd.notna(v) else "—"


def _h_num(v, d=2):
    return f"{v:.{d}f}" if pd.notna(v) else "—"


def _h_clv(v, n):
    if pd.isna(v):
        return "—"
    nn = f' <span class="mono">({int(n)})</span>' if pd.notna(n) else ""
    return f'<span class="{"pos" if v >= 0 else "neg"}">{v:+.3f}</span>{nn}'


def _h_range(v):
    if pd.isna(v) or not np.isfinite(v):
        return "—"
    return f"{v:.0f}×"


def _score_color(s):
    if s >= 75:
        return "#3fb950"
    if s >= 65:
        return "#56d364"
    if s >= 55:
        return "#d29922"
    return "#8b949e"


def _report_row(rank, r):
    w = str(r.get("wallet", ""))
    short = (w[:6] + "…" + w[-4:]) if len(w) >= 10 else w
    label = str(r.get("label", "") or "")
    star = ""
    if label.startswith("★"):
        star = '<span class="star">★</span> '
        label = label[1:]
    name = f'{star}<span class="name">{_esc(label)}</span>' if (label or star) else \
           '<span class="name">—</span>'
    score = float(r.get("score", 0) or 0)
    col = _score_color(score)
    width = min(max(score, 0), 100)
    score_cell = (f'<div class="scorecell"><span class="scoreval" style="color:{col}">'
                  f'{score:.0f}</span><span class="track"><i style="width:{width:.0f}%;'
                  f'background:{col}"></i></span></div>')
    bal = r.get("total_balance")
    if pd.isna(bal):
        bal = r.get("portfolio_value")
    style = str(r.get("copy_style", "?") or "?")
    style_cls = {"resolution": "res", "swing": "swing", "scalp": "scalp"}.get(style, "")
    prof = f"https://polymarket.com/profile/{_esc(w)}"
    return ("<tr>"
            f'<td class="mono">{rank}</td>'
            f'<td><a class="prof" href="{prof}" target="_blank">{name}</a>'
            f'<div class="mono">{short}</div></td>'
            f'<td>{score_cell}</td>'
            f'<td>{_h_money_c(r.get("total_pnl"))}</td>'
            f'<td>{_h_money(bal)}</td>'
            f'<td>{_h_pct(r.get("roi"))}</td>'
            f'<td>{_h_num(r.get("profit_factor"), 2)}</td>'
            f'<td>{_h_pct(r.get("max_drawdown"))}</td>'
            f'<td>{_h_clv(r.get("skill_clv"), r.get("n_clv"))}</td>'
            f'<td><span class="chip {style_cls}">{_esc(style)}</span></td>'
            f'<td class="cat">{_esc(r.get("top_category", "") or "")}</td>'
            f'<td>{_h_num(r.get("trades_per_week"), 0)}</td>'
            f'<td>{_h_range(r.get("stake_range_ratio"))}</td>'
            f'<td class="mono">{_h_num(r.get("n_settled"), 0)}</td>'
            "</tr>")


def render_report_html(final, mode, stamp_human, csv_name, live=False):
    from collections import Counter
    q = final[final["disqualified"] == ""]
    dq = final[final["disqualified"] != ""]
    n_screen, n_qual = len(final), len(q)
    avg = q["score"].mean() if n_qual else 0.0
    cards = [("Qualified", f"{n_qual}"), ("Screened", f"{n_screen}")]
    if n_qual:
        cards.append(("Top score", f"{q['score'].iloc[0]:.0f}"))
        cards.append(("Avg score", f"{avg:.1f}"))
    cards.append(("Mode", mode))
    cards_html = "".join(
        f'<div class="card"><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div></div>'
        for k, v in cards)

    if n_qual:
        rows = "".join(_report_row(i + 1, r) for i, (_, r) in enumerate(q.iterrows()))
        table = ('<div class="tablewrap"><table><thead><tr>'
                 '<th>#</th><th>Wallet</th><th>Score</th><th>Total P/L</th><th>Balance</th>'
                 '<th>ROI</th><th>PF</th><th>Max DD</th><th>CLV (n)</th><th>Style</th>'
                 '<th>Category</th><th>Bets/wk</th><th>Stake&nbsp;range</th><th>Settled</th>'
                 f'</tr></thead><tbody>{rows}</tbody></table></div>')
    else:
        table = '<div class="empty">No wallets cleared the bar this run.</div>'

    dq_html = ""
    if len(dq):
        reasons = Counter(re.sub(r"\(.*?\)", "", str(s).split(";")[0]).strip()
                          for s in dq["disqualified"])
        chips = "".join(f'<span class="dq"><b>{v}</b> {_esc(k)}</span>'
                        for k, v in reasons.most_common(14))
        dq_html = f'<div class="section">Filtered out ({len(dq)})</div><div class="dqbar">{chips}</div>'

    live_meta = "<meta http-equiv='refresh' content='20'>" if live else ""
    live_tag = ("<span style='color:#3fb950'>● live — refreshing</span> &nbsp;·&nbsp; "
                if live else "")
    badge = (f"<div class='badge' style='border-color:#1c3a26;color:#7ee787'>"
             f"● running · {n_qual} qualified</div>") if live else \
            f"<div class='badge'>{n_qual} qualified / {n_screen} screened</div>"
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"{live_meta}"
            f"<title>Screener — {_esc(stamp_human)}</title><style>{REPORT_CSS}</style></head>"
            "<body><div class='wrap'>"
            "<div class='head'><div><h1 class='title'>Polymarket <span>Copy-Trade</span> Screener</h1>"
            f"<div class='sub'>{live_tag}{_esc(stamp_human)} &nbsp;·&nbsp; mode: {_esc(mode)} &nbsp;·&nbsp; "
            f"<a href='{_esc(csv_name)}'>download CSV</a> &nbsp;·&nbsp; <a href='index.html'>all runs →</a></div></div>"
            f"{badge}</div>"
            f"<div class='cards'>{cards_html}</div>{table}{dq_html}"
            "<div class='foot'>Balance = open-position value + on-chain USDC cash. "
            "Total P/L = realized + unrealized (matches the Polymarket profile). "
            "CLV is closing-line value (skill); the number in parentheses is how many markets it was measured on. "
            "Click a name to open that wallet's Polymarket profile. "
            "This is a shortlist to investigate, not financial advice — copy-trading is never passive.</div>"
            "</div></body></html>")


def rebuild_index(reports_dir, recs):
    items = sorted(recs, key=lambda r: r.get("ts", ""), reverse=True)
    rows = "".join(
        f"<a class='runrow' href='{_esc(r['file'])}'>"
        f"<div class='rl'><span class='rmode'>{_esc(r['mode'])}</span>"
        f"<span class='rtime'>{_esc(r['human'])}</span></div>"
        f"<div class='rstat'><b>{r['qualified']}</b> qualified "
        f"<span class='muted'>/ {r['screened']} screened</span></div>"
        f"<div class='rstat'>avg <b>{r['avg_score']}</b></div>"
        f"<div class='rtop'>top: {_esc(str(r.get('top_label', '')))} "
        f"<span class='muted'>{float(r.get('top_score', 0) or 0):.0f}</span></div>"
        f"<span class='go'>→</span></a>"
        for r in items)
    html = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Screener — all runs</title><style>{REPORT_CSS}{INDEX_CSS}</style></head>"
            "<body><div class='wrap'>"
            "<div class='head'><div><h1 class='title'>Screener <span>Runs</span></h1>"
            f"<div class='sub'>{len(items)} run(s) · newest first</div></div></div>"
            f"<div class='runs'>{rows}</div></div></body></html>")
    with open(os.path.join(reports_dir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(html)


def archive_report(final, args, ts=None, human=None, live=False):
    """Write this run's CSV + styled HTML dashboard into the reports/ folder and
    rebuild index.html. Pass a fixed `ts` to keep updating the SAME files as a run
    progresses (live publishing); the manifest entry is upserted, not duplicated.
    `live=True` tags the run as still in progress in the index."""
    rdir = args.reports_dir
    os.makedirs(rdir, exist_ok=True)
    if ts is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
    if human is None:
        human = time.strftime("%Y-%m-%d %H:%M")
    mode = _run_mode_str(args)
    csv_name = f"run_{ts}.csv"
    final.to_csv(os.path.join(rdir, csv_name), index=False)
    html_name = f"report_{ts}.html"
    with open(os.path.join(rdir, html_name), "w", encoding="utf-8") as fh:
        fh.write(render_report_html(final, mode, human, csv_name, live=live))

    q = final[final["disqualified"] == ""]
    top = q.iloc[0].to_dict() if len(q) else {}
    manifest = os.path.join(rdir, "index.json")
    recs = []
    if os.path.exists(manifest):
        try:
            with open(manifest) as fh:
                recs = json.load(fh)
        except (OSError, ValueError):
            recs = []
    recs = [r for r in recs if r.get("ts") != ts]          # upsert this run
    recs.append({
        "ts": ts, "human": human, "mode": mode, "file": html_name, "csv": csv_name,
        "live": bool(live),
        "screened": int(len(final)), "qualified": int(len(q)),
        "avg_score": round(float(q["score"].mean()), 1) if len(q) else 0.0,
        "top_label": str(top.get("label", "") or (str(top.get("wallet", ""))[:10])),
        "top_score": float(top.get("score", 0) or 0),
    })
    with open(manifest, "w") as fh:
        json.dump(recs, fh, indent=2)
    rebuild_index(rdir, recs)
    return os.path.join(rdir, html_name)


GIT_PUSH_INTERVAL = 90          # seconds; throttle intra-run pushes to GitHub Pages
_last_git_push = [0.0]


def git_publish(reports_dir, remote=None, message="update", force=False):
    """Commit the reports folder and push it to GitHub so GitHub Pages serves the
    live dashboard. Best-effort: needs `git` on PATH and a remote whose credentials
    are cached (GitHub Desktop / Git Credential Manager provides these). Intra-run
    calls are throttled; pass force=True for the final push."""
    import shutil
    import subprocess
    if shutil.which("git") is None:
        if force:
            print("  ! git not found on PATH — can't publish to GitHub Pages", file=sys.stderr)
        return
    now = time.time()
    if not force and (now - _last_git_push[0]) < GIT_PUSH_INTERVAL:
        return                                          # throttle live pushes
    rdir = os.path.abspath(reports_dir)

    def g(*a):
        return subprocess.run(["git", "-C", rdir, *a], capture_output=True, text=True)

    try:
        if not os.path.isdir(os.path.join(rdir, ".git")):
            g("init")
            g("branch", "-M", "main")
            open(os.path.join(rdir, ".nojekyll"), "a").close()   # serve files as-is
        if remote:
            if g("remote", "get-url", "origin").returncode != 0:
                g("remote", "add", "origin", remote)
            else:
                g("remote", "set-url", "origin", remote)
        g("add", "-A")
        g("commit", "-m", f"{message} {time.strftime('%Y-%m-%d %H:%M')}")  # no-op if unchanged
        push = g("push", "-u", "origin", "main")
        if push.returncode != 0:
            print(f"  ! git push failed: {push.stderr.strip().splitlines()[-1:] or ''}",
                  file=sys.stderr)
        else:
            _last_git_push[0] = now
            print("  ↑ dashboard pushed to GitHub Pages", file=sys.stderr)
    except Exception as e:
        print(f"  ! GitHub publish error: {e}", file=sys.stderr)


def run_hunt(args, deep, use_cache, target):
    """HUNT MODE — keep discovering and snowballing until `target` wallets QUALIFY.

    Seeds an initial pool (leaderboards + trending markets + curated web sources),
    then screens it in small batches, checking the running qualified count after
    each one and stopping the moment `target` is reached. When the pool runs dry it
    EXPANDS: every qualified wallet leads to the other big holders of the markets it
    holds, so the search frontier grows round over round until the target is hit, no
    new wallets can be found, or the round cap is reached."""
    seed_client = PolymarketClient(use_gamma=True, use_clv=False, cache=use_cache,
                                   cache_ttl_hours=args.cache_ttl)
    print(f"HUNT MODE — searching until {target} wallets qualify "
          f"(batch {args.hunt_batch}, max {args.hunt_max_rounds} expansions)\n",
          file=sys.stderr)

    labels, pool, seen = {}, [], set()

    def _ingest(ws, ls):
        for w in ws:
            if w not in seen and w not in pool:
                pool.append(w)
        for k, v in (ls or {}).items():
            labels.setdefault(k, v)

    # initial sources (best candidates first: leaderboards, then markets, then web)
    if not args.no_leaderboard:
        _ingest(*seed_from_leaderboard_sweep(seed_client, args.lb_count))
    if not args.no_markets:
        _ingest(*seed_from_trending_markets(seed_client, args.auto_markets,
                                            args.holders_limit, args.holders_min_balance))
    sources = ([] if args.no_web else list(KNOWN_WALLET_SOURCES)) + list(args.seed_url or [])
    if sources:
        ws, ls, frags = seed_from_web(sources)
        _ingest(ws, ls)
        if frags:
            for w in resolve_truncated(frags, list(pool)):
                base = labels.get(w, "")
                if not base.startswith("★"):
                    labels[w] = ("★" + base)[:20]

    if not pool:
        print("Hunt found no candidates to start from.", file=sys.stderr)
        return

    all_frames, qualified_all, frontier, expansions = [], [], [], 0
    expanded = set()        # wallets we've already snowballed from (don't repeat)
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_human = time.strftime("%Y-%m-%d %H:%M")
    interrupted = False
    try:
        while True:
            # 1) screen the next batch, newest-best first
            batch = [w for w in pool[:args.hunt_batch] if w not in seen]
            pool = pool[args.hunt_batch:]
            if batch:
                seen.update(batch)
                print(f"\n[hunt] screening {len(batch)} candidates — "
                      f"{len(qualified_all)}/{target} qualified so far\n", file=sys.stderr)
                dfb = screen(batch, use_gamma=deep, use_clv=deep, labels=labels,
                             workers=args.workers, cache=use_cache, cache_ttl=args.cache_ttl)
                if not dfb.empty:
                    all_frames.append(dfb)
                    newq = list(dfb[dfb["disqualified"] == ""]["wallet"])
                    qualified_all.extend(newq)
                    frontier.extend(newq)
                publish_progress(all_frames, args, run_ts, run_human)   # live update
                if len(qualified_all) >= target:
                    print(f"\n✓ Target reached: {len(qualified_all)} wallets qualified.",
                          file=sys.stderr)
                    break

            # 2) pool empty -> keep the hunt alive by snowballing. Expand off qualified
            #    wallets when we have them, but FALL BACK to the most promising
            #    near-misses so a run with 0 qualified so far doesn't just give up.
            if not pool:
                if expansions >= args.hunt_max_rounds:
                    print(f"\nHunt stopping: hit the {args.hunt_max_rounds}-expansion cap "
                          f"({len(qualified_all)}/{target} found). Raise --hunt-max-rounds "
                          f"to keep going.", file=sys.stderr)
                    break
                seeds = [w for w in frontier if w not in expanded]
                frontier = []
                lead_kind = "qualified wallets"
                if not seeds:                   # nothing qualified yet -> near-misses
                    seeds = _best_near_misses(all_frames, exclude=expanded, k=args.hunt_batch)
                    lead_kind = "near-miss leads"
                if not seeds:
                    print(f"\nHunt stopping: ran out of leads to expand "
                          f"({len(qualified_all)}/{target} found).", file=sys.stderr)
                    break
                expansions += 1
                expanded.update(seeds)
                print(f"\n[hunt] pool empty — expansion {expansions}/{args.hunt_max_rounds}: "
                      f"snowballing off {len(seeds)} {lead_kind} "
                      f"({len(qualified_all)}/{target} qualified) ...", file=sys.stderr)
                exp_w, exp_l = expand_from_qualified(
                    seeds, seed_client, args.holders_limit, args.holders_min_balance,
                    exclude=seen)
                _ingest(exp_w, exp_l)
                # if this lead set produced nothing new, the loop tries the next
                # batch of leads on the following iteration (or stops when none remain).
    except KeyboardInterrupt:
        interrupted = True
        print("\n\n[Ctrl+C] stopping the hunt — keeping everything found so far …",
              file=sys.stderr)

    _finalize_results(all_frames, args, ts=run_ts, human=run_human, interrupted=interrupted)


def _best_near_misses(all_frames, exclude, k):
    """Pick the most promising DISQUALIFIED wallets to snowball off when nothing has
    qualified yet — ranked by net profit, since real, profitable traders tend to
    co-hold markets with other serious traders. Lets the hunt keep generating leads
    instead of dead-ending at 0 found."""
    if not all_frames:
        return []
    df = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=["wallet"])
    dq = df[df["disqualified"] != ""].copy()
    if dq.empty:
        return []
    tp = pd.to_numeric(dq.get("total_pnl"), errors="coerce")
    rp = pd.to_numeric(dq.get("realized_pnl"), errors="coerce")
    dq["_lead"] = tp.fillna(rp).fillna(0.0)
    dq = dq[~dq["wallet"].isin(exclude)].sort_values("_lead", ascending=False)
    return list(dq["wallet"].head(k))


def _to_condition_ids(client, market_args):
    """Turn a mix of condition ids / market slugs / Polymarket URLs into a
    deduped list of condition ids (resolving slugs via Gamma)."""
    cids = []
    for arg in market_args:
        for piece in str(arg).split(","):
            piece = piece.strip()
            if not piece:
                continue
            if piece.startswith("http"):                        # pull slug from a URL
                piece = piece.rstrip("/").split("/")[-1].split("?")[0]
            if re.fullmatch(r"0x[a-fA-F0-9]{64}", piece):        # already a condition id
                cids.append(piece)
            else:
                resolved = client.resolve_market_slug(piece)
                if resolved:
                    cids.extend(resolved)
                    print(f"  resolved '{piece}' -> {len(resolved)} market(s)", file=sys.stderr)
                else:
                    print(f"  ! could not resolve '{piece}'", file=sys.stderr)
    return list(dict.fromkeys(cids))


def seed_from_markets(client, market_args, limit, min_balance):
    """Seed candidates from the top HOLDERS of one or more markets — surfaces
    conviction-holders of any size, not just global-PnL whales."""
    cids = _to_condition_ids(client, market_args)
    if not cids:
        print("No valid markets to seed from.", file=sys.stderr)
        return [], {}
    holders = client.fetch_top_holders(cids, limit, min_balance)
    wallets, labels = [], {}
    for h in holders:
        w = h["wallet"]
        if w not in labels:
            wallets.append(w)
        if h["name"]:
            labels[w] = h["name"]
    print(f"Holders seed: {len(wallets)} wallets from {len(cids)} market(s) "
          f"(top {min(max(limit,1),20)}/outcome)", file=sys.stderr)
    return wallets, labels


def _print_report(df, top):
    """Console summary: funnel + DQ breakdown + ranked survivor table."""
    from collections import Counter
    dq = df[df["disqualified"] != ""]
    q = df[df["disqualified"] == ""]
    print(f"\n{'='*78}\nSCREEN COMPLETE — {len(df)} wallets: "
          f"{len(q)} qualified, {len(dq)} disqualified\n{'='*78}")
    if not dq.empty:
        reasons = Counter(re.sub(r"\(.*?\)", "", str(s).split(";")[0]).strip()
                          for s in dq["disqualified"])
        print("disqualified for:  " +
              "   ".join(f"{k} ({v})" for k, v in reasons.most_common()))

    if q.empty:
        print("\nNo wallets passed the filters.")
        return
    show = q[REPORT_COLS].head(top).copy()
    for col in ("edge", "edge_lb", "net_edge", "skill_clv", "roi",
                "profit_factor", "max_drawdown"):
        show[col] = show[col].map(lambda v: f"{v:+.3f}" if (pd.notna(v) and v != "") else "—")
    show["trades_per_week"] = show["trades_per_week"].map(
        lambda v: f"{v:.0f}" if (pd.notna(v) and v != "") else "—")
    show["n_clv"] = show["n_clv"].map(
        lambda v: f"{int(v)}" if (pd.notna(v) and v != "") else "—")
    show["stake_range_ratio"] = show["stake_range_ratio"].map(
        lambda v: f"{v:.0f}x" if (pd.notna(v) and v != "" and np.isfinite(v)) else "—")
    for col in ("stake_median", "portfolio_value", "cash_balance", "total_balance",
                "realized_pnl", "open_unrealized_pnl", "total_pnl"):
        show[col] = show[col].map(
            lambda v: f"${v:,.0f}" if (pd.notna(v) and v != "") else "—")
    show["pnl_source"] = show["pnl_source"].map(
        lambda v: "accounting" if v == "closed_positions" else "estimate")
    show["label"] = show["label"].fillna("").astype(str).str.slice(0, 14)
    show["wallet"] = show["wallet"].str.slice(0, 12) + ".."
    print(f"\nTOP {min(top, len(q))} CANDIDATES (ranked by score):\n")
    with pd.option_context("display.max_columns", None, "display.width", 320):
        print(show.to_string(index=False))
    print("\ntotal_pnl = realized + unrealized (matches the Polymarket profile). "
          "Weigh it and net_edge alongside score.")


def main():
    ap = argparse.ArgumentParser(description="Polymarket copy-trade wallet screener v2")
    ap.add_argument("wallets", nargs="*")
    ap.add_argument("--file")
    ap.add_argument("--auto", action="store_true",
                    help="AUTONOMOUS discovery: assemble candidates from a leaderboard "
                         "sweep, trending markets, and snowball expansion — no manual seeds")
    ap.add_argument("--hunt", type=int, nargs="?", const=HUNT_TARGET, default=None, metavar="N",
                    help="HUNT MODE: keep discovering & snowballing until N wallets qualify "
                         f"(default {HUNT_TARGET}); grows the search until it finds enough")
    ap.add_argument("--hunt-batch", type=int, default=HUNT_BATCH,
                    help="in --hunt, how many candidates to screen per batch before "
                         "re-checking the target")
    ap.add_argument("--hunt-max-rounds", type=int, default=HUNT_MAX_ROUNDS,
                    help="in --hunt, safety cap on snowball expansions")
    ap.add_argument("--auto-markets", type=int, default=AUTO_TRENDING_MARKETS,
                    help="how many top live markets to mine holders from in --auto")
    ap.add_argument("--expand-rounds", type=int, default=AUTO_EXPAND_ROUNDS,
                    help="snowball rounds off qualified wallets in --auto (0 to disable)")
    ap.add_argument("--seed-url", action="append", default=[], metavar="URL_OR_FILE",
                    help="extra internet source to scan for wallet addresses — a URL or a "
                         "local file path (save a JS dashboard's page and pass the file); "
                         "repeatable")
    ap.add_argument("--no-web", action="store_true",
                    help="in --auto, skip the curated internet tracker sources")
    ap.add_argument("--no-leaderboard", action="store_true",
                    help="in --auto, skip the leaderboard-sweep source")
    ap.add_argument("--no-markets", action="store_true",
                    help="in --auto, skip the trending-market holder source")
    ap.add_argument("--leaderboard", action="store_true")
    ap.add_argument("--lb-category", default=LB_DEFAULTS["category"])
    ap.add_argument("--lb-period", default=LB_DEFAULTS["timePeriod"],
                    choices=["DAY", "WEEK", "MONTH", "ALL"])
    ap.add_argument("--lb-order", default=LB_DEFAULTS["orderBy"], choices=["PNL", "VOL"])
    ap.add_argument("--lb-count", type=int, default=LB_DEFAULTS["count"])
    ap.add_argument("--fast", action="store_true",
                    help="quick screen only — skip the automatic CLV + category analysis")
    ap.add_argument("--workers", type=int, default=5,
                    help="parallel wallets to process at once (lower to 1 if you hit 429s "
                         "or run low on memory)")
    ap.add_argument("--low-mem", action="store_true",
                    help="minimise memory use: 1 worker + smaller CLV sample. Use this if "
                         "you hit MemoryError / can't allocate (often a sign C: is out of disk)")
    # accepted for backwards-compatibility; deep analysis is now on by default
    ap.add_argument("--gamma", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--clv", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--convergence", action="store_true",
                    help="instead of screening, scan input wallets for live consensus")
    ap.add_argument("--min-converge", type=int, default=CONVERGE_MIN_WALLETS)
    ap.add_argument("--from-market", action="append", default=[], metavar="MARKET",
                    help="seed from top holders of a market (condition id, slug, or URL); "
                         "repeatable and/or comma-separated")
    ap.add_argument("--holders-limit", type=int, default=20,
                    help="top holders per outcome to pull (max 20)")
    ap.add_argument("--holders-min-balance", type=int, default=1,
                    help="minimum token balance for a holder to count")
    ap.add_argument("--check-clv", metavar="TOKEN_OR_WALLET",
                    help="validate the /prices-history endpoint live, then exit")
    ap.add_argument("-o", "--out", default="wallet_scores.csv")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--reports-dir", default="reports",
                    help="folder where every run's HTML dashboard + CSV are archived "
                         "(an index.html lets you browse them one after another)")
    ap.add_argument("--publish-git", action="store_true",
                    help="commit + push the reports folder to GitHub each run, so the "
                         "dashboard is live on GitHub Pages (reachable from anywhere)")
    ap.add_argument("--git-remote", default=None, metavar="URL",
                    help="GitHub repo URL for --publish-git (set once; e.g. "
                         "https://github.com/you/polymarket-dashboard.git)")
    ap.add_argument("--min-score", type=float, default=None,
                    help=f"composite score floor to qualify (default {FILTERS['min_score']:.0f}); "
                         "raise for stricter, lower to see more")
    ap.add_argument("--no-cash-balance", action="store_true",
                    help="skip the on-chain USDC cash-balance lookup (faster; portfolio "
                         "value still shown)")
    ap.add_argument("--rpc-url", default=POLYGON_RPC,
                    help="Polygon RPC endpoint for the on-chain cash-balance lookup")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the on-disk response cache and refetch everything")
    ap.add_argument("--cache-ttl", type=float, default=CACHE_TTL_HOURS,
                    help="hours before a cached response is considered stale")
    args = ap.parse_args()

    if args.check_clv:
        check_clv(args.check_clv)
        return

    # Deep analysis (CLV + category) runs automatically unless --fast.
    deep = not args.fast
    use_cache = not args.no_cache
    if args.min_score is not None:                  # CLI override of the score floor
        FILTERS["min_score"] = args.min_score
    global FETCH_CASH, RPC_URL, CLV_SAMPLE           # on-chain cash-balance settings
    FETCH_CASH = not args.no_cash_balance
    RPC_URL = args.rpc_url
    if args.low_mem:                                 # shrink footprint on tight machines
        args.workers = 1
        CLV_SAMPLE = min(CLV_SAMPLE, 15)

    # Autonomous discovery: the screener picks its own hunting grounds.
    if args.hunt is not None:
        run_hunt(args, deep=deep, use_cache=use_cache, target=max(1, args.hunt))
        return

    if args.auto:
        run_auto(args, deep=deep, use_cache=use_cache)
        return

    # Seeding needs Gamma to resolve market slugs/URLs; that's all this client does.
    seed_client = PolymarketClient(use_gamma=True, use_clv=False, cache=use_cache,
                                   cache_ttl_hours=args.cache_ttl)
    wallets, labels = list(args.wallets), {}
    if args.file:
        with open(args.file) as fh:
            wallets += [ln for ln in fh.read().splitlines() if ln.strip() and not ln.startswith("#")]
    if args.leaderboard:
        lbw, lblabels = seed_from_leaderboard(seed_client, args.lb_category, args.lb_period,
                                              args.lb_order, args.lb_count)
        wallets += lbw; labels.update(lblabels)
    if args.from_market:
        mw, mlabels = seed_from_markets(seed_client, args.from_market,
                                        args.holders_limit, args.holders_min_balance)
        wallets += mw; labels.update(mlabels)
    if not wallets:
        ap.error("provide wallet addresses, --auto, --file, --leaderboard, or --from-market")
    wallets = list(dict.fromkeys(w.strip().lower() for w in wallets if w.strip()))

    if args.convergence:
        print(f"Scanning {len(wallets)} wallets for live consensus "
              f"(>={args.min_converge} on same side) ...", file=sys.stderr)
        hits = scan_convergence(wallets, seed_client, args.min_converge)
        if not hits:
            print("No convergence found.", file=sys.stderr); return
        print("\nLIVE CONVERGENCE (high-conviction consensus):\n")
        for title, outcome, n in hits:
            print(f"  {n} wallets  |  {outcome:<6}  |  {title}")
        return

    # One pass: cheap screen, then automatic CLV/Gamma analysis of the survivors.
    df = screen(wallets, use_gamma=deep, use_clv=deep, labels=labels,
                workers=args.workers, cache=use_cache, cache_ttl=args.cache_ttl)
    if df.empty:
        print("No usable results.", file=sys.stderr); return
    df.to_csv(args.out, index=False)
    _print_report(df, args.top)
    print(f"\nFull results -> {args.out}")


if __name__ == "__main__":
    main()
