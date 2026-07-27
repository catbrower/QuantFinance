#!/usr/bin/env python3
"""Check which ETFs in etfs_combined.csv actually have data on Polygon.

Probes each symbol with one daily-aggregate request over the past week. A symbol counts
as UNAVAILABLE if the request errors or comes back with no bars -- an unknown ticker
returns an empty list rather than raising, so both cases have to be treated the same.

Writes two files:
  etfs_polygon_check.csv  -- every symbol with its status, bar count and last close
  etfs_available.csv      -- the original rows, filtered to the available symbols

The probe report doubles as a resume cache: rerunning skips symbols already checked
unless --refresh is passed. That matters if the key is rate-limited, where 500 calls can
take an hour and a half rather than ten seconds.

Usage:
    python check_polygon_availability.py [--days 7] [--workers 8] [--refresh] [--limit N]
"""

import argparse
import csv
import datetime as dt
import os
import sys
import threading
import time
from pathlib import Path

import pandas as pd
from massive import RESTClient

HERE = Path(__file__).parent
SRC_CSV = HERE / "etfs_combined.csv"
REPORT_CSV = HERE / "etfs_polygon_check.csv"
OUT_CSV = HERE / "etfs_available.csv"

# Prefer the environment; the literal is the key already sitting in current.ipynb, kept
# only so this runs out of the box. Move it to POLYGON_API_KEY when convenient.
API_KEY = os.environ.get("POLYGON_API_KEY", ""api_key"")

RETRIES = 3
BACKOFF = 1.5          # seconds, doubled each retry
REPORT_FIELDS = ["symbol", "status", "bars", "last_close", "total_volume", "error"]


def probe(client, symbol, start, end):
    """One symbol -> a report row. status is 'ok' | 'empty' | 'error'."""
    last_err = ""
    for attempt in range(RETRIES):
        try:
            bars = client.get_aggs(ticker=symbol, multiplier=1, timespan="day",
                                   from_=str(start), to=str(end), limit=50)
            if not bars:
                return dict(symbol=symbol, status="empty", bars=0,
                            last_close="", total_volume="", error="")
            vol = sum(getattr(b, "volume", 0) or 0 for b in bars)
            return dict(symbol=symbol, status="ok", bars=len(bars),
                        last_close=getattr(bars[-1], "close", ""),
                        total_volume=vol, error="")
        except Exception as exc:                      # noqa: BLE001 - any failure = unavailable
            last_err = f"{type(exc).__name__}: {exc}"[:200]
            if attempt < RETRIES - 1:
                time.sleep(BACKOFF * (2 ** attempt))
    return dict(symbol=symbol, status="error", bars=0,
                last_close="", total_volume="", error=last_err)


def load_done():
    """Symbols already probed, from a previous run."""
    if not REPORT_CSV.exists():
        return {}
    df = pd.read_csv(REPORT_CSV, dtype=str, keep_default_na=False)
    return {r["symbol"]: dict(r) for _, r in df.iterrows()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=7, help="lookback window (default 7)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent requests (default 8)")
    ap.add_argument("--refresh", action="store_true", help="re-probe symbols already checked")
    ap.add_argument("--limit", type=int, default=0, help="only probe the first N symbols")
    args = ap.parse_args()

    if not SRC_CSV.exists():
        sys.exit(f"input not found: {SRC_CSV}")
    src = pd.read_csv(SRC_CSV, dtype=str, keep_default_na=False)
    symbols = [s for s in src["Symbol"].tolist() if s]
    if args.limit:
        symbols = symbols[: args.limit]

    done = {} if args.refresh else load_done()
    todo = [s for s in symbols if s not in done]

    end = dt.date.today()
    start = end - dt.timedelta(days=args.days)
    print(f"{len(symbols)} symbols | {len(done)} cached | {len(todo)} to probe")
    print(f"window {start} -> {end} (daily bars), {args.workers} workers\n")

    results = dict(done)
    if todo:
        # append as we go so an interrupted run is not wasted
        fresh = not REPORT_CSV.exists() or args.refresh
        fh = REPORT_CSV.open("w" if fresh else "a", encoding="utf-8", newline="")
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
        if fresh:
            writer.writeheader()
        lock = threading.Lock()
        counter = {"n": 0}

        def work(sym):
            client = RESTClient(api_key=API_KEY)      # one client per thread
            row = probe(client, sym, start, end)
            with lock:
                writer.writerow(row)
                fh.flush()
                results[sym] = row
                counter["n"] += 1
                if counter["n"] % 50 == 0 or counter["n"] == len(todo):
                    print(f"  probed {counter['n']:>4}/{len(todo)}")
            return row

        from concurrent.futures import ThreadPoolExecutor
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, todo))
        fh.close()
        print(f"  done in {time.perf_counter() - t0:.1f}s")

    # ---- summarise & filter ----
    report = pd.DataFrame([results[s] for s in symbols if s in results])
    report["bars"] = pd.to_numeric(report["bars"], errors="coerce").fillna(0).astype(int)
    counts = report["status"].value_counts().to_dict()
    available = set(report.loc[report["status"] == "ok", "symbol"])

    out = src[src["Symbol"].isin(available)].copy()
    out.to_csv(OUT_CSV, index=False)
    report.to_csv(REPORT_CSV, index=False)          # rewrite tidy, deduped

    print(f"\nstatus: {counts}")
    print(f"available : {len(available):>4} of {len(symbols)} "
          f"({len(available) / len(symbols) * 100:.1f}%)")
    missing = report.loc[report["status"] != "ok", "symbol"].tolist()
    if missing:
        print(f"unavailable ({len(missing)}): {', '.join(missing[:25])}"
              + (" ..." if len(missing) > 25 else ""))
    print(f"\nwrote {REPORT_CSV.name} ({len(report)} rows) "
          f"and {OUT_CSV.name} ({len(out)} rows x {out.shape[1]} cols)")
    if report["bars"].gt(0).any():
        print("bars returned per available symbol:",
              report.loc[report["status"] == "ok", "bars"].value_counts().to_dict())


if __name__ == "__main__":
    main()
