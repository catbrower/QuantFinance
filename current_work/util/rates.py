"""Risk-free rate for converting funded returns (SPY) into excess returns.

WHY THIS EXISTS AT ALL: Moskowitz, Ooi & Pedersen (2012) trade futures, which are
unfunded, so their returns are ALREADY excess returns and no rate is subtracted anywhere
in the paper. We need one only because SPY is a funded cash instrument standing in for
the S&P 500 future. This module disappears once real futures returns are available.

WHICH RATE: a short bill, not a long note. The risk-free rate for a one-month holding
period is the 3-month (or 1-month) T-bill. The 10-year constant-maturity yield carries
term premium and duration risk and is not a funding rate.

Measured on SPY 2021-2026 the choice happened not to matter much (Sharpe 0.571 with
DTB3 vs 0.575 with DGS10) because the two averaged 3.64% and 3.70% over that span. That
is coincidence: the curve was inverted 42% of those days and the 10y-3m spread ranged
-1.73% to +2.28%. In 2021 DGS10 read 1.45% while actual funding was 0.05%.

What DOES matter is using a rate at all: assuming zero inflates the SPY TSMOM Sharpe
from 0.57 to 0.99 and the annualised mean from 21.5% to 37.3%.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

SERIES = {
    "DTB3": "3-month Treasury bill, secondary market, daily",     # preferred
    "DTB4WK": "4-week Treasury bill, daily",
    "DGS10": "10-year Treasury constant maturity, daily",         # NOT a funding rate
}
LOCAL_FILES = {"DGS10": "treasury-rates-10-year.csv"}


def _tidy(df, series):
    date_col = next(c for c in df.columns if "date" in c.lower())
    s = pd.to_numeric(df[series], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    return (s.dropna() / 100.0).rename(series).sort_index()


def load_local(series="DGS10", data_dir=DATA_DIR):
    """Read a rate series already downloaded into data/."""
    path = Path(data_dir) / LOCAL_FILES.get(series, f"{series}.csv")
    if not path.exists():
        raise FileNotFoundError(path)
    return _tidy(pd.read_csv(path), series)


def fetch_fred(series="DTB3", data_dir=DATA_DIR, cache=True):
    """Download a daily series from FRED. No API key needed. Caches into data/."""
    cache_path = Path(data_dir) / f"{series}.csv"
    raw = urllib.request.urlopen(FRED_CSV.format(series=series), timeout=60).read()
    if cache:
        cache_path.write_bytes(raw)
    return _tidy(pd.read_csv(io.BytesIO(raw)), series)


def risk_free_daily(series="DTB3", data_dir=DATA_DIR, allow_download=True):
    """Daily risk-free rate as a decimal (0.0425 = 4.25%).

    Order: cached/local file, then FRED, then the local DGS10 as a last resort with a
    warning -- a 10-year yield is a poor funding proxy but far better than assuming zero.
    """
    cache_path = Path(data_dir) / f"{series}.csv"
    if cache_path.exists():
        return _tidy(pd.read_csv(cache_path), series)
    if allow_download:
        try:
            return fetch_fred(series, data_dir)
        except Exception as exc:                        # offline, firewalled, etc.
            print(f"  [rates] FRED fetch failed ({type(exc).__name__}); falling back")
    fallback = load_local("DGS10", data_dir)
    print("  [rates] USING DGS10 (10-year) AS THE RISK-FREE RATE -- it is not a funding"
          "\n          rate; results carry a term-premium bias. Prefer DTB3.")
    return fallback


def monthly_rate(daily, index=None):
    """Daily annualised rate -> per-month rate, aligned to a PeriodIndex if given."""
    m = daily.groupby(daily.index.to_period("M")).mean() / 12.0
    return m if index is None else m.reindex(index)


def align_daily(daily, dates):
    """Forward-fill onto a trading calendar (rates are not published every day)."""
    return daily.reindex(pd.DatetimeIndex(dates)).ffill()
