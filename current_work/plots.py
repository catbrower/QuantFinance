"""Charts for the intraday research notebook.

Every plot takes the data it draws, so any of them can be pointed at a different
session without touching the notebook:

    plots.plot_session_stats(df_min, day=3, ticker="AAPL", window=stats_window)
    plots.plot_session_jumps(df_min, day=3, ticker="AAPL", window=stats_window)

Each function accepts either a frame already sliced to one session, or the full minute
frame plus `day=`. All of them return the Figure so the caller can save or tweak it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from research import CUM, DOWN, REG_MINUTES, ROLL, TICKS, UP, clock_labels

__all__ = [
    "plot_tsmom_tstats", "plot_tsmom_equity",
    "session", "session_date",
    "plot_bar_coverage", "plot_session_stats", "plot_session_jumps",
]


# ------------------------------------------------------------------------ helpers -----
def session(df, day=None):
    """One session's rows, sorted by minute.

    `day=None` means df is already a single session. Raises if it turns out not to be,
    rather than silently plotting several days on top of each other.
    """
    out = df if day is None else df[df["index_day"] == day]
    if out.empty:
        raise ValueError(f"no rows for index_day={day}")
    n = out["index_day"].nunique()
    if n != 1:
        raise ValueError(f"expected one session, got {n}; pass day=<index_day>")
    return out.sort_values("index_minute")


def session_date(day_df):
    """Calendar date of a session, from its first timestamp (market-local)."""
    return (pd.to_datetime(day_df["timestamp"].iloc[0], unit="ms", utc=True)
            .tz_convert("America/New_York").date())


def _tidy(ax):
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _minute_axis(ax, xlabel="index_minute"):
    ax.set_xticks(TICKS)
    ax.set_xticklabels(clock_labels())
    ax.set_xlim(-2, REG_MINUTES + 1)
    ax.set_xlabel(xlabel)


def _require(day_df, cols, what):
    missing = [c for c in cols if c not in day_df.columns]
    if missing:
        raise KeyError(f"{what} needs columns not present: {missing}")


# -------------------------------------------------------------------------- charts ----
def plot_bar_coverage(index_minutes, *, ticker="", n_sessions=None, n_bars=None,
                      figsize=(12, 4), color="#4269d0"):
    """Bars per index_minute across every session -- a data-coverage sanity check.

    One bin per minute, so each column's height is how many bars landed on that minute of
    the day. Dashed lines mark the regular session; anything outside them is pre/post
    market.
    """
    mins = pd.Series(index_minutes).dropna()
    sessions = n_sessions if n_sessions is not None else "?"
    bars = n_bars if n_bars is not None else len(mins)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(mins, bins=range(int(mins.min()), int(mins.max()) + 2),
            color=color, edgecolor="none")
    ax.axvline(0, color="#888888", linestyle="--", linewidth=1)
    ax.axvline(REG_MINUTES - 1, color="#888888", linestyle="--", linewidth=1)
    ax.set_xlabel("index_minute   (0 = 9:30 open, 389 = 15:59 close)")
    ax.set_ylabel("bars")
    ax.set_title(f"{ticker}: bars per index_minute  ({sessions} sessions, {bars:,} bars)")
    _tidy(ax)
    return fig


def plot_session_stats(df, day=None, *, ticker="", window=60, figsize=(14, 9)):
    """Candles with the price envelope on top, return moments underneath.

    Mean/std are in %/bar so they cannot share the price axis -- they get their own panel
    rather than being silently rescaled onto the candles.
    """
    d = session(df, day)
    _require(d, ["open", "high", "low", "close", "price_roll_high", "price_roll_low",
                 "price_cum_high", "price_cum_low", "ret_roll_mean", "ret_roll_std",
                 "ret_cum_mean", "ret_cum_std"], "plot_session_stats")
    date = session_date(d)
    label = d["index_day"].iloc[0]
    colour = (d["close"] >= d["open"]).map({True: UP, False: DOWN}).tolist()
    x = d["index_minute"]

    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # wick then body; 2px-equivalent bodies at 390 candles stay legible
    ax.vlines(x, d["low"], d["high"], color=colour, linewidth=0.8)
    ax.bar(x, d["close"] - d["open"], bottom=d["open"], width=0.7,
           color=colour, linewidth=0)

    # running / cumulative price envelope, with a faint fill so the running channel
    # reads as one band rather than two stray lines
    ax.fill_between(x, d["price_roll_low"], d["price_roll_high"], color=ROLL,
                    alpha=0.08, linewidth=0)
    ax.plot(x, d["price_roll_high"], color=ROLL, linewidth=1.2, linestyle="--",
            label=f"running high/low ({window}m)")
    ax.plot(x, d["price_roll_low"], color=ROLL, linewidth=1.2, linestyle="--")
    ax.plot(x, d["price_cum_high"], color=CUM, linewidth=1.2, linestyle="--",
            label="cumulative high/low")
    ax.plot(x, d["price_cum_low"], color=CUM, linewidth=1.2, linestyle="--")
    if "vwap" in d.columns:
        ax.plot(x, d["vwap"], color="#666666", linewidth=1, label="minute vwap")
    ax.set_ylabel("price")
    ax.set_title(f"{ticker}  index_day={label} ({date}) -- "
                 f"{window}m running vs cumulative stats")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False)

    ax_r.axhline(0, color="#999999", linewidth=0.8)
    for m, s, colr, name in [("ret_roll_mean", "ret_roll_std", ROLL, f"running ({window}m)"),
                             ("ret_cum_mean", "ret_cum_std", CUM, "cumulative")]:
        ax_r.fill_between(x, d[m] - d[s], d[m] + d[s], color=colr, alpha=0.10, linewidth=0)
        ax_r.plot(x, d[m], color=colr, linewidth=1.8, label=f"{name} mean ±1σ")
    ax_r.set_ylabel("% close return per bar")
    ax_r.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    _minute_axis(ax_r)

    for a in (ax, ax_r):
        _tidy(a)
    return fig


def plot_session_jumps(df, day=None, *, ticker="", window=60, figsize=(14, 6)):
    """Jump amplitude and rate for one session, running vs cumulative.

    Gaps in the xi trace are windows whose variance estimate came out negative -- the
    estimator failing on too few samples, not a property of the data. The count goes in
    the title so the gaps are labelled rather than mysterious.
    """
    d = session(df, day)
    _require(d, ["ret_roll_xi", "ret_cum_xi", "ret_roll_lamb", "ret_cum_lamb",
                 "ret_roll_xi_var"], "plot_session_jumps (run compute_jump_terms first)")
    date = session_date(d)
    label = d["index_day"].iloc[0]
    x = d["index_minute"]

    fig, (ax_xi, ax_lamb) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    for ax, col_r, col_c, name in [
        (ax_xi, "ret_roll_xi", "ret_cum_xi", "jump amplitude  $\\xi$  (%)"),
        (ax_lamb, "ret_roll_lamb", "ret_cum_lamb", "jump rate  $\\lambda$"),
    ]:
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.plot(x, d[col_r], color=ROLL, linewidth=1.5, label=f"running ({window}m)")
        ax.plot(x, d[col_c], color=CUM, linewidth=2, label="cumulative")
        ax.set_ylabel(name)
        _tidy(ax)

    n_gap = int(d["ret_roll_xi_var"].notna().sum() - d["ret_roll_xi"].notna().sum())
    ax_xi.set_title(f"{ticker}  index_day={label} ({date}) -- jump-diffusion terms from "
                    f"the close % returns   ({n_gap} running windows dropped: "
                    f"negative $\\sigma_\\xi^2$)")
    ax_xi.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    _minute_axis(ax_lamb)
    return fig


# ---------------------------------------------------- Moskowitz, Ooi & Pedersen (2012)
def plot_tsmom_tstats(reg_return, reg_sign=None, *, title="", figsize=(13, 6),
                      sig=1.96):
    """Fig. 1: t-statistics of the pooled lag regressions, by month lag.

    Panel A is Eq. (2) (lagged scaled return), Panel B is Eq. (3) (sign only). The
    paper's finding is positive t-stats for lags 1-12 and negative beyond.
    """
    panels = [(reg_return, "Panel A: Eq. (2), lagged scaled return")]
    if reg_sign is not None:
        panels.append((reg_sign, "Panel B: Eq. (3), sign of lagged return"))

    fig, axes = plt.subplots(len(panels), 1, figsize=figsize, sharex=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (reg, label) in zip(axes, panels):
        t = reg["t_stat"]
        ax.bar(t.index, t.values,
               color=[UP if v >= 0 else DOWN for v in t.values], width=0.8)
        ax.axhline(0, color="#333333", linewidth=0.8)
        for s in (sig, -sig):
            ax.axhline(s, color="#999999", linestyle="--", linewidth=0.8)
        ax.axvline(12.5, color="#7570b3", linestyle=":", linewidth=1.2)
        ax.set_ylabel("t-statistic")
        ax.set_title(label, fontsize=10, loc="left")
        _tidy(ax)
    axes[-1].set_xlabel("month lag h")
    if title:
        fig.suptitle(title, y=0.98)
    fig.tight_layout()
    return fig


def plot_tsmom_equity(strategy, benchmark=None, *, title="", figsize=(13, 5),
                      labels=("TSMOM", "always long, same sizing")):
    """Cumulative growth of the vol-scaled strategy against the always-long benchmark."""
    fig, ax = plt.subplots(figsize=figsize)
    s = pd.Series(strategy).dropna()
    ax.plot(s.index.to_timestamp() if hasattr(s.index, "to_timestamp") else s.index,
            (1 + s).cumprod(), color=ROLL, linewidth=1.8, label=labels[0])
    if benchmark is not None:
        b = pd.Series(benchmark).dropna()
        ax.plot(b.index.to_timestamp() if hasattr(b.index, "to_timestamp") else b.index,
                (1 + b).cumprod(), color=CUM, linewidth=1.5, label=labels[1])
    ax.axhline(1.0, color="#999999", linewidth=0.8)
    ax.set_ylabel("growth of 1")
    ax.set_title(title or "Time series momentum, vol-scaled")
    ax.legend(frameon=False)
    _tidy(ax)
    return fig
