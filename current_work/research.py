"""Calculations for the intraday / swing research notebook.

Everything here is a pure-ish computation: frame construction, indicators, the jump
estimator, the oracle label, and the execution harness. Plotting and reporting stay in
the notebook, since those are the things you tweak interactively.

CAUSALITY RULE for anything in the indicator pipeline: a value stamped on minute m may
only use data from minute m and earlier, and cross-day profiles may only use *prior*
sessions. Nothing is normalised by an end-of-day or full-sample statistic. Anything not
yet knowable is NaN, never filled.

Two things in here are deliberately NOT causal, and are named and isolated so they
cannot be mistaken for features:
  * build_oracle()          -- uses the whole session; it is the ceiling, a label.
  * execution_benchmarks()  -- TWAP/VWAP over the whole day; scores a policy, never feeds it.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from math import gamma, pi

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ session constants --
DAY_START = 9 * 60 + 30            # 09:30 in minutes past midnight, market local
DAY_END = 16 * 60                  # 16:00
REG_MINUTES = DAY_END - DAY_START  # 390 minutes in the regular session

PRICE_COLS = ["open", "high", "low", "close"]
RETURN_COL = "close_pct"           # the single return series everything is built from
RET_COLS = [RETURN_COL]
PER_BAR = len(RET_COLS)            # return observations per second-bar

OR_WINDOWS = (15, 30, 60)          # opening-range lengths, in minutes
VR_Q = 5                           # variance-ratio aggregation, in minutes

MU_1 = np.sqrt(2.0 / pi)                          # E|Z|
MU_43 = 2 ** (2 / 3) * gamma(7 / 6) / gamma(0.5)  # E|Z|^(4/3)

# palette, shared by every chart in the notebook
UP, DOWN = "#1b9e77", "#d95f02"    # diverging pair, readable under CVD
ROLL, CUM = "#7570b3", "#e7298a"
TICKS = list(range(0, REG_MINUTES + 1, 30))


def clock_labels(ticks=TICKS):
    """['09:30', '10:00', ...] for an index_minute axis."""
    return [f"{(DAY_START + t) // 60:02d}:{(DAY_START + t) % 60:02d}" for t in ticks]


# ------------------------------------------------------------------------- prep --------
def add_time_index(df, day_start=DAY_START):
    """day / index_day / index_minute, in market-local time, on a raw agg frame."""
    ny = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    df["day"] = ny.dt.dayofweek                                   # 0 = Monday
    df["index_day"] = pd.factorize(ny.dt.normalize(), sort=True)[0]
    df["index_minute"] = ny.dt.hour * 60 + ny.dt.minute - day_start
    return df


def add_returns(df, price_cols=("open", "high", "low", "close", "vwap")):
    """Bar-to-bar % return of each price series, plus volume's own change series.

    Everything downstream works on returns, never on price levels -- a std of levels is
    a range measure, a std of returns is a volatility.
    """
    for col in price_cols:
        if col in df.columns:
            df[f"{col}_pct"] = df[col].pct_change(fill_method=None) * 100
    df["volume_pct"] = (df["volume"].pct_change(fill_method=None) * 100).replace(
        [np.inf, -np.inf], np.nan)
    return df


# ---------------------------------------------------------------------- helpers --------
def _roll(df_min, cols, how="sum", window=None):
    """Grouped rolling over minutes, per session. Never spans a day boundary."""
    r = df_min.groupby("index_day")[cols].rolling(window, min_periods=window)
    return getattr(r, how)().reset_index(level=0, drop=True)


def _expand(df_min, cols, how="sum", window=None):
    """Grouped expanding from the session open, undefined before `window` minutes."""
    e = df_min.groupby("index_day")[cols].expanding(min_periods=window)
    return getattr(e, how)().reset_index(level=0, drop=True)


def _mean_std(n, s, ss):
    """Pooled mean/std from window totals -- ddof=1, clipped against fp cancellation."""
    mean = s / n
    var = (ss - s**2 / n) / (n - 1)
    return mean, np.sqrt(var.clip(lower=0))


def _prior_day_profile(df_min, col):
    """Mean of `col` at this index_minute over PRIOR sessions only (NaN on day 0)."""
    return (
        df_min.sort_values(["index_minute", "index_day"])
        .groupby("index_minute")[col]
        .transform(lambda s: s.shift(1).expanding().mean())
        .reindex(df_min.index)
    )


# ------------------------------------------------------------- frame construction ------
def build_session_frame(df_raw):
    """Regular-session second-bars, sorted, with the return series cleaned.

    The first bar of a session has no valid predecessor *inside* the session -- its
    pct_change reaches back into premarket, or on a day with no premarket data all the
    way into the previous session -- so drop it rather than let an overnight gap
    masquerade as a one-second return.
    """
    df = (
        df_raw[(df_raw["index_minute"] >= 0) & (df_raw["index_minute"] < REG_MINUTES)]
        .sort_values(["index_day", "index_minute", "timestamp"])
        .reset_index(drop=True)
    )
    df.loc[df.groupby("index_day").cumcount() == 0, RET_COLS] = np.nan
    return df


def build_minute_frame(df_reg):
    """Collapse seconds -> minutes ONCE. This is what keeps everything else vectorised.

    Every statistic downstream is a sum (or max/min) over a window of minutes, and a sum
    is associative: summing 60 minute-blocks equals summing their ~3,600 underlying
    values. So each minute carries count / sum / sum-of-squares / max / min, and the
    windowing becomes grouped rolling arithmetic over ~390 rows per session instead of
    ~23,400. Counts are carried per minute rather than assumed, so minutes with missing
    seconds -- and the dropped first bar of each session -- pool correctly.
    """
    r = df_reg[RET_COLS]
    tmp = df_reg.assign(
        _r_n=r.notna().sum(axis=1),
        _r_sum=r.sum(axis=1),
        _r_sqsum=(r**2).sum(axis=1),
        _p_max=df_reg[PRICE_COLS].max(axis=1),
        _p_min=df_reg[PRICE_COLS].min(axis=1),
        _v_sqsum=df_reg["volume"] ** 2,
        _vw_num=df_reg["vwap"] * df_reg["volume"],
    )
    out = tmp.groupby(["index_day", "index_minute"], as_index=False).agg(
        timestamp=("timestamp", "first"),
        # the minute candle itself, rebuilt from the seconds inside it
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), bars=("close", "size"),
        # return block
        r_n=("_r_n", "sum"), r_sum=("_r_sum", "sum"), r_sqsum=("_r_sqsum", "sum"),
        # price envelope, over all four o/h/l/c levels
        p_max=("_p_max", "max"), p_min=("_p_min", "min"),
        # volume block, kept separate from the return pool
        v_sum=("volume", "sum"), v_sqsum=("_v_sqsum", "sum"),
        v_max=("volume", "max"), v_min=("volume", "min"),
        _vw_num=("_vw_num", "sum"),
    )
    out["vwap"] = out["_vw_num"] / out["volume"].where(out["volume"] > 0)
    out["v_n"] = out["bars"]
    out["m_ret"] = out.groupby("index_day")["close"].pct_change(fill_method=None) * 100
    return out.drop(columns=["_vw_num"])


# ------------------------------------------------------------------ pipeline steps -----
# Contract:  f(df_min, df_reg, window) -> df_min
# Steps never mutate df_reg, and each one only appends columns.

def add_window_moments(df_min, df_reg, window):
    """ret_{roll,cum}_{mean,std}, volume_{roll,cum}_{mean,std}, price/volume high/low.

    Mean and std are of the % returns, so the std is a volatility. High/low stay as
    price *levels*, pooled over every o/h/l/c in the window.
    """
    sums = ["r_n", "r_sum", "r_sqsum", "v_n", "v_sum", "v_sqsum"]
    hi_cols, lo_cols = ["p_max", "v_max"], ["p_min", "v_min"]

    roll = _roll(df_min, sums, "sum", window)
    cum = _expand(df_min, sums, "sum", window)
    roll_hi = _roll(df_min, hi_cols, "max", window)
    roll_lo = _roll(df_min, lo_cols, "min", window)
    cum_hi = _expand(df_min, hi_cols, "max", window)
    cum_lo = _expand(df_min, lo_cols, "min", window)

    df_min["ret_roll_mean"], df_min["ret_roll_std"] = _mean_std(
        roll["r_n"], roll["r_sum"], roll["r_sqsum"])
    df_min["ret_cum_mean"], df_min["ret_cum_std"] = _mean_std(
        cum["r_n"], cum["r_sum"], cum["r_sqsum"])
    df_min["volume_roll_mean"], df_min["volume_roll_std"] = _mean_std(
        roll["v_n"], roll["v_sum"], roll["v_sqsum"])
    df_min["volume_cum_mean"], df_min["volume_cum_std"] = _mean_std(
        cum["v_n"], cum["v_sum"], cum["v_sqsum"])

    df_min["price_roll_high"], df_min["price_roll_low"] = roll_hi["p_max"], roll_lo["p_min"]
    df_min["price_cum_high"], df_min["price_cum_low"] = cum_hi["p_max"], cum_lo["p_min"]
    df_min["volume_roll_high"], df_min["volume_roll_low"] = roll_hi["v_max"], roll_lo["v_min"]
    df_min["volume_cum_high"], df_min["volume_cum_low"] = cum_hi["v_max"], cum_lo["v_min"]
    return df_min


def add_realised_variance(df_min, df_reg, window):
    """rv, bv, rv_vol, jump_var, jump_frac, bns_z.

    RV catches total variation; BV is robust to jumps, so RV - BV isolates the jump part.
    Same question xi/lambda are asking, answered with second moments -- which is why it
    survives at sample sizes where the KM estimator does not.

    Caveat: at one-second sampling microstructure noise biases RV upward. If rv_vol
    reads implausibly high next to yz_vol, subsample the returns to 5-10s.
    """
    r = df_reg[RETURN_COL]
    g = df_reg.groupby("index_day")[RETURN_COL]
    r1, r2 = g.shift(1), g.shift(2)
    blocks = (
        df_reg[["index_day", "index_minute"]]
        .assign(_rv=r**2,
                _bv=r.abs() * r1.abs(),
                _tp=(r.abs() ** (4 / 3)) * (r1.abs() ** (4 / 3)) * (r2.abs() ** (4 / 3)))
        .groupby(["index_day", "index_minute"], as_index=False)
        .agg(rv_sum=("_rv", "sum"), bv_sum=("_bv", "sum"), tp_sum=("_tp", "sum"))
    )
    df_min = df_min.merge(blocks, on=["index_day", "index_minute"], how="left")

    w = _roll(df_min, ["rv_sum", "bv_sum", "tp_sum", "r_n"], "sum", window)
    n = w["r_n"]
    df_min["rv"] = w["rv_sum"]                                 # realised variance, %^2
    df_min["bv"] = w["bv_sum"] / MU_1**2                       # bipower variation
    df_min["rv_vol"] = np.sqrt(df_min["rv"].clip(lower=0))     # window realised vol, %
    df_min["jump_var"] = (df_min["rv"] - df_min["bv"]).clip(lower=0)
    df_min["jump_frac"] = (df_min["jump_var"] / df_min["rv"]).replace(
        [np.inf, -np.inf], np.nan)                             # relative jump, RJ

    # BNS z-statistic: how many sigmas of jump contribution, via tripower quarticity
    tq = n * MU_43 ** (-3) * w["tp_sum"]
    denom = ((pi**2 / 4) + pi - 5) / n * np.maximum(1.0, tq / df_min["bv"] ** 2)
    df_min["bns_z"] = df_min["jump_frac"] / np.sqrt(denom.replace(0, np.nan))
    return df_min


def add_range_volatility(df_min, df_reg, window):
    """park_vol, gk_vol, rs_vol, yz_vol -- all in %.

    5-8x more efficient than close-to-close, and none of these assume an h/l ordering,
    which is exactly why they are the right way to use the extremes.
    """
    o, h, l, c = (df_min[k] for k in ("open", "high", "low", "close"))
    prev_c = df_min.groupby("index_day")["close"].shift(1)
    hl, co = np.log(h / l), np.log(c / o)

    tmp = df_min.assign(
        _park=hl**2 / (4 * np.log(2)),
        _gk=0.5 * hl**2 - (2 * np.log(2) - 1) * co**2,
        _rs=np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o),
        _gap=np.log(o / prev_c),      # NaN on each session's first bar
        _co=co,
    )
    tmp["_gap_sq"], tmp["_co_sq"] = tmp["_gap"] ** 2, tmp["_co"] ** 2

    for name in ("park", "gk", "rs"):
        df_min[f"{name}_vol"] = 100 * np.sqrt(
            _roll(tmp, [f"_{name}"], "mean", window)[f"_{name}"].clip(lower=0))

    # Yang-Zhang: gap variance + k * open-to-close variance + (1-k) * Rogers-Satchell
    s = _roll(tmp, ["_gap", "_co"], "sum", window)
    ss = _roll(tmp, ["_gap_sq", "_co_sq"], "sum", window)
    n_gap = _roll(tmp, ["_gap"], "count", window)["_gap"]
    n_co = _roll(tmp, ["_co"], "count", window)["_co"]
    var_gap = (ss["_gap_sq"] - s["_gap"] ** 2 / n_gap) / (n_gap - 1)
    var_co = (ss["_co_sq"] - s["_co"] ** 2 / n_co) / (n_co - 1)
    k = 0.34 / (1.34 + (n_co + 1) / (n_co - 1))
    rs_mean = _roll(tmp, ["_rs"], "mean", window)["_rs"]
    df_min["yz_vol"] = 100 * np.sqrt((var_gap + k * var_co + (1 - k) * rs_mean).clip(lower=0))
    return df_min


def add_relative_profiles(df_min, df_reg, window):
    """rvol, rvol_vol, rtrades -- each value against its typical level at this minute.

    Volume and volatility follow a stable U-shape across the session, so the raw level is
    nearly uninformative; what matters is the level relative to this minute of day. The
    profile uses prior sessions only, so index_day 0 is NaN by construction.
    """
    for col, out in [("volume", "rvol"), ("rv_vol", "rvol_vol"), ("bars", "rtrades")]:
        prof = _prior_day_profile(df_min, col)
        df_min[f"{out}_profile"] = prof
        df_min[out] = df_min[col] / prof.replace(0, np.nan)
    return df_min


def add_day_structure(df_min, df_reg, window):
    """pos_in_range, vwap_dev, or{N}_high/low/break, sess_high/low/vwap.

    Session-to-date extremes are available from the first minute, unlike price_cum_*
    which waits for `window`. The opening-range columns are NaN until the range has
    actually completed -- that is the causality constraint.
    """
    gd = df_min.groupby("index_day")
    df_min["sess_high"] = gd["high"].cummax()
    df_min["sess_low"] = gd["low"].cummin()
    rng = (df_min["sess_high"] - df_min["sess_low"]).replace(0, np.nan)
    df_min["pos_in_range"] = (df_min["close"] - df_min["sess_low"]) / rng

    pv = df_min["vwap"].fillna(df_min["close"]) * df_min["volume"]
    df_min["sess_vwap"] = (pv.groupby(df_min["index_day"]).cumsum()
                           / gd["volume"].cumsum().replace(0, np.nan))
    df_min["vwap_dev"] = (df_min["close"] / df_min["sess_vwap"] - 1) * 100

    for n in OR_WINDOWS:
        inside, after = df_min["index_minute"] < n, df_min["index_minute"] >= n
        orh = df_min["high"].where(inside).groupby(df_min["index_day"]).transform("max")
        orl = df_min["low"].where(inside).groupby(df_min["index_day"]).transform("min")
        df_min[f"or{n}_high"], df_min[f"or{n}_low"] = orh.where(after), orl.where(after)
        # >0 broken out above the range, <0 below, in units of range width; 0 inside it
        df_min[f"or{n}_break"] = (
            np.maximum(df_min["close"] - orh, 0) + np.minimum(df_min["close"] - orl, 0)
        ).where(after) / (orh - orl).replace(0, np.nan)
    return df_min


def add_regime(df_min, df_reg, window):
    """efficiency_ratio, variance_ratio -- is today trending or chopping?

    Efficiency ratio is net move / total path length: 1 = pure trend, 0 = pure chop.
    Variance ratio > 1 means trending (returns positively autocorrelated), < 1 means
    mean-reverting.
    """
    gd = df_min.groupby("index_day")
    tmp = df_min.assign(
        _abs=df_min["m_ret"].abs(),
        _sq=df_min["m_ret"] ** 2,
        _q=gd["close"].pct_change(VR_Q, fill_method=None) * 100,
    )
    tmp["_q_sq"] = tmp["_q"] ** 2

    net = (df_min["close"] - gd["close"].shift(window - 1)).abs()
    df_min["efficiency_ratio"] = net / _roll(tmp, ["_abs"], "sum", window)["_abs"]

    v1 = _roll(tmp, ["_sq"], "mean", window)["_sq"]
    vq = _roll(tmp, ["_q_sq"], "mean", window)["_q_sq"]
    df_min["variance_ratio"] = vq / (VR_Q * v1.replace(0, np.nan))
    return df_min


def add_flow(df_min, df_reg, window):
    """flow_imbalance, amihud, avg_trade_size -- order-flow proxies, still pure TA."""
    bar_rng = (df_reg["high"] - df_reg["low"]).replace(0, np.nan)
    aggs = dict(signed_vol=("_signed", "sum"), dollar_vol=("_dollar", "sum"))
    tmp = df_reg[["index_day", "index_minute"]].assign(
        _signed=(df_reg["close"] - df_reg["open"]) / bar_rng * df_reg["volume"],
        _dollar=df_reg["close"] * df_reg["volume"],
    )
    if "transactions" in df_reg.columns:
        tmp["_trades"] = df_reg["transactions"]
        aggs["trades"] = ("_trades", "sum")
    flow = tmp.groupby(["index_day", "index_minute"], as_index=False).agg(**aggs)
    df_min = df_min.merge(flow, on=["index_day", "index_minute"], how="left")

    fw = _roll(df_min, ["signed_vol", "volume", "dollar_vol", "m_ret"], "sum", window)
    df_min["flow_imbalance"] = fw["signed_vol"] / fw["volume"].replace(0, np.nan)
    # Amihud illiquidity: price move per $1M traded
    df_min["amihud"] = fw["m_ret"].abs() / fw["dollar_vol"].replace(0, np.nan) * 1e6
    if "trades" in df_min.columns:
        df_min["avg_trade_size"] = df_min["volume"] / df_min["trades"].replace(0, np.nan)
    return df_min


def add_prior_day(df_min, df_reg, window):
    """prev_d_* and overnight_gap_* -- known at the open, constant through the session."""
    daily = df_min.groupby("index_day").agg(
        d_open=("open", "first"), d_close=("close", "last"),
        d_high=("high", "max"), d_low=("low", "min"), d_volume=("volume", "sum"),
    ).reset_index()
    daily["d_range_pct"] = (daily["d_high"] - daily["d_low"]) / daily["d_close"] * 100
    daily["d_ret_pct"] = (daily["d_close"] / daily["d_open"] - 1) * 100
    daily["d_close_loc"] = ((daily["d_close"] - daily["d_low"])
                            / (daily["d_high"] - daily["d_low"]).replace(0, np.nan))

    prior = daily.copy()                      # shift a session so each day sees yesterday
    prior["index_day"] += 1
    prior = prior.rename(columns=lambda c: f"prev_{c}" if c != "index_day" else c)
    df_min = df_min.merge(prior, on="index_day", how="left")

    df_min["prev_atr_pct"] = df_min["index_day"].map(
        daily.set_index("index_day")["d_range_pct"].shift(1).expanding().mean())
    df_min["overnight_gap_pct"] = (
        df_min.groupby("index_day")["open"].transform("first") / df_min["prev_d_close"] - 1
    ) * 100
    df_min["overnight_gap_atr"] = (df_min["overnight_gap_pct"]
                                   / df_min["prev_atr_pct"].replace(0, np.nan))
    return df_min


# The single list that says what gets calculated. Drop a line to drop that group.
# Dependencies: relative_profiles needs rv_vol (realised_variance); flow needs m_ret
# (built with the minute frame); prior_day feeds prev_atr_pct to the execution harness.
PIPELINE = [
    add_window_moments,        # ret_{roll,cum}_{mean,std}, volume_*, price/volume high/low
    add_realised_variance,     # rv, bv, rv_vol, jump_var, jump_frac, bns_z
    add_range_volatility,      # park_vol, gk_vol, rs_vol, yz_vol
    add_relative_profiles,     # rvol, rvol_vol, rtrades
    add_day_structure,         # pos_in_range, vwap_dev, or{15,30,60}_*, sess_*
    add_regime,                # efficiency_ratio, variance_ratio
    add_flow,                  # flow_imbalance, amihud, avg_trade_size
    add_prior_day,             # prev_d_*, overnight_gap_*, prev_atr_pct
]


def build_features(df_raw, window=60, pipeline=None):
    """Raw second-bars -> (df_reg, df_min) with every indicator in `pipeline` applied."""
    df_reg = build_session_frame(df_raw)
    df_min = build_minute_frame(df_reg)
    for step in (PIPELINE if pipeline is None else pipeline):
        df_min = step(df_min, df_reg, window)
    return df_reg, df_min


def feature_names(df_min):
    """The indicator columns, in pipeline order, that exist on this frame."""
    wanted = [
        "ret_roll_mean", "ret_roll_std", "ret_cum_mean", "ret_cum_std",
        "price_roll_high", "price_roll_low", "price_cum_high", "price_cum_low",
        "rv", "bv", "rv_vol", "jump_var", "jump_frac", "bns_z",
        "park_vol", "gk_vol", "rs_vol", "yz_vol",
        "rvol", "rvol_vol", "rtrades",
        "pos_in_range", "vwap_dev", *[f"or{n}_break" for n in OR_WINDOWS],
        "efficiency_ratio", "variance_ratio",
        "flow_imbalance", "amihud", "avg_trade_size",
        "prev_d_range_pct", "prev_d_ret_pct", "prev_d_close_loc", "overnight_gap_atr",
    ]
    return [c for c in wanted if c in df_min.columns]


# ------------------------------------------------------- jump-diffusion (jumpdiff) -----
JUMP_COLS = ["ret_roll_xi_var", "ret_roll_lamb", "ret_cum_xi_var", "ret_cum_lamb"]


def return_path(df):
    """Cumulative % close return within a session, starting at 0.

    jd.moments() calls np.diff() on whatever it is handed, so we must pass a *process*
    whose increments are the returns. diff() of this is exactly the % close returns.

    We deliberately do NOT interleave o/h/l/c into a path. That gives 4x the points per
    window, but measured against a simulated jump-diffusion with known parameters it
    overstates the jump variance by ~30% -- the high and low are extremes over the bar
    rather than sequential observations, so threading them into a path manufactures large
    o->l->h moves that read as jumps. Close-only sits within 5% of the true tick path.
    Averaging the four (ohlc4) is worse still: it attenuates xi by ~70%, because a jump
    landing mid-bar gets split across two half-sized increments.
    """
    return df[RETURN_COL].fillna(0.0).cumsum().to_numpy()


def jump_terms(timeseries, bins=500):
    """(xi_var, lamb) for one window of the cumulative % return path.

    Careful with jumpdiff's naming: jd.jump_amplitude returns M6/(5*M4), which is the
    jump amplitude VARIANCE sigma_xi^2 (percent-squared), not the amplitude xi. The
    formulae in the two docstrings in jumpdiff/parameters.py are swapped; the code is
    right, the prose isn't. jd.jump_rate wants that same variance, so leaving
    xi_est=None (it recomputes internally) keeps the pair consistent.
    """
    import jumpdiff as jd

    if timeseries.size < 8 or np.ptp(timeseries) <= 0:
        return np.nan, np.nan          # degenerate window -> silverman bandwidth is 0
    try:
        _, moments = jd.moments(timeseries=timeseries, bins=np.array([bins]))
        with np.errstate(all="ignore"):
            return (jd.jump_amplitude(moments=moments)[0],
                    jd.jump_rate(moments=moments)[0])
    except (AssertionError, ValueError, ZeroDivisionError):
        return np.nan, np.nan


def compute_jump_terms(df_reg, df_min, window, *, bins=500, max_threads=8):
    """Running / cumulative jump amplitude and rate, one session per thread.

    Returns (df_min, stats). jd.moments() spends its time in numpy/scipy convolution,
    which drops the GIL, so sessions genuinely overlap. Paths are built up front on this
    thread so the pool touches only numpy arrays and needs no locking.
    """
    jobs = {}
    for day, day_sec in df_reg.groupby("index_day"):
        day_min = df_min[df_min["index_day"] == day]
        path = return_path(day_sec)
        offs = np.concatenate([[0], np.cumsum(day_min["bars"].to_numpy() * PER_BAR)])
        assert offs[-1] == path.size, f"index_day {day}: minute offsets must tile the path"
        jobs[day] = (path, offs, len(day_min))

    def one_day(day):
        path, offs, n = jobs[day]
        out = np.full((4, n), np.nan)
        t0 = time.perf_counter()
        for i in range(window - 1, n):
            hi = offs[i + 1]
            # rebase each running window to its own start, so the estimate does not
            # inherit whatever cumulative level the session happened to drift to
            w = path[offs[i - window + 1]:hi]
            out[0, i], out[1, i] = jump_terms(w - w[0] if w.size else w, bins)
            out[2, i], out[3, i] = jump_terms(path[:hi], bins)
        return day, out, time.perf_counter() - t0

    days = sorted(jobs)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        results = list(pool.map(one_day, days))
    wall = time.perf_counter() - t0

    for col in JUMP_COLS:
        df_min[col] = np.nan
    for day, out, _ in results:
        mask = df_min["index_day"] == day
        for col, values in zip(JUMP_COLS, out):
            df_min.loc[mask, col] = values

    # A negative variance is the estimator failing on too short a window; sqrt turns
    # exactly those into NaN rather than propagating sqrt(negative).
    with np.errstate(invalid="ignore"):
        df_min["ret_roll_xi"] = np.sqrt(df_min["ret_roll_xi_var"].where(lambda s: s >= 0))
        df_min["ret_cum_xi"] = np.sqrt(df_min["ret_cum_xi_var"].where(lambda s: s >= 0))

    stats = {
        "sessions": len(days),
        "windows": sum(2 * max(0, jobs[d][2] - window + 1) for d in days),
        "points": sum(jobs[d][0].size for d in days),
        "wall_s": wall,
        "thread_s": sum(r[2] for r in results),
        "threads_usable": min(len(days), max_threads),
        "max_threads": max_threads,
    }
    stats["speedup"] = stats["thread_s"] / wall if wall else float("nan")
    return df_min, stats


def negative_share(df_min, cols=JUMP_COLS):
    """Share of finite jump estimates that came out negative -- should be ~0 to be usable.

    This is the number to watch: sigma_xi^2 is a variance and lamb a rate, so negatives
    are the estimator failing, not a property of the data. A close-only window carries
    ~3,600 points and 4th/6th moments are sample-hungry. If it climbs, raise the window
    or subsample the returns to 5-10s -- not back to the candle path.
    """
    valid = df_min[cols].notna()
    return (df_min[cols][valid] < 0).sum() / valid.sum()


# ------------------------------------------------------------------------- oracle ------
def _best_round_trip(prices):
    """(entry_idx, exit_idx, pct) for the best long, entry <= exit. O(n)."""
    if len(prices) < 2:
        return 0, 0, 0.0
    run_min = np.minimum.accumulate(prices)
    gains = prices / run_min - 1.0
    j = int(np.argmax(gains))
    i = int(np.argmin(prices[: j + 1]))
    return i, j, float(gains[j] * 100)


def build_oracle(df_min, *, cost_bps=3.0, degrade=5):
    """THIS IS A LABEL, NOT A FEATURE.

    Every column is computed with full knowledge of the whole session. It measures the
    CEILING -- how much profit is on the table before any strategy exists -- and serves
    as a target. It must never be fed to a model as an input, which is why it returns its
    own frame rather than columns on df_min.
    """
    rows = []
    for day, g in df_min.groupby("index_day"):
        g = g.sort_values("index_minute")
        mins = g["index_minute"].to_numpy()
        close = g["close"].to_numpy(float)
        high, low = g["high"].to_numpy(float), g["low"].to_numpy(float)

        # executable oracle: enter and exit at a minute's CLOSE, a price that actually
        # traded at a knowable moment
        li, lj, lpct = _best_round_trip(close)
        si, sj, spct = _best_round_trip(-close + 2 * close.max())     # mirror for short

        # absolute ceiling: buy the session low, sell a later high. Not executable, but
        # it bounds everything above.
        ceil_pct = float(np.max(high / np.minimum.accumulate(low) - 1.0) * 100)

        di, dj = li + degrade, lj - degrade      # give up `degrade` minutes at each end
        deg_pct = float(close[dj] / close[di] - 1.0) * 100 if dj > di else 0.0

        rows.append({
            "index_day": day,
            "date": pd.to_datetime(g["timestamp"].iloc[0], unit="ms", utc=True)
                      .tz_convert("America/New_York").date(),
            "long_entry_min": mins[li], "long_exit_min": mins[lj],
            "long_entry_price": close[li], "long_exit_price": close[lj], "long_pct": lpct,
            "short_entry_min": mins[si], "short_exit_min": mins[sj],
            "short_entry_price": close[si], "short_exit_price": close[sj], "short_pct": spct,
            "ceiling_pct": ceil_pct, "degraded_pct": deg_pct,
            "hold_minutes": int(mins[lj] - mins[li]),
            "day_range_pct": float((high.max() - low.min()) / close[-1] * 100),
        })

    out = pd.DataFrame(rows)
    out["best_pct"] = out[["long_pct", "short_pct"]].max(axis=1)
    out["best_side"] = np.where(out["long_pct"] >= out["short_pct"], "long", "short")
    for c in ("long_pct", "short_pct", "best_pct", "degraded_pct"):
        out[f"{c}_net"] = out[c] - cost_bps / 100.0
    return out


# ---------------------------------------------------------------------- execution ------
def execution_benchmarks(df_min):
    """Per-day TWAP / VWAP / arrival / close. Day-level LABELS: they score, never feed."""
    b = df_min.assign(_pv=df_min["vwap"].fillna(df_min["close"]) * df_min["volume"])
    bench = b.groupby("index_day").agg(
        twap=("close", "mean"),            # the "day mean" -- what naive execution gets
        arrival=("close", "first"),        # price when the decision is made
        close_px=("close", "last"),
        day_low=("low", "min"), day_high=("high", "max"),
        _pv=("_pv", "sum"), _v=("volume", "sum"),
    )
    bench["vwap_day"] = bench["_pv"] / bench["_v"].replace(0, np.nan)
    bench["range_pct"] = (bench["day_high"] - bench["day_low"]) / bench["twap"] * 100
    return bench.drop(columns=["_pv", "_v"])


class ExecutionHarness:
    """Score limit-order placement policies against day-level benchmarks.

    The trap this is built around: a limit order that fills at a great price only tells
    you what happened on the days it filled, and the days it did NOT fill are exactly the
    days the market ran away. So every policy carries a forced market fallback at a
    deadline, and every headline number is UNCONDITIONAL -- forced fills included. The
    conditional number is reported alongside purely to show how large the lie would be.
    """

    def __init__(self, df_reg, df_min, *, deadline=380):
        self.deadline = deadline
        self.df_min = df_min
        self.bench = execution_benchmarks(df_min)
        # scale for the initial guess: mean of PRIOR sessions' ranges (NaN on day 0)
        self.scale = df_min.groupby("index_day")["prev_atr_pct"].first() / 100.0
        self.days = [d for d in self.bench.index
                     if np.isfinite(self.scale.get(d, np.nan))]
        # repricing happens each minute; fills are checked each second
        self.sec = {d: (g["index_minute"].to_numpy(), g["low"].to_numpy(float),
                        g["high"].to_numpy(float), g["close"].to_numpy(float))
                    for d, g in df_reg.groupby("index_day")}

    def simulate(self, day, side, k0, gamma_):
        """One session, one order -> (price, minute, filled_passively).

        Offset starts at k0 x the prior-day range and decays toward the market as the
        deadline approaches -- "guess from previous ranges, then update it". A resting
        buy fills at its own limit once any second-bar trades at or below it.
        """
        mins, low, high, close = self.sec[day]
        ref, scale = self.bench.at[day, "arrival"], self.scale[day]
        frac = np.clip(1.0 - mins / self.deadline, 0.0, 1.0) ** gamma_
        off = k0 * scale * frac
        lim = ref * (1.0 - off) if side == "buy" else ref * (1.0 + off)

        ok = (mins < self.deadline) & ((low <= lim) if side == "buy" else (high >= lim))
        if ok.any():
            i = int(np.argmax(ok))
            return float(lim[i]), int(mins[i]), True
        j = min(int(np.searchsorted(mins, self.deadline)), len(close) - 1)
        return float(close[j]), int(mins[j]), False          # forced market

    def score(self, price, day, side, benchmark="twap"):
        """Basis points better than the benchmark. Positive is good for both sides."""
        ref = self.bench.at[day, benchmark]
        return ((ref - price) if side == "buy" else (price - ref)) / ref * 1e4

    def random_null(self, side, draws=400, seed=0):
        """Execute at a uniformly random minute -- the bar any policy must clear."""
        rng = np.random.default_rng(seed)
        per_day = []
        for d in self.days:
            c = self.df_min.loc[self.df_min["index_day"] == d, "close"].to_numpy(float)
            pick = rng.integers(0, len(c), draws)
            per_day.append([self.score(c[i], d, side) for i in pick])
        a = np.array(per_day)
        return a.mean(), a.mean(axis=1).std(ddof=1)

    def sweep(self, k0s=(0.0, 0.10, 0.25, 0.50, 1.00), gammas=(0.5, 1.0, 2.0),
              sides=("buy", "sell")):
        rows = []
        for side in sides:
            for k0 in k0s:
                for gamma_ in gammas:
                    res = [self.simulate(d, side, k0, gamma_) for d in self.days]
                    px = np.array([r[0] for r in res])
                    filled = np.array([r[2] for r in res])
                    uncond = np.array([self.score(p, d, side, "twap")
                                       for p, d in zip(px, self.days)])
                    uncond_v = np.array([self.score(p, d, side, "vwap_day")
                                         for p, d in zip(px, self.days)])
                    cond = uncond[filled] if filled.any() else np.array([np.nan])
                    rows.append({
                        "k0": k0, "gamma": gamma_, "side": side,
                        "fill_rate": filled.mean(),
                        "vs_twap_bps": uncond.mean(),
                        "vs_vwap_bps": uncond_v.mean(),
                        "IF_FILLED_ONLY_bps": np.nanmean(cond),   # the misleading number
                        "median_fill_min": np.median([r[1] for r in res]),
                    })
        return pd.DataFrame(rows)
