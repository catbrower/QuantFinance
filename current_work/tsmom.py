"""Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum", JFE 104(2) 228-250.

A from-the-equations reimplementation. Equation numbers refer to the paper (a copy
sits next to this file).

    Eq. (1)  ex ante annualised variance
             sigma_t^2 = 261 * sum_{i>=0} (1-d) d^i (r_{t-1-i} - rbar_t)^2
             with d chosen so the centre of mass d/(1-d) = 60 days, and rbar_t the
             exponentially weighted mean computed the same way.

    Eq. (2)  r^s_t / sigma^s_{t-1} = a + b_h * r^s_{t-h} / sigma^s_{t-h-1} + e
    Eq. (3)  r^s_t / sigma^s_{t-1} = a + b_h * sign(r^s_{t-h}) + e
             pooled panel, t-stats clustered by time (month), h = 1..60

    Eq. (5)  r^TSMOM,s_{t,t+1} = sign(r^s_{t-12,t}) * (40% / sigma^s_t) * r^s_{t,t+1}
             the diversified version averages equally across available instruments.

WHAT DIFFERS HERE, and it matters when reading the numbers:
  * The paper uses 58 futures/forwards, whose returns are already EXCESS returns
    (futures are unfunded). SPY is a funded cash instrument, so an excess return needs
    the risk-free rate subtracted -- pass `rf_annual`. Left at 0 the "excess" return is
    really a total return, which biases the 12-month sign long in flat markets.
  * One instrument is not a diversified portfolio. The paper's headline Sharpe comes
    from averaging ~58 weakly-correlated bets; a single equity index cannot reproduce
    it and should not be expected to.
  * Clustering by time with a single instrument leaves one observation per cluster, so
    the clustered covariance collapses to White/HC. It only starts to bite once more
    instruments are added -- which is why it is implemented generally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALISATION = 261        # trading days per year, as in Eq. (1)
COM_DAYS = 60              # centre of mass of the EW weights, d/(1-d) = 60
VOL_TARGET = 0.40          # the 40% annualised target in Eq. (5)
MONTHS_PER_YEAR = 12


# ------------------------------------------------------------------ ex ante volatility
def ex_ante_vol(daily_returns, com=COM_DAYS, annualisation=ANNUALISATION):
    """Eq. (1): exponentially weighted ex ante ANNUALISED volatility, from daily returns.

    Implemented exactly rather than approximately. Because the weights sum to one,

        sum w_i (r_i - rbar)^2  ==  E_w[r^2] - (E_w[r])^2

    so the whole thing is two EWMAs. Both are taken over returns lagged one day, which
    is what makes sigma_t depend only on data through t-1 (the paper's no-look-ahead
    condition).

    pandas' `com` is exactly the paper's centre of mass: alpha = 1/(1+com) = 1/61, so
    d = com/(1+com) = 60/61.
    """
    r = pd.Series(daily_returns).astype(float)
    lagged = r.shift(1)
    m1 = lagged.ewm(com=com, adjust=False).mean()
    m2 = (lagged**2).ewm(com=com, adjust=False).mean()
    var = annualisation * (m2 - m1**2)
    return np.sqrt(var.clip(lower=0))


# ------------------------------------------------------------------- monthly assembly
def to_monthly(prices, daily_returns=None, rf_annual=0.0, com=COM_DAYS):
    """Daily prices -> monthly frame with excess returns and month-end ex ante vol.

    Columns:
        ret        -- monthly EXCESS return (simple), net of rf_annual/12
        sigma      -- ex ante annualised vol as of the LAST DAY of that month
        sigma_lag  -- previous month-end sigma; this is the sigma^s_{t-1} that Eq. (2)
                      divides by and the one Eq. (5) sizes with
    """
    px = pd.Series(prices).astype(float).dropna()
    if not isinstance(px.index, pd.DatetimeIndex):
        raise TypeError("prices must be indexed by date")
    r_d = px.pct_change() if daily_returns is None else pd.Series(daily_returns).astype(float)

    sigma_d = ex_ante_vol(r_d, com=com)

    grp = px.index.to_period("M")
    monthly = pd.DataFrame({
        "price": px.groupby(grp).last(),
        "sigma": sigma_d.groupby(grp).last(),
    })
    monthly["ret"] = monthly["price"].pct_change() - rf_annual / MONTHS_PER_YEAR
    monthly["sigma_lag"] = monthly["sigma"].shift(1)
    return monthly.dropna(subset=["ret"])


def trailing_return(ret, k):
    """Cumulative excess return over the past k months, ending at t inclusive."""
    return (1.0 + ret).rolling(k).apply(np.prod, raw=True) - 1.0


# --------------------------------------------------------- pooled regressions, Eq. 2/3
def _ols_clustered(y, X, clusters):
    """OLS with standard errors clustered on `clusters`. Returns (beta, se, t, n, G).

    V = (X'X)^-1 [sum_g X_g' e_g e_g' X_g] (X'X)^-1, with the usual finite-sample
    correction G/(G-1) * (N-1)/(N-K).
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta

    meat = np.zeros((k, k))
    codes = pd.factorize(pd.Series(clusters))[0]
    for g in np.unique(codes):
        m = codes == g
        xg_eg = X[m].T @ resid[m]
        meat += np.outer(xg_eg, xg_eg)

    G = len(np.unique(codes))
    corr = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = corr * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.diag(V))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    return beta, se, t, n, G


def lag_regressions(panel, max_lag=60, spec="return"):
    """Eq. (2) (spec="return") or Eq. (3) (spec="sign") for h = 1..max_lag.

    `panel` is a long frame with columns: instrument, month, ret, sigma_lag. Returns a
    frame indexed by lag with beta, se, t_stat, n_obs.

    LHS is always r_t / sigma_{t-1}. RHS is r_{t-h} / sigma_{t-h-1} for Eq. (2), putting
    both sides on the same scale, or sign(r_{t-h}) for Eq. (3), which needs no scaling
    because it is already +/-1.
    """
    if spec not in {"return", "sign"}:
        raise ValueError("spec must be 'return' or 'sign'")
    df = panel.sort_values(["instrument", "month"]).copy()
    df["y"] = df["ret"] / df["sigma_lag"]

    rows = []
    for h in range(1, max_lag + 1):
        g = df.groupby("instrument", sort=False)
        if spec == "return":
            df["x"] = g["ret"].shift(h) / g["sigma_lag"].shift(h)
        else:
            df["x"] = np.sign(g["ret"].shift(h))
        d = df.dropna(subset=["y", "x"])
        d = d[np.isfinite(d["y"]) & np.isfinite(d["x"])]
        if len(d) < 3 or d["x"].nunique() < 2:
            rows.append({"lag": h, "beta": np.nan, "se": np.nan,
                         "t_stat": np.nan, "n_obs": len(d)})
            continue
        X = np.column_stack([np.ones(len(d)), d["x"].to_numpy()])
        beta, se, t, n, _ = _ols_clustered(d["y"], X, d["month"])
        rows.append({"lag": h, "beta": beta[1], "se": se[1],
                     "t_stat": t[1], "n_obs": n})
    return pd.DataFrame(rows).set_index("lag")


# ------------------------------------------------------------------- strategy, Eq. (5)
def tsmom_returns(monthly, lookback=12, vol_target=VOL_TARGET):
    """Eq. (5) for one instrument: sign(r_{t-k,t}) * (vol_target / sigma_t) * r_{t,t+1}.

    Everything is shifted so month t's strategy return uses only information available
    at the end of month t-1: the signal is the trailing return through t-1, and the
    sizing uses sigma_{t-1}.
    """
    out = pd.DataFrame(index=monthly.index)
    out["ret"] = monthly["ret"]
    out["signal"] = np.sign(trailing_return(monthly["ret"], lookback)).shift(1)
    out["scale"] = (vol_target / monthly["sigma"]).shift(1)
    out["tsmom"] = out["signal"] * out["scale"] * out["ret"]
    out["long_only"] = out["scale"] * out["ret"]     # the "replace sign with 1" benchmark
    return out


def kh_grid(monthly, lookbacks=(1, 3, 6, 9, 12, 24, 36, 48),
            holdings=(1, 3, 6, 9, 12, 24, 36, 48), vol_target=VOL_TARGET):
    """Sharpe ratios over the (look-back k, holding h) grid behind the paper's Table 2.

    For h > 1 the paper averages the h overlapping portfolios formed in the last h
    months, so one monthly return series comes out regardless of h.
    """
    res = pd.DataFrame(index=pd.Index(lookbacks, name="lookback"),
                       columns=pd.Index(holdings, name="holding"), dtype=float)
    for k in lookbacks:
        signal = np.sign(trailing_return(monthly["ret"], k)).shift(1)
        position = signal * (vol_target / monthly["sigma"]).shift(1)
        for h in holdings:
            # At month t you hold h sub-portfolios, formed at t-1 ... t-h. The month's
            # return is the AVERAGE POSITION across those formation dates applied to
            # this month's return. Averaging past *returns* instead would smooth the
            # series and inflate the Sharpe ratio mechanically.
            avg_pos = pd.concat([position.shift(i) for i in range(h)], axis=1).mean(axis=1)
            res.loc[k, h] = sharpe(avg_pos * monthly["ret"])
    return res


def sharpe(returns, periods_per_year=MONTHS_PER_YEAR):
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)


def summarise(returns, name="strategy", periods_per_year=MONTHS_PER_YEAR):
    r = pd.Series(returns).dropna()
    if r.empty:
        return {"name": name, "months": 0}
    cum = float((1 + r).prod())
    yrs = len(r) / periods_per_year
    return {
        "name": name,
        "months": len(r),
        "mean_ann": r.mean() * periods_per_year,
        "vol_ann": r.std(ddof=1) * np.sqrt(periods_per_year),
        "sharpe": sharpe(r, periods_per_year),
        "skew": r.skew(),
        "max_drawdown": float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()),
        "cagr": cum ** (1 / yrs) - 1 if yrs > 0 and cum > 0 else np.nan,
    }


def alpha_vs_long(strategy, benchmark):
    """Regress TSMOM on always-long, as the paper does: alpha, its t-stat, and beta."""
    d = pd.concat([pd.Series(strategy, name="y"),
                   pd.Series(benchmark, name="x")], axis=1).dropna()
    if len(d) < 3:
        return {"alpha_ann": np.nan, "alpha_t": np.nan, "beta": np.nan, "n": len(d)}
    X = np.column_stack([np.ones(len(d)), d["x"].to_numpy()])
    beta, se, t, n, _ = _ols_clustered(d["y"], X, d.index)
    return {"alpha_ann": beta[0] * MONTHS_PER_YEAR, "alpha_t": t[0],
            "beta": beta[1], "n": n}


# ------------------------------------------------------------------------- reference --
REFERENCE_CSV = "reference_mop2012_PLACEHOLDER.csv"


def load_reference(path=REFERENCE_CSV):
    """Load the reference table used for comparison.

    The shipped file is a PLACEHOLDER with invented numbers so the comparison plumbing
    can be exercised. It is shaped like the paper's Table 2 (t-stats of alphas over the
    look-back x holding grid). Replace the file with the real digitised table and
    nothing else has to change; `is_placeholder` below keys off the `source` column.
    """
    ref = pd.read_csv(path, comment="#")
    ref.attrs["is_placeholder"] = bool(
        ref["source"].astype(str).str.contains("PLACEHOLDER", case=False).any())
    return ref


def compare_to_reference(ours, reference, value_name="sharpe"):
    """Join our (lookback, holding) results onto the reference grid and diff them."""
    long = (ours.stack().rename(value_name).reset_index()
            .rename(columns={"level_0": "lookback", "level_1": "holding"}))
    long["lookback"] = long["lookback"].astype(int)
    long["holding"] = long["holding"].astype(int)
    ref = reference.copy()
    ref["lookback"] = ref["lookback"].astype(int)
    ref["holding"] = ref["holding"].astype(int)
    merged = long.merge(ref, on=["lookback", "holding"], how="left",
                        suffixes=("_ours", "_ref"))
    merged["diff"] = merged[value_name] - merged["reference_value"]
    return merged
