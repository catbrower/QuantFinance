"""Loaders for the CFTC files in data/.

Naming follows institution_category_subcategory:

    cftc_cot_disaggregated.csv    Disaggregated, Futures-and-Options Combined
    cftc_cot_financial.csv        Traders in Financial Futures (TFF), Fut-and-Opts Combined
    cftc_cot_index_trader.csv     Commodity Index Trader Supplement
    cftc_positions_large_trader.csv   weekly large-trader buys/sells by category
    cftc_positions_accounts.csv       weekly trading-account net position changes

COT vs POSITIONS -- they answer different questions
---------------------------------------------------
The cftc_cot_* files carry POSITION LEVELS and, crucially, Open_Interest_All. That is
what Moskowitz, Ooi & Pedersen (2012) Section 2.2 needs:

    Net speculator position = (spec long - spec short) / open interest

The cftc_positions_* files carry FLOWS -- buys, sells, net position changes -- with no
open-interest column, so that ratio cannot be formed from them and a level series cannot
be recovered (integrating a change series leaves an unknown constant).

WHERE THE COT FILES DIFFER FROM THE PAPER
-----------------------------------------
The paper uses the LEGACY report, which splits traders into "commercial" (hedgers) and
"non-commercial" (speculators). None of these three files carry that split:

    disaggregated : Prod_Merc, Swap, M_Money, Other_Rept
    financial     : Dealer, Asset_Mgr, Lev_Money, Other_Rept
    index_trader  : Comm_*_NoCIT / NComm_*_NoCIT  -- legacy-shaped, but with index
                    traders REMOVED, and only ~19 agricultural contracts to 2022

So a crosswalk is needed, and it is a judgement call rather than an identity. The
defaults below follow common practice; SWAP DEALERS ARE THE CONTENTIOUS ONE (they hedge
index exposure, so they sit on the commercial side here even though the flow originates
with speculative money). Override `spec`/`hedge` to test sensitivity.

For the paper's exact series, the legacy futures-only archive is a separate download:
    https://www.cftc.gov/files/dea/history/deacot<YEAR>.zip     (1986 onward)
    https://www.cftc.gov/files/dea/history/deacot1986_2016.zip  (bundle)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")

COT_FILES = {
    "disaggregated": "cftc_cot_disaggregated.csv",
    "financial": "cftc_cot_financial.csv",
    "index_trader": "cftc_cot_index_trader.csv",
}

# Crosswalk from each report's categories onto the paper's speculator/hedger split.
CROSSWALK = {
    "disaggregated": {"spec": ["M_Money", "Other_Rept"], "hedge": ["Prod_Merc", "Swap"]},
    "financial": {"spec": ["Lev_Money", "Asset_Mgr", "Other_Rept"], "hedge": ["Dealer"]},
    "index_trader": {"spec": ["NComm"], "hedge": ["Comm"]},
}

NAME_COL = "Market_and_Exchange_Names"
OI_COL = "Open_Interest_All"


def _num(s):
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def _position_col(columns, prefix, side):
    """Find e.g. Swap__Positions_Short_All -- the file has stray double underscores."""
    want = f"{prefix}_positions_{side}_all".lower().replace("__", "_")
    for c in columns:
        if c.lower().replace("__", "_") == want:
            return c
    # index_trader uses a _NoCIT suffix
    want_nocit = f"{prefix}_positions_{side}_all_nocit".lower().replace("__", "_")
    for c in columns:
        norm = c.lower().replace("__", "_").replace("postions", "positions")
        if norm == want_nocit:
            return c
    raise KeyError(f"no column for {prefix} {side} in {len(columns)} columns")


def load_cot(report="financial", contracts=None, data_dir=DATA_DIR, extra_columns=()):
    """Load one COT report, keeping only the columns the position measures need.

    `contracts` is a substring or list of substrings matched case-insensitively against
    Market_and_Exchange_Names -- worth using on the disaggregated file, which is 162 MB.
    """
    if report not in COT_FILES:
        raise ValueError(f"report must be one of {sorted(COT_FILES)}")
    path = Path(data_dir) / COT_FILES[report]
    header = pd.read_csv(path, nrows=0).columns.tolist()

    cats = CROSSWALK[report]["spec"] + CROSSWALK[report]["hedge"]
    pos_cols = [_position_col(header, c, side) for c in cats for side in ("long", "short")]
    keep = [NAME_COL, "report_date", "year", OI_COL, *pos_cols, *extra_columns]
    keep = [c for c in dict.fromkeys(keep) if c in header]

    df = pd.read_csv(path, usecols=keep, low_memory=False)
    if contracts:
        pats = [contracts] if isinstance(contracts, str) else list(contracts)
        mask = pd.Series(False, index=df.index)
        for p in pats:
            mask |= df[NAME_COL].str.contains(p, case=False, na=False, regex=False)
        df = df[mask]

    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    # Identifier columns stay strings. CFTC contract codes are alphanumeric
    # ("13874A" for the E-mini S&P 500), so coercing them to numbers nulls them.
    id_like = {NAME_COL, "report_date"} | {
        c for c in df.columns if "code" in c.lower() or "name" in c.lower()}
    for c in df.columns:
        if c not in id_like:
            df[c] = _num(df[c])
        elif c != "report_date":
            df[c] = df[c].astype(str).str.strip()
    return df.sort_values([NAME_COL, "report_date"]).reset_index(drop=True)


def net_speculator_position(df, report="financial", spec=None, hedge=None):
    """MOP Eq. in Section 2.2, using this report's crosswalk.

    Adds spec_long/spec_short/hedge_long/hedge_short and:
        net_spec_position  = (spec long - spec short) / open interest
        net_hedge_position = (hedge long - hedge short) / open interest

    The two do not sum to zero here the way the paper's do, because the non-reportable
    tail is excluded and the categories are not an exhaustive two-way split.
    """
    spec = spec or CROSSWALK[report]["spec"]
    hedge = hedge or CROSSWALK[report]["hedge"]
    out = df.copy()
    for label, cats in (("spec", spec), ("hedge", hedge)):
        for side in ("long", "short"):
            cols = [_position_col(out.columns, c, side) for c in cats]
            out[f"{label}_{side}"] = out[cols].sum(axis=1)
    oi = out[OI_COL].replace(0, pd.NA)
    out["net_spec_position"] = (out["spec_long"] - out["spec_short"]) / oi
    out["net_hedge_position"] = (out["hedge_long"] - out["hedge_short"]) / oi
    return out


def to_monthly(df, value_cols=("net_spec_position",), date_col="report_date"):
    """Weekly COT -> month-end observations, so it joins to monthly returns."""
    d = df.dropna(subset=[date_col]).copy()
    d["month"] = d[date_col].dt.to_period("M")
    agg = d.groupby("month")[list(value_cols)].last()
    agg["obs_in_month"] = d.groupby("month").size()
    return agg


# ------------------------------------------------- the flow files (position CHANGES) --
BLOCK_WIDTH = 12
HEADER_ROWS = 3
SPECULATOR_LIKE = {"Managed Money", "Leveraged Funds"}
HEDGER_LIKE = {"Producer/Merchant/Processor/User", "Dealer intermediary"}


def _flow_num(x):
    if x is None:
        return float("nan")
    s = str(x).strip().strip('"').replace(",", "")
    if s in {"", "-", "."}:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_large_trader(path=None, data_dir=DATA_DIR):
    """-> long frame: week_start, week_end, product, category, buys, sells, net.

    `net` is buys - sells: a weekly position CHANGE, not a level.
    """
    import csv

    path = Path(path or Path(data_dir) / "cftc_positions_large_trader.csv")
    rows = list(csv.reader(path.open()))
    r0, r1, body = rows[0], rows[1], rows[HEADER_ROWS:]

    out = []
    for start in range(0, len(r0), BLOCK_WIDTH):
        product = r0[start].strip()
        if not product:
            continue
        cats = [(off, r1[start + off].strip()) for off in (2, 4, 6, 8)]
        for row in body:
            if start + BLOCK_WIDTH > len(row) or not row[start].strip():
                continue
            base = {"week_start": row[start].strip(), "week_end": row[start + 1].strip(),
                    "product": product,
                    "avg_net_position_change": _flow_num(row[start + 10]),
                    "avg_trade_volume": _flow_num(row[start + 11])}
            for off, cat in cats:
                buys, sells = _flow_num(row[start + off]), _flow_num(row[start + off + 1])
                out.append({**base, "category": cat, "buys": buys, "sells": sells,
                            "net": buys - sells})

    df = pd.DataFrame(out)
    for c in ("week_start", "week_end"):
        df[c] = pd.to_datetime(df[c], format="%m/%d/%Y", errors="coerce")
    df["role"] = df["category"].map(
        lambda c: "speculator-like" if c in SPECULATOR_LIKE
        else "hedger-like" if c in HEDGER_LIKE else "other")
    return df.sort_values(["product", "week_start", "category"]).reset_index(drop=True)


def load_trading_accounts(path=None, data_dir=DATA_DIR):
    """-> long frame: week_start, week_end, product, net_position_change, trade_volume."""
    path = Path(path or Path(data_dir) / "cftc_positions_accounts.csv")
    wide = pd.read_csv(path)
    net_pfx, vol_pfx = "Trading Account Net Position Change ", "Average Trade Volume "
    out = []
    for c in wide.columns:
        if not c.startswith(net_pfx):
            continue
        p = c[len(net_pfx):]
        if vol_pfx + p not in wide.columns:
            continue
        out.append(pd.DataFrame({
            "week_start": wide["Week Start"], "week_end": wide["Week End"], "product": p,
            "net_position_change": wide[c].map(_flow_num),
            "trade_volume": wide[vol_pfx + p].map(_flow_num)}))
    df = pd.concat(out, ignore_index=True)
    for c in ("week_start", "week_end"):
        df[c] = pd.to_datetime(df[c], format="%m/%d/%Y", errors="coerce")
    return df.sort_values(["product", "week_start"]).reset_index(drop=True)


def coverage(df, by="product", date_col=None):
    date_col = date_col or ("report_date" if "report_date" in df.columns else "week_start")
    by = by if by in df.columns else NAME_COL
    g = df.groupby(by)
    return pd.DataFrame({"rows": g.size(), "first": g[date_col].min(),
                         "last": g[date_col].max()}).sort_values("rows", ascending=False)
