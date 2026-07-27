# Research plan — finding alpha with technical analysis

**Goal:** find out whether *any* alpha exists using technical analysis alone.

**Two tracks, one shared foundation:**

| Track | Horizon | Instruments | Status |
|---|---|---|---|
| **A — Swing** | 3 days to ~1 month | all securities, FX, futures/ETFs, crypto | **current focus** |
| **B — Intraday** | 0–1 entries/exits per day | single equities | parked; pipeline built |

**Design principle — this layer is algorithm-independent.** Nothing here commits to a strategy. What
this notebook produces is a *feature and evidence layer*: causal, per-period quantities plus the
measurements that say how much signal is even present. Any strategy invented later consumes these
findings rather than recomputing them. Concretely that means every column must be computable from data
strictly *before* the period it is stamped on, and no feature may be normalised using statistics from
the future (including end-of-day or full-sample statistics).

---
---

# Part 1 — Shared foundations (true at any horizon)

## Is there anything to find? What the evidence says

### Second moments are forecastable; first moments barely

The strongest replicated finding in market data is that **volatility is highly predictable** —
realised-volatility forecasts routinely reach R² of 30–60% out of sample (the HAR family is the
standard workhorse). Volume and range share the property, and it does not decay.

**Directional** predictability is a different animal: where it exists it is R² well under 1% per
period. This matters for us specifically, because almost everything computed so far — the pooled std,
the ranges, ξ and λ — is a *volatility* feature set. Expect it to forecast tomorrow's range well and
tomorrow's sign barely. Still economically useful (sizing, stops, regime gating, options), just not
directly the alpha we are hunting.

### Documented effects worth taking as priors

- **Time-series momentum** — Moskowitz, Ooi & Pedersen (2012), across every asset class.
- **Cross-sectional momentum** — Jegadeesh & Titman (1993).
- **Short-horizon reversal** — Lehmann (1990), Jegadeesh (1990).
- **Value and momentum everywhere** — Asness, Moskowitz & Pedersen (2013): both effects appear in
  equities, bonds, FX, and commodities, and they are negatively correlated with each other.
- **Market intraday momentum** — Gao, Han, Li & Zhou (JFE 2018): first half-hour predicts last
  half-hour on SPY. *(Track B)*
- **Overnight vs intraday returns differ** — Lou, Polk & Skouras, *A tug of war* (JFE 2019): momentum
  and reversal flip sign between the two segments, and a large share of equity return accrues overnight.

So the prior is not "there is nothing here". It is **"there are real effects, they are small, and most
have decayed since publication"** — McLean & Pontiff (2016) measured roughly 58% decay post-publication.

---

## The power arithmetic, and why breadth is the whole game

The standard error of an estimated annual Sharpe is roughly `1 / sqrt(years)` — **regardless of holding
period**, for a strategy that is always invested:

| Sample | SE(Sharpe) |
|---|---|
| 1 year | ~1.00 |
| 4 years | ~0.50 |
| 9 years | ~0.33 |
| 16 years | ~0.25 |

To distinguish a true Sharpe of 0.5 from zero at two standard errors on **a single instrument** takes
~16 years. No feature set fixes that; the experiment is underpowered by construction.

### Breadth is the fix, and it is quantifiable

Run the same rule on N instruments with average pairwise correlation ρ between the per-instrument P&L
streams; effective independent bets are about `N / (1 + (N−1)ρ)`. At ρ = 0.1 and N = 100 that is ~9
effective bets, so portfolio Sharpe scales by ~3×. A per-instrument Sharpe of 0.5 becomes ~1.5 at
portfolio level — 3σ detectable in **2–4 years** instead of 16.

The catch: ρ is dominated by the market factor. If every instrument's signal is really "long when the
market rises", ρ ≈ 0.7 and breadth buys almost nothing. Practical implication:

> **Evaluate signals on market-residual returns, not raw returns.** That is what makes breadth pay.

**Note the structural advantage of Track A:** swing horizons let you hold many positions *concurrently*,
so breadth is native. Track B's one-trade-per-day constraint gave breadth of one.

---

## Validation discipline — this *is* the experiment

Given the question is "is there any alpha in TA at all", the honest answer depends almost entirely on
this section.

- **Walk-forward with purge and embargo.** Overlapping multi-day holds make returns autocorrelated —
  purge a full holding period between train and test, or the leakage is severe. Consider combinatorial
  purged CV to get more out of few samples.
- **Costs inside the objective**, never subtracted afterwards. Never fill at the signal bar's close —
  use the next bar's open, or an explicit delay. Fills at the decision price are the single largest
  source of fake alpha.
- **Correct for search.** We will try hundreds of variants. White's Reality Check, Hansen's SPA, and
  the Deflated Sharpe Ratio exist precisely to price that in. A Sharpe of 1.5 after 500 trials is not
  the same as a Sharpe of 1.5 from one hypothesis.
- **Robustness across instruments and eras.** If it works on AAPL 2023 but not MSFT 2022, it is noise.
  The cheapest and most brutal test available.
- **A final holdout touched exactly once.**

Objective function: not raw return. Use Sharpe/Sortino on periodic P&L, or a robust variant (median
period P&L, or mean minus λ × downside deviation).

---

## Choosing a method: ML vs numeric optimisation vs genetic algorithms

Run them in this order — they answer different questions.

**1. Coarse numeric optimisation of a simple parametric rule.** Three to six parameters: entry
threshold, stop, target, time-stop. Grid or Bayesian. Beyond the baseline number, the real output is
the **shape of the objective surface** — a broad ridge means a real effect, an isolated spike means
overfitting. That diagnostic is worth more than the optimum itself.

**2. Meta-labelling with a small gradient-boosted tree** on top of the best simple rule. Shallow trees,
heavy regularisation, tens of features not hundreds. This is where ML earns its keep.

**3. Genetic algorithms last, and warily.** A GA searching over rule *structure* is an overfitting
engine — a machine for finding the luckiest combination in a vast space. If used: fitness must be
cross-validated rather than in-sample, complexity must be penalised, and the winner must survive a
reality check against the full search. Its legitimate role is hypothesis *generation* (which feature
families matter), not producing a deployable rule.

**Skip deep learning and RL entirely** at these sample sizes.

---

## Compute budget

Measured on this notebook, so the scaling decisions are grounded:

| Step | Cost |
|---|---|
| Pooled stats, ranges, all vectorised minute features | **0.19s** for 5 sessions |
| Kramers-Moyal jump terms (`jumpdiff`) | **12–70 ms per window**, 662 windows per session |

The KM estimator is the *only* expensive thing here, by three to four orders of magnitude. Bipower
variation delivers the same continuous-vs-jump decomposition essentially free and with better
statistical properties. **Swap BV in as the primary jump measure and keep `jumpdiff` as an occasional
cross-check.**

**For Track A the whole question is moot:** at 3-day-to-1-month horizons the signal lives at *daily*
frequency. Daily bars for thousands of instruments over 20+ years is a few hundred MB and seconds to
process. Intraday data still buys better volatility estimates and better execution, but nothing in the
signal requires it.

---
---

# Part 2 — Track A: swing horizon (3 days – 1 month) — CURRENT FOCUS

## What changes at this horizon

1. **Daily bars suffice.** The second-level pipeline becomes an optional refinement, not a prerequisite.
2. **Breadth is native** — many concurrent positions rather than one trade a day.
3. **Multi-asset unlocks return drivers that do not exist in single-stock equity**, above all *carry*.
4. **Costs become a rounding error.** A weeks-long hold on a liquid instrument is far more forgiving
   than an intraday round trip — this widens the set of viable edges considerably.

## ⚠ The sign-flip trap

There is a **reversal/momentum sign flip around the 1-month boundary**. Roughly 1–4 weeks is the
*reversal* zone; 3–12 months is the *momentum* zone. Our target window straddles the transition.

This is why standard practice uses **"12-1" momentum** — skipping the most recent month specifically to
avoid reversal contamination. Get this backwards and the result is not a weak strategy but a reliably
losing one. **Measure the crossover point empirically per asset class before committing to a direction**;
it varies by asset and has moved over time.

---

## Strategy catalogue, by return driver

### 1. Time-series momentum / trend following
Long if the trailing N-day return is positive, short if negative, across a diversified basket. Signal
via breakout (Donchian channels — the Turtle system), moving-average crossover, or plain sign-of-return.
Lookbacks of 20–100 days give holds in the weeks range. Most documented and most capacity-rich strategy
at this horizon; the edge comes from diversification across many weakly-correlated markets rather than
from any one signal being clever.

### 2. Cross-sectional momentum (relative strength)
Rank instruments by trailing return, long the top decile, short the bottom. Applies to equities,
sectors, countries, currencies, commodities. Use 12-1 to dodge the reversal zone.

### 3. Short-term reversal
Buy recent losers, sell recent winners over 3–10 days. Sits *exactly* in our target window. In liquid
names this is now largely a liquidity-provision trade — the edge is real but execution-sensitive and
substantially arbitraged.

### 4. Statistical arbitrage / pairs
Cointegrated pairs with z-score entry and exit, holding days to weeks (Gatev, Goetzmann & Rouwenhorst).
ETF-vs-constituents and related-ETF pairs are the cleaner modern version. Naturally market-neutral —
which, per the breadth arithmetic, is exactly what makes diversification pay.

### 5. Carry
The big addition from going multi-asset, and genuinely orthogonal to trend.
- **FX carry** — long high-yield, short low-yield currencies.
- **Futures carry / roll yield** — long backwardated, short contangoed, in commodities, rates, equity index.
- **Crypto carry** — perpetual funding rates and spot-vs-futures basis.
- **Cross-sectional carry within an asset class** — rank by roll yield, curve slope, or dividend yield.

Trend + carry is the classic CTA blend precisely because their correlation is low. Carry is
crash-prone: it earns steadily and loses violently, so the trend overlay acts as a partial hedge.

### 6. Value / long-horizon reversal
3–5 year price mean reversion (De Bondt & Thaler), FX purchasing-power-parity deviation, commodity
long-run price level. Signals update slowly but can be rebalanced monthly. Negatively correlated with
momentum, which is why the two combine so well.

### 7. Defensive / betting-against-beta
Long low-beta (levered), short high-beta (Frazzini & Pedersen). Works within *and* across asset
classes. Related: the low-volatility anomaly.

### 8. Volatility risk premium
Short volatility via the VIX futures term structure or the roll in vol ETPs; realised-vs-implied spread
across assets. Very tradeable at multi-day horizons. Severe left tail — size it as if it will eventually
lose 50% in a week, because it will.

### 9. Term structure and curve trades
Calendar spreads (long one expiry, short another), yield-curve steepeners/flatteners via bond futures
or ETFs, VIX front-vs-second-month spread. Lower volatility and a different driver from outright
direction.

### 10. Cross-asset lead-lag
Credit spreads (HYG/LQD) leading equity drawdowns; copper/gold as a rates signal; dollar strength into
EM equity and commodities; oil into energy equities and inversely into airlines. Slow-moving and well
suited to weeks-long holds.

### 11. Relative value between related instruments
Closed-end fund discount mean reversion (well documented, genuinely tradeable at weeks horizon);
ADR vs local listing; ETF pairs tracking the same exposure; dual-listed share classes.

### 12. Trend or reversion on *spreads and ratios* rather than outrights
Apply any of the above to gold/silver, growth/value, small/large, sector-vs-index. Spreads are naturally
market-neutral, which drives ρ down and makes breadth pay.

---

## Calendar and flow strategies

These are driven by *scheduled, predictable flows* rather than information — which is why they persist.
Nearly free to test and they fit the swing horizon naturally.

### Month and quarter boundaries
- **Turn-of-month (TOM)** — the last day plus first three days of the month historically capture a
  disproportionate share of equity returns, driven by pension and 401(k) inflows.
- **Month-end rebalancing flows** — pension funds rebalance back to target equity/bond weights at
  month-end, so **the direction is predictable from the month's return so far** (a strong month implies
  month-end equity selling). See Etula, Rinne, Suominen & Vuolteenaho.
- **Month-end bond index duration extension** — bond indices extend duration at month-end, forcing
  index funds to buy the long end. Mechanical and dated.
- **Month-end FX flows** — hedging rebalances concentrated at the London 4pm fix.
- **Quarter-end window dressing** — funds buy winners and dump losers before reporting dates.

### Annual
- **Tax-loss selling and the January reversal** — buy December's beaten-down losers, exit in January.
  The residue of the January effect that still survives.
- **Mutual fund fiscal year-end (October)** — a second, less crowded window-dressing window.
- **Halloween / "sell in May"** — Nov–Apr outperforms May–Oct (Bouman & Jacobsen 2002). Robust
  internationally, though contested. More a regime tilt than a trade.

### Scheduled events
- **Pre-FOMC announcement drift** — equity returns are startlingly concentrated in the 24 hours before
  scheduled FOMC announcements (Lucca & Moench 2015). One of the largest calendar effects on record.
- **The FOMC cycle** — even weeks of the FOMC cycle outperform odd weeks (Cieslak, Morse &
  Vissing-Jorgensen). Dates are known years ahead.
- **Treasury auction cycle** — auctions create predictable concessions into the auction and reversal
  after.
- **Earnings announcement premium** — stocks earn elevated returns around their scheduled announcement
  dates (Beaver; Frazzini & Lamont). The *date* is public and scheduled, so this is calendar-driven
  even though the content is not.

### Market-structure dates
- **Options expiry (OpEx)** — pinning into monthly expiry, gamma effects, and historically weak returns
  in the week following. Triple witching in Mar/Jun/Sep/Dec.
- **Index reconstitution** — Russell annual rebalance (June), S&P adds and deletes. Predictable flows,
  though the effect has shrunk as it got arbitraged.
- **Futures roll windows** — index futures quarterly rolls, and the GSCI/BCOM commodity roll on the
  5th–9th business day, create dated and directional pressure.

### Seasonal in the physical sense
- **Commodity seasonality** — natural gas injection/withdrawal season, heating oil, summer driving
  gasoline demand, agricultural planting and harvest cycles. Driven by physical reality rather than
  by investor behaviour, which is exactly why these persist.

### Intraday-boundary effects usable at swing horizon
- **Overnight vs intraday decomposition** — since a large share of equity return accrues overnight, a
  "hold overnight only" variant of any swing signal is worth testing as a separate leg.
- **Holiday effect** — elevated returns in the sessions before market holidays. Small but persistent.

---

## Momentum crashes — do not skip this

Momentum strategies have *predictable* crashes: they collapse after market declines when volatility is
high (Daniel & Moskowitz, *Momentum Crashes*, 2016). Scaling momentum exposure by forecast volatility
substantially improves its risk-adjusted return and removes much of the left tail. Since volatility is
the one thing we *can* forecast well, this is the highest-value application of the whole volatility
feature set.

---

## What actually drives the Sharpe

At this horizon, **position sizing and portfolio construction contribute more than signal selection**:

- **Volatility targeting** — scale each position inversely to its recent realised volatility so every
  market contributes equal risk. Worth more than any entry-rule refinement, and it is what separates
  real trend implementations from the naive version (Moreira & Muir 2017).
- **Correlation-aware sizing** — 40 instruments that are all equity beta is *one* bet, not forty.
- **Turnover discipline** — costs scale with turnover, which is far more forgiving here than intraday.

---

## Where to start

1. **Time-series momentum with volatility targeting on ~30–50 liquid ETFs** spanning equity indices,
   bonds, commodities, and FX. Best documentation, easiest data, real diversification. This is the
   baseline — if a fancier idea cannot beat it, it is not worth having.
2. **Short-term reversal, market-neutral, on liquid large-cap equities.** Fits the window precisely and
   is roughly uncorrelated with (1).
3. **Carry as the third leg**, once the first two run.
4. **Calendar overlays** — cheap to test, and they compose with everything above rather than competing.

**Instrument shortcut:** ETFs give futures-like multi-asset exposure through the equity data feed we
already have, with no separate futures source and no contract-roll handling.

**Honest expectation:** a diversified trend-plus-carry portfolio has historically run a Sharpe around
0.5–0.8, with long, painful flat stretches. That is a genuinely good outcome, and it is not exciting
month to month.

---
---

# Part 3 — Track B: intraday (0–1 entries/exits per day) — parked

*Pipeline is built; revisit after Track A. Constraint: at most one round trip per session, possibly none.*

## The constraint that governs this track: the sample size is days, not bars

With ≤1 round trip per day we get **one decision per day**. One week is 5 samples; two years is ~500.
The 390 minute-rows per day *look* like data, but they are one correlated observation of a single
decision — the effective *n* is the day count. This is the track's fatal weakness and the reason it is
parked behind Track A, where breadth is native.

## Step 0 — measure the ceiling before building anything

For each day compute the **oracle**: the best possible (entry minute, exit minute) pair — the maximum
drawup over the session (and maximum drawdown for the short side), an O(n) scan per day. Then:

1. **What is the median oracle profit, net of realistic costs?** On AAPL a round trip runs roughly
   1–3 bps against a ~1.5% daily range — but measure it, don't assume it.
2. **Degrade the oracle deliberately** — enter 5 minutes late, exit 5 minutes early. If profit
   collapses, the edge needs timing precision we will never achieve out of sample. If it degrades
   gracefully, timing is forgiving.
3. **What fraction of oracle profit lives in the top 20% of days?** If concentrated, *choosing which
   days to trade* matters more than *when within the day*.

## Where the intraday alpha probably is

1. **Day selection over intraday timing** — a filter on which days to trade at all
   (gap size × RVOL × prior-day range), with a crude entry.
2. **Regime-conditional sign flips** — momentum in low-vol trending sessions, reversal in high-vol
   ones. The *unconditional* effect is ~zero, which is exactly why it survives.
3. **The overnight/intraday boundary** — gap fade vs gap-and-go, conditioned on opening volume.
4. **The last 30 minutes**, where the flow is structural rather than informational.

## Framing: optimal stopping, not per-bar classification

With one shot per day, taking a signal at 10:00 forecloses a better one at 14:00. The real question at
each minute is: **is acting now better in expectation than continuing to wait?**

- **Triple-barrier labelling** — label each candidate entry by whichever comes first: profit target,
  stop, or end-of-day. Handles path dependence; the time barrier is native to the constraint.
- **Meta-labelling** — a simple primary rule proposes entries, a model decides only **take or skip**.
  The "0" in "0–1 trades" *is* the skip decision.

**Do not neglect the exit.** With one trade per day the exit policy (fixed vs trailing vs vol-scaled vs
time-based) often dominates total P&L.

## Intraday features worth adding

**First, a correction on what is already computed:** `price_roll_std` is the standard deviation of
**price levels** — a range/trend measure, not volatility. For signals we want the standard deviation of
**returns**. The pooled mean is effectively a moving average.

1. **Relative-to-typical, keyed on `index_minute`** — volume and volatility have a stable U-shape, so
   the informative quantity is *volume now vs typical volume at this minute of day* (RVOL), and the
   same for realised vol. Build the per-`index_minute` profile from *prior days only*.
2. **Realised variance and bipower variation** — their ratio separates continuous diffusion from jumps.
   The BNS jump test gives a clean z-statistic. *Caveat:* at second granularity microstructure noise
   biases RV upward — sample at 1–5 minutes or use a pre-averaged estimator.
3. **Range-based volatility** — Garman-Klass, Rogers-Satchell, Yang-Zhang. 5–8× more efficient than
   close-to-close, and the o/h/l/c inputs are already in hand.
4. **Position within the day's structure** — location in the cumulative high/low range, distance from
   the opening-range high/low (15/30/60 min), VWAP deviation, overnight gap normalised by ATR.
5. **Regime state** — variance ratio or Hurst, Kaufman efficiency ratio (net move ÷ sum of absolute moves).
6. **Flow proxies** — tick-rule signed volume or `(close − open)/(high − low) × volume`; average trade
   size from `volume / transactions`; Amihud illiquidity `|return| / volume`.
7. **Prior-day context** — yesterday's range, close location, whether it trended; day-of-week.

---
---

## Data scale-up

**Track A (daily, multi-asset):** thousands of instruments × 20+ years of daily bars is a few hundred
MB. Get breadth first — it is the thing that changes what is knowable.

**Track B (second-level, single symbol):** compute features at second granularity but **persist only
the minute-level feature frame** (a few hundred rows per day per symbol), keeping seconds transient.

Both: cache to parquet keyed by (symbol, date) so re-runs don't re-hit the API. Watch split adjustments
if extending back past 2020.

---

## Open questions

1. **Position spanning days.** For Track B: flat at every close, or may a position span days with at
   most one action per day? Track B above assumes intraday-flat.
2. **Shorting and leverage.** Several strategies above (BAB, pairs, carry, cross-sectional long/short)
   require both. Which are actually available in the target account?
3. **Universe definition.** Which ETF basket for Track A, and does it need survivorship-bias-free
   delisted history for the equity cross-section?
