# v2.0 Pre-registration: Regime-Aware Portfolio

**Status:** written 2026-10-28, before any portfolio code exists and before
any portfolio result has been computed. The git commit that adds this file
is the timestamp for that claim.

---

## 1. Why this document exists

Steps 1–8 made design decisions after looking at results from the same
2010–2026 period. Each decision had a training-data reason, but repeated
rounds still bend a design towards the period being looked at. v1.0 dealt
with that by freezing the design and judging it on days that did not exist
yet (`config/freeze.json`, `scripts/forward_test.py`).

v2.0 asks a new question: **does combining the per-asset regime signals into
one portfolio beat simple portfolio rules?** To keep that answer honest,
every rule that decides the answer is written down here first: which
assets, how each portfolio is built, how it is costed, which numbers are
compared, and what counts as a pass. After this file is committed, none of
these rules may change because of a result (see Section 11).

---

## 2. What is reused unchanged from v1.0

- **The per-asset HMM signals.** The regime-aware portfolio uses the
  `target_weight` column that v1.0 already produces for each asset
  (`src/models/signal.py::compute_signal`, config hash `5855083a…`).
  No model is retrained, no feature, labeling or sizing rule changes.
- **Traded instruments and cash** from `config/config.yaml`
  (`backtest.instruments`, `backtest.cash_ticker` = `^IRX`).
- **Trading costs per asset** from `src/data/loader.py::get_cost_bps`
  (spread + commission, per unit of turnover).
- **The two-day execution delay**: a weight decided at the close of day *t*
  is traded at the close of *t+1* and earns returns from *t+2*.
- **Metric definitions** from `src/backtest/metrics.py` and the Sharpe test
  from `src/stats/significance.py`.

Why: if v2.0 also changed the signals, a better or worse result could not
be attributed to the portfolio step. Holding v1.0 fixed isolates the one
new thing.

---

## 3. Assets and calendar

| Asset | Signal from | Traded as | Enters the portfolio |
|---|---|---|---|
| S&P 500 | `^GSPC` HMM | `SPY` | 2010-01-04 (start of its out-of-sample signals) |
| Gold | `GC=F` HMM | `GLD` | 2010-01-04 (start of its out-of-sample signals) |
| Bitcoin | `BTC-USD` HMM | `BTC-USD` | first US trading day on/after 2021-01-01 (start of its out-of-sample signals) |

- **NASDAQ is excluded.** It moves almost exactly with the S&P 500, so it
  would add a second copy of the same risk rather than a new asset, and
  give US equities a double share in equal weight.
- **Bitcoin joins late on purpose.** Before 2021 there is no out-of-sample
  Bitcoin signal, so the regime-aware portfolio could not hold it without
  using a model that had seen the future. From 2010 until Bitcoin joins,
  all three portfolios hold S&P 500 + Gold only; on the day it joins, all
  three switch to three assets together. The asset set on any day is always
  the same for all three portfolios, so they are compared like for like.
- **Calendar: US stock-market trading days** (the `SPY` calendar).
  Bitcoin trades every day; its return on a stock trading day is measured
  from its close on the previous stock trading day, so weekend and holiday
  moves are rolled into the next trading day rather than dropped.
- **Signal timing across calendars.** On stock trading day *t* each asset's
  signal is its latest value dated on or before *t* (carried forward from
  the past, never filled backwards). A signal more than 5 stock trading
  days stale counts as 0 exposure (cash). Bitcoin's daily close
  (00:00 UTC) falls a few hours after the US close on the same date; that
  is harmless because the trade happens at the next day's close, after the
  signal is known.
- **Missing prices.** A missing instrument price on a stock trading day
  means a 0% return for that asset that day (price carried forward). No
  price or signal is ever filled backwards.

---

## 4. The three portfolios

All weights are decided at the close of day *t* from data up to and
including *t* only. Weights are never negative (no shorting) and never add
up to more than 100% (no borrowing). Whatever is not invested earns the
cash rate.

### 4.1 Equal weight (baseline)

```
w_i(t) = 1 / N(t)          for each of the N(t) assets in the portfolio
```

Why: the simplest diversified portfolio. It needs no estimate at all, so
it cannot be wrong about the future, and it is hard to beat in practice.

### 4.2 Plain risk parity (baseline)

Inverse-volatility weights, fully invested:

```
sigma_i(t) = sample std (ddof=1) of asset i's daily returns over the
             63 stock trading days ending at t (inclusive)
w_i(t)     = (1 / sigma_i(t)) / sum_j (1 / sigma_j(t))
```

"Plain" means correlations are ignored. Why: Gold, Bitcoin and equities
have very different volatilities (Bitcoin is several times as volatile as
the S&P 500). Equal weight lets the most volatile asset dominate the risk;
inverse volatility gives each asset a roughly equal share of it. 63 days is
about one quarter: long enough to be stable, short enough to react. Every
asset has well over 63 days of instrument history before it enters, so
this never starts empty.

### 4.3 Regime-aware (the strategy being tested)

```
e_i(t) = asset i's v1.0 HMM target_weight on day t   (between 0 and 1)
w_i(t) = w_i^RP(t) * e_i(t)
cash   = 1 - sum_i w_i(t)
```

The risk-parity weights from 4.2, each scaled by that asset's own regime
exposure. When the model steps out of an asset, that slice goes to **cash**;
it is **not** moved into the other assets.

Why this design:
- It is the portfolio version of "HMM as a risk overlay": the base
  allocation is a known-good simple rule, and the regime model only
  decides how much of each slice to keep on.
- Because the regime-aware portfolio is exactly risk parity when every
  exposure is 100%, the comparison with plain risk parity measures one
  thing only: what the regime signals add or cost.
- Not re-spreading the freed weight keeps the rule from piling into
  whichever asset is left, which would raise concentration exactly when
  the model sees stress.

---

## 5. Backtest mechanics

New engine in `src/backtest/portfolio.py`. For each asset *i* and day *t*:

- **Held weight:** `h_i(t) = w_i(t-2)` (the two-day delay of Section 2).
- **Portfolio return before costs:**
  `sum_i h_i(t) * r_i(t) + (1 - sum_i h_i(t)) * cash(t)`.
- **Rebalancing:** back to the target weights every day, for all three
  portfolios alike.
- **Turnover is measured from drifted weights.** Between trades, holdings
  drift with prices: an asset that rose is a bigger share of the portfolio
  the next morning. The trade size for asset *i* is the distance from that
  drifted weight to the new held weight. Why: comparing target to target
  would make equal weight and risk parity look free to hold, when in fact
  they trade every day just to stay balanced.
- **Costs:** trade size × that asset's `get_cost_bps` / 10,000, summed over
  assets and charged on the day the new weight takes effect (the same
  convention as the v1.0 engine). Cash trades are free.
- **Cash rate:** `^IRX`, the yield known at the previous close, as in v1.0.

---

## 6. Evaluation window

- **Historical window:** 2010-01-04 to the last day of data used at the
  v2.0 freeze. This is the window the pass/fail verdict in Section 8 is
  read on first.
- **Forward window:** stock trading days after the v2.0 freeze record's
  `data_end_at_freeze` (Section 10).
- Annualization: 252 rows per year for all portfolio numbers (the
  portfolio lives on the stock calendar, including Bitcoin's slice).

---

## 7. Metrics

**Deciding metrics** (the only ones the verdict uses):

1. **Sharpe ratio above cash:** `compute_sharpe` on daily net return minus
   daily cash return, annualized with 252.
2. **Maximum drawdown:** the deepest peak-to-trough fall of the net value
   (the first value returned by `compute_drawdowns`), as a negative number.

**Reported, not deciding:** CAGR, volatility, Sortino, Calmar, annual
turnover, average invested share, calendar-year and stress-episode
breakdowns (episodes from `evaluation.stress_episodes`), and the weights
through time.

Why only two: every extra metric in a verdict is another way to pass by
luck. Sharpe asks "return per unit of risk"; max drawdown asks "how bad
did it get". Together they cover the claim a regime overlay makes: similar
or better reward with less damage in bad periods.

---

## 8. The pass/fail test

The quotable rule (Phase 4 copies this sentence word for word into the
README):

> **Pass:** the regime-aware portfolio has a strictly higher Sharpe ratio
> above cash than both equal weight and plain risk parity, and a maximum
> drawdown no deeper than either of them. **Significant pass:** a pass in
> which both Sharpe differences also have a Holm-adjusted p-value below
> 0.05.

Details fixed in advance:
- Comparisons use full-precision numbers, not rounded ones, with no
  tolerance band. A tie in Sharpe is not a pass; an exactly equal drawdown
  is allowed.
- All three portfolios use the identical date range, asset set, cash
  series and cost model.
- Possible verdicts are exactly: **Fail**, **Pass**, **Significant pass**.

---

## 9. Significance tests

- **Test:** `block_bootstrap_sharpe_diff` on daily excess returns
  (above cash): moving blocks of 60 days, 2,000 draws, seed 42,
  annualization 252. The HAC delta-method p-value
  (`ledoit_wolf_sharpe_diff`) is reported next to it as a cross-check but
  does not decide anything.
- **Family:** two tests — regime-aware vs equal weight, and regime-aware
  vs plain risk parity — Holm-corrected together with `adjust_pvalues`.
- **Kept separate from the v1.0 family** (each asset's HMM vs its
  baselines). Why: they answer different questions. Merging them would
  make each v1.0 p-value change just because v2.0 exists, and the
  portfolio result would be diluted by tests it has nothing to do with.

---

## 10. Forward test

- v2.0 gets its own freeze record (Phase 3). Its forward window is every
  stock trading day after that record's `data_end_at_freeze`.
- The same rule (Section 8) is applied to the forward window.
- Before 252 forward trading days (about one year) the forward result is
  shown as **interim**, with no verdict. Why: Sharpe over a few weeks is
  mostly noise; a verdict that early would be a coin toss presented as
  evidence.
- v1.0's forward test keeps running unchanged next to it, on the same new
  days.

The historical verdict and the forward verdict are reported separately.
Only the forward one is fully clean: the per-asset signals the
regime-aware portfolio uses were developed while looking at 2010–2026
(see Section 12).

---

## 11. Commitments

1. **The result is reported whatever it is**, in the README and the
   dashboard, including a Fail.
2. **No rescue tuning.** If v2.0 fails, its parameters (lookback, assets,
   staleness limit, weight rule, costs) are not adjusted to make it pass.
   A new idea becomes a new version with its own pre-registration, judged
   on its own forward data.
3. **Allowed after commit:** fixing implementation bugs so the code does
   what this document says. Every such fix is recorded in Section 14 with
   the date, the commit, and whether any portfolio result had already been
   seen.
4. **Not allowed after commit:** changing any rule in Sections 3–10. If a
   rule turns out to be impossible to implement as written, the problem is
   recorded in Section 14 and resolved by the smallest change that keeps
   the rule's intent, before results are looked at where possible.

---

## 12. Known limits of this test

- **The historical window is only half blind.** No portfolio result has
  been seen, but the per-asset signals were shaped while looking at
  2010–2026 results. The historical verdict is evidence about combining
  signals, not a clean out-of-sample test of the signals themselves.
- **Three assets is a small portfolio,** and only two of them before 2021.
  The diversification benefit available is limited.
- **Bitcoin's slice covers about six years,** including one large
  crypto bear market (2022). Its contribution rests on few independent
  episodes.
- **Two-asset to three-asset switch.** Adding Bitcoin in 2021 changes all
  three portfolios at once; results before and after that date describe
  slightly different portfolios.
- **Yahoo adjusted prices** shift slightly between downloads (about 1e-6 a
  day for SPY), so reruns can move the last digits of a metric; this is
  noted in the README already.

---

## 13. Parameters to be mirrored in `config.yaml` (`portfolio:` section)

Phase 3 puts these into config. They must match this table exactly.

| Setting | Value |
|---|---|
| assets | `spx`, `gold`, `bitcoin` |
| calendar | US stock trading days (`SPY` dates) |
| asset entry | first stock trading day of that asset's out-of-sample signals |
| risk-parity lookback | 63 stock trading days, sample std (ddof=1), inclusive of *t* |
| signal staleness limit | 5 stock trading days, then exposure 0 |
| execution delay | 2 days (decide at *t*, trade at *t+1*, earn from *t+2*) |
| rebalancing | daily, to target, turnover from drifted weights |
| costs | per asset, `get_cost_bps` |
| cash | `^IRX`, previous close's yield |
| annualization | 252 |
| bootstrap | block 60, 2,000 draws, seed 42 |
| Holm family | {regime-aware vs equal weight, regime-aware vs risk parity} |
| significance level | 0.05, Holm-adjusted |
| forward verdict | after 252 forward trading days |

---

## 14. Amendments log

Entries are added here, never edited or removed.

| Date | Commit | Change | Portfolio results seen before this change? |
|---|---|---|---|
| — | — | (none yet) | — |
