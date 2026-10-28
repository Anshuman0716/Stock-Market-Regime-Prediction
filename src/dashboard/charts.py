"""
Plotly chart builders for the dashboard.

Every function takes rows already stored by the pipeline (via
src/dashboard/data.py) and returns a go.Figure. Nothing here computes a
metric; charts only draw what the store holds.

Conventions (applied in _style):
- One y-scale per panel. Different measures (price vs probability, equity vs
  drawdown) get their own stacked panel with a shared date axis, never a
  second y-axis on the same plot: two scales on one plot invite readers to
  see correlations that are only an artefact of how the axes were aligned.
- Colors follow the entity (theme.py); regime names are always in the legend
  and hover so colour is never the only cue.
- Transparent background so the chart sits on Streamlit's light or dark
  theme; thin 2px lines; hairline grid.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.dashboard.theme import (
    REGIME_LABELS, REGIME_COLORS, STRATEGY_ORDER, STRATEGY_NAMES, STRATEGY_COLORS,
    SEQUENTIAL_BLUE, GRID_COLOR, AXIS_COLOR, MUTED_TEXT, FONT_FAMILY, asset_name, hex_to_rgba,
    ASSET_COLORS, PORTFOLIO_STRATEGY_ORDER, PORTFOLIO_STRATEGY_NAMES, PORTFOLIO_STRATEGY_COLORS,
)

# The strategy set a chart draws: per-asset strategies by default, or the
# v2.0 portfolio's three.
PER_ASSET = (STRATEGY_ORDER, STRATEGY_NAMES, STRATEGY_COLORS)
PORTFOLIO = (PORTFOLIO_STRATEGY_ORDER, PORTFOLIO_STRATEGY_NAMES, PORTFOLIO_STRATEGY_COLORS)

LINE_WIDTH = 2
PRICE_COLOR = MUTED_TEXT  # neutral: the regime colours carry the meaning on the price chart
REGIME_SYMBOLS = {"Bull": "circle", "Transition": "diamond", "Crisis": "triangle-down"}

RANGE_BUTTONS = [
    dict(count=1, label="1Y", step="year", stepmode="backward"),
    dict(count=3, label="3Y", step="year", stepmode="backward"),
    dict(count=5, label="5Y", step="year", stepmode="backward"),
    dict(step="all", label="All"),
]


def _style(fig, height, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36 if legend else 12, b=8),
        font=dict(family=FONT_FAMILY, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title_text=""),
        hoverlabel=dict(font=dict(family=FONT_FAMILY)),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS_COLOR, ticks="outside", tickcolor=AXIS_COLOR,
                     zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, gridwidth=1, linecolor=AXIS_COLOR, zeroline=False)
    return fig


def _range_selector(fig, axis="xaxis"):
    fig.update_layout({axis: dict(rangeselector=dict(
        buttons=RANGE_BUTTONS, x=0, y=1.0, yanchor="bottom",
        bgcolor="rgba(137,135,129,0.12)", activecolor="rgba(137,135,129,0.35)",
        font=dict(size=12)))})


def _zero_line(fig, row=None, col=None, horizontal=True):
    kw = dict(line_width=1, line_color=AXIS_COLOR)
    if row is not None:
        kw.update(row=row, col=col)
    if horizontal:
        fig.add_hline(y=0, **kw)
    else:
        fig.add_vline(x=0, **kw)


def pretty_episode(name):
    """'covid_crash_2020' -> 'Covid crash 2020'."""
    return name.replace("_", " ").capitalize()


# ── Regime timeline ──────────────────────────────────────────────────────

def regime_timeline(timeline, spells, asset_id, smoothed=True):
    """Price (log scale) shaded by regime, above a stacked regime-probability panel.

    smoothed=True shows the trailing-smoothed probabilities the strategy
    trades; False shows the raw filtered HMM output, so the reader can see
    how much the smoothing calms the signal.
    """
    prefix, label_col = ("sp_", "label") if smoothed else ("p_", "raw_label")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.64, 0.36],
                        vertical_spacing=0.05)

    fig.add_trace(go.Scatter(
        x=timeline.index, y=timeline["close"], name="Close", mode="lines",
        line=dict(color=PRICE_COLOR, width=1.5), showlegend=False,
        customdata=timeline[[label_col]].to_numpy(),
        hovertemplate="%{x|%d %b %Y}<br>Close %{y:,.2f}<br>Regime: %{customdata[0]}<extra></extra>",
    ), row=1, col=1)

    # Regime bands: each spell is shaded until the next spell starts, so
    # weekends and holidays inside a spell don't leave white gaps.
    if not spells.empty:
        ends = list(spells["start_date"].iloc[1:]) + [spells["end_date"].iloc[-1]]
        for (_, s), x1 in zip(spells.iterrows(), ends):
            fig.add_vrect(x0=s["start_date"], x1=x1, row=1, col=1, layer="below", line_width=0,
                          fillcolor=hex_to_rgba(REGIME_COLORS[s["label"]], 0.16))

    # Stacked in a fixed order (Bull at the bottom) so position is a second
    # cue besides colour.
    for lab in REGIME_LABELS:
        fig.add_trace(go.Scatter(
            x=timeline.index, y=timeline[f"{prefix}{lab.lower()}"], name=lab, mode="lines",
            stackgroup="probs", line=dict(width=0.5, color=REGIME_COLORS[lab]),
            fillcolor=hex_to_rgba(REGIME_COLORS[lab], 0.75), legendgroup=lab,
            hovertemplate=f"{lab}: %{{y:.0%}}<extra></extra>",
        ), row=2, col=1)

    _style(fig, height=600)
    fig.update_yaxes(type="log", title_text=f"{asset_name(asset_id)} close (log scale)", row=1, col=1)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title_text="Regime probability", row=2, col=1)
    fig.update_layout(hovermode="x unified")
    _range_selector(fig)
    return fig


# ── Performance ──────────────────────────────────────────────────────────

def performance_panels(daily, strategies=PER_ASSET):
    """Growth of $1 (log), drawdown and rolling 1-year Sharpe, one panel each, shared dates.

    `strategies` = (order, names, colors): PER_ASSET or PORTFOLIO.
    """
    order, names, colors = strategies
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.46, 0.27, 0.27],
                        vertical_spacing=0.05)
    for strat in order:
        d = daily[daily["strategy"] == strat]
        if d.empty:
            continue
        common = dict(x=d["date"], mode="lines", legendgroup=strat, name=names[strat],
                      line=dict(color=colors[strat], width=LINE_WIDTH))
        fig.add_trace(go.Scatter(y=d["equity"], hovertemplate="$%{y:.2f}", **common), row=1, col=1)
        fig.add_trace(go.Scatter(y=d["drawdown"], showlegend=False, hovertemplate="%{y:.1%}", **common),
                      row=2, col=1)
        fig.add_trace(go.Scatter(y=d["rolling_sharpe"], showlegend=False, hovertemplate="%{y:.2f}", **common),
                      row=3, col=1)
    _zero_line(fig, row=3, col=1)
    _style(fig, height=760)
    fig.update_yaxes(type="log", title_text="Growth of $1 (log)", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", title_text="Drawdown", row=2, col=1)
    fig.update_yaxes(title_text="Rolling 1-year Sharpe", row=3, col=1)
    fig.update_layout(hovermode="x unified")
    _range_selector(fig)
    return fig


def _strategy_bars(df, x, y, orientation="v", text_fmt=".0%", hover_extra=None):
    fig = go.Figure()
    for strat in STRATEGY_ORDER:
        d = df[df["strategy"] == strat]
        if d.empty:
            continue
        values, cats = d[y], d[x]
        custom = d[hover_extra[0]].to_numpy() if hover_extra else None
        extra = f"<br>{hover_extra[1]}: %{{customdata:.0%}}" if hover_extra else ""
        value_ref = "%{x" if orientation == "h" else "%{y"
        cat_ref = "%{y}" if orientation == "h" else "%{x}"
        fig.add_trace(go.Bar(
            x=values if orientation == "h" else cats, y=cats if orientation == "h" else values,
            orientation=orientation, name=STRATEGY_NAMES[strat], marker_color=STRATEGY_COLORS[strat],
            marker_line_width=0, customdata=custom,
            hovertemplate=f"{cat_ref}<br>{STRATEGY_NAMES[strat]}: {value_ref}:{text_fmt}}}{extra}<extra></extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    return fig


def yearly_returns_chart(period_metrics):
    years = period_metrics[period_metrics["kind"] == "year"].sort_values("period")
    fig = _strategy_bars(years, x="period", y="Total Return", hover_extra=("Avg. Exposure", "Avg. exposure"))
    _zero_line(fig)
    _style(fig, height=380)
    fig.update_yaxes(tickformat=".0%", title_text="Total return in the year")
    fig.update_xaxes(type="category", title_text="")
    return fig


def episode_chart(period_metrics, episodes):
    """Total return over each equity stress episode, in chronological order (top = oldest)."""
    eps = period_metrics[period_metrics["kind"] == "episode"].copy()
    # Sort by start date: the run's stored config is JSON with sorted keys,
    # so dict order would be alphabetical, not chronological.
    order = sorted((e for e in episodes if e in set(eps["period"])), key=lambda e: episodes[e][0])
    labels = {e: f"{pretty_episode(e)}  ({episodes[e][0]} → {episodes[e][1]})" for e in order}
    eps["label"] = eps["period"].map(labels)
    fig = _strategy_bars(eps, x="label", y="Total Return", orientation="h",
                         hover_extra=("Avg. Exposure", "Avg. exposure"))
    _zero_line(fig, horizontal=False)
    _style(fig, height=130 + 70 * max(len(order), 1))
    fig.update_layout(margin=dict(t=56))  # room for the legend above the first bar group
    fig.update_xaxes(tickformat=".0%", title_text="Total return over the episode", showgrid=True,
                     gridcolor=GRID_COLOR)
    fig.update_yaxes(categoryorder="array", categoryarray=[labels[e] for e in reversed(order)],
                     showgrid=False, title_text="")
    return fig


# ── Risk ─────────────────────────────────────────────────────────────────

def exposure_chart(daily):
    """Share of capital invested each day, one small panel per strategy.

    Buy & Hold is always 100%, so it is left out. Separate panels on a shared
    date axis keep each strategy readable; overlaid, the three series hid
    each other over 16 years of daily data.
    """
    shown = [s for s in STRATEGY_ORDER if s != "buy_hold" and (daily["strategy"] == s).any()]
    fig = make_subplots(rows=len(shown), cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=[STRATEGY_NAMES[s] for s in shown])
    for i, strat in enumerate(shown, start=1):
        d = daily[daily["strategy"] == strat]
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["weight"], name=STRATEGY_NAMES[strat], mode="lines", fill="tozeroy",
            line=dict(color=STRATEGY_COLORS[strat], width=1, shape="hv" if strat == "dma200" else "linear"),
            fillcolor=hex_to_rgba(STRATEGY_COLORS[strat], 0.3), hovertemplate="%{y:.0%}"), row=i, col=1)
        fig.update_yaxes(range=[0, 1.05], tickvals=[0, 0.5, 1], tickformat=".0%", row=i, col=1)
    _style(fig, height=110 + 120 * len(shown), legend=False)
    fig.update_annotations(font=dict(size=12), x=0, xanchor="left")
    # No range buttons here: they would sit on top of the first panel's
    # title. Drag to zoom; double-click to reset.
    fig.update_layout(hovermode="x unified", margin=dict(t=36))
    return fig


TAIL_METRICS = ["VaR 95% (Hist)", "CVaR 95% (Hist)", "VaR 99% (Hist)", "CVaR 99% (Hist)"]


def tail_risk_chart(asset_metrics):
    """Daily VaR / CVaR per strategy. asset_metrics: index strategy, columns metric names.

    Stored as positive loss sizes (0.02 = a 2% one-day loss), so taller bars
    mean more tail risk.
    """
    rows = []
    for strat in STRATEGY_ORDER:
        if strat in asset_metrics.index:
            for m in TAIL_METRICS:
                rows.append({"strategy": strat, "metric": m, "value": asset_metrics.loc[strat, m]})
    df = pd.DataFrame(rows)
    fig = _strategy_bars(df, x="metric", y="value", text_fmt=".2%")
    _style(fig, height=340)
    fig.update_yaxes(tickformat=".1%", title_text="One-day loss", rangemode="tozero")
    fig.update_xaxes(title_text="")
    return fig


# ── Model insight ────────────────────────────────────────────────────────

def transition_heatmap(probs, counts):
    """P(regime tomorrow | regime today), from the out-of-sample smoothed labels."""
    z = probs.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"→ {c}" for c in probs.columns], y=list(probs.index),
        zmin=0, zmax=1, colorscale=[[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)],
        xgap=2, ygap=2, showscale=False,
        customdata=counts.to_numpy(dtype=float),
        hovertemplate="Today %{y} %{x}<br>Probability %{z:.1%}<br>Days observed %{customdata:,.0f}<extra></extra>",
    ))
    # Cell labels: white on dark cells, dark ink on light ones, for contrast.
    for i, row in enumerate(probs.index):
        for j, col in enumerate(probs.columns):
            v = z[i, j]
            if np.isnan(v):
                continue
            fig.add_annotation(x=j, y=i, text=f"{v:.1%}", showarrow=False,
                               font=dict(color="#ffffff" if v > 0.5 else "#0b0b0b", size=14))
    _style(fig, height=300, legend=False)
    fig.update_yaxes(autorange="reversed", title_text="Regime today", showgrid=False)
    fig.update_xaxes(side="top", title_text="", showline=False, ticks="")
    return fig


def spell_duration_chart(spells):
    """How long each regime lasts once it starts (one dot per spell)."""
    fig = go.Figure()
    for lab in REGIME_LABELS:
        d = spells[spells["label"] == lab]
        if d.empty:
            continue
        fig.add_trace(go.Box(
            y=d["days"], name=lab, boxpoints="all", jitter=0.45, pointpos=0,
            marker=dict(color=REGIME_COLORS[lab], size=8, symbol=REGIME_SYMBOLS[lab], opacity=0.75,
                        line=dict(width=1, color="rgba(255,255,255,0.8)")),
            line=dict(color=REGIME_COLORS[lab], width=1.5), fillcolor="rgba(0,0,0,0)",
            customdata=d[["start_date", "end_date"]].astype(str).to_numpy(),
            hovertemplate=lab + ": %{y} trading days<br>%{customdata[0]} → %{customdata[1]}<extra></extra>",
        ))
    _style(fig, height=340, legend=False)
    # Explicit ticks: Plotly's default log ticks label minor steps ("2", "5")
    # without their decade, which reads as nonsense.
    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    fig.update_yaxes(type="log", tickvals=ticks, ticktext=[str(t) for t in ticks],
                     title_text="Spell length (trading days, log)")
    return fig


def fold_states_chart(fold_states):
    """Each yearly fold's HMM states: training volatility vs year, coloured by the label they got.

    Shows how volatility and label relate. Labels come from what each state
    earned in training (src/models/labeling.py), so for stocks the calm
    states are usually Bull, while for gold a calm drift can be Transition.
    """
    fig = go.Figure()
    for lab in REGIME_LABELS:
        d = fold_states[fold_states["label"] == lab]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["test_year"], y=d["ann_vol"], mode="markers", name=lab,
            marker=dict(color=REGIME_COLORS[lab], symbol=REGIME_SYMBOLS[lab],
                        size=8 + 30 * d["train_share"], opacity=0.85,
                        line=dict(width=2, color="rgba(255,255,255,0.9)")),
            customdata=d[["fold_idx", "state", "train_share", "ann_return", "sharpe"]].to_numpy(),
            hovertemplate=(lab + " · fold %{customdata[0]} (trades %{x}) · state %{customdata[1]}"
                           "<br>Training: vol %{y:.1%}, return %{customdata[3]:.1%}, Sharpe %{customdata[4]:.2f}"
                           "<br>Share of training days %{customdata[2]:.0%}<extra></extra>"),
        ))
    _style(fig, height=380)
    fig.update_yaxes(tickformat=".0%", title_text="State volatility in training (ann.)")
    fig.update_xaxes(title_text="Year the fold traded", dtick=1)
    return fig


# ── Overview and statistics ──────────────────────────────────────────────

def asset_dot_plot(metrics, metric, tickformat):
    """One row per asset, one dot per strategy, for a single metric (e.g. Sharpe).

    The title is left to the page (st.markdown) so it can't collide with the legend.
    """
    fig = go.Figure()
    assets = list(dict.fromkeys(metrics.index.get_level_values(0)))
    labels = [asset_name(a) for a in assets]
    # Thin hairline linking each asset's dots: makes the spread easy to read.
    for a, lab in zip(assets, labels):
        vals = metrics.loc[a, metric]
        fig.add_trace(go.Scatter(x=[vals.min(), vals.max()], y=[lab, lab], mode="lines",
                                 line=dict(color=GRID_COLOR, width=3), showlegend=False, hoverinfo="skip"))
    for strat in STRATEGY_ORDER:
        xs = [metrics.loc[(a, strat), metric] if (a, strat) in metrics.index else None for a in assets]
        fig.add_trace(go.Scatter(
            x=xs, y=labels, mode="markers", name=STRATEGY_NAMES[strat],
            marker=dict(color=STRATEGY_COLORS[strat], size=13, line=dict(width=2, color="rgba(255,255,255,0.9)")),
            hovertemplate=f"%{{y}}<br>{STRATEGY_NAMES[strat]}: %{{x:{tickformat}}}<extra></extra>",
        ))
    _style(fig, height=300)
    fig.update_xaxes(tickformat=tickformat, showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(labels)), showgrid=False)
    return fig


def significance_forest(sig):
    """Sharpe difference (HMM minus baseline) with its 95% bootstrap interval, per asset."""
    fig = go.Figure()
    sig = sig.copy()
    sig["row"] = sig["asset_id"].map(asset_name)
    for base in [s for s in STRATEGY_ORDER if s != "hmm"]:
        d = sig[sig["baseline"] == base]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["sharpe_diff"], y=[d["row"], ["vs " + STRATEGY_NAMES[base]] * len(d)],
            mode="markers", name="HMM vs " + STRATEGY_NAMES[base],
            marker=dict(color=STRATEGY_COLORS[base], size=11),
            error_x=dict(type="data", symmetric=False, array=d["ci_upper"] - d["sharpe_diff"],
                         arrayminus=d["sharpe_diff"] - d["ci_lower"], color=STRATEGY_COLORS[base],
                         thickness=2, width=6),
            customdata=d[["ci_lower", "ci_upper", "p_value", "p_value_hac", "p_value_holm"]].to_numpy(),
            hovertemplate=("%{y}<br>Sharpe difference %{x:+.2f}"
                           "<br>95% interval %{customdata[0]:+.2f} to %{customdata[1]:+.2f}"
                           "<br>p (bootstrap) %{customdata[2]:.3f} · p (HAC) %{customdata[3]:.3f}"
                           "<br>p (Holm-adjusted) %{customdata[4]:.3f}<extra></extra>"),
        ))
    _zero_line(fig, horizontal=False)
    _style(fig, height=420)
    fig.update_xaxes(title_text="Sharpe ratio difference (HMM − baseline)", showgrid=True, gridcolor=GRID_COLOR,
                     zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig


def portfolio_weights_chart(weights):
    """What each portfolio held through time: one panel per strategy, assets stacked.

    weights: index (strategy, date), one column per asset (held weights, so
    after the two-day delay). The empty space above the stack is cash, which
    is the regime-aware portfolio's whole point, so it is left visible rather
    than filled in.
    """
    shown = [s for s in PORTFOLIO_STRATEGY_ORDER if s in weights.index.get_level_values(0)]
    fig = make_subplots(rows=len(shown), cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[PORTFOLIO_STRATEGY_NAMES[s] for s in shown])
    assets = list(weights.columns)
    for i, strat in enumerate(shown, start=1):
        w = weights.loc[strat]
        for a in assets:
            color = ASSET_COLORS.get(a, MUTED_TEXT)
            fig.add_trace(go.Scatter(
                x=w.index, y=w[a], name=asset_name(a), legendgroup=a, showlegend=(i == 1),
                mode="lines", stackgroup=f"s{i}", line=dict(color=color, width=0.5),
                fillcolor=hex_to_rgba(color, 0.55), hovertemplate=f"{asset_name(a)}: %{{y:.0%}}<extra></extra>"),
                row=i, col=1)
        fig.update_yaxes(range=[0, 1.02], tickvals=[0, 0.5, 1], tickformat=".0%", row=i, col=1)
    _style(fig, height=130 + 170 * len(shown))
    fig.update_annotations(font=dict(size=12), x=0, xanchor="left")
    fig.update_layout(hovermode="x unified", margin=dict(t=48))
    return fig
