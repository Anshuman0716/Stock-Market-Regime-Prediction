"""
Names and colors shared by every dashboard chart.

Why fixed colors per entity: a reader who learns "violet = HMM strategy" or
"red = Crisis" on one chart must be able to rely on it on every other chart,
so colors follow the entity, never its position in a list.

Both palettes were checked with a colour-vision-deficiency validator against
light and dark backgrounds:
- Strategies (violet / orange / green / light violet) pass every check in
  both themes, all pairs compared.
- Regimes (blue / amber / red) pass, but amber vs red sits close for
  deuteranopes, so regime names are always shown as text (legends, labels,
  hover) and the probability chart keeps a fixed stacking order. Colour is
  never the only way to tell regimes apart.
Red-vs-green for Bull/Crisis was rejected: it fails for red-green
colour-blind readers.
"""
from src.models.signal import REGIME_LABELS

REGIME_COLORS = {
    "Bull": "#2a78d6",
    "Transition": "#c98500",
    "Crisis": "#e34948",
}

# Short icons give regimes a second, non-colour cue in text (cards, tables).
REGIME_ICONS = {"Bull": "▲", "Transition": "◆", "Crisis": "▼"}

STRATEGY_ORDER = ["hmm", "buy_hold", "dma200", "vol_target"]
STRATEGY_NAMES = {
    "hmm": "HMM regime strategy",
    "buy_hold": "Buy & Hold",
    "dma200": "200-day MA filter",
    "vol_target": "Volatility target",
}
STRATEGY_COLORS = {
    "hmm": "#6250d6",
    "buy_hold": "#d95926",
    "dma200": "#199e70",
    "vol_target": "#9085e9",
}

# config.yaml only has tickers, so display names live here.
ASSET_NAMES = {
    "spx": "S&P 500",
    "nasdaq": "NASDAQ Composite",
    "gold": "Gold",
    "bitcoin": "Bitcoin",
}

# Single-hue ramp (light -> dark blue) for magnitude, e.g. the transition
# matrix. One hue only: a rainbow would suggest categories that aren't there.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Recessive chrome: hairline grid, muted axes.
GRID_COLOR = "rgba(137, 135, 129, 0.25)"
AXIS_COLOR = "rgba(137, 135, 129, 0.6)"
MUTED_TEXT = "#898781"

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'

__all__ = [
    "REGIME_LABELS", "REGIME_COLORS", "REGIME_ICONS",
    "STRATEGY_ORDER", "STRATEGY_NAMES", "STRATEGY_COLORS",
    "ASSET_NAMES", "SEQUENTIAL_BLUE", "GRID_COLOR", "AXIS_COLOR", "MUTED_TEXT", "FONT_FAMILY",
]


def asset_name(asset_id):
    return ASSET_NAMES.get(asset_id, asset_id)


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"
