"""
Paper trading: compute today's signal exactly as the backtest does and size
an order on an Alpaca paper account.

Usage:
    python scripts/paper_trade.py --asset spx                 # dry run, no keys needed
    python scripts/paper_trade.py --asset spx --live          # Alpaca paper account

The traded symbol defaults to the instrument the backtest trades
(config backtest.instruments, e.g. SPY for the S&P 500). Every run appends
a row to data/paper_trades.csv.

Guardrails:
- refuses stale prices (latest bar more than 3 days old) or a missing
  feature row for the latest day;
- warns when the newest model is more than a year old;
- moves the weight by at most --max-weight-change per run (default 50%), so
  a large swing is spread over several days instead of done at once.
"""
import argparse
import csv
import logging
import os
import sys
from datetime import datetime

import pandas as pd
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import download_prices, load_config  # noqa: E402
from src.features.engineering import FEATURE_COLUMNS, build_features  # noqa: E402
from src.models.hmm import predict_latest_probs  # noqa: E402
from src.models.labeling import load_fold_artifact  # noqa: E402
from src.models.signal import compute_signal  # noqa: E402
from src.trading.orders import plan_rebalance  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"
# Without a timeout a stalled connection would hang the run forever.
HTTP_TIMEOUT_SECONDS = 10
MAX_DATA_AGE_DAYS = 3  # weekend + one holiday
LOG_FIELDS = ['timestamp', 'asset', 'symbol', 'target_weight', 'current_weight', 'new_weight',
              'qty_ordered', 'side', 'dry_run', 'p_bull']


def get_latest_model(asset_name="spx"):
    """The newest fold artifact (trained on everything before the current year)."""
    model_dir = os.path.join("models", asset_name)
    folds = []
    if os.path.isdir(model_dir):
        for name in os.listdir(model_dir):
            if name.startswith("hmm_fold_") and name.endswith(".joblib"):
                try:
                    folds.append(int(name[len("hmm_fold_"):-len(".joblib")]))
                except ValueError:
                    pass
    if not folds:
        raise FileNotFoundError(f"No fold models in {model_dir}; run scripts/run_pipeline.py first.")
    return load_fold_artifact(max(folds), directory=model_dir)


def _alpaca_get(url, api_key, api_secret):
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
    return requests.get(url, headers=headers, timeout=HTTP_TIMEOUT_SECONDS)


def get_current_position(symbol, api_key, api_secret):
    resp = _alpaca_get(f"{ALPACA_API_BASE_URL}/v2/positions/{symbol}", api_key, api_secret)
    if resp.status_code == 200:
        return float(resp.json()['qty'])
    if resp.status_code == 404:  # no position
        return 0.0
    raise RuntimeError(f"Could not fetch position for {symbol}: {resp.status_code} {resp.text}")


def get_account_value(api_key, api_secret):
    resp = _alpaca_get(f"{ALPACA_API_BASE_URL}/v2/account", api_key, api_secret)
    if resp.status_code == 200:
        return float(resp.json()['portfolio_value'])
    raise RuntimeError(f"Could not fetch account: {resp.status_code} {resp.text}")


def get_latest_price(symbol, api_key, api_secret):
    resp = _alpaca_get(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest", api_key, api_secret)
    if resp.status_code == 200:
        return float(resp.json()['trade']['p'])
    raise RuntimeError(f"Could not fetch latest price for {symbol}: {resp.status_code} {resp.text}")


def submit_order(symbol, qty, side, api_key, api_secret, dry_run=True):
    if dry_run:
        logger.info(f"[DRY RUN] Would submit order: {side} {qty} {symbol}")
        return {"status": "dry_run", "qty": qty, "side": side}
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
    data = {"symbol": symbol, "qty": str(qty), "side": side, "type": "market", "time_in_force": "day"}
    resp = requests.post(f"{ALPACA_API_BASE_URL}/v2/orders", headers=headers, json=data,
                         timeout=HTTP_TIMEOUT_SECONDS)
    if resp.status_code == 200:
        logger.info(f"Submitted order: {side} {qty} {symbol}")
        return resp.json()
    raise RuntimeError(f"Order rejected: {resp.status_code} {resp.text}")


def log_trade(record, log_file="data/paper_trades.csv"):
    """Append one run to the trade log.

    A log written with an older column layout is migrated first (old rows
    kept, new columns left empty), so rows never end up under the wrong
    header.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if os.path.isfile(log_file):
        with open(log_file, newline='') as f:
            header = next(csv.reader(f), [])
        if header != LOG_FIELDS:
            pd.read_csv(log_file).reindex(columns=LOG_FIELDS).to_csv(log_file, index=False)
            logger.info("Trade log migrated to the current column layout.")
    write_header = not os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def main():
    parser = argparse.ArgumentParser(description="Run one HMM paper-trading step.")
    parser.add_argument('--asset', default='spx', help="Asset key from config (e.g. spx)")
    parser.add_argument('--symbol', default=None,
                        help="Broker symbol (default: the backtest's instrument, e.g. SPY for spx)")
    parser.add_argument('--live', action='store_true', help="Submit to the Alpaca paper account")
    parser.add_argument('--max-weight-change', type=float, default=0.5,
                        help="Largest weight change made in one run (the rest follows on later runs)")
    args = parser.parse_args()

    cfg = load_config()
    symbol = args.symbol or (cfg.get("backtest", {}).get("instruments") or {}).get(args.asset)
    if not symbol:
        sys.exit(f"No instrument configured for '{args.asset}'; pass --symbol.")

    api_key = os.environ.get("ALPACA_API_KEY_ID")
    api_secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if args.live:
        if not api_key or not api_secret:
            sys.exit("ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set for live paper trading.")
        if "-" in symbol:
            sys.exit(f"'{symbol}' looks like a crypto pair; this script only places stock/ETF orders.")
        logger.info(f"LIVE paper trading {symbol}.")
    else:
        logger.info(f"DRY RUN for {symbol}.")

    # 1. Latest prices and features.
    data = download_prices(cfg)
    if args.asset not in data:
        sys.exit(f"Asset '{args.asset}' not found in the downloaded data.")
    asset_df = data[args.asset]

    latest_date = asset_df.index[-1].date()
    age = (datetime.now().date() - latest_date).days
    if age > MAX_DATA_AGE_DAYS:
        sys.exit(f"Guardrail: latest data is {age} days old ({latest_date}).")

    features = build_features(asset_df, asset_df['VIX_Close'])
    if features.empty or features.index[-1] != asset_df.index[-1]:
        sys.exit(f"Guardrail: no complete feature row for {latest_date}.")

    # 2. Newest fold model.
    try:
        artifact = get_latest_model(args.asset)
    except FileNotFoundError as e:
        sys.exit(str(e))
    train_end = artifact.get("train_end")
    if train_end is not None and pd.Timestamp(train_end).year < datetime.now().year - 1:
        logger.warning(f"Newest model was trained through {train_end}; run scripts/run_pipeline.py to retrain.")

    # 3. Today's signal, exactly as in the backtest: continuous filtered
    # forward pass over the full history, then the shared signal definition.
    probs = predict_latest_probs(features, artifact, FEATURE_COLUMNS)
    today = compute_signal(probs, cfg['model']['smoothing_window']).iloc[-1]
    target = float(today['target_weight'])
    logger.info(f"Raw probabilities: Bull={today['p_bull']:.3f} Transition={today['p_transition']:.3f} "
                f"Crisis={today['p_crisis']:.3f}")
    logger.info(f"Smoothed regime: {today['label']} | target weight (smoothed P(Bull)): {target:.4f}")

    # 4. Account state: the broker's when live, otherwise a $100k empty
    # account priced at the latest close of the traded instrument.
    if args.live:
        current_qty = get_current_position(symbol, api_key, api_secret)
        account_value = get_account_value(api_key, api_secret)
        price = get_latest_price(symbol, api_key, api_secret)
    else:
        current_qty, account_value = 0.0, 100_000.0
        price_col = "Trade_Close" if "Trade_Close" in asset_df else "Close"
        price = float(asset_df[price_col].dropna().iloc[-1])
        logger.info(f"Dry-run account: $100,000, no position, {symbol} at {price:,.2f}.")

    plan = plan_rebalance(target, current_qty, price, account_value, args.max_weight_change)
    if plan.capped:
        logger.warning(f"Step capped: moving {plan.current_weight:.0%} -> {plan.new_weight:.0%} "
                       f"(target {target:.0%}, max change {args.max_weight_change:.0%} per run).")

    if plan.qty != 0:
        submit_order(symbol, abs(plan.qty), plan.side, api_key, api_secret, dry_run=not args.live)
    else:
        logger.info("Holding matches the target. No order needed.")

    log_trade({
        'timestamp': datetime.now().isoformat(timespec="seconds"),
        'asset': args.asset,
        'symbol': symbol,
        'target_weight': round(target, 4),
        'current_weight': round(plan.current_weight, 4),
        'new_weight': round(plan.new_weight, 4),
        'qty_ordered': plan.qty,
        'side': plan.side,
        'dry_run': not args.live,
        'p_bull': round(float(today['sp_bull']), 4),
    })
    logger.info("Run logged to data/paper_trades.csv.")


if __name__ == "__main__":
    main()
