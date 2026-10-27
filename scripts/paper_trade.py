import os
import sys
import logging
import argparse
import csv
from datetime import datetime
from pathlib import Path
import requests
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_config, download_prices
from src.features.engineering import build_features, FEATURE_COLUMNS
from src.models.labeling import load_fold_artifact
from src.models.hmm import predict_latest_probs
from src.models.signal import compute_signal

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Alpaca Config
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"

def get_latest_model(asset_name="spx"):
    model_dir = Path(f"models/{asset_name}")
    if not model_dir.exists():
        raise FileNotFoundError(f"No models found for {asset_name}")
    
    folds = []
    for f in model_dir.glob("hmm_fold_*.joblib"):
        try:
            fold_idx = int(f.stem.split('_')[-1])
            folds.append(fold_idx)
        except ValueError:
            pass
    if not folds:
        raise FileNotFoundError(f"No valid fold models found in {model_dir}")
    
    latest_fold = max(folds)
    return load_fold_artifact(latest_fold, directory=str(model_dir))

def get_current_position(symbol, api_key, api_secret):
    url = f"{ALPACA_API_BASE_URL}/v2/positions/{symbol}"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret
    }
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return float(resp.json()['qty'])
    elif resp.status_code == 404:
        return 0.0
    else:
        logger.error(f"Error fetching position: {resp.text}")
        return 0.0

def get_account_value(api_key, api_secret):
    url = f"{ALPACA_API_BASE_URL}/v2/account"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret
    }
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return float(resp.json()['portfolio_value'])
    else:
        logger.error(f"Error fetching account: {resp.text}")
        return 0.0

def get_latest_price(symbol, api_key, api_secret):
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret
    }
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return float(resp.json()['trade']['p'])
    else:
        logger.error(f"Error fetching latest price: {resp.text}")
        return None

def submit_order(symbol, qty, side, api_key, api_secret, dry_run=True):
    if dry_run:
        logger.info(f"[DRY RUN] Would submit order: {side} {qty} {symbol}")
        return {"status": "dry_run", "qty": qty, "side": side}
        
    url = f"{ALPACA_API_BASE_URL}/v2/orders"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret
    }
    data = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": "day"
    }
    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code == 200:
        logger.info(f"Successfully submitted order: {side} {qty} {symbol}")
        return resp.json()
    else:
        logger.error(f"Failed to submit order: {resp.text}")
        return None

def log_trade(record, log_file="data/paper_trades.csv"):
    file_exists = os.path.isfile(log_file)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'asset', 'target_weight', 'current_weight', 'qty_ordered', 'side', 'dry_run', 'p_bull']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

def main():
    parser = argparse.ArgumentParser(description="Run HMM paper trading loop.")
    parser.add_argument('--asset', type=str, default='spx', help="Asset key to trade (e.g. spx)")
    parser.add_argument('--symbol', type=str, default='SPY', help="Broker symbol for the asset (e.g. SPY)")
    parser.add_argument('--live', action='store_true', help="Disable dry run and submit actual paper trades")
    parser.add_argument('--max-weight-change', type=float, default=0.5, help="Guardrail: max allowed weight change in one day")
    args = parser.parse_args()

    api_key = os.environ.get("ALPACA_API_KEY_ID")
    api_secret = os.environ.get("ALPACA_API_SECRET_KEY")
    
    if not args.live:
        logger.info("Running in DRY-RUN mode.")
    else:
        if not api_key or not api_secret:
            logger.error("ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set for live paper trading.")
            sys.exit(1)
        logger.info("Running in LIVE PAPER TRADING mode.")

    # 1. Pull the latest bar and compute features
    # Download prices config defaults to 'spx' etc.
    cfg = load_config()
    data = download_prices(cfg)
    
    if args.asset not in data:
        logger.error(f"Asset '{args.asset}' not found in downloaded data.")
        sys.exit(1)
        
    asset_df = data[args.asset]
    
    # Guardrail: stale data check
    latest_date = asset_df.index[-1].date()
    today = datetime.now().date()
    # allow 3 days max (weekend + 1 holiday)
    if (today - latest_date).days > 3:
        logger.error(f"Guardrail tripped: Latest data is stale ({latest_date}). Today is {today}.")
        sys.exit(1)
        
    features = build_features(asset_df, asset_df['VIX_Close'])

    # Guardrail: the latest trading day must have a full feature row.
    # build_features drops incomplete rows, so check the date survived.
    if features.empty or features.index[-1] != asset_df.index[-1]:
        logger.error(f"Guardrail tripped: no complete feature row for {latest_date} ({args.asset}).")
        sys.exit(1)

    # 2. Load the most recent fold (trained on everything before this year).
    try:
        artifact = get_latest_model(args.asset)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    train_end = artifact.get("train_end")
    if train_end is not None and pd.Timestamp(train_end).year < today.year - 1:
        logger.warning(f"Latest model was trained through {train_end}; run scripts/run_pipeline.py to retrain.")

    # 3. Today's signal, computed exactly as in the backtest: continuous
    # filtered forward pass over the full history, then the shared
    # smoothing + sizing rule in src/models/signal.py.
    probs = predict_latest_probs(features, artifact, FEATURE_COLUMNS)
    signal = compute_signal(probs, cfg['model']['smoothing_window'])
    today_signal = signal.iloc[-1]
    target_weight = float(today_signal['target_weight'])
    p_bull = float(today_signal['sp_bull'])

    logger.info(f"Raw probabilities today: Bull={today_signal['p_bull']:.3f} "
                f"Transition={today_signal['p_transition']:.3f} Crisis={today_signal['p_crisis']:.3f}")
    logger.info(f"Smoothed regime: {today_signal['label']} | target weight (smoothed P(Bull)): {target_weight:.4f}")

    # 4. Size the order against the broker account (mock values when no keys).
    if not api_key:
        current_qty = 0.0
        account_value = 100000.0
        latest_price = 450.0
        logger.info("No API keys provided, using mock account data for dry run.")
    else:
        current_qty = get_current_position(args.symbol, api_key, api_secret)
        account_value = get_account_value(api_key, api_secret)
        latest_price = get_latest_price(args.symbol, api_key, api_secret)
        if latest_price is None:
            logger.error("Could not fetch latest price. Exiting.")
            sys.exit(1)

    current_weight = (current_qty * latest_price) / account_value if account_value > 0 else 0.0

    # Guardrail: refuse unusually large one-day rebalances.
    if abs(target_weight - current_weight) > args.max_weight_change:
        logger.error(f"Guardrail tripped: Weight change from {current_weight:.2f} to {target_weight:.2f} "
                     f"exceeds max allowed ({args.max_weight_change}).")
        sys.exit(1)

    target_qty = (account_value * target_weight) / latest_price
    qty_to_order = int(target_qty - current_qty)
    side = "buy" if qty_to_order > 0 else "sell"
    qty_abs = abs(qty_to_order)

    if qty_abs > 0:
        submit_order(args.symbol, qty_abs, side, api_key, api_secret, dry_run=not args.live)
    else:
        logger.info("Target weight matches current weight. No order needed.")

    log_trade({
        'timestamp': datetime.now().isoformat(),
        'asset': args.asset,
        'target_weight': round(target_weight, 4),
        'current_weight': round(current_weight, 4),
        'qty_ordered': qty_to_order,
        'side': side if qty_abs > 0 else 'none',
        'dry_run': not args.live,
        'p_bull': round(p_bull, 4),
    })
    logger.info("Trade run logged successfully.")


if __name__ == "__main__":
    main()
