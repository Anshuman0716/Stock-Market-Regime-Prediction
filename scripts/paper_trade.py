import os
import sys
import glob
import logging
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
import requests
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_config, download_prices
from src.features.engineering import build_features, FEATURE_COLUMNS
from src.models.labeling import load_fold_artifact
from src.models.hmm import predict_proba_filtered
from src.backtest.risk import compute_target_weight_proba

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
    features['returns'] = asset_df['Close'].pct_change()
    
    # Guardrail: NaNs in the final row
    if features.iloc[-1].isna().any():
        logger.error(f"Guardrail tripped: NaNs found in the latest feature row for {args.asset}.")
        sys.exit(1)
        
    features = features.dropna()
    
    # 2. Load the persisted model
    try:
        artifact = get_latest_model(args.asset)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)
        
    model = artifact["model"]
    scaler = artifact["scaler"]
    state_map = artifact["state_map"]
    
    # 3. Produce today's filtered regime probabilities and target weight
    # predict_proba_filtered needs the historical sequence to correctly compute the forward pass.
    # Take the last 252 days to ensure the forward probabilities have stabilized.
    X_raw_window = features.iloc[-252:][FEATURE_COLUMNS].values
    X_scaled_window = scaler.transform(X_raw_window)
    probs_window = predict_proba_filtered(model, X_scaled_window)
    
    # Extract today's probabilities
    latest_probs = probs_window[-1]
    
    prob_dict = {state_map[s]: latest_probs[s] for s in range(len(state_map))}
    
    # Combine if multiple states map to Bull
    p_bull = 0.0
    for s in range(len(state_map)):
        if state_map[s] == 'Bull':
            p_bull += latest_probs[s]
            
    logger.info(f"Latest Probabilities: {prob_dict}")
    logger.info(f"Target P(Bull): {p_bull:.4f}")
    
    # Use compute_target_weight_proba interface
    target_weight = min(max(p_bull, 0.0), 1.0) # cap to 0.0 - 1.0
    
    if args.live or not args.live:
        # 4. Submit order to broker sandbox
        # For dry-run without credentials, we just simulate the order logic
        if not api_key:
            # Mock values
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
        
        # Guardrail: Check max weight change
        if abs(target_weight - current_weight) > args.max_weight_change:
            logger.error(f"Guardrail tripped: Weight change from {current_weight:.2f} to {target_weight:.2f} exceeds max allowed ({args.max_weight_change}).")
            sys.exit(1)
            
        target_qty = (account_value * target_weight) / latest_price
        qty_to_order = int(target_qty - current_qty)
        
        side = "buy" if qty_to_order > 0 else "sell"
        qty_abs = abs(qty_to_order)
        
        if qty_abs > 0:
            order_res = submit_order(args.symbol, qty_abs, side, api_key, api_secret, dry_run=not args.live)
        else:
            logger.info("Target weight matches current weight. No order needed.")
            order_res = {"status": "no_action"}
            
        # Log to file
        record = {
            'timestamp': datetime.now().isoformat(),
            'asset': args.asset,
            'target_weight': round(target_weight, 4),
            'current_weight': round(current_weight, 4),
            'qty_ordered': qty_to_order,
            'side': side if qty_abs > 0 else 'none',
            'dry_run': not args.live,
            'p_bull': round(p_bull, 4)
        }
        log_trade(record)
        logger.info("Trade run logged successfully.")

if __name__ == "__main__":
    main()
