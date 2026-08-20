import os
import sys
import json
import uuid
import subprocess
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_config, download_prices
from src.features.engineering import build_features, FEATURE_COLUMNS
from src.models.hmm import walk_forward_predict
from src.db.store import get_config_hash, get_git_sha

def train_and_persist():
    cfg = load_config('config/config.yaml')
    data = download_prices(cfg)
    
    config_hash = get_config_hash(cfg)
    git_sha = get_git_sha()
    
    manifest = {
        "created_at": datetime.now().isoformat(),
        "config_hash": config_hash,
        "git_sha": git_sha,
        "assets": {}
    }
    
    for asset_name in ['spx', 'nasdaq', 'gold', 'bitcoin']:
        if asset_name not in data:
            continue
            
        print(f"Training models for {asset_name}...")
        asset = data[asset_name]
        
        features = build_features(asset, asset['VIX_Close'])
        features['returns'] = asset['Close'].pct_change()
        features = features.dropna()
        
        # walk_forward_predict will naturally save fold artifacts because we uncommented it
        raw_labels, out_probs, folds_info = walk_forward_predict(features, FEATURE_COLUMNS, cfg, asset_name=asset_name)
        
        manifest["assets"][asset_name] = {
            "training_start": str(features.index[0].date()),
            "training_end": str(features.index[-1].date()),
            "folds": folds_info
        }
        
    os.makedirs('models', exist_ok=True)
    with open('models/manifest.json', 'w') as f:
        json.dump(manifest, f, indent=4, default=str)
        
    print("Training complete. Manifest and fold artifacts saved in models/")

if __name__ == "__main__":
    train_and_persist()
