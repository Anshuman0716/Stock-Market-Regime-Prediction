import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import logging

sys.path.insert(0, ".")
from src.data.loader import load_config, download_prices
from src.features.engineering import build_features, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO)

def main():
    cfg = load_config('config/config.yaml')
    data = download_prices(cfg)
    spx = data['spx']
    features = build_features(spx, spx['VIX_Close'])
    features['returns'] = spx['Close'].pct_change()
    features = features.dropna()
    
    # 1. Initial Training Window Only
    # Use the first min_train_days for walk-forward, which is year < 2005
    train_df = features[features.index.year < 2005]
    X_train_raw = train_df[FEATURE_COLUMNS].values
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    
    N, d = X_train.shape
    print(f"Initial training window: N={N} observations, d={d} features.")
    
    # ==========================================
    # TASK 1: BIC/AIC SWEEP
    # ==========================================
    results = []
    
    for k in range(2, 7):
        # Free parameters
        p = (k - 1) + k*(k - 1) + k*d + k*d
        
        model = GaussianHMM(n_components=k, covariance_type="diag", n_iter=2000, random_state=42)
        model.fit(X_train)
        logL = model.score(X_train)
        
        aic = 2 * p - 2 * logL
        bic = p * np.log(N) - 2 * logL
        
        results.append({"k": k, "p": p, "logL": logL, "AIC": aic, "BIC": bic})
        print(f"k={k}: p={p}, logL={logL:.2f}, AIC={aic:.2f}, BIC={bic:.2f}")
        
    df_results = pd.DataFrame(results)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_results['k'], df_results['AIC'], marker='o', label='AIC')
    ax.plot(df_results['k'], df_results['BIC'], marker='s', label='BIC')
    ax.set_xlabel('Number of States (k)')
    ax.set_ylabel('Information Criterion (Lower is Better)')
    ax.set_title(f'BIC/AIC Sweep (N={N}, d={d}, Full Covariance)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate parameter counts
    for i, row in df_results.iterrows():
        ax.annotate(f"p={int(row['p'])}", 
                    (row['k'], row['BIC']), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
        
    plt.tight_layout()
    os.makedirs('docs/figures', exist_ok=True)
    plt.savefig('docs/figures/bic_aic_sweep.png')
    plt.close()
    
    best_k_bic = df_results.loc[df_results['BIC'].idxmin()]['k']
    best_k_aic = df_results.loc[df_results['AIC'].idxmin()]['k']
    print(f"Best k according to BIC: {int(best_k_bic)}")
    print(f"Best k according to AIC: {int(best_k_aic)}")
    
    # The user's default is k=4, so let's test stability for k=4
    chosen_k = 4
    print(f"\nProceeding with Seed and Bootstrap stability tests for k={chosen_k} (README default)...")
    
    # Baseline model for chosen k
    baseline_model = GaussianHMM(n_components=chosen_k, covariance_type="diag", n_iter=2000, random_state=42)
    baseline_model.fit(X_train)
    baseline_preds = baseline_model.predict(X_train)
    
    # ==========================================
    # TASK 2: SEED STABILITY
    # ==========================================
    seeds = range(100, 120)  # 20 seeds
    ari_scores_seed = []
    
    for s in seeds:
        m = GaussianHMM(n_components=chosen_k, covariance_type="diag", n_iter=2000, random_state=s)
        m.fit(X_train)
        preds = m.predict(X_train)
        ari = adjusted_rand_score(baseline_preds, preds)
        ari_scores_seed.append(ari)
        
    ari_seed = np.array(ari_scores_seed)
    print("\n--- Seed Stability (20 seeds) ---")
    print(f"Mean ARI: {ari_seed.mean():.4f}")
    print(f"Min ARI:  {ari_seed.min():.4f}")
    print(f"Max ARI:  {ari_seed.max():.4f}")
    diff_sols = np.sum(ari_seed < 0.90)
    print(f"Fraction converging to materially different solution (ARI < 0.90): {diff_sols/len(seeds):.2%}")
    
    # ==========================================
    # TASK 3: BOOTSTRAP STABILITY
    # ==========================================
    # Moving block bootstrap. Block length = 60 days (approx 3 months).
    # Financial regimes persist, so we need a block length that captures short-term autocorrelation
    # but still shuffles the macroscopic regimes.
    block_length = 60
    n_resamples = 20
    ari_scores_boot = []
    
    np.random.seed(42)
    for _ in range(n_resamples):
        # Generate block indices
        n_blocks = int(np.ceil(N / block_length))
        start_indices = np.random.randint(0, N - block_length, size=n_blocks)
        
        resample_idx = []
        for start in start_indices:
            resample_idx.extend(range(start, start + block_length))
        resample_idx = resample_idx[:N]  # Trim to exact length N
        
        X_resampled = X_train[resample_idx]
        
        m = GaussianHMM(n_components=chosen_k, covariance_type="diag", n_iter=2000, random_state=42)
        try:
            m.fit(X_resampled)
            # Predict on the ORIGINAL training set to compare with baseline predictions
            preds = m.predict(X_train)
            ari = adjusted_rand_score(baseline_preds, preds)
            ari_scores_boot.append(ari)
        except Exception as e:
            print(f"Bootstrap fit failed: {e}")
            
    if ari_scores_boot:
        ari_boot = np.array(ari_scores_boot)
        print("\n--- Bootstrap Stability (20 resamples, block=60) ---")
        print(f"Mean ARI: {ari_boot.mean():.4f}")
        print(f"Min ARI:  {ari_boot.min():.4f}")
        print(f"Max ARI:  {ari_boot.max():.4f}")
        diff_sols_boot = np.sum(ari_boot < 0.90)
        print(f"Fraction converging to materially different solution (ARI < 0.90): {diff_sols_boot/len(ari_scores_boot):.2%}")
        
if __name__ == "__main__":
    main()
