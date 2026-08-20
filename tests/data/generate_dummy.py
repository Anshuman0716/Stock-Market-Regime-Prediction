import pandas as pd
import numpy as np
import os

os.makedirs('tests/data', exist_ok=True)

# Generate ~1500 days of data (enough for min_train_days=1008)
np.random.seed(42)
dates = pd.date_range(start='2015-01-01', periods=1500, freq='B')
close = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.01, size=len(dates))))
high = close * (1 + np.abs(np.random.normal(0.005, 0.002, size=len(dates))))
low = close * (1 - np.abs(np.random.normal(0.005, 0.002, size=len(dates))))
open_p = close * (1 + np.random.normal(0, 0.002, size=len(dates)))
volume = np.random.randint(1000000, 10000000, size=len(dates))

spx = pd.DataFrame({'Open': open_p, 'High': high, 'Low': low, 'Close': close, 'Volume': volume}, index=dates)
spx.index.name = 'Date'
spx.to_csv('tests/data/sample_spx.csv')

vix_close = 15 + np.random.normal(0, 2, size=len(dates))
vix_close = np.clip(vix_close, 10, 80)
vix = pd.DataFrame({'Close': vix_close}, index=dates)
vix.index.name = 'Date'
vix.to_csv('tests/data/sample_vix.csv')
