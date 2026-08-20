import pytest
import pandas as pd
import os

@pytest.fixture
def sample_spx():
    path = os.path.join(os.path.dirname(__file__), 'data', 'sample_spx.csv')
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    return df

@pytest.fixture
def sample_vix():
    path = os.path.join(os.path.dirname(__file__), 'data', 'sample_vix.csv')
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    return df
