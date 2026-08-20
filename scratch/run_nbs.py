import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
notebooks = ['notebooks/01_eda_and_features.ipynb', 'notebooks/02_model_development.ipynb', 'notebooks/03_backtest_and_risk.ipynb']

for nb_path in notebooks:
    print(f"Executing {nb_path}...")
    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)
    try:
        ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"SUCCESS: {nb_path}")
    except Exception as e:
        print(f"ERROR executing {nb_path}:\n{e}")
