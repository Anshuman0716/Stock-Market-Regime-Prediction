import os
import re

def grep_repo():
    search_dirs = ["notebooks", "app.py", "src/models", "src/data", "src/backtest", "src/db", "src/stats"]
    
    feature_patterns = [r"\bta\.rsi\b", r"\bta\.macd\b", r"\bta\.atr\b", r"\bta\.bbands\b", r"\bta\.obv\b", r"\.ewm\(", r"\.rolling\("]
    hmm_patterns = [r"\bGaussianHMM\b"]
    
    hits = []
    
    # 1. Search for feature math outside src/features/
    for root_dir in search_dirs:
        if root_dir == "app.py":
            files = ["app.py"]
            dir_path = "."
        else:
            dir_path = root_dir
            files = []
            for d, _, fs in os.walk(dir_path):
                for f in fs:
                    if f.endswith('.py') or f.endswith('.ipynb'):
                        files.append(os.path.join(d, f))
                        
        for fpath in files:
            full_path = fpath if root_dir == "app.py" else fpath
            if not os.path.exists(full_path):
                continue
                
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Check features
            norm_path = os.path.normpath(full_path)
            if not norm_path.startswith(os.path.normpath("src/features")):
                for p in feature_patterns:
                    matches = re.finditer(p, content)
                    for m in matches:
                        start = max(0, content.rfind("\n", 0, m.start()))
                        end = content.find("\n", m.end())
                        if end == -1: end = len(content)
                        line = content[start:end].strip()
                        # Allow rolling in labeling.py
                        if "labeling.py" in norm_path and ".rolling(" in line:
                            continue
                        hits.append(f"FEATURE HIT in {full_path}: {line}")
                        
            # Check HMM outside src/models
            if not norm_path.startswith(os.path.normpath("src/models")):
                for p in hmm_patterns:
                    matches = re.finditer(p, content)
                    for m in matches:
                        start = max(0, content.rfind("\n", 0, m.start()))
                        end = content.find("\n", m.end())
                        if end == -1: end = len(content)
                        line = content[start:end].strip()
                        hits.append(f"HMM HIT in {full_path}: {line}")

    if not hits:
        print("Static check PASSED: No duplicate implementations found.")
    else:
        print("Static check FAILED: Found duplicates:")
        for h in hits:
            print(h)

if __name__ == "__main__":
    grep_repo()
