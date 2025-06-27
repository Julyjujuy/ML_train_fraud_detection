import os
import pandas as pd
from pathlib import Path

#Paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROC_DIR = PROJECT_DIR / "data" / "processed"
SRC_DATA = PROJECT_DIR /"dataset"

def copy_raw():
    """copy CSVs from /dataset into /data/raw"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for csv in SRC_DATA.glob("*.csv"):
        dest = RAW_DIR / csv.name
        if not dest.exists():
            dest.write_bytes(csv.read_bytes())
    print(f"Copied {len(list(SRC_DATA.glob('*.csv')))} files to {RAW_DIR}")
    
def load_dataframe(filename: str) -> pd.DataFrame:
    """Load one of the raw CSVs from data/raw"""
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run copy_raw() first.")
    return pd.read_csv(path)
if __name__ == "__main__":
    copy_raw()
    #Quick sanity check
    for fname in ["creditcard.csv", "PaySim_Synthetic_Mobile-Money_Simulator_dataset.csv"]:
        df = load_dataframe(fname)
        print(f"{fname}: {df.shape} rows×cols")
