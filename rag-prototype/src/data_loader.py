from pathlib import Path
import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "leetcode_problems.csv"
)

def load_problems() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded Problems : {len(df)}")
    print(f"Number of Columns : {len(df.columns)}")

    return df