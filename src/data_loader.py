import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Loads the diabetes dataset from disk.
    """
    return pd.read_csv(path)
