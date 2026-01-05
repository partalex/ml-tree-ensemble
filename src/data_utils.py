import pandas as pd
import numpy as np


def load_csv(
        path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads CSV where the last column is the target variable.
    Args:
        path (str): Path to the CSV file.
    Returns:
        tuple[np.ndarray, np.ndarray]: Features and labels as numpy arrays.
    """
    df: pd.DataFrame = pd.read_csv(path, header=None)
    features: np.ndarray = df.iloc[:, :-1].to_numpy(dtype=float)
    labels: np.ndarray = df.iloc[:, -1].to_numpy(dtype=int)
    return features, labels
