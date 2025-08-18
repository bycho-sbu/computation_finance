import pandas as pd
import numpy as np

# internal helper: read a whitespace- or csv-delimited file and return a univariate series
def _read_target_series(file_path: str) -> pd.Series:
    """
    reads the file at file_path using flexible whitespace/csv parsing and returns the LAST column
    as a numeric pandas Series named 'value'. preserves row order and attempts to parse the first
    column as dates for the index (falls back to RangeIndex if parsing fails).
    """
    # try whitespace first; fall back to comma-separated if needed
    try:
        df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    except Exception:
        df = pd.read_csv(file_path, header=None)

    # choose last column as the target series
    y = pd.to_numeric(df.iloc[:, -1], errors="coerce")

    # try to use first column as datetime index if it parses like YYYYMMDD (or similar)
    idx = None
    try:
        idx = pd.to_datetime(df.iloc[:, 0].astype(str), format="%Y%m%d", errors="raise")
    except Exception:
        try:
            idx = pd.to_datetime(df.iloc[:, 0], errors="raise")
        except Exception:
            idx = pd.RangeIndex(start=0, stop=len(df), step=1)

    s = pd.Series(y.values, index=idx, name="value").dropna()
    # ensure chronological order if a datetime index was created
    if isinstance(s.index, pd.DatetimeIndex):
        s = s.sort_index()
    return s


def static_split(file_path: str, train_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a time series dataset stored in a file into training and testing sets based on the specified fraction.

    Parameters:
    - file_path (str): Path to the file containing the time series data (whitespace- or csv-delimited).
    - train_frac (float): Fraction of observations to allocate to the training set (0 < train_frac < 1).

    Returns:
    - (train_df, test_df): Two DataFrames with a single column 'value' and the original index.
      Indices are half-open slices [0:cut) for train and [cut:n) for test in row order.

    Raises:
    - ValueError: If train_frac is not between 0 and 1, or if the series is empty.
    """
    if not (0 < train_frac < 1):
        raise ValueError("train_frac must be between 0 and 1.")
    s = _read_target_series(file_path)
    if len(s) == 0:
        raise ValueError("no valid numeric observations found in the target column.")

    cut = int(len(s) * train_frac)
    if cut <= 0 or cut >= len(s):
        raise ValueError("train_frac results in an empty train or test split; choose a different value.")

    train_df = s.iloc[:cut].to_frame(name="value").copy()
    test_df = s.iloc[cut:].to_frame(name="value").copy()
    return train_df, test_df


def rolling_window_splits(file_path: str, train_size: int, test_size: int, slide_size: int = 1) -> list[tuple[int, int, int, int]]:
    """
    Generates index references for rolling-window train/test splits using half-open integer indices.

    Parameters:
    - file_path (str): Path to the file containing the time series data.
    - train_size (int): Number of observations in each training window (>0).
    - test_size (int): Number of observations in each test window (>0).
    - slide_size (int, optional): Step size to slide the window forward each iteration (default 1, must be >0).

    Returns:
    - List of tuples (start_train, end_train, start_test, end_test), where slices are half-open:
      train = [start_train:end_train), test = [start_test:end_test)

    Notes:
    - Windows are contiguous: start_test == end_train for each split.
    - The function does NOT read the file repeatedly during modeling; it only determines n from the file once.
    """
    if train_size <= 0 or test_size <= 0 or slide_size <= 0:
        raise ValueError("train_size, test_size, and slide_size must be positive integers.")

    n = len(_read_target_series(file_path))
    idx_refs: list[tuple[int, int, int, int]] = []

    start_train = 0
    while start_train + train_size + test_size <= n:
        end_train = start_train + train_size
        start_test = end_train
        end_test = start_test + test_size
        idx_refs.append((start_train, end_train, start_test, end_test))
        start_train += slide_size

    return idx_refs


def expanding_window_splits(file_path: str, initial_train_size: int, test_size: int, expansion_step: int = 1) -> list[tuple[int, int, int, int]]:
    """
    Generates index references for expanding-window train/test splits using half-open integer indices.

    Parameters:
    - file_path (str): Path to the file containing the time series data.
    - initial_train_size (int): Initial number of observations in the training set (>0).
    - test_size (int): Number of observations in the test set for each split (>0).
    - expansion_step (int, optional): How many observations to add to the training set each step (default 1, >0).

    Returns:
    - List of tuples (start_train, end_train, start_test, end_test) with half-open slices.

    Notes:
    - Training window always starts at 0 and expands by expansion_step each iteration.
    - Splits are generated while there are enough remaining observations to form a full test window.
    """
    if initial_train_size <= 0 or test_size <= 0 or expansion_step <= 0:
        raise ValueError("initial_train_size, test_size, and expansion_step must be positive integers.")

    n = len(_read_target_series(file_path))
    idx_refs: list[tuple[int, int, int, int]] = []

    end_train = initial_train_size
    while end_train + test_size <= n:
        start_train = 0
        start_test = end_train
        end_test = start_test + test_size
        idx_refs.append((start_train, end_train, start_test, end_test))
        end_train += expansion_step

    return idx_refs
