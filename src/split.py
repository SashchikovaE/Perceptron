import pandas as pd
import numpy as np
from pathlib import Path


def split_data(input_file,
               train_file,
               val_file,
               test_file,
               val_size=0.2,
               test_size=0.2,
               seed=42):
    data_path = Path(__file__).parent.parent / 'data'
    df = pd.read_csv(data_path / input_file)

    df.columns = [str(i) for i in range(1, len(df.columns) + 1)]

    rng = np.random.default_rng(seed)

    n = len(df)
    idx = rng.permutation(n)

    n_test = int(n * test_size)
    n_val  = int(n * val_size)

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    train_df.to_csv(data_path / train_file, index=False, header=False)
    val_df.to_csv(data_path / val_file,     index=False, header=False)
    test_df.to_csv(data_path / test_file,   index=False, header=False)

