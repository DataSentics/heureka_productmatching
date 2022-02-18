import os
import logging
import pandas as pd
import typing as t


def load_data_from_paths(dataset_paths: t.List[str]):

    if not dataset_paths:
        return pd.DataFrame()

    dfs = [
        pd.read_csv(os.path.join(path, "xgboostmatching_dataset.csv"), index_col=False)
        for path in dataset_paths
    ]
    data = pd.concat([df for df in dfs if not df.empty])
    empty_dataframes = [dataset_paths[i] for i, df in enumerate(dfs) if df.empty]

    logging.info(f"Loaded {len(data)} rows from {dataset_paths}, found empty dataframes: {empty_dataframes}")

    return data