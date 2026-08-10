from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(file_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def dataframe_overview(dataframe: pd.DataFrame) -> dict[str, object]:
    return {
        "shape": dataframe.shape,
        "columns": list(dataframe.columns),
        "missing_values": dataframe.isna().sum().to_dict(),
        "summary": dataframe.describe(include="all").to_dict(),
    }


def filter_rows(dataframe: pd.DataFrame, column: str, minimum_value: float) -> pd.DataFrame:
    return dataframe[dataframe[column] > minimum_value]


def select_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.select_dtypes(include="number")

