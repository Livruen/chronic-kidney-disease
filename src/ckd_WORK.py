"""Datenimport und technische Grundbereinigung für die CKD-EDA."""

from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wbcc", "rbcc",
]


def load_ckd_dataset():
    """Lädt zuerst die mitgelieferte CSV und nutzt UCI nur als Fallback."""

    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "chronic_kidney_disease.csv"

    if csv_path.exists():
        data = pd.read_csv(csv_path)
        data_raw = data.drop(columns="class").copy()
        classes = data[["class"]].copy()

        parameters = pd.DataFrame(
            {
                "name": [*data_raw.columns, "class"],
                "description": [*data_raw.columns, "CKD-Status"],
            }
        )
    else:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as error:
            raise ImportError(
                "CSV nicht gefunden. Installiere ucimlrepo mit "
                "'%pip install -r requirements.txt'."
            ) from error

        dataset = fetch_ucirepo(id=336)
        data_raw = dataset.data.features.copy()
        classes = dataset.data.targets.copy()
        parameters = dataset.variables.copy()

    classes.iloc[:, 0] = (
        classes.iloc[:, 0]
        .astype("string")
        .str.replace("\t", "", regex=False)
        .str.strip()
        .str.lower()
    )

    parameter_names = parameters[["name", "description"]].copy()
    return data_raw, classes, parameters, parameter_names


def clean_ckd_data(data_raw: pd.DataFrame) -> pd.DataFrame:
    """Bereinigt nur technische Schreibvarianten, nicht medizinische Werte."""

    data_clean = data_raw.copy()

    for column in data_clean.columns:
        if column in NUMERIC_COLUMNS:
            data_clean[column] = pd.to_numeric(
                data_clean[column],
                errors="coerce",
            )
        else:
            values = (
                data_clean[column]
                .astype("string")
                .str.replace("\t", "", regex=False)
                .str.strip()
                .replace({"?": pd.NA, "": pd.NA})
            )
            data_clean[column] = values.astype(object).where(
                values.notna(),
                np.nan,
            )

    return data_clean


def split_by_completeness(data_raw: pd.DataFrame):
    """Teilt die Daten in Total, Complete und Incomplete Cases."""

    total = data_raw.copy()
    complete = total.loc[total.notna().all(axis=1)].copy()
    incomplete = total.loc[total.isna().any(axis=1)].copy()

    if len(total) != len(complete) + len(incomplete):
        raise RuntimeError("Complete und Incomplete ergeben nicht Total.")

    return total, complete, incomplete

