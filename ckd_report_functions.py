"""Hilfsfunktionen für den reproduzierbaren CKD-Analysebericht.

Die Funktionen kapseln Datenimport, Bereinigung, Visualisierung und
Modellbewertung. Das zugehörige Jupyter Notebook enthält dadurch nur kurze,
gut lesbare Funktionsaufrufe.

Datenquelle
-----------
Chronic Kidney Disease, UCI Machine Learning Repository
DOI: 10.24432/C5G020
Lizenz: Creative Commons Attribution 4.0 International (CC BY 4.0)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, fisher_exact
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NAVY = "#17324D"
TEAL = "#2A9D8F"
CKD_RED = "#C94F55"
NON_CKD_GREEN = "#4E9F70"
BLUE = "#3977A8"
PALE_BLUE = "#EAF3F7"
PALE_YELLOW = "#FFF4D6"
GRAY = "#5B6777"

NUMERIC_COLUMNS = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wbcc", "rbcc",
]

CATEGORICAL_COLUMNS = [
    "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane",
]

PARAMETER_LABELS = {
    "age": "Alter",
    "bp": "Blutdruck",
    "sg": "Spezifisches Uringewicht",
    "al": "Albumin im Urin",
    "su": "Zucker im Urin",
    "rbc": "Rote Blutkörperchen im Urin",
    "pc": "Eiterzellen",
    "pcc": "Eiterzellklumpen",
    "ba": "Bakterien",
    "bgr": "Blutzucker",
    "bu": "Blutharnstoff",
    "sc": "Serumkreatinin",
    "sod": "Natrium",
    "pot": "Kalium",
    "hemo": "Hämoglobin",
    "pcv": "Packed Cell Volume",
    "wbcc": "Leukozytenzahl",
    "rbcc": "Erythrozytenzahl",
    "htn": "Bluthochdruck",
    "dm": "Diabetes mellitus",
    "cad": "Koronare Herzkrankheit",
    "appet": "Appetit",
    "pe": "Periphere Ödeme",
    "ane": "Anämie",
}


def set_report_style() -> None:
    """Setzt eine Farbwelt passend zur Projektpräsentation."""

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5.5),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5DF",
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "font.size": 11,
            "legend.frameon": False,
        }
    )


def load_ckd_dataset(
    local_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Lädt den CKD-Datensatz lokal oder über ``ucimlrepo``.

    Parameters
    ----------
    local_csv:
        Optionaler Pfad zu einer CSV-Datei mit 24 Merkmalen und ``class``.

    Returns
    -------
    features, target, variable_information
    """

    if local_csv is not None:
        path = Path(local_csv)
        if path.exists():
            data = pd.read_csv(path)
            if "class" not in data.columns:
                raise ValueError("Die lokale CSV-Datei benötigt die Spalte 'class'.")
            return (
                data.drop(columns="class").copy(),
                data["class"].copy(),
                parameter_dictionary(),
            )

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as error:
        raise ImportError(
            "Installiere ucimlrepo mit: %pip install ucimlrepo"
        ) from error

    dataset = fetch_ucirepo(id=336)
    features = dataset.data.features.copy()
    target = dataset.data.targets.iloc[:, 0].copy()
    variables = dataset.variables.copy()
    return features, target, variables


def parameter_dictionary() -> pd.DataFrame:
    """Erzeugt eine kompakte Variablenbeschreibung."""

    rows = []
    for name in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS:
        rows.append(
            {
                "name": name,
                "description": PARAMETER_LABELS[name],
                "type": "numerisch" if name in NUMERIC_COLUMNS else "kategorial",
            }
        )
    rows.append({"name": "class", "description": "CKD-Status", "type": "Zielvariable"})
    return pd.DataFrame(rows)


def clean_ckd_data(
    features: pd.DataFrame,
    target: pd.Series | pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Bereinigt technische Schreibvarianten ohne medizinische Werte zu verändern."""

    cleaned = features.copy()

    for column in cleaned.columns:
        if column in NUMERIC_COLUMNS:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        else:
            text_values = (
                cleaned[column]
                .astype("string")
                .str.replace("\t", "", regex=False)
                .str.strip()
                .replace({"?": pd.NA, "": pd.NA})
            )
            # Scikit-learn verarbeitet np.nan in Objektspalten zuverlässig.
            cleaned[column] = text_values.astype(object).where(
                text_values.notna(), np.nan
            )

    if isinstance(target, pd.DataFrame):
        if target.shape[1] != 1:
            raise ValueError("Die Zielvariable muss genau eine Spalte enthalten.")
        target_series = target.iloc[:, 0].copy()
    else:
        target_series = target.copy()

    target_series = (
        target_series
        .reindex(cleaned.index)
        .astype("string")
        .str.replace("\t", "", regex=False)
        .str.strip()
        .str.lower()
        .replace(
            {
                "not ckd": "notckd",
                "non-ckd": "notckd",
                "nonckd": "notckd",
            }
        )
    )

    valid = target_series.isin(["ckd", "notckd"])
    if not valid.all():
        invalid = target_series.loc[~valid].drop_duplicates().tolist()
        raise ValueError(f"Unbekannte Zielklassen: {invalid}")

    return cleaned, target_series


def encode_target(target: pd.Series) -> pd.Series:
    """Kodiert ``notckd`` als 0 und ``ckd`` als 1."""

    encoded = target.map({"notckd": 0, "ckd": 1})
    if encoded.isna().any():
        raise ValueError("Die Zielvariable enthält unbekannte oder fehlende Klassen.")
    return encoded.astype(int)


def split_by_completeness(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Teilt die Merkmale in Total, Complete und Incomplete Cases."""

    total = features.copy()
    complete = total.loc[total.notna().all(axis=1)].copy()
    incomplete = total.loc[total.isna().any(axis=1)].copy()

    if len(total) != len(complete) + len(incomplete):
        raise RuntimeError("Complete und Incomplete ergeben nicht Total.")

    return total, complete, incomplete


def dataset_overview(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Liefert die zentralen Fallzahlen des Datensatzes."""

    complete_mask = features.notna().all(axis=1)
    return pd.DataFrame(
        {
            "Kennzahl": [
                "Patienten",
                "Eingabeparameter",
                "CKD",
                "Non-CKD",
                "Complete Cases",
                "Incomplete Cases",
                "Fehlende Einzelwerte",
            ],
            "Wert": [
                len(features),
                features.shape[1],
                int((target == "ckd").sum()),
                int((target == "notckd").sum()),
                int(complete_mask.sum()),
                int((~complete_mask).sum()),
                int(features.isna().sum().sum()),
            ],
        }
    )


def missingness_summary(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Beschreibt fehlende Werte insgesamt und getrennt nach CKD-Status."""

    result = pd.DataFrame(index=features.columns)
    result["Fehlend"] = features.isna().sum()
    result["Fehlend (%)"] = features.isna().mean() * 100

    for status, label in [("ckd", "CKD"), ("notckd", "Non-CKD")]:
        mask = target.eq(status)
        result[f"{label} fehlend"] = features.loc[mask].isna().sum()
        result[f"{label} fehlend (%)"] = features.loc[mask].isna().mean() * 100

    result.index.name = "Parameter"
    return result.sort_values("Fehlend", ascending=False)


def missingness_group_table(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Erstellt die 2x2-Tabelle aus Klasse und Vollständigkeit."""

    group = pd.Series(
        np.where(features.notna().all(axis=1), "Complete", "Incomplete"),
        index=features.index,
        name="Missingness",
    )
    table = pd.crosstab(target.rename("CKD-Status"), group)
    return table.reindex(
        index=["ckd", "notckd"],
        columns=["Complete", "Incomplete"],
        fill_value=0,
    )


def fisher_missingness_test(table: pd.DataFrame) -> pd.Series:
    """Testet den Zusammenhang zwischen CKD-Status und Vollständigkeit."""

    result = fisher_exact(table.to_numpy(), alternative="two-sided")
    return pd.Series(
        {
            "Odds Ratio": float(result.statistic),
            "p-Wert": float(result.pvalue),
        },
        name="Fisher-Exakt-Test",
    )


def plot_class_distribution(target: pd.Series) -> plt.Figure:
    """Zeigt die Klassenverteilung als Balkendiagramm."""

    counts = target.value_counts().reindex(["ckd", "notckd"])
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    bars = axis.bar(
        ["CKD", "Non-CKD"],
        counts.values,
        color=[CKD_RED, NON_CKD_GREEN],
        width=0.62,
    )
    for bar, count in zip(bars, counts.values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{count} ({count / counts.sum() * 100:.1f} %)",
            ha="center",
            color=NAVY,
            fontweight="bold",
        )
    axis.set_title("Klassenverteilung")
    axis.set_ylabel("Anzahl Patienten")
    axis.set_ylim(0, max(counts.values) * 1.18)
    sns.despine(ax=axis)
    figure.tight_layout()
    return figure


def plot_missingness_by_parameter(summary: pd.DataFrame, top_n: int = 24) -> plt.Figure:
    """Zeigt die Missingness pro Parameter absteigend sortiert."""

    plot_data = summary.head(top_n).sort_values("Fehlend (%)", ascending=True)
    figure, axis = plt.subplots(figsize=(10, 7))
    bars = axis.barh(
        plot_data.index,
        plot_data["Fehlend (%)"],
        color=CKD_RED,
        alpha=0.9,
    )
    for bar, value in zip(bars, plot_data["Fehlend (%)"]):
        axis.text(
            value + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f} %",
            va="center",
            color=NAVY,
            fontsize=9,
        )
    axis.set_title("Fehlende Werte pro Parameter")
    axis.set_xlabel("Anteil fehlender Werte")
    axis.set_ylabel("")
    axis.set_xlim(0, max(42, plot_data["Fehlend (%)"].max() + 7))
    sns.despine(ax=axis)
    figure.tight_layout()
    return figure


def plot_completeness_by_class(table: pd.DataFrame) -> plt.Figure:
    """Vergleicht Complete und Incomplete Cases innerhalb der Klassen."""

    proportions = table.div(table.sum(axis=1), axis=0) * 100
    plot_data = proportions.rename(index={"ckd": "CKD", "notckd": "Non-CKD"})

    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    plot_data[["Complete", "Incomplete"]].plot(
        kind="barh",
        stacked=True,
        color=[NON_CKD_GREEN, CKD_RED],
        ax=axis,
    )
    for row_index, (_, row) in enumerate(plot_data.iterrows()):
        axis.text(row["Complete"] / 2, row_index, f"{row['Complete']:.1f} %", ha="center", va="center", color="white", fontweight="bold")
        axis.text(row["Complete"] + row["Incomplete"] / 2, row_index, f"{row['Incomplete']:.1f} %", ha="center", va="center", color="white", fontweight="bold")
    axis.set_title("Vollständigkeit nach CKD-Status")
    axis.set_xlabel("Anteil innerhalb der Klasse")
    axis.set_ylabel("")
    axis.set_xlim(0, 100)
    axis.legend(title="", loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=2)
    figure.tight_layout()
    return figure


def plot_numeric_distributions(
    features: pd.DataFrame,
    target: pd.Series,
    columns: Iterable[str],
    discrete_columns: Iterable[str] = ("sg", "al", "su"),
    columns_per_row: int = 2,
) -> plt.Figure:
    """Vergleicht beobachtete Werte von CKD und Non-CKD pro Parameter."""

    columns = list(columns)
    discrete = set(discrete_columns)
    rows = int(np.ceil(len(columns) / columns_per_row))
    figure, axes = plt.subplots(rows, columns_per_row, figsize=(7 * columns_per_row, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    plot_frame = features.copy()
    plot_frame["CKD-Status"] = target.map({"ckd": "CKD", "notckd": "Non-CKD"})
    palette = {"CKD": CKD_RED, "Non-CKD": NON_CKD_GREEN}

    for axis, column in zip(axes, columns):
        values = plot_frame[[column, "CKD-Status"]].dropna()
        if column in discrete:
            relative = (
                values.groupby("CKD-Status")[column]
                .value_counts(normalize=True)
                .rename("Anteil")
                .reset_index()
            )
            sns.barplot(
                data=relative,
                x=column,
                y="Anteil",
                hue="CKD-Status",
                hue_order=["CKD", "Non-CKD"],
                palette=palette,
                ax=axis,
            )
            axis.set_ylabel("Relativer Anteil")
        else:
            sns.histplot(
                data=values,
                x=column,
                hue="CKD-Status",
                hue_order=["CKD", "Non-CKD"],
                palette=palette,
                stat="density",
                common_norm=False,
                element="step",
                fill=False,
                linewidth=2,
                bins="auto",
                ax=axis,
            )
            axis.set_ylabel("Dichte")

        missing_ckd = int(features.loc[target.eq("ckd"), column].isna().sum())
        missing_non = int(features.loc[target.eq("notckd"), column].isna().sum())
        axis.set_title(
            f"{column} · {PARAMETER_LABELS.get(column, column)}\n"
            f"Fehlend: CKD {missing_ckd}, Non-CKD {missing_non}"
        )
        axis.set_xlabel(column)

    for axis in axes[len(columns):]:
        axis.remove()

    figure.suptitle("Beobachtete Verteilungen nach CKD-Status", color=NAVY, fontsize=17, fontweight="bold", y=1.01)
    figure.tight_layout()
    return figure


def _column_lists(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [column for column in NUMERIC_COLUMNS if column in features.columns]
    categorical = [column for column in features.columns if column not in numeric]
    return numeric, categorical


def _one_hot_encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)


def build_random_forest_pipeline(
    features: pd.DataFrame,
    strategy: str = "median",
    n_estimators: int = 300,
    random_state: int = 42,
) -> Pipeline:
    """Erstellt eine Random-Forest-Pipeline für eine Missingness-Strategie.

    Unterstützte Strategien: ``median``, ``indicators``, ``knn`` und ``native``.
    """

    numeric, categorical = _column_lists(features)

    if strategy == "median":
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        categorical_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _one_hot_encoder()),
        ]
    elif strategy == "indicators":
        numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
        categorical_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            ("encoder", _one_hot_encoder()),
        ]
    elif strategy == "knn":
        numeric_steps = [
            ("scaler", StandardScaler()),
            ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
        ]
        categorical_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _one_hot_encoder()),
        ]
    elif strategy == "native":
        numeric_steps = [("native_missing", "passthrough")]
        categorical_steps = [("encoder", _one_hot_encoder())]
    else:
        raise ValueError("strategy muss median, indicators, knn oder native sein.")

    numeric_transformer: Pipeline | str
    if strategy == "native":
        numeric_transformer = "passthrough"
    else:
        numeric_transformer = Pipeline(numeric_steps)

    preprocessing = ColumnTransformer(
        [
            ("numerisch", numeric_transformer, numeric),
            ("kategorial", Pipeline(categorical_steps), categorical),
        ],
        sparse_threshold=0,
    )

    return Pipeline(
        [
            ("vorverarbeitung", preprocessing),
            (
                "modell",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_logistic_pipeline(
    features: pd.DataFrame,
    add_missing_indicators: bool = False,
    random_state: int = 42,
) -> Pipeline:
    """Erstellt eine L2-regularisierte logistische Vergleichspipeline."""

    numeric, categorical = _column_lists(features)
    preprocessing = ColumnTransformer(
        [
            (
                "numerisch",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median",
                                add_indicator=add_missing_indicators,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "kategorial",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent",
                                add_indicator=add_missing_indicators,
                            ),
                        ),
                        ("encoder", _one_hot_encoder()),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )

    return Pipeline(
        [
            ("vorverarbeitung", preprocessing),
            (
                "modell",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def missingness_features(features: pd.DataFrame) -> pd.DataFrame:
    """Wandelt ausschließlich das Fehlen eines Werts in 0/1-Spalten um."""

    result = features.isna().astype(int)
    result.columns = [f"{column}_missing" for column in result.columns]
    return result


def calculate_binary_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float | int]:
    """Berechnet Klassifikationsmetriken mit CKD als positiver Klasse."""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Sensitivität": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Spezifität": specificity,
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_probability),
        "PR-AUC": average_precision_score(y_true, y_probability),
        "True Negative": int(tn),
        "False Positive": int(fp),
        "False Negative": int(fn),
        "True Positive": int(tp),
    }


def evaluate_model_loocv(
    model,
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str,
    n_jobs: int = -1,
) -> tuple[pd.Series, pd.DataFrame]:
    """Bewertet eine komplette Pipeline mit Leave-One-Out-Cross-Validation."""

    probabilities = cross_val_predict(
        estimator=model,
        X=features,
        y=target,
        cv=LeaveOneOut(),
        method="predict_proba",
        n_jobs=n_jobs,
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = pd.Series(
        {"Modell": model_name, **calculate_binary_metrics(target, predictions, probabilities)}
    )
    patient_results = pd.DataFrame(
        {
            "Tatsächliche Klasse": target,
            "CKD-Wahrscheinlichkeit": probabilities,
            "Vorhersage": predictions,
            "Korrekt": predictions == np.asarray(target),
        },
        index=features.index,
    )
    return metrics, patient_results


def run_loocv_benchmark(
    features: pd.DataFrame,
    target: pd.Series,
    n_estimators: int = 300,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Führt den vollständigen, rechenintensiven LOOCV-Vergleich aus."""

    models = {
        "RF Median/Modus": build_random_forest_pipeline(features, "median", n_estimators),
        "RF + Indikatoren": build_random_forest_pipeline(features, "indicators", n_estimators),
        "RF KNN/Modus": build_random_forest_pipeline(features, "knn", n_estimators),
        "RF Native Missing": build_random_forest_pipeline(features, "native", n_estimators),
        "LogReg Median/Modus": build_logistic_pipeline(features, False),
        "LogReg + Indikatoren": build_logistic_pipeline(features, True),
    }

    rows = []
    predictions = {}
    for name, model in models.items():
        metrics, patient_results = evaluate_model_loocv(
            model, features, target, name, n_jobs=n_jobs
        )
        rows.append(metrics)
        predictions[name] = patient_results

    return pd.DataFrame(rows).reset_index(drop=True), predictions


def run_missingness_only_loocv(
    features: pd.DataFrame,
    target: pd.Series,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Prüft, wie gut ausschließlich das Dokumentationsmuster klassifiziert."""

    X_missing = missingness_features(features)
    models = {
        "LogReg nur Missingness": LogisticRegression(
            class_weight="balanced", solver="liblinear", max_iter=5000, random_state=42
        ),
        "RF nur Missingness": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=1
        ),
    }
    rows = []
    for name, model in models.items():
        metrics, _ = evaluate_model_loocv(model, X_missing, target, name, n_jobs=n_jobs)
        rows.append(metrics)
    return pd.DataFrame(rows).reset_index(drop=True)


def reference_loocv_results() -> pd.DataFrame:
    """Dokumentiert die im Projekt ausgeführten LOOCV-Ergebnisse."""

    return pd.DataFrame(
        [
            {"Modell": "RF Median/Modus", "Balanced Accuracy": 0.991, "Fehler": 3},
            {"Modell": "RF + Indikatoren", "Balanced Accuracy": 0.998, "Fehler": 1},
            {"Modell": "RF KNN/Modus", "Balanced Accuracy": 0.984, "Fehler": 6},
            {"Modell": "RF Native Missing", "Balanced Accuracy": 1.000, "Fehler": 0},
            {"Modell": "LogReg Median/Modus", "Balanced Accuracy": 0.992, "Fehler": np.nan},
            {"Modell": "LogReg + Indikatoren", "Balanced Accuracy": 0.998, "Fehler": np.nan},
            {"Modell": "LogReg nur Missingness", "Balanced Accuracy": 0.820, "Fehler": np.nan},
            {"Modell": "RF nur Missingness", "Balanced Accuracy": 0.833, "Fehler": np.nan},
        ]
    )


def plot_model_comparison(results: pd.DataFrame) -> plt.Figure:
    """Zeigt die Balanced Accuracy der verglichenen Strategien."""

    plot_data = results.sort_values("Balanced Accuracy", ascending=True)
    colors = [
        TEAL if "Missingness" in name and "nur" in name else BLUE
        for name in plot_data["Modell"]
    ]
    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.barh(plot_data["Modell"], plot_data["Balanced Accuracy"] * 100, color=colors)
    for bar, value in zip(bars, plot_data["Balanced Accuracy"]):
        axis.text(value * 100 + 0.3, bar.get_y() + bar.get_height() / 2, f"{value * 100:.1f} %", va="center", color=NAVY, fontweight="bold")
    axis.set_title("Interne Modellleistung in der LOOCV")
    axis.set_xlabel("Balanced Accuracy")
    axis.set_xlim(75, 102)
    axis.axvline(100, color=CKD_RED, linestyle="--", linewidth=1)
    sns.despine(ax=axis)
    figure.tight_layout()
    return figure


def reference_feature_importance() -> pd.DataFrame:
    """Liefert die aggregierten Importances aus dem RF mit Indikatoren."""

    return pd.DataFrame(
        {
            "Merkmal": ["hemo", "pcv", "sc", "sg", "htn", "rbcc", "dm", "al", "bu"],
            "Importance": [0.145, 0.124, 0.107, 0.091, 0.084, 0.081, 0.056, 0.050, 0.031],
        }
    ).sort_values("Importance", ascending=False, ignore_index=True)


def plot_feature_importance(importance: pd.DataFrame) -> plt.Figure:
    """Zeigt Feature Importances absteigend nach Bedeutung."""

    plot_data = importance.sort_values("Importance", ascending=True)
    figure, axis = plt.subplots(figsize=(9, 5.7))
    bars = axis.barh(plot_data["Merkmal"], plot_data["Importance"] * 100, color=BLUE)
    for bar, value in zip(bars, plot_data["Importance"]):
        axis.text(value * 100 + 0.25, bar.get_y() + bar.get_height() / 2, f"{value * 100:.1f} %", va="center", color=NAVY, fontweight="bold")
    axis.set_title("Relative Bedeutung im Random Forest")
    axis.set_xlabel("Anteil an der gesamten Feature Importance")
    axis.set_ylabel("")
    axis.set_xlim(0, max(17.5, plot_data["Importance"].max() * 100 + 3))
    sns.despine(ax=axis)
    figure.tight_layout()
    return figure


def mcnemar_exact_from_discordant(
    first_only_correct: int,
    second_only_correct: int,
) -> float:
    """Berechnet den exakten zweiseitigen McNemar-p-Wert."""

    discordant = first_only_correct + second_only_correct
    if discordant == 0:
        return 1.0
    return float(
        binomtest(
            min(first_only_correct, second_only_correct),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )


def statistical_summary() -> pd.DataFrame:
    """Fasst die im Projekt berichteten statistischen Kontrollen zusammen."""

    return pd.DataFrame(
        [
            {
                "Prüfung": "Fisher: Missingness und CKD",
                "Ergebnis": "OR = 0,063; p = 6,65 × 10⁻³³",
                "Aussage": "Vollständigkeit und CKD-Status sind stark assoziiert.",
            },
            {
                "Prüfung": "McNemar: Native gegen Indikatoren",
                "Ergebnis": "0 gegen 1 abweichender Fehler; p = 1,00",
                "Aussage": "Ein abweichender Fall belegt keinen generellen Modellunterschied.",
            },
            {
                "Prüfung": "McNemar: Native gegen Median",
                "Ergebnis": "p = 0,25",
                "Aussage": "Kein belastbarer Unterschied.",
            },
            {
                "Prüfung": "McNemar: Native gegen KNN",
                "Ergebnis": "p = 0,031; Holm-korrigiert p = 0,094",
                "Aussage": "Nach Korrektur für mehrere Vergleiche nicht signifikant.",
            },
            {
                "Prüfung": "Exaktes 95-%-Intervall für 400/400",
                "Ergebnis": "Accuracy ungefähr 0,991 bis 1,000",
                "Aussage": "Perfekte interne Vorhersage garantiert keine perfekte Zukunftsleistung.",
            },
        ]
    )


def medical_interpretation_table() -> pd.DataFrame:
    """Formuliert die wichtigsten Modellmerkmale für Nichtmediziner."""

    return pd.DataFrame(
        [
            ["sc · Kreatinin", "höher", "Gesunde Nieren filtern Kreatinin aus dem Blut. Ein hoher Wert kann auf eine eingeschränkte Nierenfunktion hinweisen.", "zentral"],
            ["al · Albumin im Urin", "höher", "Albumin gelangt normalerweise kaum in den Urin. Größere Mengen können auf geschädigte Nierenfilter hinweisen.", "zentral"],
            ["hemo, pcv, rbcc", "niedriger", "Erkrankte Nieren können zu wenig Signal für die Bildung roter Blutkörperchen geben. Dadurch kann Blutarmut entstehen.", "plausible Folge"],
            ["htn, dm", "häufiger", "Bluthochdruck und Diabetes können die feinen Blutgefäße der Nieren langfristig schädigen.", "Risikofaktoren"],
            ["bu · Blutharnstoff", "höher", "Wenn die Nieren schlechter arbeiten, kann sich mehr Harnstoff im Blut ansammeln. Andere Faktoren beeinflussen den Wert ebenfalls.", "unterstützend"],
            ["sg · Uringewicht", "niedriger", "Der Wert zeigt, wie gut die Nieren den Urin konzentrieren. Ein niedriger Wert kann auf eine eingeschränkte Konzentrationsfähigkeit hinweisen.", "ergänzend"],
        ],
        columns=["Merkmal", "Muster bei CKD", "Einfache medizinische Erklärung", "Bedeutung"],
    )


__all__ = [
    "CKD_RED",
    "NON_CKD_GREEN",
    "NUMERIC_COLUMNS",
    "CATEGORICAL_COLUMNS",
    "set_report_style",
    "load_ckd_dataset",
    "clean_ckd_data",
    "encode_target",
    "parameter_dictionary",
    "split_by_completeness",
    "dataset_overview",
    "missingness_summary",
    "missingness_group_table",
    "fisher_missingness_test",
    "plot_class_distribution",
    "plot_missingness_by_parameter",
    "plot_completeness_by_class",
    "plot_numeric_distributions",
    "build_random_forest_pipeline",
    "build_logistic_pipeline",
    "missingness_features",
    "evaluate_model_loocv",
    "run_loocv_benchmark",
    "run_missingness_only_loocv",
    "reference_loocv_results",
    "plot_model_comparison",
    "reference_feature_importance",
    "plot_feature_importance",
    "mcnemar_exact_from_discordant",
    "statistical_summary",
    "medical_interpretation_table",
]
