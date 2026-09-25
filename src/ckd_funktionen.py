"""Visualisierungsfunktion aus dem Explorationsnotebook."""

import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display


def plot_all_ckd_distributions(
    features: pd.DataFrame,
    classes: pd.DataFrame | pd.Series,
) -> pd.DataFrame:
    """Zeigt numerische Verteilungen und gibt die Missingness-Tabelle zurück."""

    if isinstance(classes, pd.DataFrame):
        target = classes.iloc[:, 0].copy()
    else:
        target = classes.copy()

    target = (
        target.reindex(features.index)
        .astype("string")
        .str.replace("\t", "", regex=False)
        .str.strip()
        .str.lower()
    )

    complete_mask = features.notna().all(axis=1)
    missing_group = complete_mask.map(
        {True: "Complete", False: "Incomplete"}
    )

    table = pd.crosstab(target, missing_group).reindex(
        index=["ckd", "notckd"],
        columns=["Complete", "Incomplete"],
        fill_value=0,
    )

    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    if numeric_columns:
        columns_per_row = 3
        rows = math.ceil(len(numeric_columns) / columns_per_row)
        figure, axes = plt.subplots(
            rows,
            columns_per_row,
            figsize=(16, 4 * rows),
        )
        axes = axes.ravel()

        plot_data = features.copy()
        plot_data["_ckd_status"] = target
        palette = {"ckd": "firebrick", "notckd": "seagreen"}

        for axis, column in zip(axes, numeric_columns):
            values = plot_data[[column, "_ckd_status"]].dropna()

            if column in {"sg", "al", "su"}:
                relative = (
                    values.groupby("_ckd_status")[column]
                    .value_counts(normalize=True)
                    .rename("Anteil")
                    .reset_index()
                )
                sns.barplot(
                    data=relative,
                    x=column,
                    y="Anteil",
                    hue="_ckd_status",
                    hue_order=["ckd", "notckd"],
                    palette=palette,
                    ax=axis,
                )
                axis.set_ylabel("Relativer Anteil")
            else:
                sns.histplot(
                    data=values,
                    x=column,
                    hue="_ckd_status",
                    hue_order=["ckd", "notckd"],
                    palette=palette,
                    stat="density",
                    common_norm=False,
                    element="step",
                    fill=False,
                    bins="auto",
                    ax=axis,
                )
                axis.set_ylabel("Dichte")

            axis.set_title(column)
            axis.set_xlabel(column)

        for axis in axes[len(numeric_columns):]:
            axis.remove()

        figure.suptitle(
            "Verteilungen der numerischen Parameter nach CKD-Status",
            fontsize=16,
            y=1.01,
        )
        figure.tight_layout()
        plt.show()

    display(table)
    return table
