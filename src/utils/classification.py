import math
import numpy as np
import pandas as pd
import shap
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Any
from matplotlib import patches as mpatches
from matplotlib import ticker as mticker

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression, LinearRegression

from .common import convert_to_integer


def describe_input_features(
    df_input: pd.DataFrame,
    df_input_train: pd.DataFrame,
    df_input_test: pd.DataFrame,
) -> pd.DataFrame:
    df_describe = df_input.describe().T
    df_describe["count"] = convert_to_integer(df_describe["count"])
    df_describe["null_count"] = df_input.isna().sum()
    df_describe["data_type"] = df_input.dtypes.astype(str).apply(
        lambda x: "numeric" if any([tp in x for tp in ["int", "float"]]) else "categorical"
    )
    # reorder columns
    df_describe = df_describe[
        ["data_type", "count", "null_count", "min", "25%", "50%", "75%", "max", "std", "mean"]
    ]
    df_describe["mean_train"] = df_input_train.mean()
    df_describe["mean_test"] = df_input_test.mean()
    df_describe["train_test_pct_diff"] = (
        df_describe["mean_test"] - df_describe["mean_train"]
    ) / df_describe["mean_train"]

    return df_describe


def plot_roc_curve(
    y_true: pd.Series,
    y_pred_proba: pd.Series,
    title: str = "Receiver Operating Characteristic",
    figsize: tuple[int, int] = (8, 5),
    ret_optimal_thresh: bool = False,
) -> plt.Figure | tuple[plt.Figure, np.float64]:
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    optimal_thresh = thresholds[np.argmax(tpr - fpr)]
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=figsize)
    color_lst = sns.color_palette()
    margin = 5
    plt.title(title)
    plt.plot(
        fpr * 100,
        tpr * 100,
        color=color_lst[0],
        ls="-",
        lw=1,
        label=f"ROC Curve (AUC: {roc_auc:.3f})",
    )
    plt.plot(
        [0, 100],
        [0, 100],
        color=color_lst[1],
        ls="--",
        lw=0.5,
        label="Random Classifier",
    )
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    if ret_optimal_thresh:
        plt.vlines(
            x=100 * optimal_thresh,
            ymin=-margin,
            ymax=100 + margin,
            color=color_lst[2],
            ls="--",
            label=f"Optimal Threshold: {100*optimal_thresh:.1f}%",
        )
    plt.legend(loc="lower right", framealpha=1)
    ax = plt.gca()
    ax.set_xlim([-margin, 100 + margin])
    ax.set_ylim([-margin, 100 + margin])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())

    if ret_optimal_thresh:
        return fig, optimal_thresh
    else:
        return fig


def compute_metrics(
    y_true: pd.Series, y_pred: pd.Series, y_pred_proba: pd.Series
) -> dict[str, float]:
    metrics_dict = dict()

    if len(np.unique(y_true)) > 1:
        metrics_dict["KS"] = compute_ks_score(y_true=y_true, y_pred_proba=y_pred_proba)
        roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred_proba)
        metrics_dict["ROC AUC"] = roc_auc
        metrics_dict["GINI"] = 2 * roc_auc - 1
    else:
        metrics_dict["KS"] = np.nan
        metrics_dict["ROC AUC"] = np.nan
        metrics_dict["GINI"] = np.nan
    metrics_dict["Accuracy"] = accuracy_score(y_true=y_true, y_pred=y_pred)
    metrics_dict["Precision"] = precision_score(y_true=y_true, y_pred=y_pred)
    metrics_dict["Recall"] = recall_score(y_true=y_true, y_pred=y_pred)
    metrics_dict["F1 Score"] = f1_score(y_true=y_true, y_pred=y_pred)

    return metrics_dict


def _get_logit_stderror_pvalues(
    model: LogisticRegression, x: pd.DataFrame | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate z-scores for scikit-learn LogisticRegression.
    This function uses asymtptics for maximum likelihood estimates.

    Source: https://stackoverflow.com/a/47079198

    parameters:
        model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
        x:     matrix on which the model was fit
    """
    p = model.predict_proba(x)
    n = len(p)
    m = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis=1))
    ans = np.zeros((m, m))
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i, 1] * p[i, 0]
    vcov = np.linalg.inv(np.matrix(ans))
    se = np.sqrt(np.diag(vcov))
    t = coefs / se
    p_values = (1 - scipy.stats.norm.cdf(abs(t))) * 2

    return se, p_values


def build_coefficients_table(
    model: LogisticRegression | LinearRegression,
    X_train_std: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(model, (LogisticRegression, LinearRegression)):
        raise ValueError(
            "Model must be either sklearn's Linear Regression or Logistic Regression. "
            f"Got {type(model)} instead."
        )

    df_coef = pd.DataFrame(
        np.transpose(model.coef_), columns=["Coefficients"], index=model.feature_names_in_
    )
    df_coef["Absolute Coefficients"] = df_coef["Coefficients"].abs()
    # compute Standard Error and coefficients' p-values
    stderr, pvalues = _get_logit_stderror_pvalues(model, X_train_std)
    df_coef["Standard Error"] = stderr[1:]  # [1:] to skip constant
    df_coef["95% CI"] = df_coef["Standard Error"] * 1.96
    df_coef["p-values"] = pvalues[1:]  # [1:] to skip constant

    df_coef = df_coef.sort_values(by="Absolute Coefficients", ascending=False)

    return df_coef


def _get_order_of_magnitude(number: float | int) -> float:
    return math.floor(math.log(number, 10))


def plot_coefficients_values(
    df_coef: pd.DataFrame,
) -> plt.Figure:

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle("Coefficient Values with 95% CI (±1.96 Std Error)")
    max_coeff, max_ci = df_coef[["Absolute Coefficients", "95% CI"]].max().tolist()

    if _get_order_of_magnitude(max_ci) > _get_order_of_magnitude(max_coeff):
        # limit x axis range as CI is too large
        ax.set_xlim([-1, max_coeff * 1.5])

    colors_dict = {"Positive": "royalblue", "Negative": "crimson"}
    df_plot = df_coef.sort_values(by="Absolute Coefficients", ascending=True)
    ax = df_plot["Absolute Coefficients"].plot(
        kind="barh",
        color=df_plot.apply(
            lambda row: (
                colors_dict["Negative"] if row["Coefficients"] < 0 else colors_dict["Positive"]
            ),
            axis=1,
        ),
        figsize=(10, max(df_plot.shape[0] / 2, 4)),
        legend=False,
        ax=ax,
        xerr=df_plot["95% CI"],
        ecolor="black",
        error_kw={"label": "95% confidence interval", "capsize": 4, "capthick": 1},
    )

    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    legend_patches = [
        mpatches.Patch(color=colors_dict["Positive"], label="Positive coefficient"),
        mpatches.Patch(color=colors_dict["Negative"], label="Negative coefficient"),
        ax.get_legend_handles_labels()[0][0],  # confidence interval
    ]
    plt.legend(handles=legend_patches, loc="lower right", framealpha=1)

    return fig


def plot_coefficients_significance(
    df_coef: pd.DataFrame,
    alpha: float = 0.05,
    log_scale: bool = False,
) -> plt.Figure:

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(
        "Coefficients' Statistical Significance " f"({100*(1 - alpha):.0f}% Confidence Level)"
    )

    colors_dict = {"fail": "orange", "pass": "limegreen", "threshold": "crimson"}
    df_plot = df_coef.sort_values(by="Absolute Coefficients", ascending=True)
    ax = df_plot["p-values"].plot(
        kind="barh",
        color=df_plot.apply(
            lambda row: colors_dict["pass"] if row["p-values"] < alpha else colors_dict["fail"],
            axis=1,
        ),
        figsize=(10, max(df_plot.shape[0] / 2, 4)),
        legend=False,
        ax=ax,
        label=None,
    )
    if log_scale:
        ax.set_xscale("log")
    # add vertical line at alpha
    ax.vlines(
        x=alpha,
        ymin=-1,
        ymax=len(df_plot),
        colors=colors_dict["threshold"],
        ls="--",
        lw=2,
        alpha=0.75,
        label=f"{100*(1 - alpha):.0f}% Confidence Level",
    )

    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    legend_patches = [
        ax.get_legend_handles_labels()[0][0],  # confidence level line
        mpatches.Patch(color=colors_dict["pass"], label="Coefficient is statistically significant"),
        mpatches.Patch(
            color=colors_dict["fail"], label="Coefficient is not statistically significant"
        ),
    ]
    plt.legend(handles=legend_patches, framealpha=0.75)

    return fig


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    estimator: Any,
    target_classes_dict: dict,
    title: str = "Confusion matrix",
    normalize: str = None,
    figsize: tuple[int, int] = (6, 4),
) -> plt.Figure:

    target_labels = [target_classes_dict[i] for i in estimator.classes_]

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    if normalize is not None:
        cm_pct = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=normalize)

    fig = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    plt.title(title, pad=20)
    plt.colorbar(format="{x:,.0f}")

    tick_marks = np.arange(len(target_labels))
    plt.xticks(
        ticks=tick_marks,
        labels=["\n".join(lb.rsplit(" ")) for lb in target_labels],
        rotation=0,
    )
    plt.yticks(
        ticks=tick_marks,
        labels=["\n".join(lb.rsplit(" ")) for lb in target_labels],
    )

    half_threshold = cm.sum() // 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            s = f"{cm[i, j]:,.0f}"
            if normalize is not None:
                s += f"\n({(cm_pct[i, j] * 100):.1f}%)"
            plt.text(
                x=j,
                y=i,
                s=s,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > half_threshold else "black",
            )

    plt.grid(False)
    plt.tight_layout()
    plt.ylabel("True label", labelpad=10)
    plt.xlabel("Predicted label", labelpad=15)

    return fig


def build_ks_table(
    y_true: pd.Series | np.ndarray,
    y_pred_proba: pd.Series | np.ndarray,
    n_bins: int = 10,
    ret_ks: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.float64]:

    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    if isinstance(y_pred_proba, pd.Series):
        y_pred_proba = y_pred_proba.values

    df = pd.DataFrame()
    df["score"] = y_pred_proba
    df["positive"] = y_true
    df["negative"] = 1 - y_true

    df["bucket"] = pd.qcut(df["score"].rank(method="first"), q=n_bins)

    ks_table = (
        df.groupby("bucket", as_index=True, observed=True)
        .agg(
            min_score=("score", "min"),
            max_score=("score", "max"),
            n_positives=("positive", "sum"),
            n_negatives=("negative", "sum"),
        )
        .reset_index(drop=True)
    )

    ks_table["n_all"] = ks_table["n_positives"] + ks_table["n_negatives"]

    ks_table["positive_rate"] = ks_table["n_positives"] / ks_table["n_all"]
    ks_table["negative_rate"] = ks_table["n_negatives"] / ks_table["n_all"]

    ks_table["cum_positives"] = ks_table["n_positives"].cumsum()
    ks_table["cum_negatives"] = ks_table["n_negatives"].cumsum()
    ks_table["cumpct_positives"] = ks_table["cum_positives"] / ks_table["n_positives"].sum()
    ks_table["cumpct_negatives"] = ks_table["cum_negatives"] / ks_table["n_negatives"].sum()

    ks_table["diff"] = np.abs(ks_table["cumpct_positives"] - ks_table["cumpct_negatives"])

    ks = ks_table["diff"].max()

    if ret_ks:
        return ks_table, ks
    else:
        return ks_table


def beautify_ks_table(ks_table: pd.DataFrame) -> pd.DataFrame:
    ks_table = ks_table.copy()

    def flag(x):
        return "<--" if x == ks_table["diff"].max() else ""

    ks_table["KS"] = ks_table["diff"].apply(flag)

    for pct_col in ["positive_rate", "negative_rate", "cumpct_positives", "cumpct_negatives"]:
        ks_table[pct_col] = ks_table[pct_col].apply("{0:.2%}".format)
    for pp_col in ["diff"]:
        ks_table[pp_col] = ks_table[pp_col].apply(lambda x: f"{100*x:.2f} pp")

    ks_table.columns = [col.replace("_", " ") for col in ks_table.columns]

    return ks_table


def compute_ks_score(
    y_true: pd.Series | np.ndarray,
    y_pred_proba: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> np.float64:

    _, ks = build_ks_table(y_true=y_true, y_pred_proba=y_pred_proba, n_bins=n_bins, ret_ks=True)

    return ks


def plot_ks_table(ks_table: pd.DataFrame, figsize: tuple[int, int] = (7, 5)) -> plt.Figure:

    # Plot the KS Gain Chart
    df_plot_ks = (
        ks_table[["max_score", "cumpct_negatives", "cumpct_positives", "diff"]] * 100
    ).set_index("max_score")
    # add point zero
    df_plot_ks = pd.concat(
        [
            pd.DataFrame(
                data=0, index=[0], columns=["cumpct_negatives", "cumpct_positives", "diff"]
            ),
            df_plot_ks,
        ],
        axis=0,
    )
    df_plot_ks.sort_index(inplace=True)

    # Create a figure and axis instance
    color_lst = sns.color_palette()
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the cumulative distributions
    ax.plot(df_plot_ks["cumpct_negatives"], label="Cumulative Negative", color=color_lst[0])
    ax.plot(df_plot_ks["cumpct_positives"], label="Cumulative Positive", color=color_lst[1])

    # set axis limites
    margin = 5
    ax.set_xlim(-margin, 100 + margin)
    ax.set_ylim(-margin, 100 + margin)

    ks_argmax = df_plot_ks["diff"].argmax()
    ks_max = df_plot_ks["diff"].max()
    ax.axvline(
        df_plot_ks.index[ks_argmax],
        ymin=(margin + df_plot_ks["cumpct_positives"].iloc[ks_argmax]) / (100 + 2 * margin),
        ymax=(margin + df_plot_ks["cumpct_negatives"].iloc[ks_argmax]) / (100 + 2 * margin),
        color=color_lst[2],
        linestyle="--",
        linewidth=1.5,
        label=f"Max KS Statistic ({ks_max:.1f})",
    )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    # Customize the plot
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Cumulative Percentage")
    ax.set_title(f"KS Gain Plot (KS Statistic = {ks_max:.3f})")
    ax.legend()
    ax.grid(True)

    return fig


def plot_eval_metrics_xgb(eval_results, eval_metrics) -> plt.Figure:
    n_epochs = len(eval_results["validation_0"][list(eval_metrics.keys())[0]])

    fig, axes = plt.subplots(
        nrows=1, ncols=len(eval_metrics.keys()), figsize=(7 * len(eval_metrics.keys()), 5)
    )
    for ax, (metric_code, metric) in zip(axes, eval_metrics.items()):
        ax.plot(range(n_epochs), eval_results["validation_0"][metric_code], label="Train")
        ax.plot(range(n_epochs), eval_results["validation_1"][metric_code], label="Test")
        ax.set_title(metric)
        ax.set_xlabel("Iterations")
        ax.legend()
    plt.suptitle("Convergence during Model Training", y=1.05)

    return fig


def plot_shap_importance(
    shap_values: np.ndarray, title: str = "SHAP Feature Importance", **kwargs: dict
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, max(shap_values.values.shape[1] / 2, 4)))
    ax.set_title(title, pad=15)
    shap.plots.bar(shap_values, ax=ax, **kwargs)

    return fig


def plot_shap_beeswarm(
    shap_values: np.ndarray, title: str = "SHAP Summary Plot", **kwargs: dict
) -> plt.Figure:
    ax = shap.plots.beeswarm(
        shap_values, show=False, plot_size=(6, max(shap_values.values.shape[1] / 2, 4)), **kwargs
    )
    ax.set_title(title, pad=15)
    fig = plt.gcf()

    return fig


def plot_gain_metric_xgb(
    xgb_estimator: Any,
    X_test_: pd.DataFrame,
    title: str = "XGBoost Feature Importance (Gain metric)",
) -> plt.Figure:
    df_xgb_gain = pd.DataFrame(
        xgb_estimator.feature_importances_, index=X_test_.columns, columns=["Feature Gain"]
    )
    fig, ax = plt.subplots(figsize=(8, max(len(df_xgb_gain) / 2, 3)))
    ax = df_xgb_gain.sort_values("Feature Gain", ascending=True).plot(
        kind="barh", legend=False, ax=ax
    )
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    plt.title(title)

    return fig


def plot_target_rate(
    y_test: pd.Series,
    y_pred_proba: pd.Series,
    title: str = "Target rate per group of predicted probability",
) -> plt.Figure:

    df_gh = pd.concat(
        [
            y_test.rename("true_label"),
            y_pred_proba.rename("pred_proba"),
            # quartis
            pd.qcut(
                y_pred_proba, q=4, labels=[f"Q{i}" for i in range(1, 4 + 1)], duplicates="raise"
            ).rename("pred_quartile"),
            # decis
            pd.qcut(
                y_pred_proba, q=10, labels=[f"D{i}" for i in range(1, 10 + 1)], duplicates="raise"
            ).rename("pred_decile"),
        ],
        axis=1,
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    fig.suptitle(title, y=1.025)

    for ax, groupby_col, plot_title in zip(
        axes,
        ["pred_quartile", "pred_decile"],
        ["Quartiles of model's predicted probability", "Deciles of model's predicted probability"],
    ):
        ax.set_title(plot_title)
        ax = (
            100
            * (
                df_gh.groupby(groupby_col, observed=True).agg(
                    taxa_pgto_parcela=("true_label", "mean")
                )
            )
        ).plot(kind="bar", legend=False, rot=0, ax=ax)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_xlabel("")

    return fig
