import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .constants import REPO_NAME, SMALL_FONTSIZE, MEDIUM_FONTSIZE, BIG_FONTSIZE


def get_repo_root_path() -> str:
    return os.path.normpath(os.getcwd().split(REPO_NAME, maxsplit=1)[0] + REPO_NAME)


def get_data_folder_path() -> str:
    repo_path = get_repo_root_path()
    data_path = os.path.normpath(os.path.join(repo_path, "data"))
    return data_path


def convert_to_integer(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, downcast="integer", errors="raise")


def set_plot_font_sizes() -> None:
    plt.rc("font", size=SMALL_FONTSIZE)  # default font size
    plt.rc("figure", titlesize=BIG_FONTSIZE)  # figure title
    plt.rc("legend", fontsize=SMALL_FONTSIZE)  # legend
    plt.rc("axes", titlesize=MEDIUM_FONTSIZE)  # axes title
    plt.rc("axes", labelsize=SMALL_FONTSIZE)  # axes labels
    plt.rc("xtick", labelsize=SMALL_FONTSIZE)  # x tick labels
    plt.rc("ytick", labelsize=SMALL_FONTSIZE)  # y tick labels


def plot_boxplot_by_class(
    df_input: pd.DataFrame,
    class_col: str,
    class_mapping: dict = None,
    plot_cols: list[str] = None,
    plots_per_line: int = 2,
    display_order: list[str] = None,
    title: str = "Features by Class",
    share_y_axis: bool = False,
    y_lim: list[float | int] = None,
    scale_factor: float = 1.5,
) -> plt.Figure:

    df = df_input.copy()
    n_classes = df[class_col].nunique()

    if class_mapping is not None:
        df[class_col] = df[class_col].map(class_mapping)
    if plot_cols is None:
        plot_cols = [col for col in df.columns if col != class_col]
    num_lines = int(np.ceil(len(plot_cols) / plots_per_line))
    fig, axes = plt.subplots(
        nrows=num_lines,
        ncols=plots_per_line,
        figsize=(n_classes * plots_per_line * scale_factor, num_lines * scale_factor * 2),
        sharey=share_y_axis,
    )
    axes_flattend = axes.flatten()

    plt.suptitle(title, y=1)
    color_lst = sns.color_palette()

    if display_order is None:
        display_order = np.sort(df[class_col].unique()).tolist()

    for ax, col in zip(axes_flattend, plot_cols):
        sns.boxplot(
            x=df[class_col],
            y=df[col],
            order=display_order,
            ax=ax,
            fliersize=2,
            color=color_lst[0],
            medianprops=dict(linewidth=2, alpha=1.0),
            flierprops=dict(markerfacecolor="black", marker=".", alpha=0.33),
            showmeans=True,
            meanprops=dict(
                marker=5,
                markerfacecolor=color_lst[1],
                markeredgecolor=color_lst[1],
                markersize=10,
            ),
        )
        ax.set_title(col)
        ax.set_ylabel("")
        ax.set_xlabel("")
        if y_lim is not None:
            y_range = max(y_lim) - min(y_lim)
            pct_margin = 0.01
            ax.set_ylim(
                ymin=(min(y_lim) - y_range * pct_margin), ymax=(max(y_lim) + y_range * pct_margin)
            )

    # delete unused axes
    for ax in axes_flattend[len(plot_cols) :]:
        fig.delaxes(ax=ax)

    fig.tight_layout()

    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    title: str = "Features' Correlation",
    method: str = "pearson",
    fig_height: int = 8,
    annot_fontsize: int = 10,
) -> plt.Figure:
    # Compute features' correlation
    df_corr = df.corr(method=method)
    # Generate a mask to onlyshow the bottom triangle
    mask_corr = ~np.triu(np.ones_like(df_corr, dtype=bool)).T

    with sns.axes_style("whitegrid"):
        fig = plt.figure(figsize=(fig_height, fig_height))
        plt.title(title)

        # generate heatmap
        sns.heatmap(
            df_corr,
            cmap="YlGnBu",
            annot=True,
            mask=mask_corr,
            vmin=-1,
            vmax=1,
            square=True,
            annot_kws=dict(fontsize=annot_fontsize),
            fmt=".2f",
        )
        plt.grid(False)
        plt.xticks(rotation=45, ha="right")

    return fig
