import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
from .constants import REPO_NAME, SMALL_FONTSIZE, MEDIUM_FONTSIZE, BIG_FONTSIZE, FIGURE_DPI


def get_repo_root_path() -> str:
    return os.path.normpath(os.getcwd().split(REPO_NAME, maxsplit=1)[0] + REPO_NAME)


def get_data_folder_path() -> str:
    repo_path = get_repo_root_path()
    data_path = os.path.normpath(os.path.join(repo_path, "data"))
    return data_path


def convert_to_integer(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, downcast="integer", errors="raise")


def _set_plot_font_sizes() -> None:
    plt.rc("font", size=SMALL_FONTSIZE)  # default font size
    plt.rc("figure", titlesize=BIG_FONTSIZE)  # figure title
    plt.rc("legend", fontsize=SMALL_FONTSIZE)  # legend
    plt.rc("axes", titlesize=MEDIUM_FONTSIZE)  # axes title
    plt.rc("axes", labelsize=SMALL_FONTSIZE)  # axes labels
    plt.rc("xtick", labelsize=SMALL_FONTSIZE)  # x tick labels
    plt.rc("ytick", labelsize=SMALL_FONTSIZE)  # y tick labels


def _set_figure_dpi() -> None:
    mpl.rcParams["figure.dpi"] = FIGURE_DPI


def set_plotting_config() -> None:
    _set_plot_font_sizes()
    _set_figure_dpi()


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
    plt.close(fig)

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

    plt.close(fig)

    return fig


def _bring_last_n_items_to_front(lst: list, n: int) -> list:
    return [*lst[-n:], *lst[:-n]]


def _build_histogram(
    ax: plt.Axes,
    df: pd.DataFrame,
    plot_col: str,
    display_name: str = None,
    display_unit: str = None,
    legend_label: str = None,
    stratify_col: str = None,
    bin_size: int = None,
    linewidth: int = 1.5,
    histogram_opacity: float = 0.75,
    color: str = None,
    show_legend: bool = True,
    show_percentage: bool = False,
    show_mean: bool = True,
    show_median: bool = False,
    show_zero_line: bool = False,
    show_kde: bool = False,
) -> plt.Axes:

    display_name = display_name or plot_col
    if stratify_col:
        hist_label = None
    else:
        hist_label = legend_label or display_name

    ax = sns.histplot(
        df,
        x=plot_col,
        kde=show_kde,
        ax=ax,
        bins=np.arange(
            (df[plot_col].min() // bin_size) * bin_size, (df[plot_col].max() + bin_size), bin_size
        ),
        color=color,
        alpha=histogram_opacity,
        line_kws=dict(linewidth=linewidth),
        label=hist_label,
        stat="percent" if show_percentage else "count",
        hue=stratify_col,
        multiple="stack" if stratify_col else "layer",
        zorder=1,
    )

    if show_percentage:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.set_ylabel("Percentage")
    else:
        ax.set_ylabel("Count")
    # add thousands separator to x-axis
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel(display_name + (f" ({display_unit})" if display_unit else ""))

    if show_mean or show_median:
        s = df[plot_col]

        if not color:
            color_lst = sns.color_palette()
            lines_color_idx = df[stratify_col].nunique() if stratify_col else 1
            lines_color = color_lst[lines_color_idx]
        else:
            lines_color = color

        def format_line_label(
            name: str,
            value: float,
            legend_label: str = legend_label,
            display_unit: str = display_unit,
        ) -> str:
            lines_label = f"{legend_label}: " if legend_label else ""
            unit = f" {display_unit}" if display_unit else ""
            return f"{lines_label}{name} ({value:,.0f}{unit})"

        if show_mean is True:
            # Plot vertical line for mean
            mean = s.mean()
            ax.axvline(
                mean,
                linestyle="-",
                linewidth=linewidth,
                color=lines_color,
                label=format_line_label("Mean", mean),
                zorder=3,
            )
        if show_median is True:
            # Plot dotted vertical line for median
            median = s.median()
            ax.axvline(
                median,
                linestyle="--",
                linewidth=linewidth,
                color=lines_color,
                label=format_line_label("Median", median),
                zorder=4,
            )

    if show_zero_line is True:
        # Plot vertical line in zero
        ax.axvline(0, linestyle=":", linewidth=linewidth / 2, color="black", label=None, zorder=2)

    if show_legend is True:
        # configure legend
        legend = ax.get_legend()
        handles, labels = ax.get_legend_handles_labels()
        if legend:
            ax.legend(
                handles=list(legend.get_patches()) + handles,
                labels=[
                    f"{stratify_col.title()}: {txt.get_text().title()}"
                    for txt in legend.get_texts()
                ]
                + labels,
                title=None,
                fontsize=SMALL_FONTSIZE,
            )
        else:
            # bring last legend item (histogram) to the front
            n = 1
            ax.legend(
                handles=_bring_last_n_items_to_front(lst=handles, n=n),
                labels=_bring_last_n_items_to_front(lst=labels, n=n),
                title=None,
                fontsize=SMALL_FONTSIZE,
            )

    return ax


def plot_histogram(
    df: pd.DataFrame,
    plot_col: str,
    title: str,
    histogram_title: str = None,
    figsize: tuple = (8, 6),
    **kwargs: dict,
) -> plt.Figure:

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle(title)
    if histogram_title:
        ax.set_title(histogram_title)

    ax = _build_histogram(ax=ax, df=df, plot_col=plot_col, **kwargs)

    plt.close(fig)

    return fig


def plot_comparison_histograms(
    title: str,
    left_title: str,
    right_title: str,
    df: pd.DataFrame,
    plot_col_before: str,
    plot_col_after: str,
    plot_col_diff: str,
    display_name: str,
    figsize: tuple = (14, 6),
    **kwargs: dict,
) -> plt.Figure:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title)

    # LEFT PLOT (before and after)
    # before plot
    before_plot_kwargs = kwargs.copy()
    before_plot_kwargs.update(
        dict(
            ax=ax1,
            df=df,
            plot_col=plot_col_before,
            histogram_opacity=0.5,
            color="red",
            legend_label="Before",
            display_name=display_name,
            show_legend=False,
        )
    )
    _build_histogram(**before_plot_kwargs)
    # after plot
    after_plot_kwargs = kwargs.copy()
    after_plot_kwargs.update(
        dict(
            ax=ax1,
            df=df,
            plot_col=plot_col_after,
            histogram_opacity=0.5,
            color="blue",
            legend_label="After",
            display_name=display_name,
            show_legend=False,
        )
    )
    _build_histogram(**after_plot_kwargs)
    ax1.set_title(left_title)
    # configure legend
    handles, labels = ax1.get_legend_handles_labels()
    # bring last 2 legend items (histogram) to the front
    n = 2
    ax1.legend(
        handles=_bring_last_n_items_to_front(lst=handles, n=n),
        labels=_bring_last_n_items_to_front(lst=labels, n=n),
        title=None,
        fontsize=SMALL_FONTSIZE,
    )

    # RIGHT PLOT (difference)
    _build_histogram(
        ax=ax2,
        df=df,
        plot_col=plot_col_diff,
        histogram_opacity=0.5,
        color="green",
        legend_label="Difference",
        display_name=display_name,
        show_zero_line=True,
        show_legend=True,
        **kwargs,
    )
    ax2.set_title(right_title)

    plt.tight_layout()
    plt.close(fig)

    return fig
