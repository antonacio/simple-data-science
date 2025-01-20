import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .constants import RANDOM_SEED


def search_kmeans(df_cl: pd.DataFrame, max_n_clusters: int) -> pd.DataFrame:

    kmeans_search_lst = []

    for i in tqdm(range(1, max_n_clusters + 1)):
        kmeans_dict = dict()
        kmeans_dict["n_clusters"] = i
        kmeans_model = KMeans(n_clusters=i, verbose=0, random_state=RANDOM_SEED)
        kmeans_model.fit(df_cl)
        # save within cluster sum of squares
        kmeans_dict["wcss"] = kmeans_model.inertia_
        if i > 1:
            # save silhouette score
            kmeans_dict["silhouette_score"] = silhouette_score(df_cl, kmeans_model.labels_)

        kmeans_search_lst.append(kmeans_dict)

    # consolidate results in a dataframe
    df_kmeans = pd.DataFrame(kmeans_search_lst)

    return df_kmeans


def plot_kmeans_search(
    df_kmeans: pd.DataFrame,
    elbow: int,
    title: str = "K-means Clustering",
    figsize: tuple[int] = (9, 7),
):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        sharex=True,
    )

    plt.suptitle(title)
    color_lst = sns.color_palette()

    for ax, plot_col, plot_title in zip(
        axes,
        ["wcss", "silhouette_score"],
        ["Within Cluster Sum of Squared Distances (WCSS)", "Silhouette Score"],
    ):

        ax.plot(
            df_kmeans["n_clusters"].values,
            df_kmeans[plot_col].values,
            color=color_lst[0],
            marker="o",
            linestyle="--",
            zorder=2,
        )
        ax.scatter(
            elbow,
            df_kmeans.loc[(df_kmeans["n_clusters"] == elbow), plot_col],
            marker="o",
            s=250,
            color=color_lst[1],
            label=f"Elbow Method's optimal\nnumber of clusters (n={elbow})",
            alpha=0.75,
            zorder=1,
        )
        ax.set_title(plot_title)
        ax.set_xticks(df_kmeans["n_clusters"].values)
        ax.set_ylabel(None)

    axes[0].legend()
    axes[1].set_xlabel("Number of clusters")

    fig.tight_layout()

    return fig


def plot_cluster_boxplots(
    df: pd.DataFrame,
    cluster_col: str,
    plot_cols: list[str] = None,
    plots_per_line: int = 2,
    display_order: list[str] = None,
    title: str = "Features by Cluster",
    share_y_axis: bool = False,
    y_lim: list[float | int] = None,
    scale_factor: float = 1.5,
):
    n_clusters = df[cluster_col].nunique()

    if plot_cols is None:
        plot_cols = [col for col in df.columns if col != cluster_col]
    num_lines = int(np.ceil(len(plot_cols) / plots_per_line))
    fig, axes = plt.subplots(
        nrows=num_lines,
        ncols=plots_per_line,
        figsize=(n_clusters * plots_per_line * scale_factor, num_lines * scale_factor * 2),
        sharey=share_y_axis,
    )
    axes_flattend = axes.flatten()

    plt.suptitle(title, y=1)
    color_lst = sns.color_palette()

    if display_order is None:
        display_order = np.sort(df[cluster_col].unique()).tolist()

    for ax, col in zip(axes_flattend, plot_cols):
        sns.boxplot(
            x=df[cluster_col],
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
