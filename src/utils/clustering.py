import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .constants import RANDOM_SEED


def search_kmeans(df_cl: pd.DataFrame, max_n_clusters: int) -> pd.DataFrame:

    kmeans_search_lst = []

    for i in range(1, max_n_clusters + 1):
        kmeans_dict = dict()
        kmeans_dict["n_clusters"] = i
        kmeans_model = KMeans(n_clusters=i, verbose=0, random_state=RANDOM_SEED)
        kmeans_model.fit(df_cl)
        # save within cluster sum of squares
        kmeans_dict["wcss"] = kmeans_model.inertia_
        if i > 1:
            # save silhouette score
            kmeans_dict["silhouette_score"] = silhouette_score(
                df_cl, kmeans_model.labels_, random_state=RANDOM_SEED
            )

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
    plt.close(fig)

    return fig
