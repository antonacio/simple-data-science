import os
import pandas as pd
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
