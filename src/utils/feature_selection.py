import logging
import pandas as pd
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)


def _get_high_vif_features(
    X: pd.DataFrame, threshold: int, break_threshold: int = 1e6
) -> list[str]:
    features = list(X.columns)
    max_len = max([len(f) for f in features])
    high_vif_feats = []

    logger.info(f"Computing the Variance Inflation Factor (VIF) for {len(features)} features...")

    count = 1
    max_vif = threshold
    while max_vif >= threshold:
        max_vif = 0
        for i, feat in enumerate(features):
            with warnings.catch_warnings(action="ignore"):
                vif_feat = variance_inflation_factor(X[features].values, i)
            if vif_feat > max_vif:
                max_vif = vif_feat
                max_vif_idx = i
                max_vif_feat = feat
                # break for loop before checking all features to save time
                if vif_feat > break_threshold:
                    break

        if max_vif > threshold:
            high_vif_feat = features.pop(max_vif_idx)
            assert high_vif_feat == max_vif_feat
            high_vif_feats.append(high_vif_feat)
            logger.info(
                f'{(str(count) + ".").rjust(4)} Removing feature: '
                f'{(high_vif_feat + " ").ljust(max_len+2, ".")} VIF: {max_vif:,.2f}'
            )
            count += 1
        else:
            max_vif_col = features.pop(max_vif_idx)
            logger.info(
                f'  >> Stopping at feat: {(max_vif_col + " ").ljust(max_len+2, ".")} '
                f"VIF: {max_vif:,.2f}  (threshold: {threshold:,})"
            )

    return high_vif_feats


def _run_manual_filter(df: pd.DataFrame, target_col: str, params: dict) -> list[str]:
    orig_shp = df.shape
    cols_to_exclude = params["cols_to_exclude"]

    if not isinstance(cols_to_exclude, list):
        cols_to_exclude = []
    elif len(cols_to_exclude) > 0:
        # cannot remove target
        if target_col in cols_to_exclude:
            cols_to_exclude = [col for col in cols_to_exclude if col != target_col]

    logger.info(
        f" - Removing {len(cols_to_exclude)} "
        f"({100 * len(cols_to_exclude) / (orig_shp[1] - 1):.2f}%) feature(s) manually..."
    )
    return cols_to_exclude


def _run_variance_filter(df: pd.DataFrame, target_col: str, params: dict) -> list[str]:
    orig_shp = df.shape
    var_threshold = params["threshold"]

    s_var = df.drop(columns=[target_col]).var(axis=0)
    low_var_cols = s_var[s_var <= var_threshold].index.tolist()

    logger.info(
        f" - Removing {len(low_var_cols):,} ({100 * len(low_var_cols) / (orig_shp[1] - 1):.1f}%)"
        f" feature(s) with variance <= {var_threshold} ..."
    )

    return low_var_cols


def _run_correlation_filter(df: pd.DataFrame, target_col: str, params: dict) -> list[str]:
    orig_shp = df.shape
    corr_threshold = params["threshold"]

    # compute pearson correlation with target
    df_feats = df.drop(columns=[target_col])
    s_corr_target = df_feats.corrwith(df[target_col])
    # rank features based on correlation with target (from worst to best)
    ranked_feats = s_corr_target.dropna().abs().sort_values(ascending=True).index.tolist()

    logger.info(f"  Running Correlation filter with threshold of {corr_threshold}")
    high_corr_cols = []
    for feat in ranked_feats:
        s_corr_feat = df_feats.drop(columns=[feat]).corrwith(df_feats[feat]).dropna()
        feat_max_corr = s_corr_feat.abs().max()
        feat_idxmax_corr = s_corr_feat.abs().idxmax()

        if feat_max_corr > corr_threshold:
            logger.info(
                f" - Removing feature '{feat}' with correlation "
                f"{s_corr_feat.loc[feat_idxmax_corr]:+.4f} to '{feat_idxmax_corr}'"
            )
            high_corr_cols.append(feat)
            df_feats = df_feats.drop(columns=[feat])

    logger.info(
        f" - Removing {len(high_corr_cols):,} "
        f"({100 * len(high_corr_cols) / (orig_shp[1] - 1):.1f}%) feature(s) with "
        f"abs(correlation) > {corr_threshold} ..."
    )

    return high_corr_cols


def _run_vif_filter(df: pd.DataFrame, target_col: str, params: dict) -> list[str]:
    orig_shp = df.shape

    vif_threshold = params["threshold"]
    high_vif_feats = _get_high_vif_features(df.drop(columns=[target_col]), threshold=vif_threshold)
    logger.info(
        f" - Removing {len(high_vif_feats):,} "
        f"({100 * len(high_vif_feats) / (orig_shp[1] - 1):.1f}%)"
        f" feature(s) with VIF >= {vif_threshold:,.0f} ..."
    )

    return high_vif_feats


def run_feature_selection_steps(
    df_input: pd.DataFrame, target_col: str, fs_steps: dict
) -> tuple[list[str], pd.DataFrame]:
    # define available filter functions
    fs_functions = {
        "manual": _run_manual_filter,
        "variance": _run_variance_filter,
        "correlation": _run_correlation_filter,
        "vif": _run_vif_filter,
    }
    # check if provided steps are valid
    for filter_name, filter_params in fs_steps.items():
        if filter_name not in fs_functions.keys():
            raise ValueError(
                f"Filter name must be one of {list(fs_functions.keys())}. "
                f"Got {filter_name} instead"
            )

    # run feature selection steps
    logger.info(
        "--> Starting the Feature Selection process with "
        f"{df_input.drop(columns=[target_col]).shape[1]:,} features"
    )
    # build feature selection log table
    df_fs = pd.DataFrame(index=df_input.columns).assign(filter="", step=0)
    df = df_input.copy()
    for step, (filter_name, filter_params) in enumerate(fs_steps.items(), start=1):
        logger.info(f"{step}. {filter_name.upper()} FILTER")
        removed_feats = fs_functions[filter_name](df, target_col, params=filter_params)
        df = df.drop(columns=removed_feats)
        df_fs.loc[removed_feats, ["filter", "step"]] = (filter_name, step)

    selected_feats = df.drop(columns=[target_col]).columns.tolist()
    logger.info(
        "--> Completed the Feature Selection process with "
        f"{len(selected_feats):,} selected features"
    )
    df_fs.loc[selected_feats, ["filter", "step"]] = ("Selected feature", -1)

    return selected_feats, df_fs
