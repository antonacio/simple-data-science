import logging
import pandas as pd
import numpy as np
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import root_mean_squared_error, f1_score
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def _remove_features_with_l1_regularization(
    X: pd.DataFrame,
    y: pd.Series,
    l1_params: dict,
) -> list[str]:

    # carregar parametros
    problem = l1_params["problem"]
    train_test_split_params = l1_params["train_test_split_params"]
    logspace_search = l1_params["logspace_search"]
    error_tolerance_pct = l1_params["error_tolerance_pct"]
    min_feats_to_keep = l1_params["min_feats_to_keep"]
    random_seed = l1_params["random_seed"]

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        **train_test_split_params,
        random_state=random_seed,
    )

    # Standardize X_train
    stdscaler = StandardScaler()
    X_train_std = pd.DataFrame(
        stdscaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_std = pd.DataFrame(stdscaler.transform(X_test), columns=X.columns, index=X_test.index)

    # define search space
    logspace_values = np.logspace(**logspace_search)
    coef_lst = []
    metrics_dict = dict()

    # define L1-based linear model ad its evaluation metric
    match problem.lower().strip():
        case "classification":
            LinearModel = LinearSVC
            model_params = dict(penalty="l1")
            search_arg = "C"
            eval_metric_fn = f1_score
            eval_metric_params = dict(average="weighted")
            eval_metric_greater_is_better = True
        case "regression":
            LinearModel = Lasso
            model_params = dict()
            search_arg = "alpha"
            eval_metric_fn = root_mean_squared_error
            eval_metric_params = dict()
            eval_metric_greater_is_better = False
        case _:
            raise ValueError(
                "Argument 'problem' must be either 'classification' or 'regression'. "
                f"Got {problem} instead."
            )

    for i, search_val in enumerate(logspace_values, start=1):
        # Fit model and make predictions
        model_params[search_arg] = search_val
        model = LinearModel(**model_params, random_state=random_seed)
        model.fit(X_train_std, y_train)
        y_pred = model.predict(X_test_std)
        eval_metric = eval_metric_fn(y_test, y_pred, **eval_metric_params)

        s_coef = pd.Series(data=np.mean(model.coef_, axis=0), index=model.feature_names_in_, name=i)
        coef_lst.append(s_coef)
        metrics_dict[i] = dict(
            search_val=search_val, n_zero_coefs=len(s_coef[s_coef == 0]), eval_metric=eval_metric
        )

    df_coef = pd.concat(coef_lst, axis=1)
    df_coef.columns.name = "iteration"
    df_coef.index.name = "feature"
    df_iter_metrics = pd.DataFrame.from_dict(metrics_dict, orient="index")
    df_iter_metrics.index.name = "iteration"

    # select the model that removes the most features while satisfying the following conditions:
    #  - the selected model's metric score must be within the specified tolerance with respect to
    #    the best score among all models
    #  - the number of removed features must not exceed the specified number
    if eval_metric_greater_is_better is True:
        best_metric_score = df_iter_metrics["eval_metric"].max()
        eval_metric_filter = df_iter_metrics["eval_metric"] > (
            best_metric_score * (1 - error_tolerance_pct)
        )
    else:
        best_metric_score = df_iter_metrics["eval_metric"].min()
        eval_metric_filter = df_iter_metrics["eval_metric"] < (
            best_metric_score * (1 + error_tolerance_pct)
        )
    min_feats_filter = (X.shape[1] - df_iter_metrics["n_zero_coefs"]) >= min_feats_to_keep
    df_iter_best = df_iter_metrics[eval_metric_filter & min_feats_filter]

    if len(df_iter_best) > 0:
        best_iter = df_iter_best["n_zero_coefs"].idxmax()
        s_coef_best_iter = df_coef[best_iter]
        l1_feats_to_drop = s_coef_best_iter[s_coef_best_iter == 0].index.tolist()
    else:
        l1_feats_to_drop = []

    return l1_feats_to_drop


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


def _run_manual_filter(X: pd.DataFrame, y: pd.Series, params: dict) -> list[str]:
    target_col = y.name
    orig_shp = X.shape
    cols_to_exclude = params["cols_to_exclude"]

    if not isinstance(cols_to_exclude, list):
        cols_to_exclude = [cols_to_exclude]
    elif len(cols_to_exclude) > 0:
        # cannot remove target
        if target_col in cols_to_exclude:
            cols_to_exclude = [col for col in cols_to_exclude if col != target_col]

    # check if any of the features to exclude are not in the dataframe
    if not set(cols_to_exclude).issubset(X.columns.tolist()):
        raise ValueError(
            "Some features to exclude in the 'manual_filter' are not present in the input table: "
            f"{set(cols_to_exclude) - set(X.columns.tolist())}"
        )

    logger.info(
        f" - Removing {len(cols_to_exclude)} ({100 * len(cols_to_exclude) / orig_shp[1]:.1f}%) "
        f"feature(s) manually: {cols_to_exclude}"
    )
    return cols_to_exclude


def _run_null_variance_filter(X: pd.DataFrame, y: pd.Series, params: dict) -> list[str]:
    orig_shp = X.shape

    s_var = X.var(axis=0)
    low_var_cols = s_var[s_var == 0].index.tolist()

    logger.info(
        f" - Removing {len(low_var_cols):,} ({100 * len(low_var_cols) / orig_shp[1]:.1f}%) "
        f"feature(s) with null variance (var == 0): {low_var_cols}"
    )

    return low_var_cols


def _run_correlation_filter(X: pd.DataFrame, y: pd.Series, params: dict) -> list[str]:
    orig_shp = X.shape
    corr_threshold = params["threshold"]

    # compute pearson correlation with target
    s_corr_target = X.corrwith(y)
    # rank features based on correlation with target (from worst to best)
    ranked_feats = s_corr_target.dropna().abs().sort_values(ascending=True).index.tolist()

    logger.info(f"  Running Correlation filter with threshold of {corr_threshold}")
    high_corr_cols = []
    for feat in ranked_feats:
        s_corr_feat = X.drop(columns=[feat]).corrwith(X[feat]).dropna()
        feat_max_corr = s_corr_feat.abs().max()
        feat_idxmax_corr = s_corr_feat.abs().idxmax()

        if feat_max_corr > corr_threshold:
            logger.info(
                f" - Removing feature '{feat}' with correlation "
                f"{s_corr_feat.loc[feat_idxmax_corr]:+.4f} to '{feat_idxmax_corr}'"
            )
            high_corr_cols.append(feat)
            X = X.drop(columns=[feat])

    logger.info(
        f" - Removing {len(high_corr_cols):,} ({100 * len(high_corr_cols) / orig_shp[1]:.1f}%) "
        f"feature(s) with abs(correlation) > {corr_threshold}"
    )

    return high_corr_cols


def _run_l1_filter(X: pd.DataFrame, y: pd.Series, params: dict) -> list[str]:
    orig_shp = X.shape
    l1_feats_to_drop = _remove_features_with_l1_regularization(X=X, y=y, l1_params=params)
    logger.info(
        f" - Removing {len(l1_feats_to_drop):,} ({100 * len(l1_feats_to_drop) / orig_shp[1]:.1f}%)"
        f" feature(s) with null coefficient after L1 regularization: {l1_feats_to_drop}"
    )

    return l1_feats_to_drop


def _run_vif_filter(X: pd.DataFrame, y: pd.Series, params: dict) -> list[str]:
    orig_shp = X.shape

    vif_threshold = params["threshold"]
    high_vif_feats = _get_high_vif_features(X=X, threshold=vif_threshold)
    logger.info(
        f" - Removing {len(high_vif_feats):,} ({100 * len(high_vif_feats) / orig_shp[1]:.1f}%) "
        f"feature(s) with VIF >= {vif_threshold:,.0f}"
    )

    return high_vif_feats


def run_feature_selection_steps(
    X: pd.DataFrame, y: pd.Series, fs_steps: dict
) -> tuple[list[str], pd.DataFrame]:

    # build feature selection log table
    target_col = y.name
    orig_shp = X.shape
    df_fs = pd.DataFrame(index=X.columns.tolist() + [target_col]).assign(filter="", step=0)

    # define available filter functions
    fs_functions = {
        "manual": _run_manual_filter,
        "null_variance": _run_null_variance_filter,
        "correlation": _run_correlation_filter,
        "l1_regularization": _run_l1_filter,
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
    logger.info(f"--> Starting Feature Selection with {orig_shp[1]:,} features")

    for step, (filter_name, filter_params) in enumerate(fs_steps.items(), start=1):
        logger.info(f"{step}. {filter_name.upper()} FILTER")
        removed_feats = fs_functions[filter_name](X=X.copy(), y=y.copy(), params=filter_params)
        X = X.drop(columns=removed_feats)
        df_fs.loc[removed_feats, ["filter", "step"]] = (filter_name, step)

    selected_feats = X.columns.tolist()
    logger.info(
        f"--> Completed Feature Selection with {len(selected_feats):,} selected features "
        f"({100 * len(selected_feats) / orig_shp[1]:.1f}% of the original {orig_shp[1]} features): "
        f"{selected_feats}"
    )
    df_fs.loc[selected_feats, ["filter", "step"]] = ("Selected feature", -1)
    df_fs.loc[target_col, ["filter", "step"]] = ("Target Column", -1)

    return selected_feats, df_fs
