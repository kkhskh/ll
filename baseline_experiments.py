from __future__ import annotations

import argparse
import json
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ID_COL = "id"
TARGET_COL = "target"
PRIMARY_TIME_COL = "di"
PRIMARY_GROUP_COL = "si"
META_COLS = ["si", "di", "industry", "sector", "top2000", "top1000", "top500"]
BASELINE_OUTPUT_DIR = Path("artifacts")


@contextmanager
def _ignore_sklearn_imputer_all_nan_feature_warnings():
    """Walk-forward train slices often leave some f_* columns 100% NaN; median imputer warns."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Skipping features without any observed values",
            category=UserWarning,
        )
        yield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Trexquant local baselines.")
    parser.add_argument(
        "--train-path",
        default="train.csv",
        help="Path to local train.csv",
    )
    parser.add_argument(
        "--test-path",
        default="test.csv",
        help="Path to local test.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory for reports and dry-run submission output",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--min-train-groups",
        type=int,
        default=252,
        help="Minimum number of di groups required before the first validation window",
    )
    parser.add_argument(
        "--embargo-groups",
        type=int,
        default=5,
        help="Number of di groups to drop from training immediately before each validation window",
    )
    return parser.parse_args()


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def oof_pearson_on_covered(
    y: np.ndarray, oof: np.ndarray, covered: np.ndarray
) -> tuple[float, float]:
    """
    OOF Pearson only on rows that appeared in at least one validation fold.
    Warmup rows (never validated) must not be scored as if prediction were zero.
    Returns (pearson_corr, coverage_frac).
    """
    covered = np.asarray(covered, dtype=bool)
    y = np.asarray(y, dtype=np.float64)
    oof = np.asarray(oof, dtype=np.float64)
    coverage_frac = float(covered.mean())
    if int(covered.sum()) < 2:
        return float("nan"), coverage_frac
    return pearson_corr(y[covered], oof[covered]), coverage_frac


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


def feature_columns(df_train: pd.DataFrame) -> list[str]:
    return [c for c in df_train.columns if c not in [ID_COL, TARGET_COL]]


def usable_feature_columns(df_train: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    dropped = [col for col in cols if df_train[col].notna().sum() == 0]
    kept = [col for col in cols if col not in dropped]
    return kept, dropped


def prefix_family_sizes(cols: Iterable[str]) -> dict[str, int]:
    families: dict[str, int] = {}
    for col in cols:
        prefix = col.split("_")[0]
        families[prefix] = families.get(prefix, 0) + 1
    return dict(sorted(families.items(), key=lambda item: (-item[1], item[0])))


def top_missingness(df: pd.DataFrame, cols: list[str], head: int = 20) -> list[dict[str, float]]:
    missing = df[cols].isna().mean().sort_values(ascending=False).head(head)
    return [{"column": col, "missing_frac": float(value)} for col, value in missing.items()]


def summarize_target(target: pd.Series) -> dict[str, float]:
    summary = target.describe().to_dict()
    summary["skew"] = float(target.skew())
    summary["kurtosis"] = float(target.kurt())
    return {key: float(value) for key, value in summary.items()}


def audit_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cols: list[str],
    dropped_cols: list[str],
) -> dict[str, object]:
    common_feature_cols = [c for c in df_test.columns if c != ID_COL]
    return {
        "train_shape": list(df_train.shape),
        "test_shape": list(df_test.shape),
        "feature_count": len(cols),
        "dropped_all_missing_features": dropped_cols,
        "train_columns_match_test_plus_target": list(df_train.columns) == list(df_test.columns) + [TARGET_COL],
        "first_20_columns": list(df_train.columns[:20]),
        "train_id_unique": bool(df_train[ID_COL].is_unique),
        "test_id_unique": bool(df_test[ID_COL].is_unique),
        "train_id_duplicates": int(df_train[ID_COL].duplicated().sum()),
        "test_id_duplicates": int(df_test[ID_COL].duplicated().sum()),
        "train_di_unique": int(df_train[PRIMARY_TIME_COL].nunique()),
        "test_di_unique": int(df_test[PRIMARY_TIME_COL].nunique()),
        "train_si_unique": int(df_train[PRIMARY_GROUP_COL].nunique()),
        "test_si_unique": int(df_test[PRIMARY_GROUP_COL].nunique()),
        "dtypes": {str(key): int(value) for key, value in df_train.dtypes.astype(str).value_counts().items()},
        "top_missingness": top_missingness(df_train, cols),
        "target_summary": summarize_target(df_train[TARGET_COL]),
        "prefix_family_sizes": prefix_family_sizes(cols),
        "manual_feature_groups": {
            "meta": [col for col in META_COLS if col in cols],
            "anonymous": [col for col in cols if col.startswith("f_")],
            "all": cols,
            "test_features": common_feature_cols,
        },
    }


def prepare_linear_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    frame = df[cols].copy()
    bool_cols = frame.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        frame[bool_cols] = frame[bool_cols].astype(np.int8)
    return frame


def shuffled_kfold_splits(n_rows: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = np.arange(n_rows)
    return [(train_idx, valid_idx) for train_idx, valid_idx in splitter.split(indices)]


def grouped_kfold_splits(groups: pd.Series, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = GroupKFold(n_splits=n_splits)
    row_idx = np.arange(len(groups))
    return [(train_idx, valid_idx) for train_idx, valid_idx in splitter.split(row_idx, groups=groups)]


def walk_forward_time_splits(
    time_values: pd.Series,
    n_splits: int,
    min_train_groups: int = 252,
    embargo_groups: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    True walk-forward (expanding-window) splits on a time index.

    For each fold k:
      - validation = the kth contiguous chunk of di groups after the warmup window
      - training   = all rows with di strictly less than the earliest validation di,
                     minus the last `embargo_groups` di groups immediately before it
      - no training row may have di >= the first validation di
    """
    import math

    unique_di = np.sort(time_values.dropna().unique())
    U = len(unique_di)

    warmup_size = max(min_train_groups, math.ceil(0.30 * U))
    if warmup_size >= U:
        raise ValueError(
            f"min_train_groups={min_train_groups} exhausts all {U} unique di values. "
            "Lower --min-train-groups."
        )

    remaining_di = unique_di[warmup_size:]
    if len(remaining_di) < n_splits:
        raise ValueError(
            f"Only {len(remaining_di)} di groups remain after warmup; "
            f"cannot form {n_splits} validation chunks."
        )

    chunks = np.array_split(remaining_di, n_splits)
    row_idx = np.arange(len(time_values))
    tv = time_values.to_numpy()

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        valid_start = chunk[0]
        valid_mask = time_values.isin(chunk).to_numpy()
        valid_idx = row_idx[valid_mask]

        # all di strictly before the validation window
        pre_valid_di = unique_di[unique_di < valid_start]
        if embargo_groups > 0 and len(pre_valid_di) > embargo_groups:
            embargoed_di = pre_valid_di[-embargo_groups:]
            allowed_di = pre_valid_di[:-embargo_groups]
        else:
            embargoed_di = pre_valid_di  # too small: embargo everything before valid
            allowed_di = np.array([], dtype=pre_valid_di.dtype)

        train_mask = np.isin(tv, allowed_di)
        train_idx = row_idx[train_mask]

        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue
        splits.append((train_idx, valid_idx))

    if len(splits) < 2:
        raise ValueError(
            f"Only {len(splits)} valid fold(s) survived warmup/embargo constraints. "
            "Lower --min-train-groups or --embargo-groups."
        )
    return splits


def summarize_splits(
    df_train: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, object]]:
    """Per-fold diagnostic table for walk-forward splits."""
    rows = []
    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        tr = df_train.iloc[tr_idx]
        va = df_train.iloc[va_idx]
        tr_di = tr[PRIMARY_TIME_COL]
        va_di = va[PRIMARY_TIME_COL]
        tr_si = set(tr[PRIMARY_GROUP_COL].dropna().unique())
        va_si = set(va[PRIMARY_GROUP_COL].dropna().unique())
        overlap = len(tr_si & va_si) / max(len(va_si), 1)
        rows.append({
            "fold": fold_id,
            "n_train_rows": int(len(tr_idx)),
            "n_valid_rows": int(len(va_idx)),
            "train_di_min": int(tr_di.min()),
            "train_di_max": int(tr_di.max()),
            "valid_di_min": int(va_di.min()),
            "valid_di_max": int(va_di.max()),
            "n_train_di": int(tr_di.nunique()),
            "n_valid_di": int(va_di.nunique()),
            "n_train_si": int(len(tr_si)),
            "n_valid_si": int(len(va_si)),
            "si_overlap_frac": round(overlap, 4),
            "embargo_ok": bool(int(tr_di.max()) < int(va_di.min())),
        })
    return rows


def audit_si_utility(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    """Audit whether stock id (`si`) carries usable temporal signal."""
    out_dir = BASELINE_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    train_si = set(df_train[PRIMARY_GROUP_COL].dropna().unique())
    test_si = set(df_test[PRIMARY_GROUP_COL].dropna().unique())
    inter = train_si & test_si
    test_si_unique = max(len(test_si), 1)
    test_rows_seen = df_test[PRIMARY_GROUP_COL].isin(train_si).to_numpy()

    obs_per_si = df_train.groupby(PRIMARY_GROUP_COL).size().astype(int)
    quantiles = obs_per_si.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).to_dict()
    dist = {
        "count": float(obs_per_si.shape[0]),
        "mean": float(obs_per_si.mean()),
        "std": float(obs_per_si.std()),
        "min": float(obs_per_si.min()),
        "p01": float(quantiles.get(0.01, np.nan)),
        "p05": float(quantiles.get(0.05, np.nan)),
        "p10": float(quantiles.get(0.10, np.nan)),
        "p25": float(quantiles.get(0.25, np.nan)),
        "p50": float(quantiles.get(0.50, np.nan)),
        "p75": float(quantiles.get(0.75, np.nan)),
        "p90": float(quantiles.get(0.90, np.nan)),
        "p95": float(quantiles.get(0.95, np.nan)),
        "p99": float(quantiles.get(0.99, np.nan)),
        "max": float(obs_per_si.max()),
        "stocks_ge_1": int((obs_per_si >= 1).sum()),
        "stocks_ge_2": int((obs_per_si >= 2).sum()),
        "stocks_ge_4": int((obs_per_si >= 4).sum()),
        "stocks_ge_8": int((obs_per_si >= 8).sum()),
        "stocks_ge_12": int((obs_per_si >= 12).sum()),
    }
    pd.DataFrame(
        [{"metric": k, "value": v} for k, v in dist.items()]
    ).to_csv(out_dir / "si_obs_distribution.csv", index=False)

    # Forward-safe persistence on raw target + cross-sectional target.
    ord_train = df_train.sort_values([PRIMARY_GROUP_COL, PRIMARY_TIME_COL, ID_COL]).copy()
    ord_train["lag1_target"] = ord_train.groupby(PRIMARY_GROUP_COL, sort=False)[TARGET_COL].shift(1)
    ord_train["lag2_target"] = ord_train.groupby(PRIMARY_GROUP_COL, sort=False)[TARGET_COL].shift(2)
    valid1 = ord_train["lag1_target"].notna().to_numpy()
    valid2 = ord_train["lag2_target"].notna().to_numpy()
    y = ord_train[TARGET_COL].to_numpy(np.float64)
    lag1 = ord_train["lag1_target"].to_numpy(np.float64)
    lag2 = ord_train["lag2_target"].to_numpy(np.float64)

    raw_persist = {
        "corr_target_lag1": pearson_corr(y[valid1], lag1[valid1]) if int(valid1.sum()) >= 2 else float("nan"),
        "corr_target_lag2": pearson_corr(y[valid2], lag2[valid2]) if int(valid2.sum()) >= 2 else float("nan"),
        "corr_abs_target_abs_lag1": (
            pearson_corr(np.abs(y[valid1]), np.abs(lag1[valid1])) if int(valid1.sum()) >= 2 else float("nan")
        ),
        "sign_agreement_lag1": (
            float((np.sign(y[valid1]) == np.sign(lag1[valid1])).mean()) if int(valid1.sum()) else float("nan")
        ),
    }

    cs_mean = ord_train.groupby(PRIMARY_TIME_COL)[TARGET_COL].transform("mean")
    ord_train["target_cs"] = ord_train[TARGET_COL] - cs_mean
    ord_train["lag1_target_cs"] = ord_train.groupby(PRIMARY_GROUP_COL, sort=False)["target_cs"].shift(1)
    ord_train["lag2_target_cs"] = ord_train.groupby(PRIMARY_GROUP_COL, sort=False)["target_cs"].shift(2)
    v1 = ord_train["lag1_target_cs"].notna().to_numpy()
    v2 = ord_train["lag2_target_cs"].notna().to_numpy()
    y_cs = ord_train["target_cs"].to_numpy(np.float64)
    lag1_cs = ord_train["lag1_target_cs"].to_numpy(np.float64)
    lag2_cs = ord_train["lag2_target_cs"].to_numpy(np.float64)
    cs_persist = {
        "corr_target_lag1": pearson_corr(y_cs[v1], lag1_cs[v1]) if int(v1.sum()) >= 2 else float("nan"),
        "corr_target_lag2": pearson_corr(y_cs[v2], lag2_cs[v2]) if int(v2.sum()) >= 2 else float("nan"),
        "corr_abs_target_abs_lag1": (
            pearson_corr(np.abs(y_cs[v1]), np.abs(lag1_cs[v1])) if int(v1.sum()) >= 2 else float("nan")
        ),
        "sign_agreement_lag1": (
            float((np.sign(y_cs[v1]) == np.sign(lag1_cs[v1])).mean()) if int(v1.sum()) else float("nan")
        ),
    }

    y_train = df_train[TARGET_COL].to_numpy(np.float64)
    fold_rows: list[dict[str, object]] = []
    print("\nsi fold utility (walk-forward):")
    print(
        f"{'fold':>4}  {'valid_rows':>10}  {'seen_frac':>9}  {'count_ge_2':>10}  "
        f"{'si_mean_corr':>12}  {'si_log_count_corr':>16}"
    )
    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        tr = df_train.iloc[tr_idx]
        va = df_train.iloc[va_idx]
        tr_y = y_train[tr_idx]
        tr_global = float(np.mean(tr_y))
        si_mean = tr.groupby(PRIMARY_GROUP_COL)[TARGET_COL].mean()
        si_count = tr.groupby(PRIMARY_GROUP_COL).size()
        va_si = va[PRIMARY_GROUP_COL]
        va_mean_pred = va_si.map(si_mean).fillna(tr_global).to_numpy(np.float64)
        va_count_pred = va_si.map(np.log1p(si_count)).fillna(0.0).to_numpy(np.float64)
        seen_mask = va_si.isin(si_mean.index).to_numpy()
        ge2_mask = va_si.map(si_count).fillna(0).to_numpy() >= 2
        mean_corr = pearson_corr(y_train[va_idx], va_mean_pred)
        cnt_corr = pearson_corr(y_train[va_idx], va_count_pred)
        row = {
            "fold": fold_id,
            "valid_rows": int(len(va_idx)),
            "seen_frac": float(seen_mask.mean()),
            "count_ge_2_frac": float(ge2_mask.mean()),
            "si_mean_encode_corr": mean_corr,
            "si_log_count_corr": cnt_corr,
            "global_mean_baseline_corr": 0.0,
            "si_mean_minus_global_baseline": (mean_corr - 0.0) if np.isfinite(mean_corr) else float("nan"),
        }
        fold_rows.append(row)
        print(
            f"{fold_id:>4}  {len(va_idx):>10,}  {row['seen_frac']:>9.4f}  {row['count_ge_2_frac']:>10.4f}  "
            f"{mean_corr:>12.6f}  {cnt_corr:>16.6f}"
        )

    pd.DataFrame(fold_rows).to_csv(out_dir / "si_fold_coverage.csv", index=False)

    mean_gain = float(np.nanmean([r["si_mean_minus_global_baseline"] for r in fold_rows]))
    lag_strength = np.nanmean(
        [
            np.abs(raw_persist["corr_target_lag1"]),
            np.abs(raw_persist["corr_target_lag2"]),
            np.abs(cs_persist["corr_target_lag1"]),
            np.abs(cs_persist["corr_target_lag2"]),
        ]
    )
    dead_end = bool((mean_gain < 0.002) and (lag_strength < 0.01))
    decision_note = (
        "si target-history appears weak: do not prioritize si target-history for mainline alpha."
        if dead_end
        else "si shows measurable persistence and/or fold-safe utility: keep selective si work in research."
    )
    print(f"\nSI decision note: {decision_note}")

    payload = {
        "global_overlap": {
            "train_si_unique": int(len(train_si)),
            "test_si_unique": int(len(test_si)),
            "global_test_si_seen_frac": float(len(inter) / test_si_unique),
            "test_rows_seen_si_frac": float(test_rows_seen.mean()),
        },
        "obs_per_si_distribution": dist,
        "target_persistence_raw": raw_persist,
        "target_persistence_cross_sectional": cs_persist,
        "fold_utility": fold_rows,
        "decision": {
            "mean_si_mean_encode_gain_vs_global_baseline": mean_gain,
            "lag_persistence_strength_abs_mean": float(lag_strength),
            "dead_end_for_mainline_alpha": dead_end,
            "note": decision_note,
        },
    }
    (out_dir / "si_audit.json").write_text(json.dumps(payload, indent=2))
    return payload


def evaluate_linear_model(
    df_train: pd.DataFrame,
    cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    X = prepare_linear_frame(df_train, cols)
    y = df_train[TARGET_COL].to_numpy(dtype=np.float64)
    oof = np.zeros(len(df_train), dtype=np.float64)
    covered = np.zeros(len(df_train), dtype=bool)
    fold_rows = []

    print(
        "  (Ridge) Some features are all-NaN in certain train folds; sklearn imputer skips them — expected."
    )

    for fold_id, (train_idx, valid_idx) in enumerate(splits):
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        with _ignore_sklearn_imputer_all_nan_feature_warnings():
            model.fit(X.iloc[train_idx], y[train_idx])
            valid_pred = model.predict(X.iloc[valid_idx])
        oof[valid_idx] = valid_pred
        covered[valid_idx] = True
        fold_rows.append(
            {
                "fold": fold_id,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "fold_corr": pearson_corr(y[valid_idx], valid_pred),
            }
        )

    oof_corr, cov_frac = oof_pearson_on_covered(y, oof, covered)
    return {
        "oof_corr": oof_corr,
        "oof_coverage_frac": cov_frac,
        "folds": fold_rows,
        "oof_predictions": oof,
        "oof_covered": covered,
    }


def fit_linear_submission(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cols: list[str],
) -> np.ndarray:
    X_train = prepare_linear_frame(df_train, cols)
    X_test = prepare_linear_frame(df_test, cols)
    y_train = df_train[TARGET_COL].to_numpy(dtype=np.float64)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    with _ignore_sklearn_imputer_all_nan_feature_warnings():
        model.fit(X_train, y_train)
        return model.predict(X_test)


def evaluate_catboost_like(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    y = df_train[TARGET_COL].to_numpy(dtype=np.float64)
    cat_cols = [col for col in ["si", "industry", "sector", "top2000", "top1000", "top500"] if col in cols]

    try:
        from catboost import CatBoostRegressor
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        X = prepare_linear_frame(df_train, cols)
        X_test = prepare_linear_frame(df_test, cols)
        oof = np.zeros(len(df_train), dtype=np.float64)
        covered = np.zeros(len(df_train), dtype=bool)
        test_pred = np.zeros(len(df_test), dtype=np.float64)
        fold_rows = []
        for fold_id, (train_idx, valid_idx) in enumerate(splits):
            model = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=6,
                max_iter=400,
                l2_regularization=0.1,
                random_state=42 + fold_id,
            )
            model.fit(X.iloc[train_idx], y[train_idx])
            valid_pred = model.predict(X.iloc[valid_idx])
            test_fold_pred = model.predict(X_test)
            oof[valid_idx] = valid_pred
            covered[valid_idx] = True
            test_pred += test_fold_pred / len(splits)
            fold_rows.append(
                {
                    "fold": fold_id,
                    "n_train": int(len(train_idx)),
                    "n_valid": int(len(valid_idx)),
                    "fold_corr": pearson_corr(y[valid_idx], valid_pred),
                }
            )
        oof_corr, cov_frac = oof_pearson_on_covered(y, oof, covered)
        return {
            "model_name": "HistGradientBoostingRegressor",
            "oof_corr": oof_corr,
            "oof_coverage_frac": cov_frac,
            "folds": fold_rows,
            "oof_predictions": oof,
            "oof_covered": covered,
            "test_predictions": test_pred,
        }

    train_cb = df_train[cols].copy()
    test_cb = df_test[cols].copy()
    for col in cat_cols:
        train_cb[col] = train_cb[col].astype(str)
        test_cb[col] = test_cb[col].astype(str)

    oof = np.zeros(len(df_train), dtype=np.float64)
    covered = np.zeros(len(df_train), dtype=bool)
    test_pred = np.zeros(len(df_test), dtype=np.float64)
    fold_rows = []

    for fold_id, (train_idx, valid_idx) in enumerate(splits):
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            depth=6,
            learning_rate=0.03,
            iterations=300,
            l2_leaf_reg=8.0,
            random_seed=42 + fold_id,
            subsample=0.8,
            bootstrap_type="Bernoulli",
            od_type="Iter",
            od_wait=50,
            verbose=False,
        )
        model.fit(
            train_cb.iloc[train_idx],
            y[train_idx],
            cat_features=cat_cols if cat_cols else None,
            eval_set=(train_cb.iloc[valid_idx], y[valid_idx]),
            use_best_model=True,
        )
        valid_pred = model.predict(train_cb.iloc[valid_idx])
        test_fold_pred = model.predict(test_cb)
        oof[valid_idx] = valid_pred
        covered[valid_idx] = True
        test_pred += test_fold_pred / len(splits)
        fold_rows.append(
            {
                "fold": fold_id,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "fold_corr": pearson_corr(y[valid_idx], valid_pred),
            }
        )

    oof_corr, cov_frac = oof_pearson_on_covered(y, oof, covered)
    return {
        "model_name": "CatBoostRegressor",
        "oof_corr": oof_corr,
        "oof_coverage_frac": cov_frac,
        "folds": fold_rows,
        "oof_predictions": oof,
        "oof_covered": covered,
        "test_predictions": test_pred,
    }


def feature_group_columns(all_cols: list[str]) -> dict[str, list[str]]:
    meta_cols = [col for col in META_COLS if col in all_cols]
    anon_cols = [col for col in all_cols if col.startswith("f_")]
    return {
        "meta_only": meta_cols,
        "anonymous_only": anon_cols,
        "all_features": all_cols,
    }


def dry_run_submission(ids: pd.Series, preds: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    safe_preds = np.asarray(preds, dtype=np.float64).copy()
    if float((np.abs(safe_preds) > 0).mean()) < 0.10:
        safe_preds += 1e-9

    submission = pd.DataFrame({ID_COL: ids, TARGET_COL: safe_preds})
    checks = {
        "row_count": int(len(submission)),
        "finite_predictions": bool(np.isfinite(submission[TARGET_COL]).all()),
        "non_zero_fraction": float((submission[TARGET_COL].abs() > 0).mean()),
        "target_std": float(submission[TARGET_COL].std()),
    }
    if not checks["finite_predictions"]:
        raise ValueError("Submission contains non-finite predictions.")
    if checks["non_zero_fraction"] < 0.10:
        raise ValueError("Submission has fewer than 10% non-zero predictions.")
    return submission, checks


def ensure_output_dir(path_str: str) -> Path:
    output_dir = Path(path_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    global BASELINE_OUTPUT_DIR
    BASELINE_OUTPUT_DIR = output_dir

    df_train, df_test = load_data(args.train_path, args.test_path)
    all_cols = feature_columns(df_train)
    cols, dropped_cols = usable_feature_columns(df_train, all_cols)
    audit = audit_data(df_train, df_test, cols, dropped_cols)

    strategy_splits = {
        "shuffled_kfold": shuffled_kfold_splits(len(df_train), args.splits),
        "group_kfold_si": grouped_kfold_splits(df_train[PRIMARY_GROUP_COL], args.splits),
        "walk_forward_di": walk_forward_time_splits(
            df_train[PRIMARY_TIME_COL],
            n_splits=args.splits,
            min_train_groups=args.min_train_groups,
            embargo_groups=args.embargo_groups,
        ),
    }

    primary_strategy = "walk_forward_di"
    primary_splits = strategy_splits[primary_strategy]
    si_audit = audit_si_utility(df_train, df_test, primary_splits)

    # ── fold diagnostics ──────────────────────────────────────────────────────
    fold_diagnostics = summarize_splits(df_train, primary_splits)

    # print concise table to stdout
    header = (
        f"{'fold':>4}  {'n_train':>8}  {'n_valid':>7}  "
        f"{'tr_di_min':>9}  {'tr_di_max':>9}  "
        f"{'va_di_min':>9}  {'va_di_max':>9}  "
        f"{'n_tr_di':>7}  {'n_va_di':>7}  "
        f"{'si_ovlp':>7}  {'embargo_ok':>10}"
    )
    print("\nWalk-Forward Fold Diagnostics")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in fold_diagnostics:
        print(
            f"{row['fold']:>4}  {row['n_train_rows']:>8,}  {row['n_valid_rows']:>7,}  "
            f"{row['train_di_min']:>9}  {row['train_di_max']:>9}  "
            f"{row['valid_di_min']:>9}  {row['valid_di_max']:>9}  "
            f"{row['n_train_di']:>7}  {row['n_valid_di']:>7}  "
            f"{row['si_overlap_frac']:>7.3f}  {str(row['embargo_ok']):>10}"
        )
    print("=" * len(header))

    # acceptance check: no training di may be >= first validation di
    for row in fold_diagnostics:
        assert row["embargo_ok"], (
            f"Fold {row['fold']}: train_di_max={row['train_di_max']} >= "
            f"valid_di_min={row['valid_di_min']} — embargo violated!"
        )

    pd.DataFrame(fold_diagnostics).to_csv(
        output_dir / "walk_forward_fold_diagnostics.csv", index=False
    )

    strategy_results = []
    for strategy_name, splits in strategy_splits.items():
        result = evaluate_linear_model(df_train, cols, splits)
        strategy_results.append(
            {
                "strategy": strategy_name,
                "oof_corr": result["oof_corr"],
                "oof_coverage_frac": result["oof_coverage_frac"],
                "mean_fold_corr": float(np.nanmean([row["fold_corr"] for row in result["folds"]])),
            }
        )

    grouped_results = []
    for group_name, group_cols in feature_group_columns(cols).items():
        if len(group_cols) < 2:
            continue
        result = evaluate_linear_model(df_train, group_cols, primary_splits)
        grouped_results.append(
            {
                "feature_group": group_name,
                "n_cols": len(group_cols),
                "oof_corr": result["oof_corr"],
                "oof_coverage_frac": result["oof_coverage_frac"],
                "mean_fold_corr": float(np.nanmean([row["fold_corr"] for row in result["folds"]])),
            }
        )

    linear_primary = evaluate_linear_model(df_train, cols, primary_splits)
    print(
        f"\nOOF scoring: Pearson on validated rows only "
        f"(coverage_frac={linear_primary['oof_coverage_frac']:.4f}; "
        f"warmup rows excluded from OOF correlation).\n"
    )
    shuffled_target = df_train[TARGET_COL].sample(frac=1.0, random_state=123).reset_index(drop=True)
    shuffled_frame = df_train.copy()
    shuffled_frame[TARGET_COL] = shuffled_target
    shuffled_result = evaluate_linear_model(shuffled_frame, cols, primary_splits)

    nonlinear_result = evaluate_catboost_like(df_train, df_test, cols, primary_splits)
    linear_test_pred = fit_linear_submission(df_train, df_test, cols)

    best_model_name = "linear_ridge"
    best_score = linear_primary["oof_corr"]
    best_test_pred = linear_test_pred
    if nonlinear_result["oof_corr"] > best_score:
        best_model_name = nonlinear_result["model_name"]
        best_score = nonlinear_result["oof_corr"]
        best_test_pred = nonlinear_result["test_predictions"]

    submission, submission_checks = dry_run_submission(df_test[ID_COL], best_test_pred)

    summary = {
        "primary_strategy": primary_strategy,
        "number_of_primary_folds": len(primary_splits),
        "walk_forward_min_train_groups": args.min_train_groups,
        "walk_forward_embargo_groups": args.embargo_groups,
        "audit": audit,
        "walk_forward_fold_diagnostics": fold_diagnostics,
        "cv_strategy_results": strategy_results,
        "feature_group_results": grouped_results,
        "si_audit": si_audit,
        "linear_primary_strategy": {
            "strategy": primary_strategy,
            "oof_corr": linear_primary["oof_corr"],
            "oof_coverage_frac": linear_primary["oof_coverage_frac"],
            "folds": linear_primary["folds"],
        },
        "nonlinear_primary_strategy": {
            "model_name": nonlinear_result["model_name"],
            "strategy": primary_strategy,
            "oof_corr": nonlinear_result["oof_corr"],
            "oof_coverage_frac": nonlinear_result["oof_coverage_frac"],
            "folds": nonlinear_result["folds"],
        },
        "shuffled_target_check": {
            "strategy": primary_strategy,
            "oof_corr": shuffled_result["oof_corr"],
            "oof_coverage_frac": shuffled_result["oof_coverage_frac"],
            "folds": shuffled_result["folds"],
        },
        "chosen_submission_model": {
            "model_name": best_model_name,
            "oof_corr": best_score,
            "oof_coverage_frac": (
                nonlinear_result["oof_coverage_frac"]
                if best_model_name != "linear_ridge"
                else linear_primary["oof_coverage_frac"]
            ),
        },
        "submission_checks": submission_checks,
    }

    (output_dir / "baseline_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(strategy_results).to_csv(output_dir / "cv_strategy_results.csv", index=False)
    pd.DataFrame(grouped_results).to_csv(output_dir / "feature_group_results.csv", index=False)
    submission.to_csv(output_dir / "local_submission.csv", index=False)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved reports to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
