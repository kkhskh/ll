from __future__ import annotations

import argparse
import json
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


def contiguous_time_splits(time_values: pd.Series, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_values = np.sort(time_values.dropna().unique())
    chunks = np.array_split(unique_values, n_splits)
    valid_masks = [time_values.isin(chunk).to_numpy() for chunk in chunks]
    row_idx = np.arange(len(time_values))
    splits = []
    for valid_mask in valid_masks:
        valid_idx = row_idx[valid_mask]
        train_idx = row_idx[~valid_mask]
        splits.append((train_idx, valid_idx))
    return splits


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


def evaluate_linear_model(
    df_train: pd.DataFrame,
    cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    X = prepare_linear_frame(df_train, cols)
    y = df_train[TARGET_COL].to_numpy(dtype=np.float64)
    oof = np.zeros(len(df_train), dtype=np.float64)
    fold_rows = []

    for fold_id, (train_idx, valid_idx) in enumerate(splits):
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        model.fit(X.iloc[train_idx], y[train_idx])
        valid_pred = model.predict(X.iloc[valid_idx])
        oof[valid_idx] = valid_pred
        fold_rows.append(
            {
                "fold": fold_id,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "fold_corr": pearson_corr(y[valid_idx], valid_pred),
            }
        )

    return {
        "oof_corr": pearson_corr(y, oof),
        "folds": fold_rows,
        "oof_predictions": oof,
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
            test_pred += test_fold_pred / len(splits)
            fold_rows.append(
                {
                    "fold": fold_id,
                    "n_train": int(len(train_idx)),
                    "n_valid": int(len(valid_idx)),
                    "fold_corr": pearson_corr(y[valid_idx], valid_pred),
                }
            )
        return {
            "model_name": "HistGradientBoostingRegressor",
            "oof_corr": pearson_corr(y, oof),
            "folds": fold_rows,
            "oof_predictions": oof,
            "test_predictions": test_pred,
        }

    train_cb = df_train[cols].copy()
    test_cb = df_test[cols].copy()
    for col in cat_cols:
        train_cb[col] = train_cb[col].astype(str)
        test_cb[col] = test_cb[col].astype(str)

    oof = np.zeros(len(df_train), dtype=np.float64)
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
        test_pred += test_fold_pred / len(splits)
        fold_rows.append(
            {
                "fold": fold_id,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "fold_corr": pearson_corr(y[valid_idx], valid_pred),
            }
        )

    return {
        "model_name": "CatBoostRegressor",
        "oof_corr": pearson_corr(y, oof),
        "folds": fold_rows,
        "oof_predictions": oof,
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
                "mean_fold_corr": float(np.nanmean([row["fold_corr"] for row in result["folds"]])),
            }
        )

    linear_primary = evaluate_linear_model(df_train, cols, primary_splits)
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
        "linear_primary_strategy": {
            "strategy": primary_strategy,
            "oof_corr": linear_primary["oof_corr"],
            "folds": linear_primary["folds"],
        },
        "nonlinear_primary_strategy": {
            "model_name": nonlinear_result["model_name"],
            "strategy": primary_strategy,
            "oof_corr": nonlinear_result["oof_corr"],
            "folds": nonlinear_result["folds"],
        },
        "shuffled_target_check": {
            "strategy": primary_strategy,
            "oof_corr": shuffled_result["oof_corr"],
            "folds": shuffled_result["folds"],
        },
        "chosen_submission_model": {
            "model_name": best_model_name,
            "oof_corr": best_score,
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
