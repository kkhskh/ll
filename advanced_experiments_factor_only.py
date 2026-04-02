"""
Factor-only follow-up to the ET baseline.

Runs exactly two branches:
  A) D7_et_base_control
  F1) D7_et_factor_direction_only

The control preserves the incumbent ET representation and training target.
F1 keeps the same base representation, adds latent family factors, and trains
on the cross-sectional z-score of target within di.
"""
from __future__ import annotations

import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_experiments import (
    TRAIN_PATH,
    TEST_PATH,
    DEFAULT_FAMILY_REPS,
    ET_SETTINGS,
    ID_COL,
    META_COL_CANDIDATES,
    TARGET_COL,
    TIME_COL,
    _fit_numeric_preprocessor,
    add_rank_features,
    add_sparse_indicators,
    build_fold_local_family_interactions,
    cross_sectional_zscore_anonymous,
    get_feature_sets,
    get_sparse_anon_cols,
    oof_pearson_on_covered,
    pearson,
    walk_forward_time_splits,
)

warnings.filterwarnings("ignore")

ROOT_OUT = Path(".")
ARTIFACT_DIR = Path("artifacts_factor_only")
ARTIFACT_DIR.mkdir(exist_ok=True)

BRANCH_CONTROL = "D7_et_base_control"
BRANCH_FACTOR = "D7_et_factor_direction_only"

FAMILY_STANDARDIZATION_STATS: dict[str, tuple[float, float]] = {}


def get_default_reps(existing_cols: list[str], n_reps: int = 12) -> list[str]:
    reps = [c for c in DEFAULT_FAMILY_REPS if c in existing_cols]
    for c in existing_cols:
        if c not in reps:
            reps.append(c)
        if len(reps) >= n_reps:
            break
    return reps[:n_reps]


def save_oof_artifact(path: Path, y: np.ndarray, oof: np.ndarray, covered: np.ndarray) -> None:
    pd.DataFrame(
        {
            "oof_pred": np.asarray(oof, np.float64),
            "target": np.asarray(y, np.float64),
            "oof_covered": np.asarray(covered, np.int8),
        }
    ).to_csv(path, index=False)


def save_submission(path: Path, test_ids: np.ndarray, preds: np.ndarray, clip: bool = True) -> None:
    p = np.asarray(preds, np.float64).copy()
    if clip:
        lo, hi = np.nanpercentile(p, [0.5, 99.5])
        p = np.clip(p, lo, hi)
    if (np.abs(p) > 0).mean() < 0.10:
        p += 1e-9
    sub = pd.DataFrame({ID_COL: np.asarray(test_ids), TARGET_COL: p})
    assert np.isfinite(sub[TARGET_COL]).all(), "Non-finite predictions"
    assert (sub[TARGET_COL].abs() > 0).mean() >= 0.10, "< 10% non-zero"
    sub.to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(sub):,} rows)")


def build_feature_family_map(
    df_train: pd.DataFrame,
    anon_cols: list[str],
    sparse_threshold: float = 0.20,
    n_sparse_clusters: int = 6,
    n_dense_clusters: int = 6,
) -> dict[str, list[str]]:
    from sklearn.cluster import AgglomerativeClustering

    def _cluster_side(cols: list[str], prefix: str, max_clusters: int) -> dict[str, list[str]]:
        if not cols:
            return {}
        cols = sorted(cols)
        n_clusters = min(max_clusters, len(cols))
        if len(cols) == 1:
            return {f"{prefix}_family_01": cols}

        value_corr = df_train[cols].corr().abs().fillna(0.0)
        miss_corr = df_train[cols].isna().astype(np.float32).corr().abs().fillna(0.0)
        sim = (0.7 * value_corr + 0.3 * miss_corr).clip(lower=0.0, upper=1.0)
        np.fill_diagonal(sim.values, 1.0)
        if n_clusters >= len(cols):
            labels = np.arange(len(cols), dtype=int)
        else:
            dist = np.maximum(1.0 - sim.to_numpy(np.float64), 0.0)
            labels = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="precomputed",
                linkage="average",
            ).fit_predict(dist)
        grouped: dict[int, list[str]] = {}
        for col, label in zip(cols, labels):
            grouped.setdefault(int(label), []).append(col)
        ordered_groups = sorted(grouped.values(), key=lambda group: (group[0], len(group)))
        return {
            f"{prefix}_family_{i:02d}": sorted(group)
            for i, group in enumerate(ordered_groups, start=1)
        }

    miss = df_train[anon_cols].isna().mean()
    sparse_cols = sorted([c for c in anon_cols if float(miss.get(c, 0.0)) >= sparse_threshold])
    dense_cols = sorted([c for c in anon_cols if float(miss.get(c, 0.0)) < sparse_threshold])

    family_map: dict[str, list[str]] = {}
    family_map.update(_cluster_side(sparse_cols, "sparse", n_sparse_clusters))
    family_map.update(_cluster_side(dense_cols, "dense", n_dense_clusters))
    assigned = sorted([c for cols in family_map.values() for c in cols])
    assert assigned == sorted(anon_cols), "Every anonymous feature must belong to exactly one family"
    return family_map


def set_family_standardization_stats(df_train: pd.DataFrame, anon_cols: list[str]) -> None:
    global FAMILY_STANDARDIZATION_STATS
    stats: dict[str, tuple[float, float]] = {}
    for col in anon_cols:
        arr = pd.to_numeric(df_train[col], errors="coerce").to_numpy(np.float64)
        mu = float(np.nanmean(arr))
        sd = float(np.nanstd(arr))
        if not np.isfinite(mu):
            mu = 0.0
        if (not np.isfinite(sd)) or sd == 0.0:
            sd = 1.0
        stats[col] = (mu, sd)
    FAMILY_STANDARDIZATION_STATS = stats


def build_family_factor_block(
    df: pd.DataFrame,
    family_map: dict[str, list[str]],
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    family_factor_cols: list[str] = []
    for family_name, raw_cols in family_map.items():
        avail_raw = [c for c in raw_cols if c in out.columns]
        if not avail_raw:
            continue
        raw_arr = out[avail_raw].to_numpy(np.float64)
        finite = np.isfinite(raw_arr)
        present_count = finite.sum(axis=1).astype(np.float32)
        family_size = max(len(avail_raw), 1)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            with np.errstate(invalid="ignore"):
                value_mean = np.nanmean(raw_arr, axis=1)
                value_std = np.nanstd(raw_arr, axis=1)
                absmax = np.nanmax(np.abs(raw_arr), axis=1)

        std_arr = raw_arr.copy()
        for j, col in enumerate(avail_raw):
            mu, sd = FAMILY_STANDARDIZATION_STATS.get(col, (0.0, 1.0))
            std_arr[:, j] = (std_arr[:, j] - mu) / sd
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.", category=RuntimeWarning)
            with np.errstate(invalid="ignore"):
                disagreement = np.nanstd(std_arr, axis=1)

        factor_block = {
            f"fam_{family_name}_present_frac": (present_count / family_size).astype(np.float32),
            f"fam_{family_name}_present_count": present_count.astype(np.float32),
            f"fam_{family_name}_value_mean": np.where(np.isfinite(value_mean), value_mean, np.nan).astype(np.float32),
            f"fam_{family_name}_value_std": np.where(np.isfinite(value_std), value_std, np.nan).astype(np.float32),
            f"fam_{family_name}_absmax": np.where(np.isfinite(absmax), absmax, np.nan).astype(np.float32),
            f"fam_{family_name}_row_disagreement": np.where(finite.sum(axis=1) >= 2, disagreement, np.nan).astype(np.float32),
        }
        for name, values in factor_block.items():
            out[name] = values
            family_factor_cols.append(name)
    return out, family_factor_cols


def build_cross_sectional_target(df_train: pd.DataFrame) -> np.ndarray:
    y = pd.to_numeric(df_train[TARGET_COL], errors="coerce").to_numpy(np.float64)
    groups = df_train[TIME_COL].to_numpy()
    out = np.zeros(len(df_train), dtype=np.float64)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        vals = y[idx]
        mu = float(np.nanmean(vals))
        sd = float(np.nanstd(vals))
        if not np.isfinite(mu):
            mu = 0.0
        if (not np.isfinite(sd)) or sd == 0.0:
            out[idx] = 0.0
        else:
            out[idx] = (vals - mu) / sd
    return out


def run_et_branch(
    branch_name: str,
    df_train_model: pd.DataFrame,
    df_test_model: pd.DataFrame,
    base_feat_cols: list[str],
    y_fit: np.ndarray,
    y_eval: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    meta_cols: list[str],
    reps: list[str],
) -> dict[str, object]:
    from sklearn.ensemble import ExtraTreesRegressor

    oof = np.zeros(len(df_train_model), dtype=np.float64)
    covered = np.zeros(len(df_train_model), dtype=bool)
    test_pred = np.zeros(len(df_test_model), dtype=np.float64)
    fold_scores: list[float] = []
    n_features: list[float] = []

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        tr_di = df_train_model.iloc[tr_idx][TIME_COL]
        va_di = df_train_model.iloc[va_idx][TIME_COL]
        assert tr_di.max() < va_di.min(), f"[{branch_name}] fold {fold_id}: walk-forward violated"

        df_aug_tr, df_aug_te, ix_cols = build_fold_local_family_interactions(
            df_train_model,
            df_test_model,
            tr_idx,
            va_idx,
            reps,
        )
        feat_cols = [c for c in (base_feat_cols + ix_cols) if c in df_aug_tr.columns and c in df_aug_te.columns]
        cat_cols = [c for c in meta_cols if c in feat_cols and c != TIME_COL]
        num_cols = [
            c for c in feat_cols
            if c not in cat_cols and pd.api.types.is_numeric_dtype(df_aug_tr[c])
        ]
        Xtr = df_aug_tr.iloc[tr_idx][num_cols]
        Xva = df_aug_tr.iloc[va_idx][num_cols]
        Xte = df_aug_te[num_cols]
        Xtr_imp, Xva_imp, Xte_imp = _fit_numeric_preprocessor(Xtr, Xva, Xte, standardize=False)
        n_features.append(float(len(num_cols)))

        fold_pred_sum = np.zeros(len(va_idx), dtype=np.float64)
        fold_test_sum = np.zeros(len(df_test_model), dtype=np.float64)
        for seed, max_features, min_leaf in ET_SETTINGS:
            model = ExtraTreesRegressor(
                n_estimators=1200,
                min_samples_leaf=min_leaf,
                bootstrap=True,
                max_features=max_features,
                random_state=seed + fold_id,
                n_jobs=-1,
            )
            model.fit(Xtr_imp, y_fit[tr_idx])
            fold_pred_sum += model.predict(Xva_imp)
            fold_test_sum += model.predict(Xte_imp)
            del model
            gc.collect()

        fold_pred = fold_pred_sum / len(ET_SETTINGS)
        fold_test = fold_test_sum / len(ET_SETTINGS)
        oof[va_idx] = fold_pred
        covered[va_idx] = True
        test_pred += fold_test / len(splits)
        fold_score = pearson(y_eval[va_idx], fold_pred)
        fold_scores.append(fold_score)
        print(
            f"  [{branch_name}] fold {fold_id}: Pearson={fold_score:.6f}  "
            f"n_features={len(num_cols)}  interactions={len(ix_cols)}"
        )

    oof_score, cov_frac = oof_pearson_on_covered(y_eval, oof, covered)
    return {
        "model_name": branch_name,
        "oof_pred": oof,
        "test_pred": test_pred,
        "oof_corr": oof_score,
        "coverage_frac": cov_frac,
        "covered": covered,
        "target": y_eval,
        "test_ids": df_test_model[ID_COL].to_numpy(),
        "fold_scores": fold_scores,
        "recent_fold_mean": float(np.nanmean(fold_scores[3:5])),
        "last_fold_score": float(fold_scores[4]),
        "n_features_mean": float(np.mean(n_features)) if n_features else float("nan"),
    }


def build_production_model_summary(results: list[dict[str, object]]) -> pd.DataFrame:
    by_name = {str(res["model_name"]): res for res in results}
    control = by_name[BRANCH_CONTROL]
    rows = []
    for res in results:
        delta_oof = float(res["oof_corr"]) - float(control["oof_corr"])
        delta_recent = float(res["recent_fold_mean"]) - float(control["recent_fold_mean"])
        row = {
            "model_name": str(res["model_name"]),
            "oof_pearson": float(res["oof_corr"]),
            "coverage_frac": float(res["coverage_frac"]),
            "recent_fold_mean": float(res["recent_fold_mean"]),
            "last_fold_score": float(res["last_fold_score"]),
            "n_features_mean": float(res["n_features_mean"]),
            "delta_vs_control_oof": delta_oof,
            "delta_vs_control_recent": delta_recent,
        }
        for i, score in enumerate(res["fold_scores"]):
            row[f"fold_{i}"] = float(score)
        rows.append(row)
    col_order = [
        "model_name",
        "oof_pearson",
        "coverage_frac",
        "fold_0",
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "recent_fold_mean",
        "last_fold_score",
        "n_features_mean",
        "delta_vs_control_oof",
        "delta_vs_control_recent",
    ]
    df = (
        pd.DataFrame(rows)[col_order]
        .sort_values(["recent_fold_mean", "oof_pearson"], ascending=[False, False])
        .reset_index(drop=True)
    )
    df.to_csv(ARTIFACT_DIR / "production_model_summary.csv", index=False)
    return df


def main() -> None:
    print("Loading data...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    print(f"  train: {df_train.shape}  test: {df_test.shape}")

    meta_cols, anon_cols, anon_numeric_cols = get_feature_sets(df_train)
    print(f"\n  meta_cols ({len(meta_cols)}): {meta_cols}")
    print(f"  anon_cols: {len(anon_cols)}  |  anon_numeric: {len(anon_numeric_cols)}")

    splits = walk_forward_time_splits(df_train[TIME_COL])
    print("\nWalk-Forward Folds:")
    for i, (tr, va) in enumerate(splits):
        tr_di = df_train.iloc[tr][TIME_COL]
        va_di = df_train.iloc[va][TIME_COL]
        print(
            f"  fold {i}: train di [{tr_di.min()}–{tr_di.max()}]  "
            f"valid di [{va_di.min()}–{va_di.max()}]  "
            f"n_train={len(tr):,}  n_valid={len(va):,}  "
            f"embargo_ok={tr_di.max() < va_di.min()}"
        )

    print("\n" + "=" * 60)
    print("Building fixed D7 representation")
    print("=" * 60)
    base_cols = meta_cols + anon_cols
    df_train_rank = add_rank_features(df_train, anon_numeric_cols, group_col=TIME_COL, suffix="_csrank")
    df_test_rank = add_rank_features(df_test, anon_numeric_cols, group_col=TIME_COL, suffix="_csrank")
    rank_cols = [f"{c}_csrank" for c in anon_numeric_cols if f"{c}_csrank" in df_train_rank.columns]
    df_train_z_raw = cross_sectional_zscore_anonymous(df_train, anon_numeric_cols)
    df_test_z_raw = cross_sectional_zscore_anonymous(df_test, anon_numeric_cols)
    z_cols = [f"{c}_z" for c in anon_numeric_cols]
    df_train_z_block = df_train_z_raw[anon_numeric_cols].copy()
    df_test_z_block = df_test_z_raw[anon_numeric_cols].copy()
    df_train_z_block.columns = z_cols
    df_test_z_block.columns = z_cols

    sparse_cols = get_sparse_anon_cols(df_train, anon_cols, nan_frac_threshold=0.20)
    df_train_sparse = add_sparse_indicators(df_train, sparse_cols)
    df_test_sparse = add_sparse_indicators(df_test, sparse_cols)
    sparse_derived: list[str] = []
    for c in sparse_cols:
        sparse_derived.extend([f"{c}_isna", f"{c}_filled0", f"{c}_present_x_value"])

    reps = get_default_reps(anon_cols, n_reps=12)
    print(f"  sparse anonymous cols (>=20% NaN): {len(sparse_cols)}")
    print(f"  family representatives for fold-local interactions: {reps}")

    df_train_d4 = pd.concat([df_train_rank, df_train_z_block], axis=1)
    df_test_d4 = pd.concat([df_test_rank, df_test_z_block], axis=1)
    df_train_d6 = pd.concat([df_train_d4, df_train_sparse[sparse_derived]], axis=1)
    df_test_d6 = pd.concat([df_test_d4, df_test_sparse[sparse_derived]], axis=1)
    d7_base_feat_cols = base_cols + rank_cols + z_cols + sparse_derived
    print(f"  D7 base feature count before fold-local interactions: {len(d7_base_feat_cols)}")

    print("\n" + "=" * 60)
    print("Discovering latent anonymous-feature families")
    print("=" * 60)
    family_map = build_feature_family_map(df_train, anon_cols)
    (ARTIFACT_DIR / "family_map.json").write_text(json.dumps(family_map, indent=2))
    set_family_standardization_stats(df_train, anon_cols)
    df_train_family, family_factor_cols = build_family_factor_block(df_train_d6, family_map)
    df_test_family, factor_cols_test = build_family_factor_block(df_test_d6, family_map)
    assert family_factor_cols == factor_cols_test, "Family factor columns must match between train and test"
    print(f"  discovered families: {len(family_map)}")
    print(f"  family factor columns: {len(family_factor_cols)}")

    raw_target = df_train[TARGET_COL].to_numpy(np.float64)
    cs_target = build_cross_sectional_target(df_train)

    print("\n" + "=" * 60)
    print("Running control and factor-direction branches")
    print("=" * 60)
    control_result = run_et_branch(
        BRANCH_CONTROL,
        df_train_d6,
        df_test_d6,
        d7_base_feat_cols,
        y_fit=raw_target,
        y_eval=raw_target,
        splits=splits,
        meta_cols=meta_cols,
        reps=reps,
    )
    factor_result = run_et_branch(
        BRANCH_FACTOR,
        df_train_family,
        df_test_family,
        d7_base_feat_cols + family_factor_cols,
        y_fit=cs_target,
        y_eval=raw_target,
        splits=splits,
        meta_cols=meta_cols,
        reps=reps,
    )

    save_oof_artifact(ROOT_OUT / "oof_D7_et_base_control.csv", raw_target, control_result["oof_pred"], control_result["covered"])
    save_oof_artifact(ROOT_OUT / "oof_D7_et_factor_direction_only.csv", raw_target, factor_result["oof_pred"], factor_result["covered"])
    save_submission(ROOT_OUT / "submission_D7_et_base_control.csv", control_result["test_ids"], control_result["test_pred"], clip=True)
    save_submission(ROOT_OUT / "submission_D7_et.csv", control_result["test_ids"], control_result["test_pred"], clip=True)
    save_submission(ROOT_OUT / "submission_D7_et_factor_direction_only.csv", factor_result["test_ids"], factor_result["test_pred"], clip=True)

    results = [control_result, factor_result]
    df_summary = build_production_model_summary(results)
    print("\n" + "=" * 60)
    print("PRODUCTION MODEL SUMMARY")
    print("=" * 60)
    print(df_summary.to_string(index=False))

    delta_oof = float(factor_result["oof_corr"]) - float(control_result["oof_corr"])
    delta_recent = float(factor_result["recent_fold_mean"]) - float(control_result["recent_fold_mean"])
    unstable = (
        float(factor_result["last_fold_score"]) > float(control_result["last_fold_score"])
        and float(factor_result["recent_fold_mean"]) <= float(control_result["recent_fold_mean"])
    )

    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATION")
    print("=" * 60)
    if delta_oof >= 0.002 and not unstable:
        recommendation = f"promote {BRANCH_FACTOR}"
    else:
        recommendation = "keep control"
    if unstable:
        print("F1 is unstable: it improved fold 4 without improving recent_fold_mean.")
    print(recommendation)


if __name__ == "__main__":
    main()
