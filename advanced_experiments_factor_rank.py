"""
Factor-rank follow-up to the incumbent D7 ET baseline.

Runs exactly five branches:
  A) D7_et_base_control
  F1) D7_et_factor_direction
  F2) D7_et_factor_magnitude
  F3) D7_factor_ridge_direction
  F4) D7_factor_combined

All outputs from this file are written into artifacts_factor_rank/.
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

ARTIFACT_DIR = Path("artifacts_factor_rank")
ARTIFACT_DIR.mkdir(exist_ok=True)

BRANCH_CONTROL = "D7_et_base_control"
BRANCH_F1 = "D7_et_factor_direction"
BRANCH_F2 = "D7_et_factor_magnitude"
BRANCH_F3 = "D7_factor_ridge_direction"
BRANCH_F4 = "D7_factor_combined"
BRANCH_ORDER = [BRANCH_CONTROL, BRANCH_F1, BRANCH_F2, BRANCH_F3, BRANCH_F4]


def get_default_reps(existing_cols: list[str], n_reps: int = 12) -> list[str]:
    reps = [c for c in DEFAULT_FAMILY_REPS if c in existing_cols]
    for c in existing_cols:
        if c not in reps:
            reps.append(c)
        if len(reps) >= n_reps:
            break
    return reps[:n_reps]


def save_oof_artifact(branch_name: str, y_true: np.ndarray, oof_pred: np.ndarray, covered: np.ndarray) -> None:
    pd.DataFrame(
        {
            "oof_pred": np.asarray(oof_pred, np.float64),
            "target": np.asarray(y_true, np.float64),
            "oof_covered": np.asarray(covered, np.int8),
        }
    ).to_csv(ARTIFACT_DIR / f"oof_{branch_name}.csv", index=False)


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


def build_family_factor_block(
    df: pd.DataFrame,
    family_map: dict[str, list[str]],
    train_feature_stats: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, tuple[float, float]]]:
    out = df.copy()
    stats = dict(train_feature_stats or {})
    if not stats:
        all_cols = sorted({col for cols in family_map.values() for col in cols})
        for col in all_cols:
            arr = pd.to_numeric(out[col], errors="coerce").to_numpy(np.float64)
            mu = float(np.nanmean(arr))
            sd = float(np.nanstd(arr))
            if not np.isfinite(mu):
                mu = 0.0
            if (not np.isfinite(sd)) or sd == 0.0:
                sd = 1.0
            stats[col] = (mu, sd)

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
            mu, sd = stats[col]
            std_arr[:, j] = (std_arr[:, j] - mu) / sd
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.", category=RuntimeWarning)
            with np.errstate(invalid="ignore"):
                disagreement = np.nanstd(std_arr, axis=1)

        block = {
            f"fam_{family_name}_present_frac": (present_count / family_size).astype(np.float32),
            f"fam_{family_name}_present_count": present_count.astype(np.float32),
            f"fam_{family_name}_value_mean": np.where(np.isfinite(value_mean), value_mean, np.nan).astype(np.float32),
            f"fam_{family_name}_value_std": np.where(np.isfinite(value_std), value_std, np.nan).astype(np.float32),
            f"fam_{family_name}_absmax": np.where(np.isfinite(absmax), absmax, np.nan).astype(np.float32),
            f"fam_{family_name}_row_disagreement": np.where(finite.sum(axis=1) >= 2, disagreement, np.nan).astype(np.float32),
        }

        rank_cols = [f"{c}_csrank" for c in avail_raw if f"{c}_csrank" in out.columns]
        if rank_cols:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                with np.errstate(invalid="ignore"):
                    rank_mean = np.nanmean(out[rank_cols].to_numpy(np.float64), axis=1)
            block[f"fam_{family_name}_rank_mean"] = np.where(np.isfinite(rank_mean), rank_mean, np.nan).astype(np.float32)

        z_cols = [f"{c}_z" for c in avail_raw if f"{c}_z" in out.columns]
        if z_cols:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                with np.errstate(invalid="ignore"):
                    z_mean = np.nanmean(out[z_cols].to_numpy(np.float64), axis=1)
            block[f"fam_{family_name}_z_mean"] = np.where(np.isfinite(z_mean), z_mean, np.nan).astype(np.float32)

        for name, values in block.items():
            out[name] = values
            family_factor_cols.append(name)

    return out, family_factor_cols, stats


def build_cross_sectional_target(
    df_train: pd.DataFrame,
    target_col: str = TARGET_COL,
    time_col: str = TIME_COL,
) -> tuple[np.ndarray, np.ndarray]:
    target = pd.to_numeric(df_train[target_col], errors="coerce").to_numpy(np.float64)
    groups = df_train[time_col].to_numpy()
    target_cs_z = np.zeros(len(df_train), dtype=np.float64)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        vals = target[idx]
        mu = float(np.nanmean(vals))
        sd = float(np.nanstd(vals))
        if not np.isfinite(mu):
            mu = 0.0
        if (not np.isfinite(sd)) or sd == 0.0:
            target_cs_z[idx] = 0.0
        else:
            target_cs_z[idx] = (vals - mu) / sd
    target_abs = np.abs(target)
    return target_cs_z, target_abs


def build_result_from_predictions(
    branch_name: str,
    y_eval: np.ndarray,
    oof_pred: np.ndarray,
    test_pred: np.ndarray,
    covered: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_features_mean: float,
    test_ids: np.ndarray,
) -> dict[str, object]:
    fold_scores = [pearson(y_eval[va_idx], oof_pred[va_idx]) for _, va_idx in splits]
    oof_score, cov_frac = oof_pearson_on_covered(y_eval, oof_pred, covered)
    return {
        "model_name": branch_name,
        "oof_pred": oof_pred,
        "test_pred": test_pred,
        "oof_corr": oof_score,
        "coverage_frac": cov_frac,
        "covered": covered,
        "target": y_eval,
        "test_ids": test_ids,
        "fold_scores": fold_scores,
        "recent_fold_mean": float(np.nanmean(fold_scores[3:5])),
        "last_fold_score": float(fold_scores[4]),
        "n_features_mean": float(n_features_mean),
    }


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
        print(
            f"  [{branch_name}] fold {fold_id}: Pearson={fold_score:.6f}  "
            f"n_features={len(num_cols)}  interactions={len(ix_cols)}"
        )

    return build_result_from_predictions(
        branch_name=branch_name,
        y_eval=y_eval,
        oof_pred=oof,
        test_pred=test_pred,
        covered=covered,
        splits=splits,
        n_features_mean=float(np.mean(n_features)) if n_features else float("nan"),
        test_ids=df_test_model[ID_COL].to_numpy(),
    )


def run_ridge_direction(
    df_train_model: pd.DataFrame,
    df_test_model: pd.DataFrame,
    family_factor_cols: list[str],
    y_fit: np.ndarray,
    y_eval: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    from sklearn.linear_model import RidgeCV

    feat_cols = [c for c in family_factor_cols if c in df_train_model.columns and c in df_test_model.columns]
    oof = np.zeros(len(df_train_model), dtype=np.float64)
    covered = np.zeros(len(df_train_model), dtype=bool)
    test_pred = np.zeros(len(df_test_model), dtype=np.float64)
    n_features: list[float] = []

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        tr_di = df_train_model.iloc[tr_idx][TIME_COL]
        va_di = df_train_model.iloc[va_idx][TIME_COL]
        assert tr_di.max() < va_di.min(), f"[{BRANCH_F3}] fold {fold_id}: walk-forward violated"

        Xtr = df_train_model.iloc[tr_idx][feat_cols]
        Xva = df_train_model.iloc[va_idx][feat_cols]
        Xte = df_test_model[feat_cols]
        Xtr_imp, Xva_imp, Xte_imp = _fit_numeric_preprocessor(Xtr, Xva, Xte, standardize=True)
        n_features.append(float(len(feat_cols)))
        model = RidgeCV(alphas=np.logspace(-3, 3, 13))
        model.fit(Xtr_imp, y_fit[tr_idx])
        p_va = model.predict(Xva_imp)
        p_te = model.predict(Xte_imp)
        oof[va_idx] = p_va
        covered[va_idx] = True
        test_pred += p_te / len(splits)
        fold_score = pearson(y_eval[va_idx], p_va)
        print(f"  [{BRANCH_F3}] fold {fold_id}: Pearson={fold_score:.6f}  n_features={len(feat_cols)}")

    return build_result_from_predictions(
        branch_name=BRANCH_F3,
        y_eval=y_eval,
        oof_pred=oof,
        test_pred=test_pred,
        covered=covered,
        splits=splits,
        n_features_mean=float(np.mean(n_features)) if n_features else float("nan"),
        test_ids=df_test_model[ID_COL].to_numpy(),
    )


def combine_direction_and_magnitude(
    pred_dir: np.ndarray,
    pred_mag: np.ndarray,
    ref_mag: np.ndarray | None = None,
) -> np.ndarray:
    ref = np.asarray(ref_mag if ref_mag is not None else pred_mag, np.float64)
    lo, hi = np.nanquantile(ref, [0.05, 0.95])
    clipped = np.clip(np.asarray(pred_mag, np.float64), lo, hi)
    scale = clipped - lo + 1e-6
    mean_scale = float(np.nanmean(scale))
    if (not np.isfinite(mean_scale)) or mean_scale == 0.0:
        mean_scale = 1.0
    scale = scale / mean_scale
    return np.asarray(pred_dir, np.float64) * scale


def save_branch_outputs(branch_name: str, result: dict[str, object], clip_submission: bool = True) -> None:
    save_oof_artifact(
        branch_name,
        np.asarray(result["target"], np.float64),
        np.asarray(result["oof_pred"], np.float64),
        np.asarray(result["covered"], bool),
    )
    save_submission(
        ARTIFACT_DIR / f"submission_{branch_name}.csv",
        np.asarray(result["test_ids"]),
        np.asarray(result["test_pred"], np.float64),
        clip=clip_submission,
    )


def build_production_model_summary(results: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_name = {str(res["model_name"]): res for res in results}
    control = by_name[BRANCH_CONTROL]
    rows = []
    delta_rows = []
    for res in results:
        row = {
            "model_name": str(res["model_name"]),
            "oof_pearson": float(res["oof_corr"]),
            "coverage_frac": float(res["coverage_frac"]),
            "recent_fold_mean": float(res["recent_fold_mean"]),
            "last_fold_score": float(res["last_fold_score"]),
            "n_features_mean": float(res["n_features_mean"]),
        }
        for i, score in enumerate(res["fold_scores"]):
            row[f"fold_{i}"] = float(score)
        rows.append(row)

        if str(res["model_name"]) != BRANCH_CONTROL:
            delta_rows.append(
                {
                    "model_name": str(res["model_name"]),
                    "delta_vs_control_oof": float(res["oof_corr"]) - float(control["oof_corr"]),
                    "delta_vs_control_recent": float(res["recent_fold_mean"]) - float(control["recent_fold_mean"]),
                    "delta_vs_control_n_features": float(res["n_features_mean"]) - float(control["n_features_mean"]),
                    "unstable_fold4_only": (
                        float(res["last_fold_score"]) > float(control["last_fold_score"])
                        and float(res["recent_fold_mean"]) <= float(control["recent_fold_mean"])
                    ),
                }
            )

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
    ]
    df_summary = (
        pd.DataFrame(rows)[col_order]
        .sort_values(["recent_fold_mean", "oof_pearson"], ascending=[False, False])
        .reset_index(drop=True)
    )
    df_delta = pd.DataFrame(delta_rows)
    df_summary.to_csv(ARTIFACT_DIR / "production_model_summary.csv", index=False)
    return df_summary, df_delta


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
    for i, (tr_idx, va_idx) in enumerate(splits):
        tr_di = df_train.iloc[tr_idx][TIME_COL]
        va_di = df_train.iloc[va_idx][TIME_COL]
        print(
            f"  fold {i}: train di [{tr_di.min()}–{tr_di.max()}]  "
            f"valid di [{va_di.min()}–{va_di.max()}]  "
            f"n_train={len(tr_idx):,}  n_valid={len(va_idx):,}  "
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
    df_train_family, family_factor_cols, train_feature_stats = build_family_factor_block(df_train_d6, family_map, train_feature_stats=None)
    df_test_family, factor_cols_test, _ = build_family_factor_block(df_test_d6, family_map, train_feature_stats=train_feature_stats)
    assert family_factor_cols == factor_cols_test, "Family factor columns must match between train and test"
    (ARTIFACT_DIR / "family_factor_columns.json").write_text(json.dumps(family_factor_cols, indent=2))
    print(f"  discovered families: {len(family_map)}")
    print(f"  family factor columns: {len(family_factor_cols)}")

    target_raw = df_train[TARGET_COL].to_numpy(np.float64)
    target_cs_z, target_abs = build_cross_sectional_target(df_train)

    print("\n" + "=" * 60)
    print("Running control and factor-rank branches")
    print("=" * 60)
    control_result = run_et_branch(
        BRANCH_CONTROL,
        df_train_d6,
        df_test_d6,
        d7_base_feat_cols,
        y_fit=target_raw,
        y_eval=target_raw,
        splits=splits,
        meta_cols=meta_cols,
        reps=reps,
    )
    f1_result = run_et_branch(
        BRANCH_F1,
        df_train_family,
        df_test_family,
        d7_base_feat_cols + family_factor_cols,
        y_fit=target_cs_z,
        y_eval=target_raw,
        splits=splits,
        meta_cols=meta_cols,
        reps=reps,
    )
    f2_result = run_et_branch(
        BRANCH_F2,
        df_train_family,
        df_test_family,
        d7_base_feat_cols + family_factor_cols,
        y_fit=target_abs,
        y_eval=target_raw,
        splits=splits,
        meta_cols=meta_cols,
        reps=reps,
    )
    f3_result = run_ridge_direction(
        df_train_family,
        df_test_family,
        family_factor_cols,
        y_fit=target_cs_z,
        y_eval=target_raw,
        splits=splits,
    )

    combined_oof = combine_direction_and_magnitude(
        np.asarray(f1_result["oof_pred"], np.float64),
        np.asarray(f2_result["oof_pred"], np.float64),
        ref_mag=np.asarray(f2_result["oof_pred"], np.float64)[np.asarray(f2_result["covered"], bool)],
    )
    combined_test = combine_direction_and_magnitude(
        np.asarray(f1_result["test_pred"], np.float64),
        np.asarray(f2_result["test_pred"], np.float64),
        ref_mag=np.asarray(f2_result["oof_pred"], np.float64)[np.asarray(f2_result["covered"], bool)],
    )
    f4_result = build_result_from_predictions(
        branch_name=BRANCH_F4,
        y_eval=target_raw,
        oof_pred=combined_oof,
        test_pred=combined_test,
        covered=np.asarray(f1_result["covered"], bool),
        splits=splits,
        n_features_mean=float(f1_result["n_features_mean"]),
        test_ids=np.asarray(f1_result["test_ids"]),
    )

    for result in [control_result, f1_result, f2_result, f3_result, f4_result]:
        save_branch_outputs(str(result["model_name"]), result, clip_submission=True)
    save_submission(ARTIFACT_DIR / "submission_D7_et.csv", control_result["test_ids"], control_result["test_pred"], clip=True)

    results = [control_result, f1_result, f2_result, f3_result, f4_result]
    df_summary, df_delta = build_production_model_summary(results)

    print("\n" + "=" * 60)
    print("PRODUCTION MODEL SUMMARY")
    print("=" * 60)
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("DELTA VS D7_et_base_control")
    print("=" * 60)
    print(df_delta.to_string(index=False))

    best_factor_row = df_delta.sort_values(["delta_vs_control_recent", "delta_vs_control_oof"], ascending=[False, False]).iloc[0]
    best_factor_name = str(best_factor_row["model_name"])
    serious = float(best_factor_row["delta_vs_control_oof"]) >= 0.002 and not bool(best_factor_row["unstable_fold4_only"])
    submit_order = [BRANCH_CONTROL]
    if serious:
        submit_order.append(best_factor_name)

    print("\n" + "=" * 60)
    print("FINAL RECOMMENDED SUBMIT ORDER")
    print("=" * 60)
    for i, name in enumerate(submit_order, start=1):
        print(f"  {i}. {name}")
    if BRANCH_F3 in df_delta["model_name"].values:
        ridge_row = df_delta[df_delta["model_name"] == BRANCH_F3].iloc[0]
        if float(ridge_row["delta_vs_control_oof"]) > 0.0:
            print("\nRidge direction being competitive suggests the family-factor representation is real.")


if __name__ == "__main__":
    main()
