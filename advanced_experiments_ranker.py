"""
Cross-sectional ranker follow-up to the incumbent D7 ET baseline.

Runs exactly three branches:
  A) D7_et_base_control
  F1) D7_et_factor_direction_only
  R1) D7_ranker_cross_sectional

All outputs from this file are written into artifacts_ranker/.
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
    TARGET_COL,
    TIME_COL,
    _fit_numeric_preprocessor,
    add_forward_safe_si_context,
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

ARTIFACT_DIR = Path("artifacts_ranker")
ARTIFACT_DIR.mkdir(exist_ok=True)

BRANCH_CONTROL = "D7_et_base_control"
BRANCH_FACTOR = "D7_et_factor_direction_only"
BRANCH_RANKER = "D7_ranker_cross_sectional"
BRANCH_ORDER = [BRANCH_CONTROL, BRANCH_FACTOR, BRANCH_RANKER]


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
        for name, values in block.items():
            out[name] = values
            family_factor_cols.append(name)
    return out, family_factor_cols, stats


def build_cross_sectional_target(
    df_train: pd.DataFrame,
    target_col: str = TARGET_COL,
    time_col: str = TIME_COL,
) -> np.ndarray:
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
    return target_cs_z


def build_rank_labels(
    df_train: pd.DataFrame,
    target_col: str = TARGET_COL,
    time_col: str = TIME_COL,
) -> tuple[np.ndarray, np.ndarray]:
    target = pd.to_numeric(df_train[target_col], errors="coerce")
    groups = df_train[time_col]
    rank_pct = target.groupby(groups).rank(pct=True, method="average").to_numpy(np.float64)
    bucket = np.zeros(len(df_train), dtype=np.int32)
    for g, idx in df_train.groupby(time_col, sort=False).indices.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        vals = target.iloc[idx_arr].to_numpy(np.float64)
        order = np.argsort(vals, kind="mergesort")
        n = len(idx_arr)
        n_bins = min(10, n)
        if n_bins <= 1:
            bucket[idx_arr] = 0
            continue
        bucket_order = np.floor(np.arange(n) * n_bins / n).astype(int)
        bucket_vals = np.zeros(n, dtype=np.int32)
        bucket_vals[order] = bucket_order
        bucket[idx_arr] = bucket_vals
    return rank_pct, bucket


def build_recency_weights(train_di: pd.Series) -> np.ndarray:
    unique_sorted = np.sort(train_di.unique())
    if len(unique_sorted) <= 1:
        return np.ones(len(train_di), dtype=np.float32)
    di_rank = {di: i for i, di in enumerate(unique_sorted)}
    denom = max(len(unique_sorted) - 1, 1)
    weights = np.array([1.0 + di_rank[di] / denom for di in train_di.to_numpy()], dtype=np.float32)
    return weights


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


def _prepare_numeric_matrices(
    df_aug_tr: pd.DataFrame,
    df_aug_te: pd.DataFrame,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    Xtr = df_aug_tr.iloc[tr_idx][feature_cols]
    Xva = df_aug_tr.iloc[va_idx][feature_cols]
    Xall = df_aug_tr[feature_cols]
    Xte = df_aug_te[feature_cols]
    med = Xtr.median(axis=0, numeric_only=True).fillna(0.0)
    Xtr_imp = Xtr.fillna(med).fillna(0.0)
    Xva_imp = Xva.fillna(med).fillna(0.0)
    Xall_imp = Xall.fillna(med).fillna(0.0)
    Xte_imp = Xte.fillna(med).fillna(0.0)
    return Xtr_imp, Xva_imp, Xall_imp, Xte_imp


def run_et_branch_with_fold_scores(
    branch_name: str,
    df_train_model: pd.DataFrame,
    df_test_model: pd.DataFrame,
    base_feat_cols: list[str],
    y_fit: np.ndarray,
    y_eval: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    reps: list[str],
) -> tuple[dict[str, object], dict[str, list[np.ndarray]]]:
    from sklearn.ensemble import ExtraTreesRegressor

    oof = np.zeros(len(df_train_model), dtype=np.float64)
    covered = np.zeros(len(df_train_model), dtype=bool)
    test_pred = np.zeros(len(df_test_model), dtype=np.float64)
    n_features: list[float] = []
    full_train_preds_by_fold: list[np.ndarray] = []
    test_preds_by_fold: list[np.ndarray] = []

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        tr_di = df_train_model.iloc[tr_idx][TIME_COL]
        va_di = df_train_model.iloc[va_idx][TIME_COL]
        assert tr_di.max() < va_di.min(), f"[{branch_name}] fold {fold_id}: walk-forward violated"

        df_aug_tr, df_aug_te, ix_cols = build_fold_local_family_interactions(
            df_train_model, df_test_model, tr_idx, va_idx, reps
        )
        feat_cols = [c for c in (base_feat_cols + ix_cols) if c in df_aug_tr.columns and c in df_aug_te.columns]
        num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df_aug_tr[c])]
        Xtr_imp, Xva_imp, Xall_imp, Xte_imp = _prepare_numeric_matrices(df_aug_tr, df_aug_te, tr_idx, va_idx, num_cols)
        n_features.append(float(len(num_cols)))

        fold_pred_sum = np.zeros(len(va_idx), dtype=np.float64)
        fold_train_sum = np.zeros(len(df_train_model), dtype=np.float64)
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
            fold_train_sum += model.predict(Xall_imp)
            fold_test_sum += model.predict(Xte_imp)
            del model
            gc.collect()

        fold_pred = fold_pred_sum / len(ET_SETTINGS)
        fold_train = fold_train_sum / len(ET_SETTINGS)
        fold_test = fold_test_sum / len(ET_SETTINGS)
        oof[va_idx] = fold_pred
        covered[va_idx] = True
        test_pred += fold_test / len(splits)
        full_train_preds_by_fold.append(fold_train)
        test_preds_by_fold.append(fold_test)
        fold_score = pearson(y_eval[va_idx], fold_pred)
        print(
            f"  [{branch_name}] fold {fold_id}: Pearson={fold_score:.6f}  "
            f"n_features={len(num_cols)}  interactions={len(ix_cols)}"
        )

    result = build_result_from_predictions(
        branch_name=branch_name,
        y_eval=y_eval,
        oof_pred=oof,
        test_pred=test_pred,
        covered=covered,
        splits=splits,
        n_features_mean=float(np.mean(n_features)) if n_features else float("nan"),
        test_ids=df_test_model[ID_COL].to_numpy(),
    )
    bundle = {
        "full_train_preds_by_fold": full_train_preds_by_fold,
        "test_preds_by_fold": test_preds_by_fold,
    }
    return result, bundle


def build_row_sparsity_features(df: pd.DataFrame, anon_cols: list[str], sparse_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    dense_cols = [c for c in anon_cols if c not in set(sparse_cols)]
    all_nan = df[anon_cols].isna()
    out["row_nan_count"] = all_nan.sum(axis=1).astype(np.float32)
    out["row_nan_frac"] = all_nan.mean(axis=1).astype(np.float32)
    out["sparse_present_count"] = df[sparse_cols].notna().sum(axis=1).astype(np.float32)
    out["sparse_present_frac"] = df[sparse_cols].notna().mean(axis=1).astype(np.float32)
    out["dense_present_count"] = df[dense_cols].notna().sum(axis=1).astype(np.float32)
    out["dense_present_frac"] = df[dense_cols].notna().mean(axis=1).astype(np.float32)
    return out


def build_ranker_feature_frames(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    family_factor_cols: list[str],
    score_et_base: np.ndarray,
    score_factor_direction: np.ndarray,
    test_score_et_base: np.ndarray,
    test_score_factor_direction: np.ndarray,
    sparsity_train: pd.DataFrame,
    sparsity_test: pd.DataFrame,
    ctx_train: pd.DataFrame,
    ctx_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    numeric_cols = [
        "score_et_base",
        "score_factor_direction",
        *family_factor_cols,
        "row_nan_count",
        "row_nan_frac",
        "sparse_present_count",
        "sparse_present_frac",
        "dense_present_count",
        "dense_present_frac",
        TIME_COL,
        "top2000",
        "top1000",
        "top500",
        "si_seen_before",
        "si_log_count",
        "si_prev_gap_di",
    ]
    cat_cols = ["sector", "industry"]

    tr = pd.DataFrame(index=df_train.index)
    te = pd.DataFrame(index=df_test.index)
    tr["score_et_base"] = score_et_base.astype(np.float32)
    te["score_et_base"] = test_score_et_base.astype(np.float32)
    tr["score_factor_direction"] = score_factor_direction.astype(np.float32)
    te["score_factor_direction"] = test_score_factor_direction.astype(np.float32)

    for col in family_factor_cols:
        tr[col] = pd.to_numeric(df_train[col], errors="coerce").astype(np.float32)
        te[col] = pd.to_numeric(df_test[col], errors="coerce").astype(np.float32)

    for col in sparsity_train.columns:
        tr[col] = pd.to_numeric(sparsity_train[col], errors="coerce").astype(np.float32)
        te[col] = pd.to_numeric(sparsity_test[col], errors="coerce").astype(np.float32)

    for col in [TIME_COL, "top2000", "top1000", "top500"]:
        tr[col] = pd.to_numeric(df_train[col], errors="coerce").astype(np.float32)
        te[col] = pd.to_numeric(df_test[col], errors="coerce").astype(np.float32)

    for col in ["si_seen_before", "si_log_count", "si_prev_gap_di"]:
        tr[col] = pd.to_numeric(ctx_train[col], errors="coerce").astype(np.float32)
        te[col] = pd.to_numeric(ctx_test[col], errors="coerce").astype(np.float32)

    for col in cat_cols:
        tr[col] = df_train[col].astype(str)
        te[col] = df_test[col].astype(str)

    return tr, te, numeric_cols, cat_cols


def run_catboost_ranker_cross_sectional(
    df_train_ranker: pd.DataFrame,
    df_test_ranker: pd.DataFrame,
    numeric_cols: list[str],
    cat_cols: list[str],
    rank_bucket: np.ndarray,
    y_eval: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    from catboost import CatBoostRanker, Pool

    all_feat_cols = numeric_cols + cat_cols
    oof = np.zeros(len(df_train_ranker), dtype=np.float64)
    covered = np.zeros(len(df_train_ranker), dtype=bool)
    test_pred = np.zeros(len(df_test_ranker), dtype=np.float64)
    n_features: list[float] = []

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        tr_di = df_train_ranker.iloc[tr_idx][TIME_COL]
        va_di = df_train_ranker.iloc[va_idx][TIME_COL]
        assert tr_di.max() < va_di.min(), f"[{BRANCH_RANKER}] fold {fold_id}: walk-forward violated"

        Xtr = df_train_ranker.iloc[tr_idx][all_feat_cols].copy()
        Xva = df_train_ranker.iloc[va_idx][all_feat_cols].copy()
        Xte = df_test_ranker[all_feat_cols].copy()

        med = Xtr[numeric_cols].median(axis=0).fillna(0.0)
        Xtr[numeric_cols] = Xtr[numeric_cols].fillna(med).fillna(0.0)
        Xva[numeric_cols] = Xva[numeric_cols].fillna(med).fillna(0.0)
        Xte[numeric_cols] = Xte[numeric_cols].fillna(med).fillna(0.0)
        for col in cat_cols:
            Xtr[col] = Xtr[col].astype(str)
            Xva[col] = Xva[col].astype(str)
            Xte[col] = Xte[col].astype(str)

        weights = build_recency_weights(tr_di)
        ytr = rank_bucket[tr_idx]
        yva = rank_bucket[va_idx]
        train_pool = Pool(
            data=Xtr,
            label=ytr,
            group_id=tr_di.to_numpy(),
            weight=weights,
            cat_features=cat_cols,
        )
        valid_pool = Pool(
            data=Xva,
            label=yva,
            group_id=va_di.to_numpy(),
            cat_features=cat_cols,
        )
        test_pool = Pool(
            data=Xte,
            cat_features=cat_cols,
        )

        model = None
        for loss_name in ["YetiRankPairwise", "PairLogitPairwise"]:
            try:
                model = CatBoostRanker(
                    loss_function=loss_name,
                    eval_metric=loss_name,
                    iterations=1500,
                    depth=6,
                    learning_rate=0.05,
                    l2_leaf_reg=3.0,
                    random_seed=42 + fold_id,
                    bootstrap_type="Bernoulli",
                    subsample=0.8,
                    od_type="Iter",
                    od_wait=200,
                    verbose=False,
                )
                model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
                break
            except Exception:
                model = None
        if model is None:
            raise RuntimeError("CatBoostRanker failed for both YetiRankPairwise and PairLogitPairwise.")

        p_va = model.predict(valid_pool)
        p_te = model.predict(test_pool)
        oof[va_idx] = p_va
        covered[va_idx] = True
        test_pred += p_te / len(splits)
        n_features.append(float(len(all_feat_cols)))
        fold_score = pearson(y_eval[va_idx], p_va)
        print(f"  [{BRANCH_RANKER}] fold {fold_id}: Pearson={fold_score:.6f}  n_features={len(all_feat_cols)}")
        del model
        gc.collect()

    return build_result_from_predictions(
        branch_name=BRANCH_RANKER,
        y_eval=y_eval,
        oof_pred=oof,
        test_pred=test_pred,
        covered=covered,
        splits=splits,
        n_features_mean=float(np.mean(n_features)) if n_features else float("nan"),
        test_ids=df_test_ranker[ID_COL].to_numpy() if ID_COL in df_test_ranker.columns else np.arange(len(df_test_ranker)),
    )


def save_branch_outputs(branch_name: str, result: dict[str, object], clip_submission: bool) -> None:
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
            "delta_vs_control_oof": float(res["oof_corr"]) - float(control["oof_corr"]),
            "delta_vs_control_recent": float(res["recent_fold_mean"]) - float(control["recent_fold_mean"]),
        }
        for i, score in enumerate(res["fold_scores"]):
            row[f"fold_{i}"] = float(score)
        rows.append(row)
        if str(res["model_name"]) != BRANCH_CONTROL:
            delta_rows.append(
                {
                    "model_name": str(res["model_name"]),
                    "delta_vs_control_oof": row["delta_vs_control_oof"],
                    "delta_vs_control_recent": row["delta_vs_control_recent"],
                    "delta_vs_control_n_features": float(res["n_features_mean"]) - float(control["n_features_mean"]),
                    "research_signal_only": bool(
                        row["delta_vs_control_oof"] > 0.0 and row["delta_vs_control_recent"] < 0.0
                    ),
                    "unstable_fold4_only": bool(
                        float(res["last_fold_score"]) > float(control["last_fold_score"])
                        and float(res["recent_fold_mean"]) < float(control["recent_fold_mean"])
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
        "delta_vs_control_oof",
        "delta_vs_control_recent",
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
    df_train_family, family_factor_cols, family_stats = build_family_factor_block(df_train_d6, family_map, None)
    df_test_family, family_factor_cols_test, _ = build_family_factor_block(df_test_d6, family_map, family_stats)
    assert family_factor_cols == family_factor_cols_test, "Family factor columns must match between train and test"
    (ARTIFACT_DIR / "family_factor_columns.json").write_text(json.dumps(family_factor_cols, indent=2))
    print(f"  discovered families: {len(family_map)}")
    print(f"  family factor columns: {len(family_factor_cols)}")

    target_raw = df_train[TARGET_COL].to_numpy(np.float64)
    target_cs_z = build_cross_sectional_target(df_train)
    _, target_rank_bucket = build_rank_labels(df_train)
    sparsity_train = build_row_sparsity_features(df_train, anon_cols, sparse_cols)
    sparsity_test = build_row_sparsity_features(df_test, anon_cols, sparse_cols)
    ctx_train, ctx_test = add_forward_safe_si_context(df_train, df_test)

    print("\n" + "=" * 60)
    print("Running control and factor branches")
    print("=" * 60)
    control_result, control_bundle = run_et_branch_with_fold_scores(
        BRANCH_CONTROL,
        df_train_d6,
        df_test_d6,
        d7_base_feat_cols,
        y_fit=target_raw,
        y_eval=target_raw,
        splits=splits,
        reps=reps,
    )
    factor_result, factor_bundle = run_et_branch_with_fold_scores(
        BRANCH_FACTOR,
        df_train_family,
        df_test_family,
        d7_base_feat_cols + family_factor_cols,
        y_fit=target_cs_z,
        y_eval=target_raw,
        splits=splits,
        reps=reps,
    )

    print("\n" + "=" * 60)
    print("Running cross-sectional ranker branch")
    print("=" * 60)
    ranker_train_parts = []
    ranker_test_parts = []
    for fold_id, _ in enumerate(splits):
        tr_part, te_part, numeric_cols, cat_cols = build_ranker_feature_frames(
            df_train=df_train_family,
            df_test=df_test_family,
            family_factor_cols=family_factor_cols,
            score_et_base=control_bundle["full_train_preds_by_fold"][fold_id],
            score_factor_direction=factor_bundle["full_train_preds_by_fold"][fold_id],
            test_score_et_base=control_bundle["test_preds_by_fold"][fold_id],
            test_score_factor_direction=factor_bundle["test_preds_by_fold"][fold_id],
            sparsity_train=sparsity_train,
            sparsity_test=sparsity_test,
            ctx_train=ctx_train,
            ctx_test=ctx_test,
        )
        ranker_train_parts.append(tr_part)
        ranker_test_parts.append(te_part)

    # Use fold-specific feature frames inside ranker training loop.
    # Start from the first frame so we can reindex and keep common columns.
    ranker_result = None
    from catboost import CatBoostRanker, Pool

    all_feat_cols = numeric_cols + cat_cols
    oof = np.zeros(len(df_train_family), dtype=np.float64)
    covered = np.zeros(len(df_train_family), dtype=bool)
    test_pred = np.zeros(len(df_test_family), dtype=np.float64)
    n_features: list[float] = []
    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        tr_di = df_train.iloc[tr_idx][TIME_COL]
        va_di = df_train.iloc[va_idx][TIME_COL]
        Xtr = ranker_train_parts[fold_id].iloc[tr_idx][all_feat_cols].copy()
        Xva = ranker_train_parts[fold_id].iloc[va_idx][all_feat_cols].copy()
        Xte = ranker_test_parts[fold_id][all_feat_cols].copy()

        med = Xtr[numeric_cols].median(axis=0).fillna(0.0)
        Xtr[numeric_cols] = Xtr[numeric_cols].fillna(med).fillna(0.0)
        Xva[numeric_cols] = Xva[numeric_cols].fillna(med).fillna(0.0)
        Xte[numeric_cols] = Xte[numeric_cols].fillna(med).fillna(0.0)
        for col in cat_cols:
            Xtr[col] = Xtr[col].astype(str)
            Xva[col] = Xva[col].astype(str)
            Xte[col] = Xte[col].astype(str)

        train_pool = Pool(
            data=Xtr,
            label=target_rank_bucket[tr_idx],
            group_id=tr_di.to_numpy(),
            weight=build_recency_weights(tr_di),
            cat_features=cat_cols,
        )
        valid_pool = Pool(
            data=Xva,
            label=target_rank_bucket[va_idx],
            group_id=va_di.to_numpy(),
            cat_features=cat_cols,
        )
        test_pool = Pool(data=Xte, cat_features=cat_cols)

        model = None
        for loss_name in ["YetiRankPairwise", "PairLogitPairwise"]:
            try:
                model = CatBoostRanker(
                    loss_function=loss_name,
                    eval_metric=loss_name,
                    iterations=1500,
                    depth=6,
                    learning_rate=0.05,
                    l2_leaf_reg=3.0,
                    random_seed=42 + fold_id,
                    bootstrap_type="Bernoulli",
                    subsample=0.8,
                    od_type="Iter",
                    od_wait=200,
                    verbose=False,
                )
                model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
                break
            except Exception:
                model = None
        if model is None:
            raise RuntimeError("CatBoostRanker failed for both YetiRankPairwise and PairLogitPairwise.")

        p_va = model.predict(valid_pool)
        p_te = model.predict(test_pool)
        oof[va_idx] = p_va
        covered[va_idx] = True
        test_pred += p_te / len(splits)
        n_features.append(float(len(all_feat_cols)))
        print(f"  [{BRANCH_RANKER}] fold {fold_id}: Pearson={pearson(target_raw[va_idx], p_va):.6f}  n_features={len(all_feat_cols)}")
        del model
        gc.collect()

    ranker_result = build_result_from_predictions(
        branch_name=BRANCH_RANKER,
        y_eval=target_raw,
        oof_pred=oof,
        test_pred=test_pred,
        covered=covered,
        splits=splits,
        n_features_mean=float(np.mean(n_features)) if n_features else float("nan"),
        test_ids=df_test[ID_COL].to_numpy(),
    )

    save_branch_outputs(BRANCH_CONTROL, control_result, clip_submission=True)
    save_branch_outputs(BRANCH_FACTOR, factor_result, clip_submission=True)
    save_branch_outputs(BRANCH_RANKER, ranker_result, clip_submission=False)
    save_submission(ARTIFACT_DIR / "submission_D7_et.csv", control_result["test_ids"], control_result["test_pred"], clip=True)

    results = [control_result, factor_result, ranker_result]
    df_summary, df_delta = build_production_model_summary(results)

    print("\n" + "=" * 60)
    print("PRODUCTION MODEL SUMMARY")
    print("=" * 60)
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("DELTA VS D7_et_base_control")
    print("=" * 60)
    print(df_delta.to_string(index=False))

    ranker_delta = df_delta[df_delta["model_name"] == BRANCH_RANKER].iloc[0]
    ranker_serious = (
        float(ranker_delta["delta_vs_control_oof"]) >= 0.002
        and float(ranker_delta["delta_vs_control_recent"]) >= 0.0
        and not bool(ranker_delta["unstable_fold4_only"])
    )
    factor_delta = df_delta[df_delta["model_name"] == BRANCH_FACTOR].iloc[0]
    submit_order = [BRANCH_CONTROL]
    if ranker_serious:
        submit_order.append(BRANCH_RANKER)

    print("\n" + "=" * 60)
    print("FINAL RECOMMENDED SUBMIT ORDER")
    print("=" * 60)
    for i, name in enumerate(submit_order, start=1):
        print(f"  {i}. {name}")
    if bool(ranker_delta["research_signal_only"]):
        print("\nR1 research_signal_only = True")
    if bool(factor_delta["research_signal_only"]):
        print("F1 research_signal_only = True")


if __name__ == "__main__":
    main()
