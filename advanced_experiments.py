"""
Advanced experiments toward 0.3+ Pearson.

Phases:
  1. Per-feature raw correlation with target (find silver-bullet features)
  2. Cross-sectional normalization within each di group
  3. Fold-safe per-si historical mean target feature
  4. Stronger CatBoost (2000 iter) + LightGBM ensemble
  5. Save best submission to artifacts/
"""
from __future__ import annotations

import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# ── constants ──────────────────────────────────────────────────────────────────
TRAIN_PATH  = "train.csv"
TEST_PATH   = "test.csv"
OUTPUT_DIR  = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)

ID_COL      = "id"
TARGET_COL  = "target"
TIME_COL    = "di"
STOCK_COL   = "si"
CAT_COLS    = ["si", "industry", "sector", "top2000", "top1000", "top500"]
N_SPLITS    = 5


# ── helpers ────────────────────────────────────────────────────────────────────
def pearson(y_true, y_pred):
    y, p = np.asarray(y_true, np.float64), np.asarray(y_pred, np.float64)
    if y.size < 2 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def contiguous_time_splits(time_series, n_splits=N_SPLITS):
    unique_vals = np.sort(time_series.dropna().unique())
    chunks = np.array_split(unique_vals, n_splits)
    idx = np.arange(len(time_series))
    return [(idx[~time_series.isin(c).to_numpy()], idx[time_series.isin(c).to_numpy()])
            for c in chunks]


# ── Phase 1: per-feature correlation ──────────────────────────────────────────
def per_feature_correlation(df_train, feature_cols):
    print("\n" + "="*60)
    print("PHASE 1: Per-feature raw Pearson with target")
    print("="*60)
    y = df_train[TARGET_COL].to_numpy(np.float64)
    rows = []
    for col in feature_cols:
        vals = df_train[col]
        valid = vals.notna()
        if valid.sum() < 100:
            rows.append({"feature": col, "corr": 0.0, "n_valid": int(valid.sum())})
            continue
        c = pearson(y[valid.to_numpy()], vals[valid].to_numpy())
        rows.append({"feature": col, "corr": abs(c) if not np.isnan(c) else 0.0,
                     "corr_signed": c if not np.isnan(c) else 0.0,
                     "n_valid": int(valid.sum())})

    df_corr = pd.DataFrame(rows).sort_values("corr", ascending=False)
    df_corr.to_csv(OUTPUT_DIR / "feature_correlations.csv", index=False)

    print(f"\nTop 30 features by |corr| with target:")
    print(df_corr.head(30).to_string(index=False))
    top_signal = df_corr[df_corr["corr"] > 0.05]["feature"].tolist()
    print(f"\nFeatures with |corr| > 0.05: {len(top_signal)}")
    print(f"Max single-feature corr: {df_corr['corr'].max():.6f}")
    return df_corr


# ── Phase 2: cross-sectional normalization ────────────────────────────────────
def cross_sectional_normalize(df, feature_cols, group_col=TIME_COL):
    """
    Fast vectorized z-score within each date group.
    Computes group stats once via groupby().transform() with numpy backend,
    then does a single array subtract/divide — no per-column loops.
    """
    print(f"\nApplying cross-sectional z-score within {group_col} groups...")
    out = df.copy()
    f_cols = [c for c in feature_cols if c in df.columns and c != group_col]

    # compute group means/stds for all numeric feature cols at once
    grp = out.groupby(group_col, sort=False)
    means = grp[f_cols].transform("mean")
    stds  = grp[f_cols].transform("std").fillna(1.0)
    stds  = stds.replace(0, 1.0)

    out[f_cols] = (out[f_cols] - means) / (stds + 1e-8)
    print("Done.")
    return out


def add_rank_features(df, feature_cols, group_col=TIME_COL, top_n=30):
    """
    Fast vectorized within-date percentile rank for the top_n features.
    Uses a single groupby rank call per feature (rank is unavoidably per-column).
    """
    out = df.copy()
    rank_cols = [c for c in feature_cols[:top_n] if c in df.columns and c != group_col]
    grp = out.groupby(group_col, sort=False)
    for col in rank_cols:
        out[f"{col}_rank"] = grp[col].rank(pct=True, na_option="keep")
    return out


# ── Phase 3: fold-safe per-si historical mean ─────────────────────────────────
def add_si_target_history(df_train, df_test, splits):
    """
    For each fold: compute mean(target) per si on TRAIN portion only.
    This is fold-safe because we never see validation targets.
    For test: use full train mean per si.
    """
    print("\nBuilding fold-safe per-si historical mean target feature...")
    si_oof = np.zeros(len(df_train))
    global_si_mean = df_train.groupby(STOCK_COL)[TARGET_COL].mean()

    for tr_idx, va_idx in splits:
        tr = df_train.iloc[tr_idx]
        va = df_train.iloc[va_idx]
        si_means = tr.groupby(STOCK_COL)[TARGET_COL].mean()
        # fill val stocks not seen in train with global grand mean
        si_oof[va_idx] = va[STOCK_COL].map(si_means).fillna(df_train[TARGET_COL].mean()).to_numpy()

    df_train = df_train.copy()
    df_test  = df_test.copy()
    df_train["si_hist_mean"] = si_oof
    df_test["si_hist_mean"]  = df_test[STOCK_COL].map(global_si_mean).fillna(df_train[TARGET_COL].mean()).to_numpy()
    print("Done.")
    return df_train, df_test


# ── CatBoost CV ────────────────────────────────────────────────────────────────
def run_catboost(df_train, df_test, feature_cols, splits, iterations=1500, tag="cb"):
    from catboost import CatBoostRegressor

    cat_features = [c for c in CAT_COLS if c in feature_cols]
    train_cb = df_train[feature_cols].copy()
    test_cb  = df_test[feature_cols].copy()
    for c in cat_features:
        train_cb[c] = train_cb[c].astype(str)
        test_cb[c]  = test_cb[c].astype(str)

    y = df_train[TARGET_COL].to_numpy(np.float64)
    oof  = np.zeros(len(df_train))
    test_pred = np.zeros(len(df_test))

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            depth=7,
            learning_rate=0.02,
            iterations=iterations,
            l2_leaf_reg=5.0,
            random_seed=42 + fold_id,
            subsample=0.8,
            bootstrap_type="Bernoulli",
            od_type="Iter",
            od_wait=150,
            verbose=False,
        )
        model.fit(
            train_cb.iloc[tr_idx], y[tr_idx],
            cat_features=cat_features if cat_features else None,
            eval_set=(train_cb.iloc[va_idx], y[va_idx]),
            use_best_model=True,
        )
        oof[va_idx]  = model.predict(train_cb.iloc[va_idx])
        test_pred   += model.predict(test_cb) / len(splits)
        fold_corr    = pearson(y[va_idx], oof[va_idx])
        print(f"  [{tag}] fold {fold_id}: Pearson={fold_corr:.6f}  best_iter={model.best_iteration_}")
        del model; gc.collect()

    oof_corr = pearson(y, oof)
    print(f"  [{tag}] OOF Pearson: {oof_corr:.6f}")
    return oof, test_pred, oof_corr


# ── LightGBM CV ────────────────────────────────────────────────────────────────
def run_lgbm(df_train, df_test, feature_cols, splits, tag="lgbm"):
    import lightgbm as lgb

    cat_features = [c for c in CAT_COLS if c in feature_cols]
    train_f = df_train[feature_cols].copy()
    test_f  = df_test[feature_cols].copy()
    for c in cat_features:
        train_f[c] = train_f[c].astype("category")
        test_f[c]  = test_f[c].astype("category")

    y = df_train[TARGET_COL].to_numpy(np.float64)
    oof  = np.zeros(len(df_train))
    test_pred = np.zeros(len(df_test))

    params = dict(
        objective="regression",
        metric="rmse",
        num_leaves=127,
        learning_rate=0.02,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        verbose=-1,
        random_state=42,
    )

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        ds_tr = lgb.Dataset(train_f.iloc[tr_idx], label=y[tr_idx],
                            categorical_feature=cat_features if cat_features else "auto")
        ds_va = lgb.Dataset(train_f.iloc[va_idx], label=y[va_idx], reference=ds_tr)
        cb = lgb.log_evaluation(period=-1)
        es = lgb.early_stopping(100, verbose=False)
        model = lgb.train(params, ds_tr, num_boost_round=2000,
                          valid_sets=[ds_va], callbacks=[cb, es])
        oof[va_idx]  = model.predict(train_f.iloc[va_idx])
        test_pred   += model.predict(test_f) / len(splits)
        fold_corr    = pearson(y[va_idx], oof[va_idx])
        print(f"  [{tag}] fold {fold_id}: Pearson={fold_corr:.6f}  best_iter={model.best_iteration}")
        del model; gc.collect()

    oof_corr = pearson(y, oof)
    print(f"  [{tag}] OOF Pearson: {oof_corr:.6f}")
    return oof, test_pred, oof_corr


# ── submission helper ──────────────────────────────────────────────────────────
def make_submission(test_ids, preds, name):
    p = np.asarray(preds, np.float64).copy()
    lo, hi = np.nanpercentile(p, [0.5, 99.5])
    p = np.clip(p, lo, hi)
    if (np.abs(p) > 0).mean() < 0.10:
        p += 1e-9
    sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: p})
    assert np.isfinite(sub[TARGET_COL]).all()
    assert (sub[TARGET_COL].abs() > 0).mean() >= 0.10
    path = OUTPUT_DIR / f"submission_{name}.csv"
    sub.to_csv(path, index=False)
    print(f"  Saved: {path}  rows={len(sub):,}")
    return sub


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test  = pd.read_csv(TEST_PATH)
    print(f"  train: {df_train.shape}  test: {df_test.shape}")

    feature_cols = [c for c in df_train.columns if c not in [ID_COL, TARGET_COL]]
    splits = contiguous_time_splits(df_train[TIME_COL])

    # ── Phase 1: per-feature correlation ──────────────────────────────────────
    df_corr = per_feature_correlation(df_train, feature_cols)
    top_features = df_corr["feature"].tolist()  # sorted by |corr|

    # ── Phase 2: cross-sectional normalization ─────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 2: Cross-sectional normalization per di")
    print("="*60)
    df_train_cs = cross_sectional_normalize(df_train, feature_cols)
    df_test_cs  = cross_sectional_normalize(df_test,  feature_cols)

    # add rank features for top 30 most correlated
    df_train_cs = add_rank_features(df_train_cs, top_features, top_n=30)
    df_test_cs  = add_rank_features(df_test_cs,  top_features, top_n=30)

    # ── Phase 3: fold-safe si history ─────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 3: Fold-safe per-si historical mean target")
    print("="*60)
    df_train_cs, df_test_cs = add_si_target_history(df_train_cs, df_test_cs, splits)

    feature_cols_enhanced = [c for c in df_train_cs.columns
                              if c not in [ID_COL, TARGET_COL]]

    # ── Phase 4: models ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 4a: CatBoost (1500 iter, enhanced features)")
    print("="*60)
    cb_oof, cb_test, cb_score = run_catboost(
        df_train_cs, df_test_cs, feature_cols_enhanced, splits,
        iterations=1500, tag="cb"
    )

    print("\n" + "="*60)
    print("PHASE 4b: LightGBM (up to 2000 iter, enhanced features)")
    print("="*60)
    lgbm_available = True
    try:
        lgbm_oof, lgbm_test, lgbm_score = run_lgbm(
            df_train_cs, df_test_cs, feature_cols_enhanced, splits, tag="lgbm"
        )
    except ImportError:
        print("  LightGBM not installed, skipping.")
        lgbm_available = False

    # ── ensemble ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ENSEMBLE")
    print("="*60)
    y = df_train[TARGET_COL].to_numpy(np.float64)

    results = {"catboost_oof_corr": cb_score}
    best_test = cb_test
    best_score = cb_score
    best_name = "catboost"

    if lgbm_available:
        results["lgbm_oof_corr"] = lgbm_score
        # simple 50/50 blend
        blend_oof  = 0.5 * cb_oof  + 0.5 * lgbm_oof
        blend_test = 0.5 * cb_test + 0.5 * lgbm_test
        blend_score = pearson(y, blend_oof)
        results["ensemble_50_50_oof_corr"] = blend_score
        print(f"  CatBoost OOF: {cb_score:.6f}")
        print(f"  LightGBM OOF: {lgbm_score:.6f}")
        print(f"  50/50 blend:  {blend_score:.6f}")

        # pick best
        best_candidates = [
            ("catboost", cb_score, cb_test),
            ("lgbm",     lgbm_score, lgbm_test),
            ("ensemble", blend_score, blend_test),
        ]
        best_name, best_score, best_test = max(best_candidates, key=lambda x: x[1])
        print(f"  → Using: {best_name}  OOF={best_score:.6f}")
    else:
        print(f"  CatBoost only. OOF: {cb_score:.6f}")

    # ── save ───────────────────────────────────────────────────────────────────
    make_submission(df_test[ID_COL], best_test, best_name)

    results["chosen_model"] = best_name
    results["chosen_oof_corr"] = best_score
    (OUTPUT_DIR / "advanced_summary.json").write_text(json.dumps(results, indent=2))

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(json.dumps(results, indent=2))
    print(f"\nAll outputs in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
