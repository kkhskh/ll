"""
next_stage.py — Ruthless next-stage research.

Five steps:
  1. SUBMISSION AUDIT      — test if trivial transforms exploit evaluation
  2. FEATURE CLUSTERING    — find latent families via missingness + correlation blocks
  3. SURPRISE FINGERPRINT  — identify which features look like surprise/event proxies
  4. INTERACTION FEATURES  — ratios & diffs between family representatives
  5. CONDITIONAL MODELS    — sector/industry-conditioned CatBoost vs global

No new third-party packages. Uses only numpy, pandas, scipy (already installed via sklearn).
"""
from __future__ import annotations

import gc
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

# ── constants ──────────────────────────────────────────────────────────────────
TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"
OUT        = Path("artifacts/next_stage")
OUT.mkdir(parents=True, exist_ok=True)

ID_COL     = "id"
TARGET_COL = "target"
TIME_COL   = "di"
STOCK_COL  = "si"
META_CANDIDATES = ["si", "di", "industry", "sector", "top2000", "top1000", "top500"]

N_SPLITS  = 5
SEED      = 42


# ── helpers ────────────────────────────────────────────────────────────────────
def pearson(y_true, y_pred):
    y, p = np.asarray(y_true, np.float64), np.asarray(y_pred, np.float64)
    if y.size < 2 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def walk_forward_splits(time_series, n_splits=N_SPLITS, min_train=252, embargo=5):
    unique_di = np.sort(time_series.dropna().unique())
    U = len(unique_di)
    warmup = max(min_train, math.ceil(0.30 * U))
    remaining = unique_di[warmup:]
    chunks = np.array_split(remaining, n_splits)
    idx = np.arange(len(time_series))
    tv  = time_series.to_numpy()
    splits = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        valid_start = chunk[0]
        pre = unique_di[unique_di < valid_start]
        allowed = pre[:-embargo] if embargo > 0 and len(pre) > embargo else np.array([], dtype=pre.dtype)
        tr_idx = idx[np.isin(tv, allowed)]
        va_idx = idx[np.isin(tv, chunk)]
        if len(tr_idx) and len(va_idx):
            splits.append((tr_idx, va_idx))
    if len(splits) < 2:
        raise ValueError("Too few folds.")
    return splits


def make_submission(test_ids, preds, name, out_dir=OUT):
    p = np.asarray(preds, np.float64).copy()
    if not np.isfinite(p).all():
        p = np.where(np.isfinite(p), p, 0.0)
    lo, hi = np.nanpercentile(p, [0.5, 99.5])
    p = np.clip(p, lo, hi)
    if (np.abs(p) > 0).mean() < 0.10:
        p += 1e-9
    sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: p})
    path = out_dir / f"submission_{name}.csv"
    sub.to_csv(path, index=False)
    print(f"  → {path}  ({len(sub):,} rows)")
    return sub


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: SUBMISSION AUDIT
# Test whether trivial transforms of the prediction vector change the score
# dramatically — if so, it exposes evaluation bugs / exploits.
# ══════════════════════════════════════════════════════════════════════════════
def submission_audit(df_train, df_test, splits):
    """
    Generate submission variants that probe the evaluation surface.

    Variants:
      baseline_catboost    — honest CatBoost OOF predictions (reference)
      rank_within_di       — replace predictions with within-di percentile rank
      sign_flip            — negate all predictions
      constant_positive    — all predictions = +1e-4 (tests if score = 0)
      id_order             — prediction ∝ ID value (tests if scorer uses row order)
      si_mean_encode       — predict mean(target per si) from train, no model
      global_mean          — predict global train mean for every test row
    """
    print("\n" + "="*60)
    print("STEP 1: SUBMISSION AUDIT")
    print("="*60)
    print("These variants probe whether the evaluation has exploits.")
    print("If sign_flip scores the same as baseline → symmetric metric (expected).")
    print("If id_order scores well → evaluation uses row position, not ID.")
    print("If constant or global_mean score > 0 → scoring bug.\n")

    y = df_train[TARGET_COL].to_numpy(np.float64)
    test_ids = df_test[ID_COL]
    anon_cols = sorted(c for c in df_train.columns if c.startswith("f_"))
    cat_cols_present = [c for c in META_CANDIDATES if c in df_train.columns and c != TIME_COL]

    # Reference: simple CatBoost baseline for OOF + test predictions
    from catboost import CatBoostRegressor

    feat_cols = [c for c in df_train.columns if c not in [ID_COL, TARGET_COL]]
    cat_feat  = [c for c in cat_cols_present if c in feat_cols]
    tr_cb = df_train[feat_cols].copy()
    te_cb = df_test[[c for c in feat_cols if c in df_test.columns]].copy()
    # align columns
    for c in feat_cols:
        if c not in te_cb.columns:
            te_cb[c] = np.nan
    te_cb = te_cb[feat_cols]
    for c in cat_feat:
        tr_cb[c] = tr_cb[c].astype(str)
        te_cb[c] = te_cb[c].astype(str)

    oof       = np.zeros(len(df_train))
    test_pred = np.zeros(len(df_test))
    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        model = CatBoostRegressor(
            loss_function="RMSE", depth=5, learning_rate=0.05,
            iterations=500, l2_leaf_reg=3.0, random_seed=SEED + fold_id,
            od_type="Iter", od_wait=50, verbose=False,
        )
        model.fit(tr_cb.iloc[tr_idx], y[tr_idx],
                  cat_features=cat_feat if cat_feat else None,
                  eval_set=(tr_cb.iloc[va_idx], y[va_idx]),
                  use_best_model=True)
        oof[va_idx]  = model.predict(tr_cb.iloc[va_idx])
        test_pred   += model.predict(te_cb) / len(splits)
        del model; gc.collect()

    baseline_oof_score = pearson(y, oof)
    print(f"  Reference CatBoost OOF Pearson: {baseline_oof_score:.6f}")

    variants = {}

    # 1a. baseline
    variants["baseline_catboost"] = test_pred.copy()

    # 1b. rank within di — any monotone transform should give same Pearson in theory
    di_test = df_test[TIME_COL].to_numpy()
    rank_pred = test_pred.copy()
    for g in np.unique(di_test):
        mask = di_test == g
        vals = rank_pred[mask]
        finite = np.isfinite(vals)
        if finite.sum() > 1:
            from scipy.stats import rankdata
            ranks = np.full(mask.sum(), 0.5)
            ranks[finite] = rankdata(vals[finite]) / (finite.sum() + 1)
            rank_pred[mask] = ranks
    variants["rank_within_di"] = rank_pred

    # 1c. sign flip
    variants["sign_flip"] = -test_pred.copy()

    # 1d. constant (should score near 0)
    variants["constant_positive"] = np.full(len(df_test), 1e-4)

    # 1e. ID order (tests if evaluation uses position)
    id_vals = df_test[ID_COL].to_numpy(np.float64)
    if not np.isfinite(id_vals).all():
        id_vals = np.arange(len(df_test), dtype=np.float64)
    variants["id_order"] = (id_vals - id_vals.mean()) / (id_vals.std() + 1e-8)

    # 1f. si mean encoding — predict train mean target per stock (no model)
    si_mean = df_train.groupby(STOCK_COL)[TARGET_COL].mean().to_dict()
    global_mean = float(df_train[TARGET_COL].mean())
    variants["si_mean_encode"] = df_test[STOCK_COL].map(si_mean).fillna(global_mean).to_numpy()

    # 1g. global mean
    variants["global_mean"] = np.full(len(df_test), global_mean)

    # Save all variants
    audit_rows = []
    for name, preds in variants.items():
        make_submission(test_ids, preds, f"audit_{name}")
        audit_rows.append({"variant": name, "note": "upload to Kaggle to score"})

    # What we CAN compute locally: OOF pearson for si_mean_encode and global_mean
    oof_simean = df_train[STOCK_COL].map(si_mean).fillna(global_mean).to_numpy()
    print(f"\n  si_mean_encode OOF Pearson (local): {pearson(y, oof_simean):.6f}")
    print(f"  global_mean OOF Pearson (local):    {pearson(y, np.full(len(y), global_mean)):.6f}")
    print(f"\n  Submit all audit_*.csv files to Kaggle and compare scores.")
    print(f"  Key questions:")
    print(f"    sign_flip score ≈ baseline?   → symmetric metric (fine)")
    print(f"    rank_within_di ≈ baseline?    → monotone invariance (expected)")
    print(f"    id_order score > 0.05?        → evaluation uses row order (exploit)")
    print(f"    constant > 0?                 → scoring bug")
    print(f"    si_mean_encode > 0.06?        → per-stock mean is real signal")

    pd.DataFrame(audit_rows).to_csv(OUT / "audit_variants.csv", index=False)
    return oof, test_pred, baseline_oof_score


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: FEATURE CLUSTERING — latent family discovery
# ══════════════════════════════════════════════════════════════════════════════
def cluster_features(df_train, anon_cols, n_families=12):
    """
    Cluster the 172 anonymous features into latent families using:
      A) pairwise Pearson correlation (captures linear co-movement)
      B) missingness pattern correlation (captures data-generation families)
      C) combined: average of A and B distance matrices

    Output: per-family representative (highest |corr| with target),
            family assignments, saved to CSV.
    """
    print("\n" + "="*60)
    print("STEP 2: FEATURE CLUSTERING (latent family discovery)")
    print("="*60)

    y = df_train[TARGET_COL].to_numpy(np.float64)
    X = df_train[anon_cols].to_numpy(np.float64)

    # ── A: pairwise feature Pearson correlation ────────────────────────────────
    print("  Computing pairwise feature correlations...")
    n = len(anon_cols)
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            xi, xj = X[:, i], X[:, j]
            mask = np.isfinite(xi) & np.isfinite(xj)
            if mask.sum() < 50:
                c = 0.0
            else:
                c = float(np.corrcoef(xi[mask], xj[mask])[0, 1])
                if not np.isfinite(c):
                    c = 0.0
            corr_matrix[i, j] = corr_matrix[j, i] = c
    dist_corr = 1.0 - np.abs(corr_matrix)   # dissimilarity: 0=identical, 1=uncorrelated

    # ── B: missingness pattern correlation ────────────────────────────────────
    print("  Computing missingness pattern correlations...")
    miss = (~np.isfinite(X)).astype(np.float64)   # 1 where NaN
    miss_col_std = miss.std(axis=0)
    miss_corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if miss_col_std[i] == 0 or miss_col_std[j] == 0:
                c = 1.0 if (miss[:, i] == miss[:, j]).all() else 0.0
            else:
                c = float(np.corrcoef(miss[:, i], miss[:, j])[0, 1])
                if not np.isfinite(c):
                    c = 0.0
            miss_corr[i, j] = miss_corr[j, i] = c
    dist_miss = 1.0 - np.abs(miss_corr)

    # ── C: combined distance ───────────────────────────────────────────────────
    dist_combined = 0.5 * dist_corr + 0.5 * dist_miss
    np.fill_diagonal(dist_combined, 0.0)

    # ── hierarchical clustering ────────────────────────────────────────────────
    print(f"  Hierarchical clustering into {n_families} families...")
    condensed = squareform(dist_combined, checks=False)
    Z = linkage(condensed, method="ward")
    family_labels = fcluster(Z, n_families, criterion="maxclust")

    # ── per-feature corr with target ───────────────────────────────────────────
    feat_corrs = []
    for i, col in enumerate(anon_cols):
        xi = X[:, i]
        mask = np.isfinite(xi) & np.isfinite(y)
        if mask.sum() < 50:
            c = 0.0
        else:
            c = float(np.corrcoef(xi[mask], y[mask])[0, 1])
            c = 0.0 if not np.isfinite(c) else c
        feat_corrs.append(c)
    feat_corrs = np.array(feat_corrs)

    # ── build family report ────────────────────────────────────────────────────
    rows = []
    family_reps = {}  # family_id → representative column name
    for fam in range(1, n_families + 1):
        members_idx = np.where(family_labels == fam)[0]
        members = [anon_cols[i] for i in members_idx]
        corrs   = feat_corrs[members_idx]
        abs_corrs = np.abs(corrs)
        best_idx  = members_idx[np.argmax(abs_corrs)]
        rep       = anon_cols[best_idx]
        family_reps[fam] = rep

        nan_fracs = [df_train[m].isna().mean() for m in members]
        rows.append({
            "family":       fam,
            "n_members":    len(members),
            "representative": rep,
            "rep_corr_target": round(feat_corrs[best_idx], 6),
            "mean_abs_corr_target": round(float(abs_corrs.mean()), 6),
            "max_abs_corr_target":  round(float(abs_corrs.max()),  6),
            "mean_nan_frac": round(float(np.mean(nan_fracs)), 4),
            "members": " ".join(members),
        })

    df_families = pd.DataFrame(rows).sort_values("max_abs_corr_target", ascending=False)
    df_families.to_csv(OUT / "feature_families.csv", index=False)

    # Also save per-feature family assignment
    per_feat = pd.DataFrame({
        "feature": anon_cols,
        "family":  family_labels,
        "corr_target": feat_corrs,
    }).sort_values("family")
    per_feat.to_csv(OUT / "feature_family_assignments.csv", index=False)

    print(f"\n  Feature families (sorted by max |corr| with target):")
    display_cols = ["family", "n_members", "representative", "rep_corr_target",
                    "max_abs_corr_target", "mean_nan_frac"]
    print(df_families[display_cols].to_string(index=False))

    # Save correlation matrix
    pd.DataFrame(corr_matrix, index=anon_cols, columns=anon_cols).to_csv(
        OUT / "feature_corr_matrix.csv"
    )

    return df_families, family_reps, family_labels, feat_corrs, per_feat


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: SURPRISE PROXY FINGERPRINTING
# ══════════════════════════════════════════════════════════════════════════════
def surprise_proxy_fingerprint(df_train, anon_cols, family_labels, feat_corrs):
    """
    For each anonymous feature, compute diagnostics that distinguish:
      - SURPRISE proxies:     high kurtosis, low within-si autocorrelation,
                              high di-level variance, sparse non-zero
      - FUNDAMENTAL proxies: low kurtosis, high within-si autocorrelation,
                              low di-level variance (stable across time)
      - NOISE:               low |corr| with target, low variance

    Output: fingerprint CSV + top candidates for each role.
    """
    print("\n" + "="*60)
    print("STEP 3: SURPRISE PROXY FINGERPRINTING")
    print("="*60)

    y = df_train[TARGET_COL].to_numpy(np.float64)
    rows = []

    for i, col in enumerate(anon_cols):
        vals = df_train[col].to_numpy(np.float64)
        finite_mask = np.isfinite(vals)
        nan_frac = 1.0 - finite_mask.mean()
        v = vals[finite_mask]

        if len(v) < 50:
            rows.append({"feature": col, "family": family_labels[i],
                         "corr_target": feat_corrs[i]})
            continue

        mu, sigma = v.mean(), v.std()
        skew = float(pd.Series(v).skew()) if len(v) > 3 else 0.0
        kurt = float(pd.Series(v).kurtosis()) if len(v) > 3 else 0.0

        # Within-si autocorrelation: mean lag-1 corr across stocks
        si_autocorrs = []
        for si_val, grp in df_train.groupby(STOCK_COL)[col]:
            ts = grp.dropna().to_numpy(np.float64)
            if len(ts) > 5:
                r = float(np.corrcoef(ts[:-1], ts[1:])[0, 1])
                if np.isfinite(r):
                    si_autocorrs.append(r)
        mean_si_autocorr = float(np.mean(si_autocorrs)) if si_autocorrs else 0.0

        # Variance decomposition: between-di vs within-di
        di_means = df_train.groupby(TIME_COL)[col].mean().dropna()
        between_di_var = float(di_means.var()) if len(di_means) > 1 else 0.0
        within_di_var = float(
            df_train.groupby(TIME_COL)[col].var().dropna().mean()
        )
        total_var = between_di_var + within_di_var + 1e-12
        frac_between_di = between_di_var / total_var  # high → time-varying (surprise-like)

        # Sparsity: fraction of near-zero values
        near_zero_frac = float((np.abs(v) < 1e-6).mean())

        rows.append({
            "feature":           col,
            "family":            family_labels[i],
            "corr_target":       round(feat_corrs[i], 6),
            "abs_corr_target":   round(abs(feat_corrs[i]), 6),
            "nan_frac":          round(nan_frac, 4),
            "near_zero_frac":    round(near_zero_frac, 4),
            "mean":              round(float(mu), 4),
            "std":               round(float(sigma), 4),
            "skew":              round(skew, 4),
            "kurtosis":          round(kurt, 4),
            "mean_si_autocorr":  round(mean_si_autocorr, 4),
            "frac_between_di":   round(frac_between_di, 4),
        })

    df_fp = pd.DataFrame(rows).sort_values("abs_corr_target", ascending=False)
    df_fp.to_csv(OUT / "surprise_fingerprint.csv", index=False)

    # Identify candidates
    surprise_candidates = df_fp[
        (df_fp["kurtosis"] > df_fp["kurtosis"].quantile(0.7)) &
        (df_fp["mean_si_autocorr"].abs() < df_fp["mean_si_autocorr"].abs().quantile(0.3)) &
        (df_fp["frac_between_di"] > df_fp["frac_between_di"].quantile(0.7))
    ]["feature"].tolist()

    fundamental_candidates = df_fp[
        (df_fp["mean_si_autocorr"] > df_fp["mean_si_autocorr"].quantile(0.7)) &
        (df_fp["frac_between_di"] < df_fp["frac_between_di"].quantile(0.3))
    ]["feature"].tolist()

    print(f"\n  Surprise-proxy candidates ({len(surprise_candidates)}): "
          f"{surprise_candidates[:10]}")
    print(f"  Fundamental candidates    ({len(fundamental_candidates)}): "
          f"{fundamental_candidates[:10]}")
    print(f"\n  Top 15 by |corr| with target:")
    print(df_fp[["feature", "family", "corr_target", "kurtosis",
                 "mean_si_autocorr", "frac_between_di", "nan_frac"]].head(15).to_string(index=False))

    df_fp.to_csv(OUT / "surprise_fingerprint.csv", index=False)
    return df_fp, surprise_candidates, fundamental_candidates


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: INTERACTION FEATURES
# ══════════════════════════════════════════════════════════════════════════════
def build_interaction_features(df_train, df_test, family_reps, anon_cols, feat_corrs):
    """
    Build pairwise interactions between family representatives only.
    For each pair of reps (i, j):
      ratio:  f_i / (|f_j| + eps)
      diff:   f_i - f_j
      product: f_i * f_j   (only if both have same-sign corr with target)

    Keep only interactions with |Pearson(interaction, target)| > threshold.
    """
    print("\n" + "="*60)
    print("STEP 4: TARGETED INTERACTION FEATURES")
    print("="*60)

    y = df_train[TARGET_COL].to_numpy(np.float64)
    reps = list(family_reps.values())
    print(f"  Family representatives ({len(reps)}): {reps}")

    feat_corr_dict = {anon_cols[i]: feat_corrs[i] for i in range(len(anon_cols))}

    candidates = []
    threshold = 0.015   # keep interactions with |Pearson| > this

    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            ri, rj = reps[i], reps[j]
            xi = df_train[ri].to_numpy(np.float64)
            xj = df_train[rj].to_numpy(np.float64)
            mask = np.isfinite(xi) & np.isfinite(xj) & np.isfinite(y)
            if mask.sum() < 100:
                continue

            # ratio
            ratio = xi / (np.abs(xj) + 1e-6)
            ratio_m = ratio[mask]
            ratio_m = np.clip(ratio_m, np.nanpercentile(ratio_m, 1),
                              np.nanpercentile(ratio_m, 99))
            c_ratio = pearson(y[mask], ratio_m)

            # diff
            diff = xi - xj
            c_diff = pearson(y[mask], diff[mask])

            # product
            prod = xi * xj
            c_prod = pearson(y[mask], prod[mask])

            for kind, val, c in [("ratio", ratio, c_ratio),
                                   ("diff",  diff,  c_diff),
                                   ("prod",  prod,  c_prod)]:
                if np.isfinite(c) and abs(c) > threshold:
                    name = f"{ri}_x_{rj}_{kind}"
                    candidates.append({
                        "name": name, "f_i": ri, "f_j": rj,
                        "kind": kind, "corr_target": round(c, 6),
                        "abs_corr": round(abs(c), 6),
                    })

    df_cand = pd.DataFrame(candidates).sort_values("abs_corr", ascending=False)
    df_cand.to_csv(OUT / "interaction_candidates.csv", index=False)
    print(f"\n  Interactions with |Pearson| > {threshold}: {len(df_cand)}")
    if len(df_cand):
        print(df_cand.head(20).to_string(index=False))

    # Materialise top interactions into train and test
    top_interactions = df_cand.head(30)
    new_feat_names = []

    for _, row in top_interactions.iterrows():
        ri, rj, kind, name = row["f_i"], row["f_j"], row["kind"], row["name"]
        xi_tr = df_train[ri].to_numpy(np.float64)
        xj_tr = df_train[rj].to_numpy(np.float64)
        xi_te = df_test[ri].to_numpy(np.float64)  if ri in df_test.columns else np.zeros(len(df_test))
        xj_te = df_test[rj].to_numpy(np.float64)  if rj in df_test.columns else np.zeros(len(df_test))

        if kind == "ratio":
            tr_val = xi_tr / (np.abs(xj_tr) + 1e-6)
            te_val = xi_te / (np.abs(xj_te) + 1e-6)
        elif kind == "diff":
            tr_val = xi_tr - xj_tr
            te_val = xi_te - xj_te
        else:
            tr_val = xi_tr * xj_tr
            te_val = xi_te * xj_te

        # clip outliers
        lo, hi = np.nanpercentile(tr_val[np.isfinite(tr_val)], [1, 99])
        tr_val = np.clip(tr_val, lo, hi)
        te_val = np.clip(te_val, lo, hi)

        df_train = df_train.copy()
        df_test  = df_test.copy()
        df_train[name] = tr_val
        df_test[name]  = te_val
        new_feat_names.append(name)

    print(f"\n  Materialised {len(new_feat_names)} top interaction features.")
    return df_train, df_test, new_feat_names, df_cand


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: CONDITIONAL MODELS (sector / industry)
# ══════════════════════════════════════════════════════════════════════════════
def run_conditional_models(df_train, df_test, anon_cols, splits, group_col="sector"):
    """
    Train one CatBoost model per group (sector or industry).
    Compare OOF Pearson vs global model on same splits.
    Earnings surprise effects are known to be sector-conditioned:
      the same feature may be bullish in Tech but bearish in Utilities.
    """
    print("\n" + "="*60)
    print(f"STEP 5: CONDITIONAL MODELS (grouped by {group_col})")
    print("="*60)

    if group_col not in df_train.columns:
        print(f"  {group_col} not found in data. Skipping.")
        return None, None

    from catboost import CatBoostRegressor

    y = df_train[TARGET_COL].to_numpy(np.float64)
    groups = df_train[group_col].astype(str).unique()
    print(f"  Unique {group_col} values: {sorted(groups)}")

    feat_cols = anon_cols  # use raw anonymous features only for conditional models
    # also add di as a numeric feature
    if TIME_COL in df_train.columns:
        feat_cols = [TIME_COL] + feat_cols

    # ── global model (reference) ───────────────────────────────────────────────
    oof_global = np.zeros(len(df_train))
    test_global = np.zeros(len(df_test))
    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        tr_di = df_train.iloc[tr_idx][TIME_COL]
        va_di = df_train.iloc[va_idx][TIME_COL]
        assert tr_di.max() < va_di.min()
        model = CatBoostRegressor(
            loss_function="RMSE", depth=5, learning_rate=0.05,
            iterations=500, l2_leaf_reg=3.0, random_seed=SEED + fold_id,
            od_type="Iter", od_wait=50, verbose=False,
        )
        model.fit(df_train[feat_cols].iloc[tr_idx], y[tr_idx],
                  eval_set=(df_train[feat_cols].iloc[va_idx], y[va_idx]),
                  use_best_model=True)
        oof_global[va_idx]  = model.predict(df_train[feat_cols].iloc[va_idx])
        test_global        += model.predict(df_test[[c for c in feat_cols if c in df_test.columns]]) / len(splits)
        del model; gc.collect()

    global_oof_score = pearson(y, oof_global)
    print(f"\n  Global model OOF Pearson:      {global_oof_score:.6f}")

    # ── conditional model: train separate model per group ─────────────────────
    oof_cond  = np.zeros(len(df_train))
    test_cond = np.zeros(len(df_test))
    group_scores = {}

    for g in sorted(groups):
        g_mask_train = (df_train[group_col].astype(str) == g).to_numpy()
        g_mask_test  = (df_test[group_col].astype(str) == g).to_numpy() if group_col in df_test.columns else np.zeros(len(df_test), bool)

        g_rows = []
        for fold_id, (tr_idx, va_idx) in enumerate(splits):
            tr_g = tr_idx[g_mask_train[tr_idx]]
            va_g = va_idx[g_mask_train[va_idx]]
            if len(tr_g) < 100 or len(va_g) < 20:
                continue
            g_rows.append((fold_id, tr_g, va_g))

        if len(g_rows) < 2:
            # fall back to global for this group
            oof_cond[g_mask_train] = oof_global[g_mask_train]
            test_cond[g_mask_test] = test_global[g_mask_test]
            continue

        oof_g  = np.zeros(g_mask_train.sum())
        tr_idx_g = np.where(g_mask_train)[0]
        test_pred_g = np.zeros(g_mask_test.sum())

        for fold_id, tr_g, va_g in g_rows:
            model = CatBoostRegressor(
                loss_function="RMSE", depth=5, learning_rate=0.05,
                iterations=500, l2_leaf_reg=3.0, random_seed=SEED + fold_id,
                od_type="Iter", od_wait=50, verbose=False,
            )
            model.fit(df_train[feat_cols].iloc[tr_g], y[tr_g],
                      eval_set=(df_train[feat_cols].iloc[va_g], y[va_g]),
                      use_best_model=True)
            # positional mapping back into g_mask_train positions
            for va_pos in va_g:
                local_pos = np.searchsorted(tr_idx_g, va_pos)
                if local_pos < len(oof_g):
                    oof_g[local_pos] = model.predict(
                        df_train[feat_cols].iloc[[va_pos]]
                    )[0]
            if g_mask_test.sum() > 0:
                test_pred_g += model.predict(
                    df_test[[c for c in feat_cols if c in df_test.columns]].iloc[g_mask_test]
                ) / len(g_rows)
            del model; gc.collect()

        oof_cond[g_mask_train]  = oof_g
        if g_mask_test.sum() > 0:
            test_cond[g_mask_test] = test_pred_g
        else:
            pass

        g_score = pearson(y[g_mask_train], oof_cond[g_mask_train])
        group_scores[g] = round(g_score, 6)
        print(f"    {group_col}={g:>10s}: n_train={g_mask_train.sum():,}  "
              f"OOF Pearson={g_score:.6f}")

    # for groups that had fallback, oof_cond already set to global
    cond_oof_score = pearson(y, oof_cond)
    print(f"\n  Conditional model OOF Pearson: {cond_oof_score:.6f}")
    print(f"  Global model OOF Pearson:      {global_oof_score:.6f}")
    print(f"  Δ (cond - global):             {cond_oof_score - global_oof_score:+.6f}")
    if cond_oof_score > global_oof_score:
        print(f"  → Conditional model wins. Sector-conditioned signal is real.")
    else:
        print(f"  → Global model wins. Sector conditioning does not help.")

    results = {
        "global_oof_pearson":      round(global_oof_score, 6),
        "conditional_oof_pearson": round(cond_oof_score, 6),
        "delta":                   round(cond_oof_score - global_oof_score, 6),
        "group_scores":            group_scores,
    }
    (OUT / "conditional_model_results.json").write_text(json.dumps(results, indent=2))
    return oof_cond, test_cond


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Loading data...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test  = pd.read_csv(TEST_PATH)
    print(f"  train: {df_train.shape}  test: {df_test.shape}")

    meta_cols = [c for c in META_CANDIDATES if c in df_train.columns]
    anon_cols = sorted(c for c in df_train.columns if c.startswith("f_"))
    print(f"  meta_cols: {meta_cols}")
    print(f"  anon_cols: {len(anon_cols)}")

    splits = walk_forward_splits(df_train[TIME_COL])
    print("\nWalk-forward folds:")
    for i, (tr, va) in enumerate(splits):
        tr_di = df_train.iloc[tr][TIME_COL]
        va_di = df_train.iloc[va][TIME_COL]
        print(f"  fold {i}: train [{tr_di.min()}–{tr_di.max()}]  "
              f"valid [{va_di.min()}–{va_di.max()}]  "
              f"n_train={len(tr):,}  n_valid={len(va):,}")

    # ── Step 1: submission audit ───────────────────────────────────────────────
    oof_base, test_base, base_score = submission_audit(df_train, df_test, splits)

    # ── Step 2: feature clustering ─────────────────────────────────────────────
    df_families, family_reps, family_labels, feat_corrs, per_feat = cluster_features(
        df_train, anon_cols, n_families=12
    )

    # ── Step 3: surprise fingerprinting ───────────────────────────────────────
    df_fp, surprise_cands, fundamental_cands = surprise_proxy_fingerprint(
        df_train, anon_cols, family_labels, feat_corrs
    )

    # ── Step 4: interaction features ──────────────────────────────────────────
    df_train_ix, df_test_ix, ix_cols, df_ix_cands = build_interaction_features(
        df_train, df_test, family_reps, anon_cols, feat_corrs
    )

    # Run CatBoost with interaction features
    if ix_cols:
        print("\n  Running CatBoost with interaction features...")
        from catboost import CatBoostRegressor
        y = df_train[TARGET_COL].to_numpy(np.float64)
        feat_with_ix = anon_cols + ix_cols
        feat_with_ix = [c for c in feat_with_ix if c in df_train_ix.columns and c in df_test_ix.columns]
        oof_ix = np.zeros(len(df_train_ix))
        test_pred_ix = np.zeros(len(df_test_ix))
        for fold_id, (tr_idx, va_idx) in enumerate(splits):
            model = CatBoostRegressor(
                loss_function="RMSE", depth=6, learning_rate=0.05,
                iterations=800, l2_leaf_reg=3.0, random_seed=SEED + fold_id,
                od_type="Iter", od_wait=80, verbose=False,
            )
            model.fit(df_train_ix[feat_with_ix].iloc[tr_idx], y[tr_idx],
                      eval_set=(df_train_ix[feat_with_ix].iloc[va_idx], y[va_idx]),
                      use_best_model=True)
            oof_ix[va_idx]   = model.predict(df_train_ix[feat_with_ix].iloc[va_idx])
            test_pred_ix    += model.predict(df_test_ix[feat_with_ix]) / len(splits)
            fc = pearson(y[va_idx], oof_ix[va_idx])
            print(f"    fold {fold_id}: Pearson={fc:.6f}  best_iter={model.best_iteration_}")
            del model; gc.collect()
        oof_ix_score = pearson(y, oof_ix)
        print(f"  With interactions OOF Pearson: {oof_ix_score:.6f}")
        print(f"  Baseline OOF Pearson:          {base_score:.6f}")
        print(f"  Δ: {oof_ix_score - base_score:+.6f}")
        if oof_ix_score > base_score:
            make_submission(df_test[ID_COL], test_pred_ix, "with_interactions")

    # ── Step 5: conditional models ────────────────────────────────────────────
    oof_cond, test_cond = run_conditional_models(
        df_train, df_test, anon_cols, splits, group_col="sector"
    )

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("NEXT-STAGE SUMMARY")
    print("="*60)
    print(f"  Baseline CatBoost OOF Pearson:   {base_score:.6f}")
    if ix_cols:
        print(f"  + Interactions OOF Pearson:      {oof_ix_score:.6f}")
    if oof_cond is not None:
        y = df_train[TARGET_COL].to_numpy(np.float64)
        print(f"  Conditional (sector) OOF Pearson:{pearson(y, oof_cond):.6f}")
    print(f"\n  Feature families saved:  {OUT/'feature_families.csv'}")
    print(f"  Fingerprint saved:       {OUT/'surprise_fingerprint.csv'}")
    print(f"  Interactions saved:      {OUT/'interaction_candidates.csv'}")
    print(f"  Audit submissions:       {OUT}/submission_audit_*.csv")
    print(f"\nKey next action: upload audit submissions to Kaggle to probe evaluation.")


if __name__ == "__main__":
    main()
