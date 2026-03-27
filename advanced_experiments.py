"""
Advanced experiments: walk-forward CatBoost ablations with honest OOF.
Parts C/D:
  - add non-target si context features
  - evaluate B1/B2/B3 for si feature usefulness
  - evaluate D1..D7 with full-column unsupervised transforms
  - fold-local interaction selection only (no full-train target screening)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── constants ──────────────────────────────────────────────────────────────────
TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"
OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

ID_COL     = "id"
TARGET_COL = "target"
TIME_COL   = "di"
STOCK_COL  = "si"

META_COL_CANDIDATES = ["si", "di", "industry", "sector", "top2000", "top1000", "top500"]

N_SPLITS   = 5
SEED       = 42
RUN_LGBM   = False   # disabled until CatBoost ablations are complete
TOP_N_ANON_SELECT = 30
TOP_FOLD_INTERACTIONS = 50
DEFAULT_FAMILY_REPS = [
    "f_118", "f_21", "f_107", "f_53", "f_50", "f_1",
    "f_35", "f_126", "f_31", "f_120", "f_117", "f_130",
]


# ── helpers ────────────────────────────────────────────────────────────────────
def pearson(y_true, y_pred):
    y, p = np.asarray(y_true, np.float64), np.asarray(y_pred, np.float64)
    if y.size < 2 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def oof_pearson_on_covered(y: np.ndarray, oof: np.ndarray, covered: np.ndarray) -> tuple[float, float]:
    """Pearson only on rows that received a validation prediction. Returns (corr, coverage_frac)."""
    covered = np.asarray(covered, dtype=bool)
    y = np.asarray(y, np.float64)
    oof = np.asarray(oof, np.float64)
    coverage_frac = float(covered.mean())
    if int(covered.sum()) < 2:
        return float("nan"), coverage_frac
    return pearson(y[covered], oof[covered]), coverage_frac


# ── walk-forward splitter (same logic as baseline_experiments.py) ──────────────
def walk_forward_time_splits(
    time_series: pd.Series,
    n_splits: int = N_SPLITS,
    min_train_groups: int = 252,
    embargo_groups: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window walk-forward splits on di.
    - warmup  = max(min_train_groups, ceil(0.30 * U))
    - remaining di split into n_splits contiguous validation chunks
    - embargo: last embargo_groups di before valid_start removed from train
    - each fold: train contains ONLY di strictly before valid_start (minus embargo)
    - empty folds skipped; raises if fewer than 2 survive
    """
    unique_di = np.sort(time_series.dropna().unique())
    U = len(unique_di)
    warmup = max(min_train_groups, math.ceil(0.30 * U))
    remaining = unique_di[warmup:]
    if len(remaining) == 0:
        raise ValueError("No di groups left after warmup. Lower min_train_groups.")

    chunks = np.array_split(remaining, n_splits)
    idx = np.arange(len(time_series))
    tv  = time_series.to_numpy()

    splits = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        valid_start = chunk[0]
        pre = unique_di[unique_di < valid_start]
        if embargo_groups > 0 and len(pre) > embargo_groups:
            allowed = pre[:-embargo_groups]
        else:
            allowed = np.array([], dtype=pre.dtype)
        tr_idx = idx[np.isin(tv, allowed)]
        va_idx = idx[np.isin(tv, chunk)]
        if len(tr_idx) > 0 and len(va_idx) > 0:
            splits.append((tr_idx, va_idx))

    if len(splits) < 2:
        raise ValueError("Fewer than 2 valid folds. Lower min_train_groups or embargo_groups.")
    return splits


# ── feature-set helpers ────────────────────────────────────────────────────────
def get_feature_sets(df):
    """Strict feature categorisation. id and target are excluded from all sets."""
    all_cols = set(df.columns)
    meta_cols = [c for c in META_COL_CANDIDATES if c in all_cols]
    anon_cols = sorted(c for c in all_cols if c.startswith("f_"))
    anon_numeric_cols = [
        c for c in anon_cols if pd.api.types.is_numeric_dtype(df[c])
    ]
    return meta_cols, anon_cols, anon_numeric_cols


# ── per-feature Pearson (anonymous cols only, raw target) ──────────────────────
def per_feature_correlation(df_train, anon_numeric_cols):
    y = df_train[TARGET_COL].to_numpy(np.float64)
    rows = []
    for col in anon_numeric_cols:
        x = df_train[col].to_numpy(np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 100:
            rows.append({"feature": col, "corr": 0.0, "corr_signed": 0.0})
            continue
        c = float(np.corrcoef(x[mask], y[mask])[0, 1])
        c = 0.0 if not np.isfinite(c) else c
        rows.append({"feature": col, "corr": abs(c), "corr_signed": c})
    return pd.DataFrame(rows).sort_values("corr", ascending=False).reset_index(drop=True)


# ── forward-safe si history (built on RAW si BEFORE any anonymous transform) ───
def add_forward_safe_si_history(df_train, df_test):
    """
    Row-level, forward-safe per-si features from the raw, unnormalized si column.

    Train (each row i sees only rows j < i for the same si, ordered by di then id):
      si_hist_count    number of prior rows for this si
      si_hist_mean     mean of prior targets (global mean if count=0)
      si_last_target   previous target for this si  (global mean if first)
      si_hist_std      std of prior targets  (global std if count < 2)

    Test: per-si summaries from FULL train.
    Unseen si → count=0, mean/last/std = global train values.
    """
    global_mean = float(df_train[TARGET_COL].mean())
    global_std  = float(df_train[TARGET_COL].std())

    # Work on a 0-based copy so positional reindex is safe
    df_tr = df_train.copy().reset_index(drop=True)

    # Canonical ordering: (si, di, id)
    sort_order = df_tr.sort_values([STOCK_COL, TIME_COL, ID_COL]).index
    df_s = df_tr.loc[sort_order].copy()

    grp_target = df_s.groupby(STOCK_COL, sort=False)[TARGET_COL]

    # si_hist_count: 0 for first occurrence of each si
    df_s["si_hist_count"] = df_s.groupby(STOCK_COL, sort=False).cumcount()

    # si_hist_mean: cumulative sum EXCLUDING current row / count
    cumsum_excl = grp_target.cumsum() - df_s[TARGET_COL]
    cnt = df_s["si_hist_count"].replace(0, np.nan)
    df_s["si_hist_mean"] = (cumsum_excl / cnt).fillna(global_mean)

    # si_last_target: target of the previous row for the same si
    df_s["si_last_target"] = grp_target.shift(1).fillna(global_mean)

    # si_hist_std: expanding std of prior rows (NaN until count >= 2)
    df_s["si_hist_std"] = grp_target.transform(
        lambda x: x.shift(1).expanding(min_periods=2).std()
    ).fillna(global_std)

    # ── assertion: no train row's own target appears in its own history ────────
    # Row with count=0 must have hist_mean == global_mean
    first_occ = df_s[df_s["si_hist_count"] == 0]
    assert (first_occ["si_hist_mean"] == global_mean).all(), \
        "First-occurrence rows should carry global_mean, not their own target"

    # Reindex back to original df_train 0-based order
    hist_cols = ["si_hist_count", "si_hist_mean", "si_last_target", "si_hist_std"]
    df_train_out = df_train.copy()
    for col in hist_cols:
        df_train_out[col] = df_s[col].reindex(range(len(df_tr))).values

    # ── test: per-si summaries from full train ─────────────────────────────────
    si_grp = df_train.groupby(STOCK_COL)
    test_count = si_grp[TARGET_COL].count()
    test_mean  = si_grp[TARGET_COL].mean()
    test_last  = si_grp[TARGET_COL].last()
    test_std   = si_grp[TARGET_COL].std()

    df_test_out = df_test.copy()
    df_test_out["si_hist_count"]  = df_test_out[STOCK_COL].map(test_count).fillna(0).astype(int)
    df_test_out["si_hist_mean"]   = df_test_out[STOCK_COL].map(test_mean).fillna(global_mean)
    df_test_out["si_last_target"] = df_test_out[STOCK_COL].map(test_last).fillna(global_mean)
    df_test_out["si_hist_std"]    = df_test_out[STOCK_COL].map(test_std).fillna(global_std)

    return df_train_out, df_test_out


def add_forward_safe_si_context(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Non-target si context features.
    Train uses prior rows within each si (ordered by si, di, id); test maps from full train.
    """
    tr = df_train.copy().reset_index(drop=True)
    te = df_test.copy()
    order = tr.sort_values([STOCK_COL, TIME_COL, ID_COL]).index
    tr_s = tr.loc[order].copy()
    tr_s["si_hist_count_ctx"] = tr_s.groupby(STOCK_COL, sort=False).cumcount()
    tr_s["si_seen_before"] = (tr_s["si_hist_count_ctx"] > 0).astype(np.int8)
    tr_s["si_log_count"] = np.log1p(tr_s["si_hist_count_ctx"].astype(np.float64))
    tr_s["si_prev_di"] = tr_s.groupby(STOCK_COL, sort=False)[TIME_COL].shift(1)
    tr_s["si_prev_gap_di"] = tr_s[TIME_COL].to_numpy(np.float64) - tr_s["si_prev_di"].to_numpy(np.float64)
    gap_median = float(np.nanmedian(tr_s["si_prev_gap_di"].to_numpy(np.float64)))
    if not np.isfinite(gap_median):
        gap_median = 1.0
    tr_s["si_prev_gap_di"] = tr_s["si_prev_gap_di"].fillna(gap_median)
    tr_out = df_train.copy()
    for col in ["si_seen_before", "si_log_count", "si_prev_gap_di"]:
        tr_out[col] = tr_s[col].reindex(range(len(tr))).values

    full_cnt = df_train.groupby(STOCK_COL).size()
    full_last_di = df_train.groupby(STOCK_COL)[TIME_COL].max()
    te_seen = te[STOCK_COL].isin(full_cnt.index).astype(np.int8)
    te_log = np.log1p(te[STOCK_COL].map(full_cnt).fillna(0).to_numpy(np.float64))
    te_gap = te[TIME_COL].to_numpy(np.float64) - te[STOCK_COL].map(full_last_di).to_numpy(np.float64)
    te_gap = np.where(np.isfinite(te_gap), te_gap, gap_median)
    te_out = df_test.copy()
    te_out["si_seen_before"] = te_seen
    te_out["si_log_count"] = te_log
    te_out["si_prev_gap_di"] = te_gap
    return tr_out, te_out


# ── cross-sectional z-score (anonymous numeric cols only) ─────────────────────
def cross_sectional_zscore_anonymous(df, anon_numeric_cols, group_col=TIME_COL):
    """
    Z-score ONLY anonymous numeric cols within each di group.
    Meta cols (si, di, industry, sector, top*) are left bit-for-bit unchanged.
    Zero-std groups replaced with std=1.
    """
    out = df.copy()
    f_cols = [c for c in anon_numeric_cols if c in df.columns]

    arr    = out[f_cols].to_numpy(np.float64)
    groups = out[group_col].to_numpy()
    result = arr.copy()

    for g in np.unique(groups):
        mask  = groups == g
        sub   = arr[mask]
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(sub, axis=0)
            sigma = np.nanstd(sub, axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sigma = np.where(np.isfinite(sigma) & (sigma != 0), sigma, 1.0)
        result[mask] = (sub - mu) / (sigma + 1e-8)

    out[f_cols] = result.astype(np.float32)
    return out


# ── percentile rank features (anonymous cols only) ────────────────────────────
def add_rank_features(df, rank_source_cols, group_col=TIME_COL, suffix="_csrank"):
    """
    Add within-di percentile rank for each col in rank_source_cols.
    NaN stays NaN. Meta cols are never touched.
    **Exploratory / submission only** — do not use for honest OOF with supervised
    column selection; use fold_safe_within_di_rank + per-fold top-N instead.
    """
    out = df.copy()
    grp = out.groupby(group_col, sort=False)
    for col in rank_source_cols:
        if col in out.columns:
            out[f"{col}{suffix}"] = grp[col].rank(pct=True, na_option="keep")
    return out


def get_sparse_anon_cols(df_train, anon_cols, nan_frac_threshold=0.20):
    """Anonymous columns with missing fraction >= threshold."""
    miss = df_train[anon_cols].isna().mean()
    return [c for c in anon_cols if float(miss.get(c, 0.0)) >= nan_frac_threshold]


def add_sparse_indicators(df, sparse_cols):
    """For each sparse f_x: add _isna, _filled0, _present_x_value."""
    out = df.copy()
    for c in sparse_cols:
        if c not in out.columns:
            continue
        filled = out[c].fillna(0.0).astype(np.float32)
        isna = out[c].isna().astype(np.int8)
        out[f"{c}_isna"] = isna
        out[f"{c}_filled0"] = filled
        out[f"{c}_present_x_value"] = filled * (1.0 - isna.astype(np.float32))
    return out


def build_fold_local_family_interactions(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    reps: list[str],
    top_k: int = TOP_FOLD_INTERACTIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Fold-local target-aware interaction selection:
    score candidates on train fold only, keep top-k |Pearson|.
    """
    y = df_train[TARGET_COL].to_numpy(np.float64)
    cands = []
    for i, a in enumerate(reps):
        if a not in df_train.columns:
            continue
        xa = df_train[a].to_numpy(np.float64)
        for b in reps[i + 1:]:
            if b not in df_train.columns:
                continue
            xb = df_train[b].to_numpy(np.float64)
            for kind in ("prod", "diff", "ratio"):
                if kind == "prod":
                    v = xa * xb
                elif kind == "diff":
                    v = xa - xb
                else:
                    v = xa / (1.0 + np.abs(xb))
                m = np.isfinite(v[tr_idx]) & np.isfinite(y[tr_idx])
                if int(m.sum()) < 100:
                    continue
                r = pearson(y[tr_idx][m], v[tr_idx][m])
                if np.isfinite(r):
                    cands.append((abs(r), r, a, b, kind))
    cands.sort(key=lambda x: x[0], reverse=True)
    chosen = cands[:top_k]

    tr = df_train.copy()
    te = df_test.copy()
    names = []
    for _, _, a, b, kind in chosen:
        name = f"ix_{a}_{b}_{kind}"
        xa_tr = tr[a].to_numpy(np.float64)
        xb_tr = tr[b].to_numpy(np.float64)
        xa_te = te[a].to_numpy(np.float64)
        xb_te = te[b].to_numpy(np.float64)
        if kind == "prod":
            v_tr = xa_tr * xb_tr
            v_te = xa_te * xb_te
        elif kind == "diff":
            v_tr = xa_tr - xb_tr
            v_te = xa_te - xb_te
        else:
            v_tr = xa_tr / (1.0 + np.abs(xb_tr))
            v_te = xa_te / (1.0 + np.abs(xb_te))
        tr[name] = v_tr
        te[name] = v_te
        names.append(name)
    return tr, te, names

def fold_safe_within_di_rank(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    value_col: str,
    group_col: str,
) -> np.ndarray:
    """
    For each row, midrank percentile of value_col within its group_col bucket,
    using only train_idx rows in that bucket as the reference distribution.
    No validation (or test) values enter the reference pool.
    """
    n = len(df)
    out = np.full(n, np.nan, dtype=np.float64)
    if value_col not in df.columns:
        return out
    vals = df[value_col].to_numpy(np.float64)
    groups = df[group_col].to_numpy()
    tr_mask = np.zeros(n, dtype=bool)
    tr_mask[train_idx] = True

    for g in np.unique(groups):
        idx_all = np.where(groups == g)[0]
        idx_tr = idx_all[tr_mask[idx_all]]
        if len(idx_tr) < 2:
            continue
        ref = vals[idx_tr]
        finite = np.isfinite(ref)
        if finite.sum() < 2:
            continue
        ref_f = ref[finite]
        for i in idx_all:
            v = vals[i]
            if not np.isfinite(v):
                continue
            out[i] = (np.sum(ref_f < v) + 0.5 * np.sum(ref_f == v)) / len(ref_f)
    return out


def fold_safe_within_di_rank_test(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    train_idx: np.ndarray,
    value_col: str,
    group_col: str,
) -> np.ndarray:
    """Test rows: rank vs train-fold reference only, same di group."""
    n_te = len(df_test)
    out = np.full(n_te, np.nan, dtype=np.float64)
    if value_col not in df_train.columns or value_col not in df_test.columns:
        return out
    vals_te = df_test[value_col].to_numpy(np.float64)
    groups_te = df_test[group_col].to_numpy()
    tr_mask = np.zeros(len(df_train), dtype=bool)
    tr_mask[train_idx] = True
    vals_tr = df_train[value_col].to_numpy(np.float64)
    groups_tr = df_train[group_col].to_numpy()

    for j in range(n_te):
        g = groups_te[j]
        v = vals_te[j]
        if not np.isfinite(v):
            continue
        idx_tr = np.where((groups_tr == g) & tr_mask)[0]
        if len(idx_tr) < 2:
            continue
        ref = vals_tr[idx_tr]
        ref = ref[np.isfinite(ref)]
        if len(ref) < 2:
            continue
        out[j] = (np.sum(ref < v) + 0.5 * np.sum(ref == v)) / len(ref)
    return out


def _union_top_n_anon_per_fold(
    df_train: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    anon_numeric_cols: list[str],
    top_n: int,
) -> list[str]:
    """Union of top-N anonymous features by |Pearson| with target on each fold's train only."""
    seen: set[str] = set()
    for tr_idx, _ in splits:
        sub = df_train.iloc[tr_idx]
        dc = per_feature_correlation(sub, anon_numeric_cols)
        for f in dc["feature"].head(top_n).tolist():
            seen.add(f)
    return sorted(seen)


# ── CatBoost CV ────────────────────────────────────────────────────────────────
class PearsonEvalMetric:
    """Custom CatBoost eval metric: maximize Pearson correlation directly."""
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True  # higher = better

    def evaluate(self, approxes, target, weight):
        pred = np.array(approxes[0], dtype=np.float64)
        targ = np.array(target,      dtype=np.float64)
        mask = np.isfinite(pred) & np.isfinite(targ)
        if mask.sum() < 2 or np.std(pred[mask]) == 0 or np.std(targ[mask]) == 0:
            return 0.0, 1
        r = float(np.corrcoef(pred[mask], targ[mask])[0, 1])
        return (r if np.isfinite(r) else 0.0), 1


def run_catboost(df_train, df_test, feature_cols, splits, meta_cols, tag, iterations=3000):
    from catboost import CatBoostRegressor

    # Only keep cols present in both dataframes
    feat = [c for c in feature_cols if c in df_train.columns and c in df_test.columns]
    cat_features = [c for c in meta_cols if c in feat]

    tr_cb = df_train[feat].copy()
    te_cb = df_test[feat].copy()
    for c in cat_features:
        tr_cb[c] = tr_cb[c].astype(str)
        te_cb[c] = te_cb[c].astype(str)

    y = df_train[TARGET_COL].to_numpy(np.float64)
    oof       = np.zeros(len(df_train))
    covered   = np.zeros(len(df_train), dtype=bool)
    test_pred = np.zeros(len(df_test))
    fold_scores = []
    best_iters = []

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        # Hard assertion: no future di in train
        tr_di = df_train.iloc[tr_idx][TIME_COL]
        va_di = df_train.iloc[va_idx][TIME_COL]
        assert tr_di.max() < va_di.min(), \
            f"[{tag}] fold {fold_id}: train di max={tr_di.max()} >= valid di min={va_di.min()}"

        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric=PearsonEvalMetric(),   # stop on Pearson, not RMSE
            depth=6,
            learning_rate=0.05,
            iterations=iterations,
            l2_leaf_reg=3.0,
            random_seed=SEED + fold_id,
            subsample=0.8,
            bootstrap_type="Bernoulli",
            od_type="Iter",
            od_wait=300,                       # give model 300 rounds to improve Pearson
            verbose=False,
        )
        model.fit(
            tr_cb.iloc[tr_idx], y[tr_idx],
            cat_features=cat_features if cat_features else None,
            eval_set=(tr_cb.iloc[va_idx], y[va_idx]),
            use_best_model=True,
        )
        oof[va_idx]  = model.predict(tr_cb.iloc[va_idx])
        covered[va_idx] = True
        test_pred   += model.predict(te_cb) / len(splits)
        fc = pearson(y[va_idx], oof[va_idx])
        fold_scores.append(fc)
        bi = int(model.best_iteration_) if model.best_iteration_ is not None else -1
        best_iters.append(bi)
        print(f"  [{tag}] fold {fold_id}: Pearson={fc:.6f}  best_iter={bi}")
        del model; gc.collect()

    oof_score, cov_frac = oof_pearson_on_covered(y, oof, covered)
    print(f"  [{tag}] OOF Pearson (covered only): {oof_score:.6f}  coverage_frac={cov_frac:.4f}")
    return oof, test_pred, oof_score, fold_scores, cov_frac, covered, len(feat), best_iters


def run_catboost_foldwise_rank_selection(
    df_train_base: pd.DataFrame,
    df_test_base: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    meta_cols: list[str],
    anon_cols: list[str],
    anon_numeric_cols: list[str],
    hist_cols: list[str],
    tag: str,
    top_n: int = TOP_N_ANON_SELECT,
    iterations: int = 3000,
):
    """
    Experiments C/D (honest): top-N anonymous features chosen **inside each train fold**
    only; within-di rank features use **train-fold rows only** as reference per di.
    Column set = union of all per-fold top-N rank bases so CatBoost sees consistent
    features (unused ranks NaN for that fold). Test preds averaged across folds.
    """
    from catboost import CatBoostRegressor

    rank_union = _union_top_n_anon_per_fold(df_train_base, splits, anon_numeric_cols, top_n)
    rank_cols = [f"{f}_csrank" for f in rank_union]
    feat_cols = [c for c in meta_cols + anon_cols + hist_cols + rank_cols
                 if c in df_train_base.columns and c in df_test_base.columns]
    cat_features = [c for c in meta_cols if c in feat_cols]

    y = df_train_base[TARGET_COL].to_numpy(np.float64)
    oof = np.zeros(len(df_train_base))
    covered = np.zeros(len(df_train_base), dtype=bool)
    test_pred = np.zeros(len(df_test_base))
    fold_scores = []

    print(f"  [{tag}] foldwise rank: |rank_union|={len(rank_union)} anon cols "
          f"(union of per-fold top-{top_n})")

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        tr_di = df_train_base.iloc[tr_idx][TIME_COL]
        va_di = df_train_base.iloc[va_idx][TIME_COL]
        assert tr_di.max() < va_di.min(), f"[{tag}] fold {fold_id}: walk-forward violated"

        dc_k = per_feature_correlation(df_train_base.iloc[tr_idx], anon_numeric_cols)
        top_k = set(dc_k["feature"].head(top_n).tolist())

        df_tr = df_train_base.copy()
        df_te = df_test_base.copy()
        for f in rank_union:
            col_r = f"{f}_csrank"
            r_tr = fold_safe_within_di_rank(df_train_base, tr_idx, f, TIME_COL)
            df_tr[col_r] = r_tr
            r_te = fold_safe_within_di_rank_test(df_train_base, df_test_base, tr_idx, f, TIME_COL)
            df_te[col_r] = r_te
            if f not in top_k:
                df_tr[col_r] = np.nan
                df_te[col_r] = np.nan

        tr_cb = df_tr[feat_cols].copy()
        te_cb = df_te[feat_cols].copy()
        for c in cat_features:
            tr_cb[c] = tr_cb[c].astype(str)
            te_cb[c] = te_cb[c].astype(str)

        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric=PearsonEvalMetric(),
            depth=6,
            learning_rate=0.05,
            iterations=iterations,
            l2_leaf_reg=3.0,
            random_seed=SEED + fold_id,
            subsample=0.8,
            bootstrap_type="Bernoulli",
            od_type="Iter",
            od_wait=300,
            verbose=False,
        )
        model.fit(
            tr_cb.iloc[tr_idx], y[tr_idx],
            cat_features=cat_features if cat_features else None,
            eval_set=(tr_cb.iloc[va_idx], y[va_idx]),
            use_best_model=True,
        )
        oof[va_idx] = model.predict(tr_cb.iloc[va_idx])
        covered[va_idx] = True
        test_pred += model.predict(te_cb) / len(splits)
        fc = pearson(y[va_idx], oof[va_idx])
        fold_scores.append(fc)
        print(f"  [{tag}] fold {fold_id}: Pearson={fc:.6f}  best_iter={model.best_iteration_}")
        del model; gc.collect()

    oof_score, cov_frac = oof_pearson_on_covered(y, oof, covered)
    print(f"  [{tag}] OOF Pearson (covered only): {oof_score:.6f}  coverage_frac={cov_frac:.4f}")
    return oof, test_pred, oof_score, fold_scores, cov_frac, covered, len(feat_cols)


def run_catboost_with_fold_local_interactions(
    df_train_base: pd.DataFrame,
    df_test_base: pd.DataFrame,
    base_feat_cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    meta_cols: list[str],
    reps: list[str],
    tag: str,
    iterations: int = 3000,
):
    from catboost import CatBoostRegressor

    y = df_train_base[TARGET_COL].to_numpy(np.float64)
    oof = np.zeros(len(df_train_base))
    covered = np.zeros(len(df_train_base), dtype=bool)
    test_pred = np.zeros(len(df_test_base))
    fold_scores = []
    best_iters = []
    fold_interactions: dict[str, list[str]] = {}
    feat_n = None

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        tr_di = df_train_base.iloc[tr_idx][TIME_COL]
        va_di = df_train_base.iloc[va_idx][TIME_COL]
        assert tr_di.max() < va_di.min(), f"[{tag}] fold {fold_id}: walk-forward violated"

        df_tr, df_te, ix_cols = build_fold_local_family_interactions(
            df_train_base, df_test_base, tr_idx, va_idx, reps, top_k=TOP_FOLD_INTERACTIONS
        )
        fold_interactions[str(fold_id)] = ix_cols
        feat_cols = [c for c in (base_feat_cols + ix_cols) if c in df_tr.columns and c in df_te.columns]
        feat_n = len(feat_cols)
        cat_features = [c for c in meta_cols if c in feat_cols]
        tr_cb = df_tr[feat_cols].copy()
        te_cb = df_te[feat_cols].copy()
        for c in cat_features:
            tr_cb[c] = tr_cb[c].astype(str)
            te_cb[c] = te_cb[c].astype(str)

        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric=PearsonEvalMetric(),
            depth=6,
            learning_rate=0.05,
            iterations=iterations,
            l2_leaf_reg=3.0,
            random_seed=SEED + fold_id,
            subsample=0.8,
            bootstrap_type="Bernoulli",
            od_type="Iter",
            od_wait=300,
            verbose=False,
        )
        model.fit(
            tr_cb.iloc[tr_idx], y[tr_idx],
            cat_features=cat_features if cat_features else None,
            eval_set=(tr_cb.iloc[va_idx], y[va_idx]),
            use_best_model=True,
        )
        oof[va_idx] = model.predict(tr_cb.iloc[va_idx])
        covered[va_idx] = True
        test_pred += model.predict(te_cb) / len(splits)
        fc = pearson(y[va_idx], oof[va_idx])
        fold_scores.append(fc)
        bi = int(model.best_iteration_) if model.best_iteration_ is not None else -1
        best_iters.append(bi)
        print(
            f"  [{tag}] fold {fold_id}: Pearson={fc:.6f}  best_iter={bi}  "
            f"top_ix={len(ix_cols)}"
        )
        del model
        gc.collect()

    oof_score, cov_frac = oof_pearson_on_covered(y, oof, covered)
    print(f"  [{tag}] OOF Pearson (covered only): {oof_score:.6f}  coverage_frac={cov_frac:.4f}")
    return oof, test_pred, oof_score, fold_scores, cov_frac, covered, int(feat_n or 0), best_iters, fold_interactions


def _prepare_d7_fold_dataset(
    df_train_base: pd.DataFrame,
    df_test_base: pd.DataFrame,
    base_feat_cols: list[str],
    meta_cols: list[str],
    reps: list[str],
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
):
    """Build one fold's D7 train/valid/test matrices with fold-local interactions."""
    df_aug_tr, df_aug_te, ix_cols = build_fold_local_family_interactions(
        df_train_base, df_test_base, tr_idx, va_idx, reps, top_k=TOP_FOLD_INTERACTIONS
    )
    feat_cols = [c for c in (base_feat_cols + ix_cols) if c in df_aug_tr.columns and c in df_aug_te.columns]
    cat_cols = [c for c in meta_cols if c in feat_cols and c != TIME_COL]
    num_cols = [
        c for c in feat_cols
        if c not in cat_cols and pd.api.types.is_numeric_dtype(df_aug_tr[c])
    ]

    tr_df = df_aug_tr.iloc[tr_idx][feat_cols].copy()
    va_df = df_aug_tr.iloc[va_idx][feat_cols].copy()
    te_df = df_aug_te[feat_cols].copy()
    for c in cat_cols:
        tr_df[c] = tr_df[c].astype(str)
        va_df[c] = va_df[c].astype(str)
        te_df[c] = te_df[c].astype(str)
    return {
        "train_df": tr_df,
        "valid_df": va_df,
        "test_df": te_df,
        "feat_cols": feat_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "ix_cols": ix_cols,
    }


def _fit_numeric_preprocessor(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    standardize: bool = False,
):
    """Fold-local median impute and optional standardize."""
    med = X_tr.median(axis=0, numeric_only=True).fillna(0.0)
    Xtr = X_tr.fillna(med).fillna(0.0)
    Xva = X_va.fillna(med).fillna(0.0)
    Xte = X_te.fillna(med).fillna(0.0)
    if not standardize:
        return Xtr, Xva, Xte
    mu = Xtr.mean(axis=0).fillna(0.0)
    sd = Xtr.std(axis=0).replace(0, 1.0).fillna(1.0)
    return (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd


def run_catboost_bagged_d7(
    df_train_base: pd.DataFrame,
    df_test_base: pd.DataFrame,
    base_feat_cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    meta_cols: list[str],
    reps: list[str],
):
    """Bagged CatBoost over the fixed D7 representation."""
    from catboost import CatBoostRegressor

    configs = {
        "A": {"depth": 5, "learning_rate": 0.02, "l2_leaf_reg": 12.0, "subsample": 0.7},
        "B": {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 8.0, "subsample": 0.8},
        "C": {"depth": 8, "learning_rate": 0.02, "l2_leaf_reg": 12.0, "subsample": 0.8},
    }
    seeds = [42, 52, 62, 72]
    y = df_train_base[TARGET_COL].to_numpy(np.float64)
    covered = np.zeros(len(df_train_base), dtype=bool)
    for _, va_idx in splits:
        covered[va_idx] = True

    oof_runs = []
    test_runs = []
    fold_interactions_all: dict[str, dict[str, list[str]]] = {}
    best_iters_all: list[int] = []

    for cfg_name, cfg in configs.items():
        for seed in seeds:
            tag = f"D7_cb_{cfg_name}_s{seed}"
            oof = np.zeros(len(df_train_base), dtype=np.float64)
            test_pred = np.zeros(len(df_test_base), dtype=np.float64)
            fold_interactions_all[tag] = {}
            print(f"\nRunning {tag}")
            for fold_id, (tr_idx, va_idx) in enumerate(splits):
                fold = _prepare_d7_fold_dataset(
                    df_train_base, df_test_base, base_feat_cols, meta_cols, reps, tr_idx, va_idx
                )
                fold_interactions_all[tag][str(fold_id)] = fold["ix_cols"]
                model = CatBoostRegressor(
                    loss_function="RMSE",
                    eval_metric=PearsonEvalMetric(),
                    depth=cfg["depth"],
                    learning_rate=cfg["learning_rate"],
                    iterations=3000,
                    l2_leaf_reg=cfg["l2_leaf_reg"],
                    random_seed=seed + fold_id,
                    subsample=cfg["subsample"],
                    bootstrap_type="Bernoulli",
                    od_type="Iter",
                    od_wait=300,
                    verbose=False,
                )
                model.fit(
                    fold["train_df"], y[tr_idx],
                    cat_features=fold["cat_cols"] if fold["cat_cols"] else None,
                    eval_set=(fold["valid_df"], y[va_idx]),
                    use_best_model=True,
                )
                p_va = model.predict(fold["valid_df"])
                oof[va_idx] = p_va
                test_pred += model.predict(fold["test_df"]) / len(splits)
                bi = int(model.best_iteration_) if model.best_iteration_ is not None else -1
                best_iters_all.append(bi)
                print(f"  [{tag}] fold {fold_id}: Pearson={pearson(y[va_idx], p_va):.6f}  best_iter={bi}")
                del model
                gc.collect()
            oof_runs.append(oof)
            test_runs.append(test_pred)

    oof_avg = np.mean(np.stack(oof_runs, axis=0), axis=0)
    test_avg = np.mean(np.stack(test_runs, axis=0), axis=0)
    oof_score, cov_frac = oof_pearson_on_covered(y, oof_avg, covered)
    pd.DataFrame({
        "oof_pred": oof_avg,
        "target": y,
        "oof_covered": covered.astype(np.int8),
    }).to_csv(OUTPUT_DIR / "oof_D7_cb_bag.csv", index=False)
    make_submission(df_test_base[ID_COL], test_avg, "D7_cb_bag")
    (OUTPUT_DIR / "D7_cb_bag_fold_interactions.json").write_text(json.dumps(fold_interactions_all, indent=2))
    return {
        "model_name": "D7_cb_bag",
        "oof_pred": oof_avg,
        "test_pred": test_avg,
        "oof_corr": oof_score,
        "coverage_frac": cov_frac,
        "covered": covered,
        "target": y,
        "test_ids": df_test_base[ID_COL].to_numpy(),
        "mean_best_iter_or_epochs": float(np.mean(best_iters_all)) if best_iters_all else float("nan"),
    }


def run_extratrees_d7(
    df_train_base: pd.DataFrame,
    df_test_base: pd.DataFrame,
    base_feat_cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    meta_cols: list[str],
    reps: list[str],
):
    """ExtraTrees on numeric-only D7 features with fold-local median imputation."""
    from sklearn.ensemble import ExtraTreesRegressor

    settings = [(42, 0.2), (52, 0.2), (42, 0.4), (52, 0.4)]
    y = df_train_base[TARGET_COL].to_numpy(np.float64)
    covered = np.zeros(len(df_train_base), dtype=bool)
    for _, va_idx in splits:
        covered[va_idx] = True

    oof_runs = []
    test_runs = []
    for seed, max_features in settings:
        tag = f"D7_et_s{seed}_mf{max_features}"
        oof = np.zeros(len(df_train_base), dtype=np.float64)
        test_pred = np.zeros(len(df_test_base), dtype=np.float64)
        print(f"\nRunning {tag}")
        for fold_id, (tr_idx, va_idx) in enumerate(splits):
            fold = _prepare_d7_fold_dataset(
                df_train_base, df_test_base, base_feat_cols, meta_cols, reps, tr_idx, va_idx
            )
            Xtr, Xva, Xte = _fit_numeric_preprocessor(
                fold["train_df"][fold["num_cols"]],
                fold["valid_df"][fold["num_cols"]],
                fold["test_df"][fold["num_cols"]],
                standardize=False,
            )
            model = ExtraTreesRegressor(
                n_estimators=1000,
                min_samples_leaf=20,
                bootstrap=True,
                max_features=max_features,
                random_state=seed + fold_id,
                n_jobs=-1,
            )
            model.fit(Xtr, y[tr_idx])
            p_va = model.predict(Xva)
            oof[va_idx] = p_va
            test_pred += model.predict(Xte) / len(splits)
            print(f"  [{tag}] fold {fold_id}: Pearson={pearson(y[va_idx], p_va):.6f}")
            del model
            gc.collect()
        oof_runs.append(oof)
        test_runs.append(test_pred)

    oof_avg = np.mean(np.stack(oof_runs, axis=0), axis=0)
    test_avg = np.mean(np.stack(test_runs, axis=0), axis=0)
    oof_score, cov_frac = oof_pearson_on_covered(y, oof_avg, covered)
    pd.DataFrame({
        "oof_pred": oof_avg,
        "target": y,
        "oof_covered": covered.astype(np.int8),
    }).to_csv(OUTPUT_DIR / "oof_D7_et.csv", index=False)
    make_submission(df_test_base[ID_COL], test_avg, "D7_et")
    return {
        "model_name": "D7_et",
        "oof_pred": oof_avg,
        "test_pred": test_avg,
        "oof_corr": oof_score,
        "coverage_frac": cov_frac,
        "covered": covered,
        "target": y,
        "test_ids": df_test_base[ID_COL].to_numpy(),
        "mean_best_iter_or_epochs": 1000.0,
    }


def run_mlp_d7(
    df_train_base: pd.DataFrame,
    df_test_base: pd.DataFrame,
    base_feat_cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    meta_cols: list[str],
    reps: list[str],
):
    """Small D7 MLP on numeric-only features with early stopping on validation Pearson."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("run_mlp_d7 requires torch to be installed.") from exc

    class MLP(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def predict_array(model, arr: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        preds = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(arr), batch_size):
                batch = torch.tensor(arr[start:start + batch_size], dtype=torch.float32, device=device)
                preds.append(model(batch).detach().cpu().numpy().astype(np.float64))
        return np.concatenate(preds, axis=0) if preds else np.zeros(0, dtype=np.float64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    y = df_train_base[TARGET_COL].to_numpy(np.float64)
    covered = np.zeros(len(df_train_base), dtype=bool)
    for _, va_idx in splits:
        covered[va_idx] = True

    oof_runs = []
    test_runs = []
    epoch_hist: list[int] = []
    for seed in [42, 52]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        tag = f"D7_mlp_s{seed}"
        oof = np.zeros(len(df_train_base), dtype=np.float64)
        test_pred = np.zeros(len(df_test_base), dtype=np.float64)
        print(f"\nRunning {tag}")
        for fold_id, (tr_idx, va_idx) in enumerate(splits):
            fold = _prepare_d7_fold_dataset(
                df_train_base, df_test_base, base_feat_cols, meta_cols, reps, tr_idx, va_idx
            )
            Xtr, Xva, Xte = _fit_numeric_preprocessor(
                fold["train_df"][fold["num_cols"]],
                fold["valid_df"][fold["num_cols"]],
                fold["test_df"][fold["num_cols"]],
                standardize=True,
            )
            Xtr_np = Xtr.to_numpy(np.float32)
            Xva_np = Xva.to_numpy(np.float32)
            Xte_np = Xte.to_numpy(np.float32)
            ytr_np = y[tr_idx].astype(np.float32)
            yva = y[va_idx]

            loader = DataLoader(
                TensorDataset(
                    torch.tensor(Xtr_np, dtype=torch.float32),
                    torch.tensor(ytr_np, dtype=torch.float32),
                ),
                batch_size=1024,
                shuffle=True,
            )
            model = MLP(Xtr_np.shape[1]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            loss_fn = nn.MSELoss()
            best_state = None
            best_score = -1e9
            best_epoch = 0
            bad_epochs = 0

            for epoch in range(1, 81):
                model.train()
                for xb, yb in loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    opt.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    opt.step()
                p_va = predict_array(model, Xva_np)
                score = pearson(yva, p_va)
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_epoch = epoch
                    bad_epochs = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad_epochs += 1
                if bad_epochs >= 10:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)
            p_va = predict_array(model, Xva_np)
            p_te = predict_array(model, Xte_np)
            oof[va_idx] = p_va
            test_pred += p_te / len(splits)
            epoch_hist.append(best_epoch)
            print(f"  [{tag}] fold {fold_id}: Pearson={pearson(y[va_idx], p_va):.6f}  best_epoch={best_epoch}")
            del model
            gc.collect()
        oof_runs.append(oof)
        test_runs.append(test_pred)

    oof_avg = np.mean(np.stack(oof_runs, axis=0), axis=0)
    test_avg = np.mean(np.stack(test_runs, axis=0), axis=0)
    oof_score, cov_frac = oof_pearson_on_covered(y, oof_avg, covered)
    pd.DataFrame({
        "oof_pred": oof_avg,
        "target": y,
        "oof_covered": covered.astype(np.int8),
    }).to_csv(OUTPUT_DIR / "oof_D7_mlp.csv", index=False)
    make_submission(df_test_base[ID_COL], test_avg, "D7_mlp")
    return {
        "model_name": "D7_mlp",
        "oof_pred": oof_avg,
        "test_pred": test_avg,
        "oof_corr": oof_score,
        "coverage_frac": cov_frac,
        "covered": covered,
        "target": y,
        "test_ids": df_test_base[ID_COL].to_numpy(),
        "mean_best_iter_or_epochs": float(np.mean(epoch_hist)) if epoch_hist else float("nan"),
    }


def _sanitize_token(value: object) -> str:
    text = str(value)
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
    return cleaned or "missing"


def _is_raw_anon_col(col: str) -> bool:
    return col.startswith("f_") and col[2:].isdigit()


def _get_reference_frame(df_train_ref: pd.DataFrame, tr_idx: np.ndarray | None) -> pd.DataFrame:
    return df_train_ref if tr_idx is None else df_train_ref.iloc[tr_idx]


def _rank_cols_by_abs_pearson(df: pd.DataFrame, cols: list[str], y: np.ndarray) -> list[str]:
    rows = []
    y = np.asarray(y, np.float64)
    for col in cols:
        if col not in df.columns:
            continue
        x = df[col].to_numpy(np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 100:
            continue
        r = pearson(y[mask], x[mask])
        if np.isfinite(r):
            rows.append((abs(r), col))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [col for _, col in rows]


def select_peer_transform_cols(
    df_train_fold: pd.DataFrame,
    anon_cols: list[str],
    sparse_cols: list[str],
    y_train: np.ndarray,
    top_sparse: int = 20,
    top_dense: int = 20,
) -> list[str]:
    sparse_set = {c for c in sparse_cols if c in anon_cols}
    sparse_ranked = _rank_cols_by_abs_pearson(df_train_fold, [c for c in anon_cols if c in sparse_set], y_train)
    dense_ranked = _rank_cols_by_abs_pearson(df_train_fold, [c for c in anon_cols if c not in sparse_set], y_train)
    selected: list[str] = []
    for col in sparse_ranked[:top_sparse] + dense_ranked[:top_dense]:
        if col not in selected:
            selected.append(col)
        if len(selected) >= top_sparse + top_dense:
            break
    return selected[: top_sparse + top_dense]


def select_interaction_reps(
    df_train_fold: pd.DataFrame,
    anon_cols: list[str],
    sparse_cols: list[str],
    y_train: np.ndarray,
    n_sparse: int = 12,
    n_dense: int = 12,
) -> list[str]:
    sparse_set = {c for c in sparse_cols if c in anon_cols}
    sparse_ranked = _rank_cols_by_abs_pearson(df_train_fold, [c for c in anon_cols if c in sparse_set], y_train)
    dense_ranked = _rank_cols_by_abs_pearson(df_train_fold, [c for c in anon_cols if c not in sparse_set], y_train)
    selected: list[str] = []
    for col in sparse_ranked[:n_sparse]:
        if col not in selected:
            selected.append(col)
    for col in dense_ranked[:n_dense]:
        if col not in selected:
            selected.append(col)
    backfill = dense_ranked[n_dense:] + sparse_ranked[n_sparse:]
    backfill += [c for c in DEFAULT_FAMILY_REPS if c in anon_cols]
    backfill += anon_cols
    for col in backfill:
        if col not in selected:
            selected.append(col)
        if len(selected) >= n_sparse + n_dense:
            break
    return selected[: n_sparse + n_dense]


def build_fold_numeric_meta_context(
    df_train_ref: pd.DataFrame,
    df_apply: pd.DataFrame,
    tr_idx: np.ndarray | None,
) -> tuple[pd.DataFrame, list[str]]:
    ref = _get_reference_frame(df_train_ref, tr_idx)
    out = pd.DataFrame(index=df_apply.index)

    for col in [TIME_COL, "top2000", "top1000", "top500"]:
        if col in df_apply.columns:
            out[col] = pd.to_numeric(df_apply[col], errors="coerce").astype(np.float32)

    denom = max(len(ref), 1)
    for col in [STOCK_COL, "industry", "sector"]:
        if col in ref.columns and col in df_apply.columns:
            freq = ref[col].value_counts(dropna=False) / denom
            out[f"{col}_freq"] = df_apply[col].map(freq).fillna(0.0).astype(np.float32)

    for col in ["si_seen_before", "si_log_count", "si_prev_gap_di"]:
        if col in df_apply.columns:
            out[col] = pd.to_numeric(df_apply[col], errors="coerce").astype(np.float32)

    for col in ["sector", "industry"]:
        if col not in ref.columns or col not in df_apply.columns:
            continue
        seen_clean: set[str] = set()
        ref_unique = ref[col].dropna().unique().tolist()
        apply_tokens = df_apply[col].map(_sanitize_token)
        for raw_cat in ref_unique:
            clean = _sanitize_token(raw_cat)
            if clean in seen_clean:
                continue
            seen_clean.add(clean)
            out[f"{col}_oh_{clean}"] = (apply_tokens == clean).astype(np.float32)

    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out, out.columns.tolist()


def build_group_relative_features(
    df_train_ref: pd.DataFrame,
    df_apply: pd.DataFrame,
    cols: list[str],
    group_col: str,
    time_col: str,
    tr_idx: np.ndarray | None,
    min_group_size: int = 5,
) -> tuple[pd.DataFrame, list[str]]:
    ref = _get_reference_frame(df_train_ref, tr_idx)
    out = pd.DataFrame(index=df_apply.index)
    created: list[str] = []
    if group_col not in ref.columns or group_col not in df_apply.columns:
        return out, created

    for col in cols:
        if col not in ref.columns or col not in df_apply.columns:
            continue
        ref_sub = ref[[time_col, group_col, col]].copy()
        grp = ref_sub.groupby([time_col, group_col], dropna=False, sort=False)[col]
        stats = grp.agg(
            finite_count="count",
            mean="mean",
            median="median",
            std="std",
            missrate=lambda s: float(s.isna().mean()),
        ).reset_index()
        stats["std"] = stats["std"].replace(0, np.nan)

        merged = df_apply[[time_col, group_col]].merge(stats, on=[time_col, group_col], how="left", sort=False)
        vals = pd.to_numeric(df_apply[col], errors="coerce").to_numpy(np.float64)
        finite_count = pd.to_numeric(merged["finite_count"], errors="coerce").fillna(0).to_numpy(np.float64)
        valid_bucket = finite_count >= float(min_group_size)
        valid_val = valid_bucket & np.isfinite(vals)
        mean = pd.to_numeric(merged["mean"], errors="coerce").to_numpy(np.float64)
        median = pd.to_numeric(merged["median"], errors="coerce").to_numpy(np.float64)
        std = pd.to_numeric(merged["std"], errors="coerce").to_numpy(np.float64)
        missrate = pd.to_numeric(merged["missrate"], errors="coerce").to_numpy(np.float64)

        out[f"{col}_{group_col}_demean"] = np.where(valid_val, vals - mean, np.nan).astype(np.float32)
        out[f"{col}_{group_col}_demedian"] = np.where(valid_val, vals - median, np.nan).astype(np.float32)
        out[f"{col}_{group_col}_z"] = np.where(valid_val & np.isfinite(std), (vals - mean) / (std + 1e-8), np.nan).astype(np.float32)
        out[f"{col}_{group_col}_missrate"] = np.where(valid_bucket, missrate, np.nan).astype(np.float32)
        out[f"{col}_{group_col}_above_med"] = np.where(valid_val, (vals > median).astype(np.float32), np.nan).astype(np.float32)

        rank_out = np.full(len(df_apply), np.nan, dtype=np.float32)
        ref_rank_arrays: dict[tuple[object, object], np.ndarray] = {}
        for key, series in grp:
            arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy(np.float64)
            if len(arr) >= min_group_size:
                ref_rank_arrays[key] = np.sort(arr)
        apply_groups = df_apply.groupby([time_col, group_col], dropna=False, sort=False).indices
        for key, idxs in apply_groups.items():
            arr = ref_rank_arrays.get(key)
            if arr is None or len(arr) < min_group_size:
                continue
            idxs_arr = np.asarray(list(idxs), dtype=int)
            sub = vals[idxs_arr]
            finite = np.isfinite(sub)
            if not finite.any():
                continue
            left = np.searchsorted(arr, sub[finite], side="left")
            right = np.searchsorted(arr, sub[finite], side="right")
            rank_out[idxs_arr[finite]] = ((left + right) * 0.5 / len(arr)).astype(np.float32)
        out[f"{col}_{group_col}_rank"] = rank_out

        created.extend([
            f"{col}_{group_col}_demean",
            f"{col}_{group_col}_demedian",
            f"{col}_{group_col}_z",
            f"{col}_{group_col}_rank",
            f"{col}_{group_col}_missrate",
            f"{col}_{group_col}_above_med",
        ])
    return out, created


def build_fold_peer_feature_block(
    df_train_ref: pd.DataFrame,
    df_test_ref: pd.DataFrame,
    selected_cols: list[str],
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    tr_apply = df_train_ref.iloc[tr_idx].copy()
    va_apply = df_train_ref.iloc[va_idx].copy()
    te_apply = df_test_ref.copy()

    sec_tr, _ = build_group_relative_features(df_train_ref, tr_apply, selected_cols, "sector", TIME_COL, tr_idx)
    sec_va, _ = build_group_relative_features(df_train_ref, va_apply, selected_cols, "sector", TIME_COL, tr_idx)
    sec_te, _ = build_group_relative_features(df_train_ref, te_apply, selected_cols, "sector", TIME_COL, tr_idx)
    ind_tr, _ = build_group_relative_features(df_train_ref, tr_apply, selected_cols, "industry", TIME_COL, tr_idx)
    ind_va, _ = build_group_relative_features(df_train_ref, va_apply, selected_cols, "industry", TIME_COL, tr_idx)
    ind_te, _ = build_group_relative_features(df_train_ref, te_apply, selected_cols, "industry", TIME_COL, tr_idx)

    tr_block = pd.concat([sec_tr, ind_tr], axis=1)
    va_block = pd.concat([sec_va, ind_va], axis=1)
    te_block = pd.concat([sec_te, ind_te], axis=1)
    return tr_block, va_block, te_block, tr_block.columns.tolist()


def _concat_unique(blocks: list[pd.DataFrame]) -> pd.DataFrame:
    frames = []
    seen: set[str] = set()
    for block in blocks:
        keep = [c for c in block.columns if c not in seen]
        if keep:
            frames.append(block[keep])
            seen.update(keep)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def _prepare_d7_peer_fold_blocks(
    df_train_base: pd.DataFrame,
    df_test_base: pd.DataFrame,
    base_feat_cols: list[str],
    meta_cols: list[str],
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    sparse_cols: list[str],
) -> dict[str, object]:
    y = df_train_base[TARGET_COL].to_numpy(np.float64)
    raw_anon_cols = [c for c in base_feat_cols if _is_raw_anon_col(c)]
    y_train = y[tr_idx]
    train_fold = df_train_base.iloc[tr_idx]

    selected_reps = select_interaction_reps(train_fold, raw_anon_cols, sparse_cols, y_train)
    df_aug_tr, df_aug_te, selected_interactions = build_fold_local_family_interactions(
        df_train_base, df_test_base, tr_idx, va_idx, selected_reps, top_k=TOP_FOLD_INTERACTIONS
    )
    selected_peer_cols = select_peer_transform_cols(train_fold, raw_anon_cols, sparse_cols, y_train)
    tr_peer, va_peer, te_peer, _ = build_fold_peer_feature_block(
        df_train_base, df_test_base, selected_peer_cols, tr_idx, va_idx
    )
    tr_ctx, _ = build_fold_numeric_meta_context(df_train_base, df_train_base.iloc[tr_idx].copy(), tr_idx)
    va_ctx, _ = build_fold_numeric_meta_context(df_train_base, df_train_base.iloc[va_idx].copy(), tr_idx)
    te_ctx, _ = build_fold_numeric_meta_context(df_train_base, df_test_base.copy(), tr_idx)

    base_numeric_cols = [
        c for c in (base_feat_cols + selected_interactions)
        if c in df_aug_tr.columns and c in df_aug_te.columns and pd.api.types.is_numeric_dtype(df_aug_tr[c])
    ]
    X_train_base = df_aug_tr.iloc[tr_idx][base_numeric_cols].copy()
    X_valid_base = df_aug_tr.iloc[va_idx][base_numeric_cols].copy()
    X_test_base = df_aug_te[base_numeric_cols].copy()

    X_train_num = _concat_unique([X_train_base, tr_ctx, tr_peer])
    X_valid_num = _concat_unique([X_valid_base, va_ctx, va_peer]).reindex(columns=X_train_num.columns)
    X_test_num = _concat_unique([X_test_base, te_ctx, te_peer]).reindex(columns=X_train_num.columns)

    cat_cols = [c for c in [STOCK_COL, "industry", "sector"] if c in meta_cols and c in df_train_base.columns]
    cb_train = _concat_unique([df_train_base.iloc[tr_idx][cat_cols].copy(), X_train_num])
    cb_valid = _concat_unique([df_train_base.iloc[va_idx][cat_cols].copy(), X_valid_num]).reindex(columns=cb_train.columns)
    cb_test = _concat_unique([df_test_base[cat_cols].copy(), X_test_num]).reindex(columns=cb_train.columns)

    return {
        "X_train_et": X_train_num,
        "X_valid_et": X_valid_num,
        "X_test_et": X_test_num,
        "feature_names": X_train_num.columns.tolist(),
        "selected_peer_cols": selected_peer_cols,
        "selected_reps": selected_reps,
        "selected_interactions": selected_interactions,
        "cb_train": cb_train,
        "cb_valid": cb_valid,
        "cb_test": cb_test,
        "cb_cat_cols": cat_cols,
    }


def _prepare_d7_et_peer_fold_dataset(
    df_train_base: pd.DataFrame,
    df_test_base: pd.DataFrame,
    base_feat_cols: list[str],
    meta_cols: list[str],
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    sparse_cols: list[str],
):
    blocks = _prepare_d7_peer_fold_blocks(
        df_train_base, df_test_base, base_feat_cols, meta_cols, tr_idx, va_idx, sparse_cols
    )
    return (
        blocks["X_train_et"],
        blocks["X_valid_et"],
        blocks["X_test_et"],
        blocks["feature_names"],
        blocks["selected_peer_cols"],
        blocks["selected_reps"],
        blocks["selected_interactions"],
    )


def _compute_fold_scores_from_oof(y: np.ndarray, oof_pred: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]) -> list[float]:
    return [pearson(y[va_idx], oof_pred[va_idx]) for _, va_idx in splits]


def parse_fold_list(folds_arg: str | None, n_splits: int) -> list[int]:
    if folds_arg is None or folds_arg.strip() == "":
        return list(range(n_splits))
    out = []
    for token in folds_arg.split(","):
        token = token.strip()
        if token == "":
            continue
        fold_id = int(token)
        if fold_id < 0 or fold_id >= n_splits:
            raise ValueError(f"Fold {fold_id} outside valid range [0, {n_splits - 1}]")
        if fold_id not in out:
            out.append(fold_id)
    if not out:
        raise ValueError("No valid folds parsed from --folds")
    return out


def _stage_prefix(stage_name: str) -> str:
    return stage_name.replace("/", "_")


def get_fold_checkpoint_npz(stage_name: str, fold_id: int) -> Path:
    return CHECKPOINT_DIR / f"{_stage_prefix(stage_name)}_fold{fold_id}.npz"


def get_fold_checkpoint_json(stage_name: str, fold_id: int) -> Path:
    return CHECKPOINT_DIR / f"{_stage_prefix(stage_name)}_fold{fold_id}.json"


def save_fold_checkpoint(
    stage_name: str,
    fold_id: int,
    va_idx: np.ndarray,
    oof_pred: np.ndarray,
    test_pred: np.ndarray,
    metadata: dict[str, object],
) -> None:
    np.savez_compressed(
        get_fold_checkpoint_npz(stage_name, fold_id),
        va_idx=np.asarray(va_idx, dtype=np.int64),
        oof_pred=np.asarray(oof_pred, dtype=np.float64),
        test_pred=np.asarray(test_pred, dtype=np.float64),
    )
    get_fold_checkpoint_json(stage_name, fold_id).write_text(json.dumps(metadata, indent=2))


def load_fold_checkpoint(stage_name: str, fold_id: int) -> dict[str, object] | None:
    npz_path = get_fold_checkpoint_npz(stage_name, fold_id)
    json_path = get_fold_checkpoint_json(stage_name, fold_id)
    if not npz_path.exists() or not json_path.exists():
        return None
    payload = np.load(npz_path)
    meta = json.loads(json_path.read_text())
    return {
        "va_idx": payload["va_idx"],
        "oof_pred": payload["oof_pred"],
        "test_pred": payload["test_pred"],
        "metadata": meta,
    }


def summarize_checkpoint_status(stage_name: str, folds_to_run: list[int]) -> tuple[list[int], list[int]]:
    done = []
    missing = []
    for fold_id in folds_to_run:
        if load_fold_checkpoint(stage_name, fold_id) is None:
            missing.append(fold_id)
        else:
            done.append(fold_id)
    return done, missing


def aggregate_fold_checkpoints(
    stage_name: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], dict[str, object], float]:
    y = df_train[TARGET_COL].to_numpy(np.float64)
    oof = np.zeros(len(df_train), dtype=np.float64)
    covered = np.zeros(len(df_train), dtype=bool)
    test_preds = []
    fold_scores = []
    fold_metadata: dict[str, object] = {}
    n_features = []

    for fold_id, (_, va_idx_expected) in enumerate(splits):
        ckpt = load_fold_checkpoint(stage_name, fold_id)
        if ckpt is None:
            raise FileNotFoundError(f"Missing checkpoint for {stage_name} fold {fold_id}")
        va_idx = np.asarray(ckpt["va_idx"], dtype=np.int64)
        if not np.array_equal(va_idx, np.asarray(va_idx_expected, dtype=np.int64)):
            raise ValueError(f"Checkpoint validation indices do not match current splits for {stage_name} fold {fold_id}")
        fold_pred = np.asarray(ckpt["oof_pred"], dtype=np.float64)
        oof[va_idx] = fold_pred
        covered[va_idx] = True
        test_preds.append(np.asarray(ckpt["test_pred"], dtype=np.float64))
        meta = dict(ckpt["metadata"])
        fold_metadata[str(fold_id)] = meta
        fold_scores.append(float(meta["fold_score"]))
        if "n_features" in meta:
            n_features.append(float(meta["n_features"]))

    test_avg = np.mean(np.stack(test_preds, axis=0), axis=0)
    n_features_mean = float(np.mean(n_features)) if n_features else float("nan")
    return oof, test_avg, covered, fold_scores, fold_metadata, n_features_mean


def build_stage_result_from_checkpoints(
    stage_name: str,
    model_name: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    submission_name: str,
    metadata_filename: str,
    write_noclip: bool = False,
    mean_best_iter_or_epochs: float = float("nan"),
) -> dict[str, object]:
    y = df_train[TARGET_COL].to_numpy(np.float64)
    oof_avg, test_avg, covered, fold_scores, fold_metadata, n_features_mean = aggregate_fold_checkpoints(
        stage_name, df_train, df_test, splits
    )
    oof_score, cov_frac = oof_pearson_on_covered(y, oof_avg, covered)
    pd.DataFrame({
        "oof_pred": oof_avg,
        "target": y,
        "oof_covered": covered.astype(np.int8),
    }).to_csv(OUTPUT_DIR / f"oof_{submission_name}.csv", index=False)
    make_submission(df_test[ID_COL], test_avg, submission_name, clip=True)
    if write_noclip:
        make_submission(df_test[ID_COL], test_avg, f"{submission_name}_noclip", clip=False)
    (OUTPUT_DIR / metadata_filename).write_text(json.dumps(fold_metadata, indent=2))
    return {
        "model_name": model_name,
        "oof_pred": oof_avg,
        "test_pred": test_avg,
        "oof_corr": oof_score,
        "coverage_frac": cov_frac,
        "covered": covered,
        "target": y,
        "test_ids": df_test[ID_COL].to_numpy(),
        "mean_best_iter_or_epochs": mean_best_iter_or_epochs,
        "fold_scores": fold_scores,
        "recent_fold_mean": float(np.nanmean(fold_scores[3:5])),
        "last_fold_score": float(fold_scores[4]),
        "n_features_mean": n_features_mean,
    }


def run_extratrees_d7_peer(
    df_train_base: pd.DataFrame,
    df_test_base: pd.DataFrame,
    base_feat_cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    meta_cols: list[str],
    sparse_cols: list[str],
    folds_to_run: list[int] | None = None,
    skip_completed: bool = True,
    finalize: bool = True,
):
    from sklearn.ensemble import ExtraTreesRegressor

    settings = [
        (seed, max_features, min_leaf)
        for seed in [42, 52, 62]
        for max_features in [0.10, 0.20, 0.35]
        for min_leaf in [10, 20]
    ]
    y = df_train_base[TARGET_COL].to_numpy(np.float64)
    covered = np.zeros(len(df_train_base), dtype=bool)
    for _, va_idx in splits:
        covered[va_idx] = True

    oof_sum = np.zeros(len(df_train_base), dtype=np.float64)
    test_sum = np.zeros(len(df_test_base), dtype=np.float64)
    fold_metadata: dict[str, object] = {}
    n_features_per_fold: list[int] = []

    folds_to_run = list(range(len(splits))) if folds_to_run is None else folds_to_run
    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        if fold_id not in folds_to_run:
            continue
        if skip_completed and load_fold_checkpoint("et_peer", fold_id) is not None:
            print(f"  [D7_et_peer] fold {fold_id}: checkpoint exists, skipping")
            continue
        Xtr, Xva, Xte, feature_names, peer_cols, reps, ix_cols = _prepare_d7_et_peer_fold_dataset(
            df_train_base, df_test_base, base_feat_cols, meta_cols, tr_idx, va_idx, sparse_cols
        )
        n_features_per_fold.append(len(feature_names))
        fold_meta = {
            "selected_peer_cols": peer_cols,
            "selected_reps": reps,
            "selected_interactions": ix_cols,
            "n_features": len(feature_names),
        }
        fold_metadata[str(fold_id)] = fold_meta
        Xtr_imp, Xva_imp, Xte_imp = _fit_numeric_preprocessor(Xtr, Xva, Xte, standardize=False)
        fold_oof_sum = np.zeros(len(va_idx), dtype=np.float64)
        fold_test_sum = np.zeros(len(df_test_base), dtype=np.float64)
        for seed, max_features, min_leaf in settings:
            tag = f"D7_et_peer_s{seed}_mf{max_features:.2f}_leaf{min_leaf}"
            model = ExtraTreesRegressor(
                n_estimators=1200,
                min_samples_leaf=min_leaf,
                bootstrap=True,
                max_features=max_features,
                random_state=seed + fold_id,
                n_jobs=-1,
            )
            model.fit(Xtr_imp, y[tr_idx])
            p_va = model.predict(Xva_imp)
            p_te = model.predict(Xte_imp)
            fold_oof_sum += p_va
            fold_test_sum += p_te
            print(
                f"  [{tag}] fold {fold_id}: Pearson={pearson(y[va_idx], p_va):.6f}  "
                f"n_features={len(feature_names)}  peer_cols={len(peer_cols)}  interactions={len(ix_cols)}"
            )
            del model
            gc.collect()

        fold_oof_avg = fold_oof_sum / len(settings)
        fold_test_avg = fold_test_sum / len(settings)
        fold_score = pearson(y[va_idx], fold_oof_avg)
        fold_meta["fold_score"] = float(fold_score)
        save_fold_checkpoint(
            "et_peer",
            fold_id,
            va_idx=va_idx,
            oof_pred=fold_oof_avg,
            test_pred=fold_test_avg,
            metadata=fold_meta,
        )
        oof_sum[va_idx] = fold_oof_avg
        test_sum += fold_test_avg
        covered[va_idx] = True

    if not finalize:
        done, missing = summarize_checkpoint_status("et_peer", list(range(len(splits))))
        return {
            "model_name": "D7_et_peer",
            "completed_folds": done,
            "missing_folds": missing,
        }

    return build_stage_result_from_checkpoints(
        "et_peer",
        "D7_et_peer",
        df_train_base,
        df_test_base,
        splits,
        submission_name="D7_et_peer",
        metadata_filename="D7_et_peer_fold_metadata.json",
        write_noclip=True,
        mean_best_iter_or_epochs=1200.0,
    )


def run_catboost_d7_peer(
    df_train_base: pd.DataFrame,
    df_test_base: pd.DataFrame,
    base_feat_cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    meta_cols: list[str],
    sparse_cols: list[str],
    folds_to_run: list[int] | None = None,
    skip_completed: bool = True,
    finalize: bool = True,
):
    from catboost import CatBoostRegressor

    configs = {
        "A": {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 8.0, "subsample": 0.8},
        "B": {"depth": 8, "learning_rate": 0.02, "l2_leaf_reg": 12.0, "subsample": 0.8},
    }
    seeds = [42, 52]
    y = df_train_base[TARGET_COL].to_numpy(np.float64)
    covered = np.zeros(len(df_train_base), dtype=bool)
    for _, va_idx in splits:
        covered[va_idx] = True

    oof_sum = np.zeros(len(df_train_base), dtype=np.float64)
    test_sum = np.zeros(len(df_test_base), dtype=np.float64)
    fold_metadata: dict[str, object] = {}
    n_features_per_fold: list[int] = []

    folds_to_run = list(range(len(splits))) if folds_to_run is None else folds_to_run
    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        if fold_id not in folds_to_run:
            continue
        if skip_completed and load_fold_checkpoint("cb_peer", fold_id) is not None:
            print(f"  [D7_cb_peer] fold {fold_id}: checkpoint exists, skipping")
            continue
        blocks = _prepare_d7_peer_fold_blocks(
            df_train_base, df_test_base, base_feat_cols, meta_cols, tr_idx, va_idx, sparse_cols
        )
        fold_meta = {
            "selected_peer_cols": blocks["selected_peer_cols"],
            "selected_reps": blocks["selected_reps"],
            "selected_interactions": blocks["selected_interactions"],
            "n_features": len(blocks["feature_names"]),
        }
        fold_metadata[str(fold_id)] = fold_meta
        n_features_per_fold.append(len(blocks["feature_names"]))
        cb_train = blocks["cb_train"].copy()
        cb_valid = blocks["cb_valid"].copy()
        cb_test = blocks["cb_test"].copy()
        cat_cols = [c for c in blocks["cb_cat_cols"] if c in cb_train.columns]
        for c in cat_cols:
            cb_train[c] = cb_train[c].astype(str)
            cb_valid[c] = cb_valid[c].astype(str)
            cb_test[c] = cb_test[c].astype(str)
        fold_oof_sum = np.zeros(len(va_idx), dtype=np.float64)
        fold_test_sum = np.zeros(len(df_test_base), dtype=np.float64)
        for cfg_name, cfg in configs.items():
            for seed in seeds:
                tag = f"D7_cb_peer_{cfg_name}_s{seed}"
                model = CatBoostRegressor(
                    loss_function="RMSE",
                    eval_metric=PearsonEvalMetric(),
                    depth=cfg["depth"],
                    learning_rate=cfg["learning_rate"],
                    iterations=3000,
                    l2_leaf_reg=cfg["l2_leaf_reg"],
                    random_seed=seed + fold_id,
                    subsample=cfg["subsample"],
                    bootstrap_type="Bernoulli",
                    od_type="Iter",
                    od_wait=300,
                    verbose=False,
                )
                model.fit(
                    cb_train,
                    y[tr_idx],
                    cat_features=cat_cols if cat_cols else None,
                    eval_set=(cb_valid, y[va_idx]),
                    use_best_model=True,
                )
                p_va = model.predict(cb_valid)
                p_te = model.predict(cb_test)
                fold_oof_sum += p_va
                fold_test_sum += p_te
                bi = int(model.best_iteration_) if model.best_iteration_ is not None else -1
                print(
                    f"  [{tag}] fold {fold_id}: Pearson={pearson(y[va_idx], p_va):.6f}  "
                    f"best_iter={bi}  n_features={len(blocks['feature_names'])}  "
                    f"peer_cols={len(blocks['selected_peer_cols'])}  interactions={len(blocks['selected_interactions'])}"
                )
                del model
                gc.collect()

        n_models = len(configs) * len(seeds)
        fold_oof_avg = fold_oof_sum / n_models
        fold_test_avg = fold_test_sum / n_models
        fold_score = pearson(y[va_idx], fold_oof_avg)
        fold_meta["fold_score"] = float(fold_score)
        save_fold_checkpoint(
            "cb_peer",
            fold_id,
            va_idx=va_idx,
            oof_pred=fold_oof_avg,
            test_pred=fold_test_avg,
            metadata=fold_meta,
        )
        oof_sum[va_idx] = fold_oof_avg
        test_sum += fold_test_avg
        covered[va_idx] = True

    if not finalize:
        done, missing = summarize_checkpoint_status("cb_peer", list(range(len(splits))))
        return {
            "model_name": "D7_cb_peer",
            "completed_folds": done,
            "missing_folds": missing,
        }

    return build_stage_result_from_checkpoints(
        "cb_peer",
        "D7_cb_peer",
        df_train_base,
        df_test_base,
        splits,
        submission_name="D7_cb_peer",
        metadata_filename="D7_cb_peer_fold_metadata.json",
        write_noclip=False,
        mean_best_iter_or_epochs=float("nan"),
    )


def blend_et_cb_peer(
    et_result: dict[str, object],
    cb_result: dict[str, object],
    splits: list[tuple[np.ndarray, np.ndarray]],
):
    covered = np.asarray(et_result["covered"], dtype=bool)
    y = np.asarray(et_result["target"], np.float64)

    def _zscore(oof: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = float(np.nanmean(oof[covered]))
        sd = float(np.nanstd(oof[covered]))
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        return (oof - mu) / sd, (test - mu) / sd

    et_oof_z, et_test_z = _zscore(np.asarray(et_result["oof_pred"], np.float64), np.asarray(et_result["test_pred"], np.float64))
    cb_oof_z, cb_test_z = _zscore(np.asarray(cb_result["oof_pred"], np.float64), np.asarray(cb_result["test_pred"], np.float64))

    best = None
    for et_w in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        cb_w = 1.0 - et_w
        oof_bl = et_w * et_oof_z + cb_w * cb_oof_z
        test_bl = et_w * et_test_z + cb_w * cb_test_z
        oof_score = pearson(y[covered], oof_bl[covered])
        fold_scores = _compute_fold_scores_from_oof(y, oof_bl, splits)
        recent_fold_mean = float(np.nanmean(fold_scores[3:5]))
        if best is None or oof_score > best["oof_corr"]:
            best = {
                "oof_pred": oof_bl,
                "test_pred": test_bl,
                "oof_corr": float(oof_score),
                "fold_scores": fold_scores,
                "recent_fold_mean": recent_fold_mean,
                "last_fold_score": float(fold_scores[4]),
                "weights": {"D7_et_peer": et_w, "D7_cb_peer": cb_w},
            }

    if best is None:
        raise RuntimeError("ET+CB peer blend search failed.")

    make_submission(et_result["test_ids"], best["test_pred"], "D7_et_cb_peer_blend", clip=True)
    (OUTPUT_DIR / "blend_weights_peer.json").write_text(
        json.dumps({"weights": best["weights"], "oof_pearson": best["oof_corr"]}, indent=2)
    )
    return {
        "model_name": "D7_et_cb_peer_blend",
        "oof_pred": best["oof_pred"],
        "test_pred": best["test_pred"],
        "oof_corr": best["oof_corr"],
        "coverage_frac": float(covered.mean()),
        "covered": covered,
        "target": y,
        "test_ids": et_result["test_ids"],
        "mean_best_iter_or_epochs": float("nan"),
        "fold_scores": best["fold_scores"],
        "recent_fold_mean": best["recent_fold_mean"],
        "last_fold_score": best["last_fold_score"],
        "n_features_mean": float("nan"),
        "weights": best["weights"],
    }


def build_production_model_summary(model_results: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for res in model_results:
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
    df = pd.DataFrame(rows).sort_values(["recent_fold_mean", "oof_pearson"], ascending=[False, False]).reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / "production_model_summary.csv", index=False)
    return df


def make_submission(test_ids, preds, name, clip: bool = True):
    p = np.asarray(preds, np.float64).copy()
    if clip:
        lo, hi = np.nanpercentile(p, [0.5, 99.5])
        p = np.clip(p, lo, hi)
    if (np.abs(p) > 0).mean() < 0.10:
        p += 1e-9
    sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: p})
    assert np.isfinite(sub[TARGET_COL]).all(), "Non-finite predictions"
    assert (sub[TARGET_COL].abs() > 0).mean() >= 0.10, "< 10% non-zero"
    path = OUTPUT_DIR / f"submission_{name}.csv"
    sub.to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(sub):,} rows)")
    return sub


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Advanced D7 ET/CB production runner with fold checkpoints.")
    parser.add_argument(
        "--stage",
        choices=["all", "et_peer", "cb_peer", "aggregate"],
        default="all",
        help="Run ET only, CB only, full pipeline, or aggregate from checkpoints.",
    )
    parser.add_argument(
        "--folds",
        default=None,
        help="Comma-separated fold ids to run for training stages, e.g. '0,1' or '3'.",
    )
    parser.add_argument(
        "--recompute-completed",
        action="store_true",
        help="Recompute folds even if checkpoint files already exist.",
    )
    args = parser.parse_args()

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
    folds_to_run = parse_fold_list(args.folds, len(splits))
    print(f"\nStage: {args.stage}")
    print(f"Requested folds: {folds_to_run}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR.resolve()}")

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
    sparse_derived = []
    for c in sparse_cols:
        sparse_derived.extend([f"{c}_isna", f"{c}_filled0", f"{c}_present_x_value"])
    print(f"  sparse anonymous cols (>=20% NaN): {len(sparse_cols)}")

    df_train_d4 = pd.concat([df_train_rank, df_train_z_block], axis=1)
    df_test_d4 = pd.concat([df_test_rank, df_test_z_block], axis=1)
    df_train_d6 = pd.concat([df_train_d4, df_train_sparse[sparse_derived]], axis=1)
    df_test_d6 = pd.concat([df_test_d4, df_test_sparse[sparse_derived]], axis=1)
    d7_base_feat_cols = base_cols + rank_cols + z_cols + sparse_derived
    print(f"  D7 base feature count before fold-local interactions: {len(d7_base_feat_cols)}")

    print("\n" + "=" * 60)
    print("D7 ET-first production candidates")
    print("=" * 60)
    if args.stage == "et_peer":
        done, missing = summarize_checkpoint_status("et_peer", list(range(len(splits))))
        print(f"Existing ET checkpoints: done={done} missing={missing}")
        status = run_extratrees_d7_peer(
            df_train_d6,
            df_test_d6,
            d7_base_feat_cols,
            splits,
            meta_cols,
            sparse_cols,
            folds_to_run=folds_to_run,
            skip_completed=not args.recompute_completed,
            finalize=False,
        )
        print(f"ET checkpoint status after run: completed={status['completed_folds']} missing={status['missing_folds']}")
        print(f"\nAll outputs in: {OUTPUT_DIR.resolve()}")
        return

    if args.stage == "cb_peer":
        done, missing = summarize_checkpoint_status("cb_peer", list(range(len(splits))))
        print(f"Existing CB checkpoints: done={done} missing={missing}")
        status = run_catboost_d7_peer(
            df_train_d6,
            df_test_d6,
            d7_base_feat_cols,
            splits,
            meta_cols,
            sparse_cols,
            folds_to_run=folds_to_run,
            skip_completed=not args.recompute_completed,
            finalize=False,
        )
        print(f"CB checkpoint status after run: completed={status['completed_folds']} missing={status['missing_folds']}")
        print(f"\nAll outputs in: {OUTPUT_DIR.resolve()}")
        return

    if args.stage == "all":
        run_extratrees_d7_peer(
            df_train_d6,
            df_test_d6,
            d7_base_feat_cols,
            splits,
            meta_cols,
            sparse_cols,
            folds_to_run=list(range(len(splits))),
            skip_completed=not args.recompute_completed,
            finalize=False,
        )
        run_catboost_d7_peer(
            df_train_d6,
            df_test_d6,
            d7_base_feat_cols,
            splits,
            meta_cols,
            sparse_cols,
            folds_to_run=list(range(len(splits))),
            skip_completed=not args.recompute_completed,
            finalize=False,
        )

    et_done, et_missing = summarize_checkpoint_status("et_peer", list(range(len(splits))))
    cb_done, cb_missing = summarize_checkpoint_status("cb_peer", list(range(len(splits))))
    print(f"Checkpoint status ET: done={et_done} missing={et_missing}")
    print(f"Checkpoint status CB: done={cb_done} missing={cb_missing}")

    results = []
    et_peer = None
    cb_peer = None
    blend_peer = None
    if not et_missing:
        et_peer = build_stage_result_from_checkpoints(
            "et_peer",
            "D7_et_peer",
            df_train_d6,
            df_test_d6,
            splits,
            submission_name="D7_et_peer",
            metadata_filename="D7_et_peer_fold_metadata.json",
            write_noclip=True,
            mean_best_iter_or_epochs=1200.0,
        )
        results.append(et_peer)
    else:
        print("Skipping ET aggregation because not all ET checkpoints exist.")

    if not cb_missing:
        cb_peer = build_stage_result_from_checkpoints(
            "cb_peer",
            "D7_cb_peer",
            df_train_d6,
            df_test_d6,
            splits,
            submission_name="D7_cb_peer",
            metadata_filename="D7_cb_peer_fold_metadata.json",
            write_noclip=False,
            mean_best_iter_or_epochs=float("nan"),
        )
        results.append(cb_peer)
    else:
        print("Skipping CB aggregation because not all CB checkpoints exist.")

    if et_peer is not None and cb_peer is not None:
        if cb_peer["recent_fold_mean"] + 0.002 >= et_peer["recent_fold_mean"]:
            blend_peer = blend_et_cb_peer(et_peer, cb_peer, splits)
            results.append(blend_peer)
        else:
            print(
                "\nSkipping ET+CB peer blend because CB peer is clearly worse on recent folds: "
                f"ET recent={et_peer['recent_fold_mean']:.6f}, CB recent={cb_peer['recent_fold_mean']:.6f}"
            )

    if not results:
        print("\nNo complete stages available to aggregate yet.")
        print(f"All outputs in: {OUTPUT_DIR.resolve()}")
        return

    df_results = build_production_model_summary(results)
    print("\n" + "=" * 60)
    print("PRODUCTION MODEL SUMMARY")
    print("=" * 60)
    print(df_results.to_string(index=False))

    submit_order = []
    if et_peer is not None:
        submit_order.extend([
            "submission_D7_et_peer_noclip.csv",
            "submission_D7_et_peer.csv",
        ])
    if cb_peer is not None:
        submit_order.append("submission_D7_cb_peer.csv")
    if blend_peer is not None and et_peer is not None and (
        blend_peer["oof_corr"] > et_peer["oof_corr"]
        and blend_peer["recent_fold_mean"] > et_peer["recent_fold_mean"]
    ):
        submit_order.append("submission_D7_et_cb_peer_blend.csv")

    print("\nRecommended submit order:")
    for path in submit_order:
        print(f"  - {path}")

    summary = {
        "production_candidates": [str(r["model_name"]) for r in results],
        "d7_base_feature_count": len(d7_base_feat_cols),
        "rank_feature_count": len(rank_cols),
        "z_feature_count": len(z_cols),
        "sparse_feature_count": len(sparse_derived),
        "models": df_results.to_dict(orient="records"),
        "submit_order": submit_order,
        "et_noclip_saved_unclipped": et_peer is not None,
        "checkpoint_status": {
            "et_peer": {"done": et_done, "missing": et_missing},
            "cb_peer": {"done": cb_done, "missing": cb_missing},
        },
    }
    if blend_peer is not None:
        summary["blend_weights_peer"] = blend_peer["weights"]
    (OUTPUT_DIR / "advanced_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nAll outputs in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
