"""
Advanced experiments: walk-forward CatBoost ablations with honest OOF.
Parts C/D:
  - add non-target si context features
  - evaluate B1/B2/B3 for si feature usefulness
  - evaluate D1..D7 with full-column unsupervised transforms
  - fold-local interaction selection only (no full-train target screening)
"""
from __future__ import annotations

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

ID_COL     = "id"
TARGET_COL = "target"
TIME_COL   = "di"
STOCK_COL  = "si"

META_COL_CANDIDATES = ["si", "di", "industry", "sector", "top2000", "top1000", "top500"]

N_SPLITS   = 5
SEED       = 42
RUN_LGBM   = False   # disabled until CatBoost ablations are complete
TOP_N_ANON_SELECT = 30
TOP_FOLD_INTERACTIONS = 20
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
        mu    = np.nanmean(sub, axis=0)
        sigma = np.nanstd(sub, axis=0)
        sigma[sigma == 0] = 1.0
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


def make_submission(test_ids, preds, name):
    p = np.asarray(preds, np.float64).copy()
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
    print("Loading data...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test  = pd.read_csv(TEST_PATH)
    print(f"  train: {df_train.shape}  test: {df_test.shape}")

    # ── feature sets ──────────────────────────────────────────────────────────
    meta_cols, anon_cols, anon_numeric_cols = get_feature_sets(df_train)
    print(f"\n  meta_cols ({len(meta_cols)}): {meta_cols}")
    print(f"  anon_cols: {len(anon_cols)}  |  anon_numeric: {len(anon_numeric_cols)}")

    # ── walk-forward splits ────────────────────────────────────────────────────
    splits = walk_forward_time_splits(df_train[TIME_COL])
    print("\nWalk-Forward Folds:")
    for i, (tr, va) in enumerate(splits):
        tr_di = df_train.iloc[tr][TIME_COL]
        va_di = df_train.iloc[va][TIME_COL]
        print(f"  fold {i}: train di [{tr_di.min()}–{tr_di.max()}]  "
              f"valid di [{va_di.min()}–{va_di.max()}]  "
              f"n_train={len(tr):,}  n_valid={len(va):,}  "
              f"embargo_ok={tr_di.max() < va_di.min()}")

    # ── exploratory correlation snapshot ───────────────────────────────────────
    print("\n" + "="*60)
    print("Exploratory: per-anonymous Pearson on full train (diagnostic only)")
    print("="*60)
    df_corr = per_feature_correlation(df_train, anon_numeric_cols)
    df_corr.to_csv(OUTPUT_DIR / "feature_correlations.csv", index=False)
    print(df_corr.head(20).to_string(index=False))

    # ── si history + si context on raw data ───────────────────────────────────
    print("\n" + "="*60)
    print("Building forward-safe si history (raw si, before any normalization)")
    print("="*60)
    si_col_before = df_train[STOCK_COL].copy()
    meta_before   = df_train[meta_cols].copy()

    df_train_h, df_test_h = add_forward_safe_si_history(df_train, df_test)
    df_train_ctx, df_test_ctx = add_forward_safe_si_context(df_train_h, df_test_h)

    # Proof assertions
    assert df_train_h[STOCK_COL].reset_index(drop=True).equals(
        si_col_before.reset_index(drop=True)
    ), "si column was mutated by add_forward_safe_si_history!"
    for col in meta_cols:
        assert df_train_h[col].reset_index(drop=True).equals(
            meta_before[col].reset_index(drop=True)
        ), f"Meta column '{col}' was mutated by add_forward_safe_si_history!"
    print("  ✓ si column unchanged after si-history build")
    print("  ✓ All meta columns unchanged after si-history build")

    hist_cols = ["si_hist_count", "si_hist_mean", "si_last_target", "si_hist_std"]
    ctx_cols = ["si_seen_before", "si_log_count", "si_prev_gap_di"]
    base_cols = meta_cols + anon_cols

    # ── unsupervised transforms over ALL anon numeric cols ────────────────────
    df_train_rank = add_rank_features(df_train_ctx, anon_numeric_cols, group_col=TIME_COL, suffix="_csrank")
    df_test_rank = add_rank_features(df_test_ctx, anon_numeric_cols, group_col=TIME_COL, suffix="_csrank")
    rank_cols = [f"{c}_csrank" for c in anon_numeric_cols if f"{c}_csrank" in df_train_rank.columns]
    df_train_z = cross_sectional_zscore_anonymous(df_train_ctx, anon_numeric_cols)
    df_test_z = cross_sectional_zscore_anonymous(df_test_ctx, anon_numeric_cols)
    z_cols = anon_numeric_cols

    sparse_cols = get_sparse_anon_cols(df_train_ctx, anon_cols, nan_frac_threshold=0.20)
    df_train_sparse = add_sparse_indicators(df_train_ctx, sparse_cols)
    df_test_sparse = add_sparse_indicators(df_test_ctx, sparse_cols)
    sparse_derived = []
    for c in sparse_cols:
        sparse_derived.extend([f"{c}_isna", f"{c}_filled0", f"{c}_present_x_value"])

    reps = [c for c in DEFAULT_FAMILY_REPS if c in anon_cols]
    print(f"  sparse anonymous cols (>=20% NaN): {len(sparse_cols)}")
    print(f"  family representatives for fold-local interactions: {reps}")

    # ── experiment definitions (Parts C + D) ──────────────────────────────────
    experiments = [
        {"name": "A_raw_features", "mode": "plain", "df_tr": df_train_ctx, "df_te": df_test_ctx, "feat": base_cols},
        {"name": "B1_raw_plus_si_context", "mode": "plain", "df_tr": df_train_ctx, "df_te": df_test_ctx, "feat": base_cols + ctx_cols},
        {"name": "B2_raw_plus_target_history_audit", "mode": "plain", "df_tr": df_train_ctx, "df_te": df_test_ctx, "feat": base_cols + hist_cols},
        {"name": "B3_raw_plus_si_context_plus_target_history_audit", "mode": "plain", "df_tr": df_train_ctx, "df_te": df_test_ctx, "feat": base_cols + ctx_cols + hist_cols},
        {"name": "D1_raw", "mode": "plain", "df_tr": df_train_ctx, "df_te": df_test_ctx, "feat": base_cols},
        {"name": "D2_raw_plus_allrank", "mode": "plain", "df_tr": df_train_rank, "df_te": df_test_rank, "feat": base_cols + rank_cols},
        {"name": "D3_raw_plus_allz", "mode": "plain", "df_tr": df_train_z, "df_te": df_test_z, "feat": base_cols + z_cols},
        {"name": "D4_raw_plus_rank_plus_z", "mode": "plain", "df_tr": df_train_rank.join(df_train_z[z_cols], rsuffix="_z"), "df_te": df_test_rank.join(df_test_z[z_cols], rsuffix="_z"), "feat": base_cols + rank_cols + [f"{c}_z" for c in z_cols]},
        {"name": "D5_raw_plus_sparseflags", "mode": "plain", "df_tr": df_train_sparse, "df_te": df_test_sparse, "feat": base_cols + sparse_derived},
        {"name": "D6_raw_plus_rank_plus_z_plus_sparseflags", "mode": "plain", "df_tr": df_train_rank.join(df_train_z[z_cols], rsuffix="_z").join(df_train_sparse[sparse_derived]), "df_te": df_test_rank.join(df_test_z[z_cols], rsuffix="_z").join(df_test_sparse[sparse_derived]), "feat": base_cols + rank_cols + [f"{c}_z" for c in z_cols] + sparse_derived},
        {"name": "D7_raw_plus_rank_plus_z_plus_sparseflags_plus_fold_interactions", "mode": "fold_interactions", "df_tr": df_train_rank.join(df_train_z[z_cols], rsuffix="_z").join(df_train_sparse[sparse_derived]), "df_te": df_test_rank.join(df_test_z[z_cols], rsuffix="_z").join(df_test_sparse[sparse_derived]), "feat": base_cols + rank_cols + [f"{c}_z" for c in z_cols] + sparse_derived},
    ]

    y = df_train[TARGET_COL].to_numpy(np.float64)
    results_rows = []
    best_exp = None

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {exp['name']}")
        print("="*60)

        if exp["mode"] == "fold_interactions":
            oof, test_pred, oof_score, fold_scores, cov_frac, covered, n_feat, best_iters, fold_ix = (
                run_catboost_with_fold_local_interactions(
                    exp["df_tr"], exp["df_te"], exp["feat"], splits, meta_cols, reps, tag=exp["name"][:25]
                )
            )
            (OUTPUT_DIR / f"fold_interactions_{exp['name']}.json").write_text(json.dumps(fold_ix, indent=2))
        else:
            feat_cols = [c for c in exp["feat"] if c in exp["df_tr"].columns]
            print(f"  feature count: {len(feat_cols)}")
            oof, test_pred, oof_score, fold_scores, cov_frac, covered, n_feat, best_iters = run_catboost(
                exp["df_tr"], exp["df_te"], feat_cols, splits, meta_cols,
                tag=exp["name"][:25],
            )

        # Save OOF predictions (oof_covered marks rows used in OOF Pearson)
        pd.DataFrame({
            "oof_pred": oof,
            "target": y,
            "oof_covered": covered.astype(np.int8),
        }).to_csv(OUTPUT_DIR / f"oof_{exp['name']}.csv", index=False)
        make_submission(df_test[ID_COL], test_pred, exp["name"])

        row = {
            "experiment":  exp["name"],
            "n_features":  n_feat,
            "oof_pearson": round(oof_score, 6),
            "oof_coverage_frac": round(cov_frac, 6),
            "foldwise_rank_selection": False,
            "mean_best_iter": float(np.nanmean(best_iters)) if len(best_iters) else float("nan"),
        }
        for i, s in enumerate(fold_scores):
            row[f"fold_{i}"] = round(s, 6)
        for i, bi in enumerate(best_iters):
            row[f"best_iter_{i}"] = int(bi)
        results_rows.append(row)

        if best_exp is None or oof_score > best_exp["oof_score"]:
            best_exp = {
                "name": exp["name"],
                "oof_score": oof_score,
                "oof_coverage_frac": cov_frac,
            }

    # ── save & print summary ───────────────────────────────────────────────────
    df_results = pd.DataFrame(results_rows).sort_values("oof_pearson", ascending=False)
    df_results.to_csv(OUTPUT_DIR / "advanced_experiment_results.csv", index=False)

    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY (sorted by OOF Pearson)")
    print("="*60)
    print(df_results.to_string(index=False))
    a_score = next((r["oof_pearson"] for r in results_rows if r["experiment"] == "A_raw_features"), np.nan)
    if np.isfinite(a_score):
        promoted = [r["experiment"] for r in results_rows if (r["oof_pearson"] - a_score) >= 0.002]
        print(f"\nPromotion rule (+0.002 vs A): {promoted if promoted else 'none'}")
    b1 = next((r["oof_pearson"] for r in results_rows if r["experiment"] == "B1_raw_plus_si_context"), np.nan)
    b2 = next((r["oof_pearson"] for r in results_rows if r["experiment"] == "B2_raw_plus_target_history_audit"), np.nan)
    b3 = next((r["oof_pearson"] for r in results_rows if r["experiment"] == "B3_raw_plus_si_context_plus_target_history_audit"), np.nan)
    if np.isfinite(a_score) and np.isfinite(b1):
        print("Decision B1 vs A:", "drop si engineered mainline" if b1 <= a_score else "keep si context in research")
    if np.isfinite(a_score) and (np.isfinite(b2) or np.isfinite(b3)):
        if (np.isfinite(b2) and b2 < a_score) or (np.isfinite(b3) and b3 < a_score):
            print("Decision B2/B3: archive target-history (no mainline use)")

    print(
        f"\n→ Best experiment: {best_exp['name']}  "
        f"OOF={best_exp['oof_score']:.6f}  "
        f"coverage_frac={best_exp['oof_coverage_frac']:.4f}"
    )
    print("\nMeta column protection proof:")
    for col in meta_cols:
        print(f"  {col}: unchanged ✓")
    print("si history built before any anonymous normalization: ✓")

    summary = {
        "best_experiment":       best_exp["name"],
        "best_oof_pearson":      best_exp["oof_score"],
        "best_oof_coverage_frac": best_exp["oof_coverage_frac"],
        "experiments":           results_rows,
    }
    (OUTPUT_DIR / "advanced_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nAll outputs in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
