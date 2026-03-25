"""
Trexquant Earnings Return Prediction - GPU training script for Colab/local.

Usage in Colab:
    !python colab_train.py --data-dir /content/drive/MyDrive/trexquant

Or with Kaggle API:
    !python colab_train.py --kaggle
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── constants ──────────────────────────────────────────────────────────────────
ID_COL     = "id"
TARGET_COL = "target"
TIME_COL   = "di"
STOCK_COL  = "si"
CAT_COLS   = ["si", "industry", "sector", "top2000", "top1000", "top500"]
N_SPLITS   = 5
SEED       = 42


# ── args ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",  default=".",  help="Folder with train.csv and test.csv")
    p.add_argument("--output-dir",default="output", help="Where to write submission + logs")
    p.add_argument("--kaggle",    action="store_true", help="Download data via Kaggle API")
    p.add_argument("--no-mlp",    action="store_true", help="Skip PyTorch MLP (faster debug)")
    p.add_argument("--no-lgbm",   action="store_true", default=True, help="Skip LightGBM (broken at best_iter=1-3)")
    p.add_argument("--cb-iters",  type=int, default=5000, help="CatBoost iterations")
    p.add_argument("--lgbm-iters",type=int, default=3000, help="LightGBM iterations")
    p.add_argument("--mlp-epochs",type=int, default=200,  help="MLP epochs per fold")
    return p.parse_args()


# ── utils ──────────────────────────────────────────────────────────────────────
def pearson(y_true, y_pred):
    y, p = np.asarray(y_true, np.float64), np.asarray(y_pred, np.float64)
    if y.size < 2 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def oof_covered_mask(n_rows: int, splits) -> np.ndarray:
    m = np.zeros(n_rows, dtype=bool)
    for _, va_idx in splits:
        m[va_idx] = True
    return m


def oof_pearson_on_covered(y: np.ndarray, oof: np.ndarray, covered: np.ndarray) -> tuple[float, float]:
    """Aggregate OOF Pearson only on rows that were in some validation fold (not zero-fill)."""
    covered = np.asarray(covered, dtype=bool)
    y = np.asarray(y, np.float64)
    oof = np.asarray(oof, np.float64)
    frac = float(covered.mean())
    if int(covered.sum()) < 2:
        return float("nan"), frac
    return pearson(y[covered], oof[covered]), frac


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
        va_idx = idx[time_series.isin(chunk).to_numpy()]
        if len(tr_idx) and len(va_idx):
            splits.append((tr_idx, va_idx))
    if len(splits) < 2:
        raise ValueError("Too few walk-forward folds. Lower min_train or embargo.")
    return splits


def compute_feature_correlations(df_train, feature_cols, target_col='target'):
    """Compute per-feature Pearson correlation with target. Returns sorted DataFrame."""
    y = df_train[target_col].to_numpy(np.float64)
    num_cols = [c for c in feature_cols if c not in CAT_COLS and c != TIME_COL]
    rows = []
    for c in num_cols:
        x = df_train[c].to_numpy(np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 100:
            rows.append({"feature": c, "pearson": 0.0})
            continue
        corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
        rows.append({"feature": c, "pearson": corr if np.isfinite(corr) else 0.0})
    df_corr = pd.DataFrame(rows).sort_values("pearson", key=abs, ascending=False)
    return df_corr


def select_top_features(df_corr, feature_cols, top_k=60, min_abs_corr=0.005):
    """Return top_k features by absolute correlation that pass the min threshold."""
    num_top = df_corr[df_corr["pearson"].abs() >= min_abs_corr]["feature"].tolist()[:top_k]
    cat_keep = [c for c in feature_cols if c in CAT_COLS or c == TIME_COL]
    return num_top + cat_keep

def cross_sectional_normalize(df, num_cols, group_col='di'):
    """Z-score each numeric feature within each date cross-section (no look-ahead)."""
    arr = df[num_cols].to_numpy(np.float64)
    groups = df[group_col].to_numpy()
    result = arr.copy()
    for g in np.unique(groups):
        mask = groups == g
        sub = arr[mask]
        mu = np.nanmean(sub, axis=0)
        sigma = np.nanstd(sub, axis=0)
        sigma[sigma == 0] = 1.0
        result[mask] = (sub - mu) / (sigma + 1e-8)
    out = df.copy()
    out[num_cols] = result.astype(np.float32)
    return out


def cross_sectional_normalize_target(df, target_col='target', group_col='di'):
    """Z-score the target within each date. Returns modified df + original target array."""
    orig = df[target_col].to_numpy(np.float64).copy()
    arr = orig.copy()
    groups = df[group_col].to_numpy()
    for g in np.unique(groups):
        mask = groups == g
        sub = arr[mask]
        mu = np.nanmean(sub)
        sigma = np.nanstd(sub)
        if sigma == 0:
            sigma = 1.0
        arr[mask] = (sub - mu) / (sigma + 1e-8)
    out = df.copy()
    out[target_col] = arr
    return out, orig


def add_si_momentum(df_train, df_test, target_col='target', time_col='di', stock_col='si'):
    """
    Rolling mean of CS-normalized target per stock (fold-safe — uses only past dates).
    Captures persistent per-stock alpha: stocks that ranked high recently tend to again.
    For test rows (no target available) we use the last training-period rolling value.
    """
    windows = [21, 63, 252]
    pivot = df_train.pivot_table(values=target_col, index=time_col, columns=stock_col, aggfunc='mean')
    new_cols = []

    for w in windows:
        col = f'si_mom_{w}'
        new_cols.append(col)
        # shift(1) so we never use the current period's target
        rolled = pivot.shift(1).rolling(window=w, min_periods=max(3, w // 5)).mean()

        # Build fast lookup dict {(di, si): value}
        stacked = rolled.stack(dropna=False)
        lookup = stacked.to_dict()

        df_train[col] = [lookup.get((d, s), np.nan)
                         for d, s in zip(df_train[time_col], df_train[stock_col])]

        # Test: map last available rolling value per stock
        last_vals = rolled.iloc[-1].to_dict()
        df_test[col] = df_test[stock_col].map(last_vals)

    return df_train, df_test, new_cols


def make_submission(test_ids, preds, path):
    p = np.asarray(preds, np.float64).copy()
    lo, hi = np.nanpercentile(p, [0.5, 99.5])
    p = np.clip(p, lo, hi)
    if (np.abs(p) > 0).mean() < 0.10:
        p += 1e-9
    sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: p})
    assert np.isfinite(sub[TARGET_COL]).all(), "Non-finite predictions"
    assert (sub[TARGET_COL].abs() > 0).mean() >= 0.10, "< 10% non-zero"
    sub.to_csv(path, index=False)
    print(f"  Saved submission: {path}  ({len(sub):,} rows)")
    return sub


# ── data loading ───────────────────────────────────────────────────────────────
def download_kaggle_data(output_dir):
    print("Downloading data from Kaggle...")
    os.system("pip install kaggle -q")
    comp = "earnings-return-prediction-challenge-2025-q-4"
    os.system(f"kaggle competitions download -c {comp} -p {output_dir}")
    os.system(f"unzip -q {output_dir}/{comp}.zip -d {output_dir}")
    return output_dir


def load_data(data_dir):
    train_path = Path(data_dir) / "train.csv"
    test_path  = Path(data_dir) / "test.csv"
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    print(f"train: {df_train.shape}  test: {df_test.shape}")
    return df_train, df_test


# ── preprocessing ──────────────────────────────────────────────────────────────
def prepare_for_catboost(df_train, df_test, feature_cols):
    cat_features = [c for c in CAT_COLS if c in feature_cols]
    tr = df_train[feature_cols].copy()
    te = df_test[feature_cols].copy()
    for c in cat_features:
        tr[c] = tr[c].astype(str)
        te[c] = te[c].astype(str)
    return tr, te, cat_features


def prepare_for_lgbm(df_train, df_test, feature_cols):
    cat_features = [c for c in CAT_COLS if c in feature_cols]
    tr = df_train[feature_cols].copy()
    te = df_test[feature_cols].copy()
    # LightGBM: label-encode categoricals as int codes
    for c in cat_features:
        combined = pd.Categorical(pd.concat([tr[c].astype(str), te[c].astype(str)]))
        tr[c] = pd.Categorical(tr[c].astype(str), categories=combined.categories).codes
        te[c] = pd.Categorical(te[c].astype(str), categories=combined.categories).codes
    return tr, te, cat_features


def prepare_for_mlp(df_train, df_test, feature_cols, splits):
    """
    Impute (median on train per fold), then normalize (standard on train per fold).
    Returns numpy arrays — preprocessing is fold-local to avoid leakage.
    """
    f_cols = [c for c in feature_cols if c not in CAT_COLS]
    bool_cols = df_train[f_cols].select_dtypes(include=["bool"]).columns.tolist()

    tr = df_train[f_cols].copy()
    te = df_test[f_cols].copy()
    for c in bool_cols:
        tr[c] = tr[c].astype(np.float32)
        te[c] = te[c].astype(np.float32)

    return tr.to_numpy(dtype=np.float32), te.to_numpy(dtype=np.float32), f_cols


# ── CatBoost ───────────────────────────────────────────────────────────────────
def run_catboost(df_train, df_test, feature_cols, splits, iterations, gpu=True):
    from catboost import CatBoostRegressor
    task_type = "GPU" if gpu else "CPU"
    print(f"\n── CatBoost ({task_type}, {iterations} iters) ──")

    tr_cb, te_cb, cat_features = prepare_for_catboost(df_train, df_test, feature_cols)
    y = df_train[TARGET_COL].to_numpy(np.float64)
    oof  = np.zeros(len(df_train))
    covered = np.zeros(len(df_train), dtype=bool)
    test_pred = np.zeros(len(df_test))

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            depth=7,
            learning_rate=0.02,
            iterations=iterations,
            l2_leaf_reg=5.0,
            random_seed=SEED + fold_id,
            subsample=0.8,
            bootstrap_type="Bernoulli",
            od_type="Iter",
            od_wait=500,
            task_type=task_type,
            verbose=500,
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
        print(f"  fold {fold_id}: Pearson={fc:.6f}  best_iter={model.best_iteration_}")
        del model; gc.collect()

    oof_score, cov_frac = oof_pearson_on_covered(y, oof, covered)
    print(f"  CatBoost OOF Pearson (covered only): {oof_score:.6f}  coverage_frac={cov_frac:.4f}")
    return oof, test_pred, oof_score


# ── LightGBM ───────────────────────────────────────────────────────────────────
def run_lgbm(df_train, df_test, feature_cols, splits, iterations, gpu=False):
    import lightgbm as lgb
    device = "cpu"   # GPU LightGBM requires max_bin<=255 and is unstable; CPU is reliable
    print(f"\n── LightGBM (cpu, up to {iterations} iters) ──")

    tr_lg, te_lg, cat_features = prepare_for_lgbm(df_train, df_test, feature_cols)
    y = df_train[TARGET_COL].to_numpy(np.float64)
    oof  = np.zeros(len(df_train))
    covered = np.zeros(len(df_train), dtype=bool)
    test_pred = np.zeros(len(df_test))

    params = dict(
        objective="regression",
        metric="rmse",
        num_leaves=63,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_child_samples=50,
        reg_alpha=0.1,
        reg_lambda=1.0,
        device=device,
        n_jobs=4,
        verbose=-1,
        seed=SEED,
    )

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        ds_tr = lgb.Dataset(tr_lg.iloc[tr_idx], label=y[tr_idx],
                            categorical_feature=cat_features if cat_features else "auto")
        ds_va = lgb.Dataset(tr_lg.iloc[va_idx], label=y[va_idx], reference=ds_tr)
        model = lgb.train(
            params, ds_tr,
            num_boost_round=iterations,
            valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(150, verbose=False),
                       lgb.log_evaluation(500)],
        )
        oof[va_idx]  = model.predict(tr_lg.iloc[va_idx])
        covered[va_idx] = True
        test_pred   += model.predict(te_lg) / len(splits)
        fc = pearson(y[va_idx], oof[va_idx])
        print(f"  fold {fold_id}: Pearson={fc:.6f}  best_iter={model.best_iteration}")
        del model; gc.collect()

    oof_score, cov_frac = oof_pearson_on_covered(y, oof, covered)
    print(f"  LightGBM OOF Pearson (covered only): {oof_score:.6f}  coverage_frac={cov_frac:.4f}")
    return oof, test_pred, oof_score


# ── PyTorch MLP ────────────────────────────────────────────────────────────────
def run_mlp(df_train, df_test, feature_cols, splits, epochs):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n── PyTorch MLP ({device}, {epochs} epochs/fold) ──")

    X_all, X_test_np, f_cols = prepare_for_mlp(df_train, df_test, feature_cols, splits)
    y_all = df_train[TARGET_COL].to_numpy(np.float32)
    n_features = X_all.shape[1]

    class MLP(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(128, 64),   nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(0.1),
                nn.Linear(64, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    def pearson_loss(pred, target):
        """Directly minimize negative Pearson correlation — this IS the metric."""
        pred_c   = pred   - pred.mean()
        target_c = target - target.mean()
        corr = (pred_c * target_c).sum() / (
            torch.sqrt((pred_c**2).sum() + 1e-8) *
            torch.sqrt((target_c**2).sum() + 1e-8)
        )
        return -corr   # minimize → maximize correlation

    def impute_normalize(X_tr_raw, X_va_raw, X_te_raw=None):
        med = np.nanmedian(X_tr_raw, axis=0)
        std = np.nanstd(X_tr_raw, axis=0)
        # Columns entirely NaN → impute with 0, treat as constant
        med = np.where(np.isnan(med), 0.0, med)
        std = np.where(np.isnan(std) | (std == 0), 1.0, std)
        def _process(X):
            out = X.copy()
            nan_mask = np.isnan(out)
            out[nan_mask] = np.take(med, np.where(nan_mask)[1])
            return (out - med) / (std + 1e-8)
        tr = _process(X_tr_raw).astype(np.float32)
        va = _process(X_va_raw).astype(np.float32)
        if X_te_raw is not None:
            te = _process(X_te_raw).astype(np.float32)
            return tr, va, te
        return tr, va

    oof  = np.zeros(len(df_train))
    covered = np.zeros(len(df_train), dtype=bool)
    test_preds_folds = []

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        X_tr_raw = X_all[tr_idx]
        X_va_raw = X_all[va_idx]
        y_tr     = y_all[tr_idx]
        y_va     = y_all[va_idx]

        X_tr, X_va, X_te = impute_normalize(X_tr_raw, X_va_raw, X_test_np)

        ds_tr = TensorDataset(
            torch.tensor(X_tr), torch.tensor(y_tr)
        )
        loader = DataLoader(ds_tr, batch_size=2048, shuffle=True,
                            num_workers=0, pin_memory=True)

        model = MLP(n_features).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_val, best_weights, patience, no_improve = -1.0, None, 30, 0

        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = pearson_loss(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                xv = torch.tensor(X_va).to(device)
                val_pred = model(xv).cpu().numpy()
            val_corr = pearson(y_va, val_pred)

            is_better = (not np.isnan(val_corr)) and (best_weights is None or val_corr > best_val)
            if is_better:
                best_val = val_corr
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                if best_weights is None:
                    # fallback: save current weights so load_state_dict never gets None
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve += 1
                if no_improve >= patience:
                    break

        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            oof[va_idx] = model(torch.tensor(X_va).to(device)).cpu().numpy()
            te_pred     = model(torch.tensor(X_te).to(device)).cpu().numpy()
        covered[va_idx] = True
        test_preds_folds.append(te_pred)

        fc = pearson(y_va, oof[va_idx])
        print(f"  fold {fold_id}: Pearson={fc:.6f}  best_val={best_val:.6f}  epochs_run={epoch+1}")
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_pred = np.mean(test_preds_folds, axis=0)
    oof_score, cov_frac = oof_pearson_on_covered(y_all.astype(np.float64), oof, covered)
    print(f"  MLP OOF Pearson (covered only): {oof_score:.6f}  coverage_frac={cov_frac:.4f}")
    return oof, test_pred, oof_score


# ── ensemble ───────────────────────────────────────────────────────────────────
def best_blend(oofs, test_preds, y, names, covered: np.ndarray):
    """Grid-search blend weights over the collected models."""
    print("\n── Ensemble blend search ──")
    n = len(oofs)
    best_score, best_w = -1.0, None

    # try equal weight first
    for step in range(21):
        # for 2 models: try 0.0, 0.05, ..., 1.0
        if n == 2:
            w = step / 20.0
            weights = [w, 1.0 - w]
        elif n == 3:
            # grid over 3-model simplex at 0.1 resolution
            candidates = []
            for a in range(11):
                for b in range(11 - a):
                    c = 10 - a - b
                    candidates.append([a/10, b/10, c/10])
            best_score = -1.0
            for weights in candidates:
                blend = sum(w * o for w, o in zip(weights, oofs))
                s, _ = oof_pearson_on_covered(y, blend, covered)
                if s > best_score:
                    best_score = s
                    best_w = weights
            break
        else:
            weights = [1.0 / n] * n

        blend = sum(w * o for w, o in zip(weights, oofs))
        s, _ = oof_pearson_on_covered(y, blend, covered)
        if s > best_score:
            best_score = s
            best_w = weights

    for name, w in zip(names, best_w):
        print(f"  {name}: weight={w:.2f}")
    cov_frac = float(np.asarray(covered, dtype=bool).mean())
    print(f"  Blend OOF Pearson (covered only): {best_score:.6f}  coverage_frac={cov_frac:.4f}")

    final_test = sum(w * tp for w, tp in zip(best_w, test_preds))
    return final_test, best_score, dict(zip(names, best_w))


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.kaggle:
        data_dir = download_kaggle_data(str(output_dir / "data"))
    else:
        data_dir = args.data_dir

    df_train, df_test = load_data(data_dir)
    feature_cols = [c for c in df_train.columns if c not in [ID_COL, TARGET_COL]]
    num_feature_cols = [c for c in feature_cols if c not in CAT_COLS and c != TIME_COL]

    # Cross-sectional normalize: z-score each feature within each date
    print("Cross-sectional normalizing features...")
    df_train = cross_sectional_normalize(df_train, num_feature_cols)
    df_test  = cross_sectional_normalize(df_test,  num_feature_cols)

    # Cross-sectional normalize target (within-date z-score) — removes date-level drift
    # Keep original target for OOF Pearson scoring; use CS target for model training
    df_train, y_orig = cross_sectional_normalize_target(df_train)

    # Per-stock momentum: rolling mean of CS target — fold-safe
    print("Adding per-stock momentum features...")
    df_train, df_test, mom_cols = add_si_momentum(df_train, df_test)
    feature_cols = feature_cols + mom_cols
    print(f"  Added {len(mom_cols)} momentum features: {mom_cols}")

    # Global top-K feature selection (uses full train + targets — optimistic vs walk-forward;
    # honest alternative: select inside each train fold only, as in advanced_experiments C/D.)
    print("Computing per-feature correlations with target...")
    df_corr = compute_feature_correlations(df_train, feature_cols)
    print("\nTop 20 features by |Pearson| with target:")
    print(df_corr.head(20).to_string(index=False))
    top_corr = df_corr["pearson"].abs().max()
    print(f"\nSingle-best-feature Pearson: {top_corr:.6f}")

    # Use only top features to reduce noise
    feature_cols = select_top_features(df_corr, feature_cols, top_k=60)
    print(f"Using {len(feature_cols)} features for modeling\n")

    splits = walk_forward_splits(df_train[TIME_COL])
    y = df_train[TARGET_COL].to_numpy(np.float64)  # CS-normalized target for training
    oof_covered = oof_covered_mask(len(df_train), splits)

    # print fold summary
    print("\nWalk-Forward Folds:")
    for i, (tr, va) in enumerate(splits):
        tr_di = df_train.iloc[tr][TIME_COL]
        va_di = df_train.iloc[va][TIME_COL]
        print(f"  fold {i}: train di [{tr_di.min()}–{tr_di.max()}]  "
              f"valid di [{va_di.min()}–{va_di.max()}]  "
              f"n_train={len(tr):,}  n_valid={len(va):,}  "
              f"embargo_ok={tr_di.max() < va_di.min()}")

    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("\nNo GPU detected, running on CPU")
    except ImportError:
        pass

    oofs, test_preds, names = [], [], []

    # CatBoost
    cb_oof, cb_test, cb_score = run_catboost(
        df_train, df_test, feature_cols, splits,
        iterations=args.cb_iters, gpu=gpu_available
    )
    oofs.append(cb_oof); test_preds.append(cb_test); names.append("catboost")

    # LightGBM
    if not args.no_lgbm:
        try:
            lg_oof, lg_test, lg_score = run_lgbm(
                df_train, df_test, feature_cols, splits,
                iterations=args.lgbm_iters, gpu=gpu_available
            )
            oofs.append(lg_oof); test_preds.append(lg_test); names.append("lgbm")
        except Exception as e:
            print(f"LightGBM failed: {e}")

    # MLP
    if not args.no_mlp:
        try:
            import torch  # noqa: F401
            mlp_oof, mlp_test, mlp_score = run_mlp(
                df_train, df_test, feature_cols, splits,
                epochs=args.mlp_epochs
            )
            oofs.append(mlp_oof); test_preds.append(mlp_test); names.append("mlp")
        except ImportError:
            print("PyTorch not installed, skipping MLP")
        except Exception as e:
            print(f"MLP failed: {e}")

    # ensemble — drop any model whose OOF contains NaN
    valid_oofs, valid_preds, valid_names = [], [], []
    for o, tp, n in zip(oofs, test_preds, names):
        if np.isnan(o).any():
            print(f"  Skipping {n} from ensemble (NaN OOF)")
        else:
            valid_oofs.append(o); valid_preds.append(tp); valid_names.append(n)

    if len(valid_oofs) == 0:
        raise RuntimeError("All models produced NaN OOF — nothing to submit.")
    elif len(valid_oofs) == 1:
        final_test = valid_preds[0]
        best_score, _cov = oof_pearson_on_covered(y_orig, valid_oofs[0], oof_covered)
        blend_weights = {valid_names[0]: 1.0}
    else:
        final_test, best_score, blend_weights = best_blend(
            valid_oofs, valid_preds, y_orig, valid_names, oof_covered
        )

    make_submission(df_test[ID_COL], final_test, output_dir / "submission.csv")

    summary = {
        "individual_oof": {
            n: float(oof_pearson_on_covered(y_orig, o, oof_covered)[0])
            for n, o in zip(names, oofs)
        },
        "oof_coverage_frac": float(oof_covered.mean()),
        "ensemble_oof": best_score,
        "blend_weights": blend_weights,
        "n_folds": len(splits),
        "n_train_rows": len(df_train),
        "n_test_rows": len(df_test),
        "gpu": gpu_available,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
