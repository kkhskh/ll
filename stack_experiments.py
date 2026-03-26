from __future__ import annotations

import gc
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_experiments import (
    ID_COL,
    TARGET_COL,
    PRIMARY_TIME_COL,
    walk_forward_time_splits,
    oof_pearson_on_covered,
)


OUT = Path("artifacts/stack")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 2024]


def pearson(y_true, y_pred):
    y = np.asarray(y_true, np.float64)
    p = np.asarray(y_pred, np.float64)
    if y.size < 2 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def get_feature_sets(df: pd.DataFrame):
    meta = [c for c in ["si", "di", "industry", "sector", "top2000", "top1000", "top500"] if c in df.columns]
    anon = sorted([c for c in df.columns if c.startswith("f_")])
    anon_num = [c for c in anon if pd.api.types.is_numeric_dtype(df[c])]
    return meta, anon, anon_num


def add_rank_features(df, cols, group_col=PRIMARY_TIME_COL, suffix="_csrank"):
    out = df.copy()
    use_cols = [c for c in cols if c in out.columns]
    if not use_cols:
        return out
    grp = out.groupby(group_col, sort=False)
    rank_block = grp[use_cols].rank(pct=True, na_option="keep")
    rank_block.columns = [f"{c}{suffix}" for c in use_cols]
    return pd.concat([out, rank_block], axis=1)


def cross_sectional_zscore_anonymous(df, anon_numeric_cols, group_col=PRIMARY_TIME_COL):
    out = df.copy()
    f_cols = [c for c in anon_numeric_cols if c in df.columns]
    arr = out[f_cols].to_numpy(np.float64)
    groups = out[group_col].to_numpy()
    result = arr.copy()
    for g in np.unique(groups):
        m = groups == g
        sub = arr[m]
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(sub, axis=0)
            sd = np.nanstd(sub, axis=0)
        mu[~np.isfinite(mu)] = 0.0
        sd[~np.isfinite(sd)] = 1.0
        sd[sd == 0] = 1.0
        result[m] = (sub - mu) / (sd + 1e-8)
    out[f_cols] = result.astype(np.float32)
    return out


def get_sparse_anon_cols(df_train, anon_cols, nan_frac_threshold=0.20):
    miss = df_train[anon_cols].isna().mean()
    return [c for c in anon_cols if float(miss.get(c, 0.0)) >= nan_frac_threshold]


def add_sparse_indicators(df, sparse_cols):
    out = df.copy()
    blocks = []
    for c in sparse_cols:
        if c not in out.columns:
            continue
        filled = out[c].fillna(0.0).to_numpy(np.float32)
        isna = out[c].isna().to_numpy(np.int8)
        block = pd.DataFrame(
            {
                f"{c}_isna": isna,
                f"{c}_filled0": filled,
                f"{c}_present_x_value": filled * (1.0 - isna.astype(np.float32)),
            },
            index=out.index,
        )
        blocks.append(block)
    if not blocks:
        return out
    return pd.concat([out] + blocks, axis=1)


def build_best_d_block(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Reconstruct best stable D block as D6:
      raw + all ranks + all z-scores + sparse indicators.
    """
    meta, anon, anon_num = get_feature_sets(df_train)
    df_tr_rank = add_rank_features(df_train, anon_num, suffix="_csrank")
    df_te_rank = add_rank_features(df_test, anon_num, suffix="_csrank")
    df_tr_z = cross_sectional_zscore_anonymous(df_train, anon_num)
    df_te_z = cross_sectional_zscore_anonymous(df_test, anon_num)
    sparse = get_sparse_anon_cols(df_train, anon, 0.20)
    df_tr_sp = add_sparse_indicators(df_train, sparse)
    df_te_sp = add_sparse_indicators(df_test, sparse)

    z_cols = [f"{c}_z" for c in anon_num]
    rank_cols = [f"{c}_csrank" for c in anon_num]
    sparse_cols = []
    for c in sparse:
        sparse_cols.extend([f"{c}_isna", f"{c}_filled0", f"{c}_present_x_value"])

    tr = df_tr_rank.copy()
    te = df_te_rank.copy()
    z_tr = df_tr_z[anon_num].copy()
    z_te = df_te_z[anon_num].copy()
    z_tr.columns = [f"{c}_z" for c in anon_num]
    z_te.columns = [f"{c}_z" for c in anon_num]
    sp_tr = df_tr_sp[sparse_cols].copy() if sparse_cols else pd.DataFrame(index=df_train.index)
    sp_te = df_te_sp[sparse_cols].copy() if sparse_cols else pd.DataFrame(index=df_test.index)
    tr = pd.concat([tr, z_tr, sp_tr], axis=1)
    te = pd.concat([te, z_te, sp_te], axis=1)
    feat = [c for c in (meta + anon + rank_cols + z_cols + sparse_cols) if c in tr.columns and c in te.columns]
    return tr, te, feat, meta


def fit_fold_imputer_scaler(X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame):
    med = X_tr.median(axis=0, numeric_only=True)
    Xtr = X_tr.fillna(med)
    Xva = X_va.fillna(med)
    Xte = X_te.fillna(med)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0).replace(0, 1.0)
    return (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd


def run_catboost_stack(df_tr, df_te, feat_cols, meta_cols, splits, y):
    from catboost import CatBoostRegressor

    depths = [6, 8]
    lrs = [0.02, 0.03]
    cat_cols = [c for c in meta_cols if c in feat_cols]
    tr_cb = df_tr[feat_cols].copy()
    te_cb = df_te[feat_cols].copy()
    for c in cat_cols:
        tr_cb[c] = tr_cb[c].astype(str)
        te_cb[c] = te_cb[c].astype(str)

    rows = []
    all_models = []
    for depth, lr, seed in itertools.product(depths, lrs, SEEDS):
        oof = np.zeros(len(df_tr), dtype=np.float64)
        covered = np.zeros(len(df_tr), dtype=bool)
        test = np.zeros(len(df_te), dtype=np.float64)
        fold_scores = []
        best_iters = []
        for fold_id, (tr_idx, va_idx) in enumerate(splits):
            model = CatBoostRegressor(
                loss_function="RMSE",
                depth=depth,
                learning_rate=lr,
                iterations=3000,
                l2_leaf_reg=3.0,
                random_seed=seed + fold_id,
                subsample=0.8,
                bootstrap_type="Bernoulli",
                od_type="Iter",
                od_wait=300,
                verbose=False,
            )
            model.fit(
                tr_cb.iloc[tr_idx], y[tr_idx],
                cat_features=cat_cols if cat_cols else None,
                eval_set=(tr_cb.iloc[va_idx], y[va_idx]),
                use_best_model=True,
            )
            p_va = model.predict(tr_cb.iloc[va_idx])
            oof[va_idx] = p_va
            covered[va_idx] = True
            test += model.predict(te_cb) / len(splits)
            fold_scores.append(pearson(y[va_idx], p_va))
            best_iters.append(int(model.best_iteration_) if model.best_iteration_ is not None else -1)
            del model
            gc.collect()
        oof_score, cov = oof_pearson_on_covered(y, oof, covered)
        name = f"catboost_d{depth}_lr{lr}_s{seed}"
        all_models.append((name, oof, covered, test, best_iters))
        rows.append({"model_name": name, "oof_pearson": oof_score, "covered_frac": cov, "mean_best_iter_or_epochs": float(np.mean(best_iters))})
        pd.DataFrame({"oof_pred": oof, "covered": covered.astype(np.int8)}).to_csv(OUT / f"oof_{name}.csv", index=False)
        pd.DataFrame({"id": df_te[ID_COL], "pred": test}).to_csv(OUT / f"test_{name}.csv", index=False)
    return rows, all_models


def run_tree_stack(df_tr, df_te, num_cols, splits, y):
    rows = []
    all_models = []
    try:
        import xgboost as xgb
        use_xgb = True
    except Exception:
        use_xgb = False
        from sklearn.ensemble import HistGradientBoostingRegressor

    for seed in SEEDS:
        oof = np.zeros(len(df_tr), dtype=np.float64)
        covered = np.zeros(len(df_tr), dtype=bool)
        test = np.zeros(len(df_te), dtype=np.float64)
        fold_steps = []
        for fold_id, (tr_idx, va_idx) in enumerate(splits):
            Xtr, Xva, Xte = fit_fold_imputer_scaler(df_tr[num_cols].iloc[tr_idx], df_tr[num_cols].iloc[va_idx], df_te[num_cols])
            if use_xgb:
                dtr = xgb.DMatrix(Xtr, label=y[tr_idx])
                dva = xgb.DMatrix(Xva, label=y[va_idx])
                dte = xgb.DMatrix(Xte)

                def feval(preds, dmat):
                    yy = dmat.get_label()
                    return "pearson", pearson(yy, preds)

                bst = xgb.train(
                    {
                        "objective": "reg:squarederror",
                        "eta": 0.03,
                        "max_depth": 6,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "seed": seed + fold_id,
                    },
                    dtr,
                    num_boost_round=3000,
                    evals=[(dva, "valid")],
                    maximize=True,
                    early_stopping_rounds=300,
                    custom_metric=feval,
                    verbose_eval=False,
                )
                p_va = bst.predict(dva)
                p_te = bst.predict(dte)
                fold_steps.append(int(bst.best_iteration))
            else:
                bst = HistGradientBoostingRegressor(
                    learning_rate=0.03, max_depth=8, max_iter=600, random_state=seed + fold_id
                )
                bst.fit(Xtr, y[tr_idx])
                p_va = bst.predict(Xva)
                p_te = bst.predict(Xte)
                fold_steps.append(600)
            oof[va_idx] = p_va
            covered[va_idx] = True
            test += p_te / len(splits)
        name = ("xgboost" if use_xgb else "histgb") + f"_s{seed}"
        oof_score, cov = oof_pearson_on_covered(y, oof, covered)
        rows.append({"model_name": name, "oof_pearson": oof_score, "covered_frac": cov, "mean_best_iter_or_epochs": float(np.mean(fold_steps))})
        all_models.append((name, oof, covered, test, fold_steps))
        pd.DataFrame({"oof_pred": oof, "covered": covered.astype(np.int8)}).to_csv(OUT / f"oof_{name}.csv", index=False)
        pd.DataFrame({"id": df_te[ID_COL], "pred": test}).to_csv(OUT / f"test_{name}.csv", index=False)
    return rows, all_models


def run_mlp_stack(df_tr, df_te, num_cols, splits, y):
    rows = []
    all_models = []
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except Exception:
        return rows, all_models

    class Net(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.20),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.20),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        oof = np.zeros(len(df_tr), dtype=np.float64)
        covered = np.zeros(len(df_tr), dtype=bool)
        test = np.zeros(len(df_te), dtype=np.float64)
        epoch_hist = []
        for fold_id, (tr_idx, va_idx) in enumerate(splits):
            Xtr, Xva, Xte = fit_fold_imputer_scaler(df_tr[num_cols].iloc[tr_idx], df_tr[num_cols].iloc[va_idx], df_te[num_cols])
            Xtr_t = torch.tensor(Xtr.to_numpy(np.float32), device=device)
            ytr_t = torch.tensor(y[tr_idx].astype(np.float32), device=device)
            Xva_t = torch.tensor(Xva.to_numpy(np.float32), device=device)
            yva = y[va_idx]
            Xte_t = torch.tensor(Xte.to_numpy(np.float32), device=device)
            net = Net(Xtr_t.shape[1]).to(device)
            opt = optim.Adam(net.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()
            best_state = None
            best_r = -1e9
            best_ep = 0
            wait = 0
            for ep in range(1, 201):
                net.train()
                opt.zero_grad()
                pred = net(Xtr_t)
                loss = loss_fn(pred, ytr_t)
                loss.backward()
                opt.step()
                net.eval()
                with torch.no_grad():
                    pva = net(Xva_t).detach().cpu().numpy().astype(np.float64)
                r = pearson(yva, pva)
                if np.isfinite(r) and r > best_r:
                    best_r = r
                    best_ep = ep
                    wait = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else:
                    wait += 1
                if wait >= 20:
                    break
            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            with torch.no_grad():
                pva = net(Xva_t).detach().cpu().numpy().astype(np.float64)
                pte = net(Xte_t).detach().cpu().numpy().astype(np.float64)
            oof[va_idx] = pva
            covered[va_idx] = True
            test += pte / len(splits)
            epoch_hist.append(best_ep)
            del net
            gc.collect()
        name = f"mlp_s{seed}"
        oof_score, cov = oof_pearson_on_covered(y, oof, covered)
        rows.append({"model_name": name, "oof_pearson": oof_score, "covered_frac": cov, "mean_best_iter_or_epochs": float(np.mean(epoch_hist))})
        all_models.append((name, oof, covered, test, epoch_hist))
        pd.DataFrame({"oof_pred": oof, "covered": covered.astype(np.int8)}).to_csv(OUT / f"oof_{name}.csv", index=False)
        pd.DataFrame({"id": df_te[ID_COL], "pred": test}).to_csv(OUT / f"test_{name}.csv", index=False)
    return rows, all_models


def zscore_with_cover(oof, covered):
    m = np.nanmean(oof[covered])
    s = np.nanstd(oof[covered])
    if (not np.isfinite(s)) or s == 0:
        s = 1.0
    return (oof - m) / s, m, s


def blend_search(models, y):
    usable = []
    for name, oof, covered, test, _ in models:
        oof_score, _ = oof_pearson_on_covered(y, oof, covered)
        if np.isfinite(oof_score):
            usable.append((name, oof, covered, test, oof_score))
    if not usable:
        raise RuntimeError("No usable models for blending.")
    covered = usable[0][2]
    z_oof = {}
    z_te = {}
    for name, oof, _, test, _ in usable:
        zo, m, s = zscore_with_cover(oof, covered)
        z_oof[name] = zo
        z_te[name] = (test - m) / s

    names = [u[0] for u in usable]
    best = (-1e9, None, None, None)
    step = 0.05
    if len(names) == 1:
        w = {names[0]: 1.0}
        boof = z_oof[names[0]]
        bte = z_te[names[0]]
        return w, pearson(y[covered], boof[covered]), bte
    for grid in itertools.product(np.arange(0, 1 + 1e-9, step), repeat=len(names)):
        if abs(sum(grid) - 1.0) > 1e-9:
            continue
        boof = np.zeros_like(usable[0][1])
        bte = np.zeros_like(usable[0][3])
        for w, n in zip(grid, names):
            boof += w * z_oof[n]
            bte += w * z_te[n]
        score = pearson(y[covered], boof[covered])
        if np.isfinite(score) and score > best[0]:
            best = (score, {n: float(w) for n, w in zip(names, grid)}, boof, bte)
    return best[1], best[0], best[3]


def main():
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    tr_feat, te_feat, feat_cols, meta_cols = build_best_d_block(df_train, df_test)
    num_cols = [c for c in feat_cols if c not in meta_cols and pd.api.types.is_numeric_dtype(tr_feat[c])]
    splits = walk_forward_time_splits(df_train[PRIMARY_TIME_COL], n_splits=5, min_train_groups=252, embargo_groups=5)
    y = df_train[TARGET_COL].to_numpy(np.float64)

    rows = []
    models = []
    r1, m1 = run_catboost_stack(tr_feat, te_feat, feat_cols, meta_cols, splits, y)
    rows.extend(r1); models.extend(m1)
    r2, m2 = run_tree_stack(tr_feat, te_feat, num_cols, splits, y)
    rows.extend(r2); models.extend(m2)
    r3, m3 = run_mlp_stack(tr_feat, te_feat, num_cols, splits, y)
    rows.extend(r3); models.extend(m3)

    table = pd.DataFrame(rows).sort_values("oof_pearson", ascending=False)
    table.to_csv(OUT / "stack_model_results.csv", index=False)

    weights, blend_oof, blend_test = blend_search(models, y)
    sub = pd.DataFrame({ID_COL: df_test[ID_COL], TARGET_COL: blend_test})
    if (sub[TARGET_COL].abs() > 0).mean() < 0.10:
        sub[TARGET_COL] += 1e-9
    sub.to_csv(OUT / "submission_stack_best.csv", index=False)
    (OUT / "stack_weights.json").write_text(json.dumps({"weights": weights, "blend_oof_pearson": blend_oof}, indent=2))

    print("\nSTACK MODEL TABLE")
    print(table.to_string(index=False))
    print("\nFinal blend weights:")
    print(json.dumps(weights, indent=2))
    print(f"Final blend OOF Pearson: {blend_oof:.6f}")
    print(f"Saved: {OUT / 'submission_stack_best.csv'}")


if __name__ == "__main__":
    main()

