# 复现论文补充材料图 S1-S9、S12-S18（S10 需要 pH/temperature 数据集，材料中没有；S11 已由 Gan_Boost_Xgb.py 复现）
# 数据：原始 Dataset.csv + WGAN 生成数据混合（与 Gan_Boost_Xgb.py 相同的特征工程）
# SHAP 计算用 xgboost 原生 pred_contribs（规避 shap 0.49 无法解析 xgboost 3.x 模型的问题）

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import shap
from sklearn.manifold import TSNE

import Gan_Boost_Xgb as G  # 复用数据准备与特征工程

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "figs_reproduced")
os.makedirs(OUT, exist_ok=True)

np.random.seed(42)

# 论文 Table S3 的树模型参数；XGB 用我们贝叶斯搜索(50 iter)得到的最优参数（与 S11 蜂群图一致）
XGB_PARAMS = {
    'n_estimators': 735, 'learning_rate': 0.03245201902430336, 'max_depth': 7,
    'min_child_weight': 5, 'gamma': 0.16048926385112766, 'subsample': 0.6521733828918832,
    'colsample_bytree': 0.6598870173617728, 'reg_alpha': 0.19079808003058807,
    'reg_lambda': 0.7326605290131484, 'random_state': 42,
}
RF_PARAMS = dict(n_estimators=240, max_depth=10, min_samples_split=5,
                 min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1)
GBR_PARAMS = dict(n_estimators=579, learning_rate=0.2366, max_depth=3,
                  min_samples_split=5, min_samples_leaf=1, subsample=0.7404, random_state=42)

TARGETS = ['Adsorption amount', 'Adsorption capacity']
ENV_ABSORBENT = ['Initial concentration', 'Dosage', 'Initial pH', 'Temperature',
                 'N/C', 'H/C', 'O+N/C', 'O/C', 'SBET', 'VTotal', 'Dp']
SA_FEATURES = ['E', 'S', 'A', 'B', 'V', 'Kow', 'pKa1', 'MW']


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------- 数据与模型 ----------------
def load_xy(target):
    """混合数据（原始+WGAN 生成），原始量纲。np.random.seed 保证与主脚本同一批 3 行验证集"""
    np.random.seed(42)
    gen = None
    for p in G.Generated_paths:
        if os.path.exists(p) and target in pd.read_csv(p, nrows=0).columns:
            gen = p
            break
    X, y, _ = G.prepare_dataset(G.Dataset_path, gen, target_variable=target)
    return X, y


def make_model(name):
    if name == 'XGBoost':
        return xgb.XGBRegressor(**XGB_PARAMS)
    if name == 'RandomForest':
        return RandomForestRegressor(**RF_PARAMS)
    if name == 'GradientBoosting':
        return GradientBoostingRegressor(**GBR_PARAMS)
    if name == 'MLR':
        return LinearRegression()
    if name == 'SVM':
        return SVR(kernel='rbf', C=10, epsilon=0.1)
    if name == 'ANN':
        return MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=3000,
                            early_stopping=True, random_state=42)
    raise ValueError(name)


SIX_MODELS = ['MLR', 'SVM', 'ANN', 'RandomForest', 'GradientBoosting', 'XGBoost']
TREE_MODELS = ['GradientBoosting', 'RandomForest', 'XGBoost']  # 与论文 S9 行序一致


def get_shap(model, X_scaled_sample, feature_names):
    """xgboost 原生 TreeSHAP"""
    booster = model.get_booster()
    dm = xgb.DMatrix(X_scaled_sample, feature_names=feature_names)
    contribs = booster.predict(dm, pred_contribs=True)
    return contribs[:, :-1], float(contribs[0, -1])


# ---------------- Fig S1 学习曲线 ----------------
def fig_s1(X, y, n_points=16, n_folds=5):
    sizes = np.linspace(50, 800, n_points).astype(int)
    fig, axes = plt.subplots(3, 1, figsize=(7, 14))
    cfg = [('XGBoost', 'XGB', 'green'), ('RandomForest', 'RF', 'blue'),
           ('GradientBoosting', 'GBR', 'red')]
    for ax, (name, short, color) in zip(axes, cfg):
        train_curve, val_curve = [], []
        for k in sizes:
            params = {**RF_PARAMS, 'n_estimators': int(k)} if name == 'RandomForest' else \
                     {**GBR_PARAMS, 'n_estimators': int(k)} if name == 'GradientBoosting' else \
                     {**XGB_PARAMS, 'n_estimators': int(k)}
            tr_rmses, va_rmses = [], []
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            for tr_i, va_i in kf.split(X):
                sc = StandardScaler()
                Xtr = sc.fit_transform(X.iloc[tr_i]); Xva = sc.transform(X.iloc[va_i])
                m = xgb.XGBRegressor(**params) if name == 'XGBoost' else \
                    RandomForestRegressor(**params) if name == 'RandomForest' else \
                    GradientBoostingRegressor(**params)
                m.fit(Xtr, y.iloc[tr_i])
                tr_rmses.append(np.sqrt(mean_squared_error(y.iloc[tr_i], m.predict(Xtr))))
                va_rmses.append(np.sqrt(mean_squared_error(y.iloc[va_i], m.predict(Xva))))
            train_curve.append(tr_rmses); val_curve.append(va_rmses)
        train_curve = np.array(train_curve); val_curve = np.array(val_curve)
        mean_tr = train_curve.mean(1); std_tr = train_curve.std(1)
        mean_va = val_curve.mean(1); std_va = val_curve.std(1)
        ax.fill_between(sizes, mean_va - 1.96 * std_va, mean_va + 1.96 * std_va,
                        color=color, alpha=0.15)
        ax.plot(sizes, mean_tr, 'o-', color='darkred' if name == 'GradientBoosting' else 'dark'+color if name != 'XGBoost' else 'darkgreen',
                ms=4, label=f'{short} Training Error')
        ax.plot(sizes, mean_va, 'o-', color=color, ms=4, label=f'{short} Validation Error')
        ax.set_xlabel('Number of Iterations (Trees)'); ax.set_ylabel('RMSE')
        ax.set_title(f'{name} Learning Curve'); ax.legend(); ax.grid(alpha=0.3)
        log(f'S1 {name} done')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'FigS1_learning_curves.png'), dpi=200)
    plt.close(fig)


# ---------------- Fig S2 单参数敏感性 ----------------
def fig_s2(X, y, n_folds=3):
    sweeps = {
        'XGBoost': [
            ('n_estimators', [50, 150, 250, 350, 450, 500]),
            ('learning_rate', [0.01, 0.05, 0.1, 0.2, 0.3]),
            ('max_depth', [1, 3, 5, 7, 9, 11]),
            ('min_child_weight', [1, 2, 3, 4, 5, 6]),
            ('colsample_bytree', [0.5, 0.6, 0.8, 0.9, 1.0]),
            ('subsample', [0.5, 0.6, 0.8, 0.9, 1.0]),
        ],
        'RandomForest': [
            ('n_estimators', [50, 150, 250, 350, 450, 550]),
            ('min_samples_split', [2, 4, 6, 8, 10]),
            ('max_depth', [1, 3, 5, 7, 9, 11]),
            ('min_samples_leaf', [1, 2, 4, 6]),
        ],
        'GradientBoosting': [
            ('learning_rate', [0.01, 0.05, 0.1, 0.2, 0.3]),
            ('max_depth', [1, 3, 5, 7, 9, 11]),
            ('min_samples_leaf', [1, 2, 4, 6]),
            ('n_estimators', [50, 150, 250, 350, 450]),
            ('min_samples_split', [2, 4, 6, 8, 10]),
        ],
    }
    base = {'XGBoost': XGB_PARAMS, 'RandomForest': RF_PARAMS, 'GradientBoosting': GBR_PARAMS}
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for name, params_list in sweeps.items():
        n = len(params_list)
        ncol = 3
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 3.5 * nrow))
        axes = np.array(axes).ravel()
        for ax, (pname, values) in zip(axes, params_list):
            r2s = []
            for v in values:
                p = {**base[name], pname: v}
                scores = []
                for tr_i, va_i in kf.split(X):
                    sc = StandardScaler()
                    Xtr = sc.fit_transform(X.iloc[tr_i]); Xva = sc.transform(X.iloc[va_i])
                    m = make_model(name) if name == 'XGBoost' else \
                        RandomForestRegressor(**p) if name == 'RandomForest' else \
                        GradientBoostingRegressor(**p)
                    if name == 'XGBoost':
                        m.set_params(**{pname: v})
                    m.fit(Xtr, y.iloc[tr_i])
                    scores.append(r2_score(y.iloc[va_i], m.predict(Xva)))
                r2s.append(np.mean(scores))
            ax.plot(values, r2s, 'o-', ms=4)
            ax.set_xlabel(pname); ax.set_ylabel('R²')
            ax.set_title(f'{name} - Sensitivity Analysis for {pname}')
            ax.grid(alpha=0.3)
        for ax in axes[n:]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'FigS2_sensitivity_{name}.png'), dpi=200)
        plt.close(fig)
        log(f'S2 {name} done')


# ---------------- Fig S3 相关性热图 ----------------
def fig_s3():
    df = pd.read_csv(G.Dataset_path).copy()
    # 论文 S3 的热图同时包含原始 C/H/O/N 和四个比值，因此在副本上追加比值列
    df['H/C'] = df['H'] / df['C']
    df['N/C'] = df['N'] / df['C']
    df['O/C'] = df['O'] / df['C']
    df['O+N/C'] = (df['O'] + df['N']) / df['C']
    cols = ['SBET', 'VTotal', 'Dp', 'O+N/C', 'O/C', 'H/C', 'N/C',
            'C', 'H', 'O', 'N', 'Dosage', 'Temperature', 'Initial pH',
            'Initial concentration', 'Adsorption amount', 'Adsorption capacity']
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right'); ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(v) > 0.6 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Pearson Correlation Coefficient Heatmap')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'FigS3_correlation_heatmap.png'), dpi=200)
    plt.close(fig)
    log('S3 done')


# ---------------- Fig S4/S5 六模型 真实vs预测 ----------------
def fig_s4_s5(target, tag):
    X, y = load_xy(target)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr); Xte = sc.transform(X_te)
    order = ['MLR', 'SVM', 'ANN', 'RandomForest', 'GradientBoosting', 'XGBoost']
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    for ax, name in zip(axes.ravel(), order):
        m = make_model(name)
        m.fit(Xtr, y_tr)
        ptr = m.predict(Xtr); pte = m.predict(Xte)
        ax.scatter(y_tr, ptr, s=18, c='#3b3b58', marker='o', alpha=0.6, label='Training Set')
        ax.scatter(y_te, pte, s=22, c='#e8a0a8', marker='s', alpha=0.7, label='Test Set')
        lim = max(y.max(), ptr.max(), pte.max()) * 1.05
        ax.plot([0, lim], [0, lim], 'k-', lw=0.8)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel('Experimental Values (mg/g)'); ax.set_ylabel('Predicted Values (mg/g)')
        ax.set_title(f'Actual vs Predicted Values - {name} Regression')
        ax.legend(fontsize=8)
        # 右侧：残差散点 + 残差直方图
        div = make_axes_locatable(ax)
        ax_r = div.append_axes('right', size='32%', pad=0.08)
        ax_r.scatter(ptr, y_tr - ptr, s=8, c='#3b3b58', alpha=0.6)
        ax_r.scatter(pte, y_te - pte, s=10, c='#e8a0a8', marker='s', alpha=0.7)
        ax_r.axhline(0, color='k', lw=0.8, ls='--')
        ax_r.set_xlabel('Prediction'); ax_r.set_ylabel('Residual'); ax_r.tick_params(labelsize=7)
        ax_h = div.append_axes('right', size='32%', pad=0.08)
        res_all = np.concatenate([y_tr - ptr, y_te - pte])
        ax_h.hist(res_all, bins=40, color='#8a7fb8', alpha=0.8)
        ax_h.set_xlabel('Residual'); ax_h.set_ylabel('Frequency'); ax_h.tick_params(labelsize=7)
        r2 = r2_score(y_te, pte)
        ax.set_title(f'Actual vs Predicted Values - {name} ($R^2$={r2:.3f})', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f'FigS{tag}_actual_vs_predicted_{tag}.png'), dpi=150)
    plt.close(fig)
    log(f'S{tag} done ({target})')


from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402


# ---------------- Fig S6 模型性能置信区间 ----------------
def fig_s6(n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    for ax, target in zip(axes, TARGETS):
        X, y = load_xy(target)
        rows = []
        for name in SIX_MODELS:
            r2s, rmses = [], []
            for tr_i, te_i in kf.split(X):
                sc = StandardScaler()
                Xtr = sc.fit_transform(X.iloc[tr_i]); Xte = sc.transform(X.iloc[te_i])
                m = make_model(name)
                m.fit(Xtr, y.iloc[tr_i])
                p = m.predict(Xte)
                r2s.append(r2_score(y.iloc[te_i], p))
                rmses.append(np.sqrt(mean_squared_error(y.iloc[te_i], p)))
            rows.append({'model': name, 'r2': np.mean(r2s), 'r2_ci': 1.96 * np.std(r2s),
                         'rmse': np.mean(rmses), 'rmse_ci': 1.96 * np.std(rmses)})
            log(f'S6 {target} {name}: R2={rows[-1]["r2"]:.3f} RMSE={rows[-1]["rmse"]:.1f}')
        d = pd.DataFrame(rows)
        x = np.arange(len(d))
        ax.errorbar(x, d['r2'], yerr=d['r2_ci'], fmt='s', color='tab:blue', capsize=4,
                    label='Test R²')
        ax.set_ylabel('Test R²', color='tab:blue')
        ax.set_ylim(0.3, 1.05)
        ax2 = ax.twinx()
        ax2.errorbar(x, d['rmse'], yerr=d['rmse_ci'], fmt='^', color='tab:orange', capsize=4,
                     label='Test RMSE')
        ax2.set_ylabel('Test RMSE', color='tab:orange')
        ax.set_xticks(x); ax.set_xticklabels([m if m != 'RandomForest' else 'RF' for m in d['model']])
        ax2.set_ylim(0, 120)
        ax.set_xlabel('Model')
        ax.set_title(f'{target} model performance with 95% confidence intervals: R² and RMSE')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'FigS6_model_performance_ci.png'), dpi=200)
    plt.close(fig)
    log('S6 done')


# ---------------- Fig S7 离群点影响 Top10 ----------------
def fig_s7():
    """定义：对每个特征，比较该特征取值为离群点(|z|>2.5)的样本与非离群样本的平均 SHAP 值之差"""
    np.random.seed(42)
    # 用含 Pollutant 独热列的特征集训练一个 XGB（与论文 S7 图中出现 Pollutant_* 一致）
    real = pd.read_csv(G.Dataset_path)
    np.random.seed(42)
    real["Number"] = range(1, len(real) + 1)
    rnd = np.random.choice(real["Number"].unique(), size=3)
    real = real[~real["Number"].isin(rnd)]
    real = pd.get_dummies(real, columns=['Pollutant'])
    real_feat = G.add_element_ratios(real)
    excl = ['Number', 'Reference', 'Adsorption amount', 'Adsorption capacity', 'pKa3', 'pKa2']
    Xr = real_feat.drop(columns=excl)
    yr = real_feat['Adsorption capacity']
    gen = pd.read_csv([p for p in G.Generated_paths if os.path.exists(p)][-1])
    gen_feat = G.add_element_ratios(gen)
    Xg = gen_feat.drop(columns=[c for c in excl if c in gen_feat.columns])
    yg = gen_feat['Adsorption capacity']
    X = pd.concat([Xr, Xg], ignore_index=True)
    y = pd.concat([yr, yg], ignore_index=True)

    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(Xs, y)
    sv, _ = get_shap(m, Xs, X.columns.tolist())

    Z = (X - X.mean()) / X.std()
    impacts = {}
    for j, col in enumerate(X.columns):
        out_mask = Z[col].abs() > 2.5
        if out_mask.sum() < 5 or out_mask.sum() >= len(X):
            impacts[col] = 0.0
            continue
        # 归一化：离群样本与正常样本的平均 SHAP 之差除以该特征 SHAP 的标准差（无量纲，与论文同量级）
        impacts[col] = (sv[out_mask.values, j].mean() - sv[~out_mask.values, j].mean()) / (sv[:, j].std() + 1e-9)
    top = pd.Series(impacts).reindex(pd.Series(impacts).abs().sort_values(ascending=False).index)[:10]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index[::-1], top.values[::-1], color='#2e8b57')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('Outlier Impact'); ax.set_ylabel('Feature')
    ax.set_title('Top 10 Features by Absolute Outlier Impact')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'FigS7_outlier_impact.png'), dpi=200)
    plt.close(fig)
    log('S7 done')


# ---------------- Fig S8 t-SNE ----------------
def fig_s8():
    Xr, _ = load_xy('Adsorption capacity')
    gen_path = [p for p in G.Generated_paths if os.path.exists(p)][-1]
    gen = pd.read_csv(gen_path)
    gen_feat = G.add_element_ratios(gen)
    Xg = gen_feat[Xr.columns]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rs = scaler.fit_transform(Xr); gs = scaler.transform(Xg)
    comb = np.vstack([rs, gs])
    emb = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto',
               random_state=42).fit_transform(comb)
    n = len(rs)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb[:n, 0], emb[:n, 1], c='tab:blue', s=18, label='Original Data')
    ax.scatter(emb[n:, 0], emb[n:, 1], c='tab:red', s=18, label='Generated Data')
    ax.set_xlabel('t-SNE Component 1'); ax.set_ylabel('t-SNE Component 2')
    ax.set_title('t-SNE Visualization of Original and Generated Data of GAN')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'FigS8_tsne.png'), dpi=200)
    plt.close(fig)
    log('S8 done')


# ---------------- Fig S9 特征重要度 2x3 网格 ----------------
def fig_s9():
    fig, axes = plt.subplots(3, 2, figsize=(13, 16))
    for row, name in enumerate(TREE_MODELS):
        for col, target in enumerate(TARGETS):
            X, y = load_xy(target)
            sc = StandardScaler()
            Xs = sc.fit_transform(X)
            m = make_model(name)
            m.fit(Xs, y)
            imp = pd.Series(m.feature_importances_, index=X.columns).sort_values()
            ax = axes[row, col]
            ax.barh(imp.index, imp.values, color='#1a4a1a')
            ax.set_xlabel('Feature Importance')
            ax.set_ylabel('Feature')
            t = 'amount' if target == 'Adsorption amount' else 'capacity'
            ax.set_title(f'Feature Importance-{name}-{t}')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'FigS9_feature_importance_grid.png'), dpi=150)
    plt.close(fig)
    log('S9 done')


# ---------------- Fig S12 SA 特征蜂群图 ----------------
def fig_s12(shap_values, X_sample_df):
    cols = [c for c in SA_FEATURES if c in X_sample_df.columns]
    idx = [X_sample_df.columns.get_loc(c) for c in cols]
    sub_sv = shap_values[:, idx]
    sub_df = X_sample_df[cols]
    plt.figure()
    shap.summary_plot(sub_sv, sub_df, show=False, sort=False, max_display=8)
    fig = plt.gcf(); fig.set_size_inches(9, 6.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'FigS12_shap_beeswarm_SA.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    log('S12 done')


# ---------------- Fig S13/S14 SHAP 依赖图 ----------------
def fig_s13_s14(shap_values, X_sample_df, feats, fname, ncols=4):
    n = len(feats)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.array(axes).ravel()
    for i, f in enumerate(feats):
        plt.sca(axes[i])
        shap.dependence_plot(f, shap_values, X_sample_df, ax=axes[i], show=False,
                             interaction_index=None)
        axes[i].set_ylabel(f'SHAP value for {f}', fontsize=8)
    for ax in axes[n:]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=200)
    plt.close(fig)
    log(f'{fname} done')


# ---------------- Fig S15/S16/S17 PDP ----------------
def pdp_grid(pipeline, X_raw, feats, fname, ncols=4, figw=4.2, figh=3.2):
    n = len(feats)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figw * ncols, figh * nrows))
    axes = np.array(axes).ravel()
    disp = PartialDependenceDisplay.from_estimator(
        pipeline, X_raw, features=[X_raw.columns.get_loc(f) for f in feats],
        kind='average', ax=axes[:n].tolist())
    for ax, f in zip(disp.axes_.ravel(), feats):
        if ax is None:
            continue
        ax.set_title(f'Partial Dependence Plot for {f}', fontsize=9)
    for ax in axes[n:]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=200)
    plt.close(fig)
    log(f'{fname} done')


# ---------------- Fig S18 H 统计 + 3D PDP ----------------
def _h_stat_quantile(model, Xs, feature_names, n_samples=1200, seed=0):
    """Friedman H 统计：两个特征在 1%~99% 分位区间内随机采样（避免 min-max 角点导致 H 饱和到 1）"""
    from itertools import combinations
    rng = np.random.default_rng(seed)
    n_features = Xs.shape[1]
    med = np.median(Xs, axis=0)
    q01 = np.quantile(Xs, 0.01, axis=0)
    q99 = np.quantile(Xs, 0.99, axis=0)
    H = {}
    for i, j in combinations(range(n_features), 2):
        Xm = np.tile(med, (n_samples, 1))
        Xm[:, i] = rng.uniform(q01[i], q99[i], n_samples)
        Xm[:, j] = rng.uniform(q01[j], q99[j], n_samples)
        F_ij = model.predict(Xm)
        Xi = Xm.copy(); Xi[:, j] = med[j]
        Fj = Xm.copy(); Fj[:, i] = med[i]
        X0 = Xm.copy(); X0[:, i] = med[i]; X0[:, j] = med[j]
        inter = F_ij - model.predict(Xi) - model.predict(Fj) + model.predict(X0)
        den = np.sum((F_ij - model.predict(X0)) ** 2)
        num = np.sum(inter ** 2)
        H[(i, j)] = 0.0 if den <= 1e-10 or np.isnan(den) else float(min(1.0, max(0.0, num / den)))
    df = pd.DataFrame(
        [{'Feature1': feature_names[i], 'Feature2': feature_names[j], 'H_statistic': v}
         for (i, j), v in H.items()])
    return df.sort_values('H_statistic', ascending=False)


def fig_s18(pipeline, X_raw):
    # (a) Top5 H 统计
    sc_arr = StandardScaler().fit_transform(X_raw)
    m = pipeline.named_steps['xgb'] if 'xgb' in pipeline.named_steps else pipeline.steps[-1][1]
    H = _h_stat_quantile(m, sc_arr, X_raw.columns.tolist())
    top5 = H.head(5)
    fig = plt.figure(figsize=(16, 12))
    ax_bar = fig.add_subplot(2, 2, 1)
    labels = [f'{a} & {b}' for a, b in zip(top5['Feature1'], top5['Feature2'])]
    ax_bar.barh(labels[::-1], top5['H_statistic'][::-1], color='orange')
    ax_bar.set_xlabel('H_statistic')
    ax_bar.set_title('Top 5 Feature Interactions by H_statistic')

    # (b)(c)(d) 前三对特征的 3D PDP
    for k, (f1, f2) in enumerate(zip(top5['Feature1'][:3], top5['Feature2'][:3])):
        ax3d = fig.add_subplot(2, 2, k + 2, projection='3d')
        g1 = np.linspace(X_raw[f1].quantile(0.01), X_raw[f1].quantile(0.99), 25)
        g2 = np.linspace(X_raw[f2].quantile(0.01), X_raw[f2].quantile(0.99), 25)
        med = X_raw.median()
        grid = np.tile(med.values, (len(g1) * len(g2), 1))
        i1, i2 = X_raw.columns.get_loc(f1), X_raw.columns.get_loc(f2)
        G1, G2 = np.meshgrid(g1, g2)
        grid[:, i1] = G1.ravel(); grid[:, i2] = G2.ravel()
        pd_vals = pipeline.predict(pd.DataFrame(grid, columns=X_raw.columns)).reshape(G1.shape)
        ax3d.plot_surface(G1, G2, pd_vals, cmap='viridis', edgecolor='k', lw=0.3)
        ax3d.set_xlabel(f1, fontsize=8); ax3d.set_ylabel(f2, fontsize=8)
        ax3d.set_zlabel('Prediction', fontsize=8)
        ax3d.set_title(f'3D PDP for {f1} and {f2}', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'FigS18_interaction_3dpdp.png'), dpi=150)
    plt.close(fig)
    log('S18 done')


# ---------------- 主流程 ----------------
def main(fast=False):
    t0 = time.time()
    target = 'Adsorption capacity'

    log('训练 capacity 最终 XGB（与 S11 一致的贝叶斯参数）...')
    X, y = load_xy(target)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr); Xte = scaler.transform(X_te)
    final_xgb = xgb.XGBRegressor(**XGB_PARAMS)
    final_xgb.fit(Xtr, y_tr)
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([('scaler', StandardScaler()), ('xgb', xgb.XGBRegressor(**XGB_PARAMS))])
    pipe.fit(X_tr, y_tr)

    # SHAP 样本（训练集前 500 条，与 S11 相同）
    X_sample = Xtr[:500]
    X_sample_df = pd.DataFrame(X_sample, columns=X.columns)
    X_sample_raw_df = X_tr.iloc[:500].reset_index(drop=True)  # 依赖图的 x 轴用原始单位（与论文一致）
    sv, _ = get_shap(final_xgb, X_sample, X.columns.tolist())

    fig_s3()
    fig_s12(sv, X_sample_raw_df)
    fig_s13_s14(sv, X_sample_raw_df, SA_FEATURES, 'FigS13_shap_dependence_SA.png')
    fig_s13_s14(sv, X_sample_raw_df, ENV_ABSORBENT, 'FigS14_shap_dependence_env_ads.png')
    pdp_grid(pipe, X, ENV_ABSORBENT, 'FigS15_pdp_env_ads.png')
    pdp_grid(pipe, X, ['Initial pH', 'Temperature'], 'FigS16_pdp_ph_temp.png', ncols=2,
             figw=6, figh=4.5)
    pdp_grid(pipe, X, SA_FEATURES, 'FigS17_pdp_SA.png')
    fig_s18(pipe, X)
    if not fast:
        fig_s1(X, y)
        fig_s2(X, y)
    fig_s4_s5('Adsorption amount', '4')
    fig_s4_s5('Adsorption capacity', '5')
    fig_s6()
    fig_s7()
    fig_s8()
    fig_s9()
    log(f'全部完成，用时 {(time.time() - t0) / 60:.1f} 分钟，输出目录: {OUT}')


if __name__ == '__main__':
    import sys
    main(fast='--fast' in sys.argv)
