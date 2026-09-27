# ================= S14~S18 一次性出图脚本 =================
import os
import numpy as np
import pandas as pd
import shap
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

# ---------------- 全局配置 ----------------
SEED = 42
TARGET = 'Adsorption capacity'          # S14/S15/S17/S18 论文用的都是 capacity
#路径区(start)
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 项目根目录
#路径区(stop)
FILE_PATH = os.path.join(base_path, "Dataset.csv")
MODEL_PATH = os.path.join(base_path, "feature_rate_xgb_model", "xgboost_model_adsorption_capacity.joblib")
OUT_DIR = os.path.join(base_path, "fig", "out")
os.makedirs(OUT_DIR, exist_ok=True)

BEST_PARAMS = dict(n_estimators=515, learning_rate=0.1466, max_depth=10,
                   min_child_weight=5, gamma=0.3639, subsample=0.8591,
                   colsample_bytree=0.9636, reg_alpha=0.0908, reg_lambda=0.0112,
                   random_state=SEED)

ENV_ADS = ['SBET', 'VTotal', 'Dp', 'H/C', 'N/C', 'O+N/C', 'O/C',
           'Initial concentration', 'Dosage', 'Temperature', 'Initial pH']
SA_FEATS = ['E', 'S', 'A', 'B', 'V', 'Kow', 'pKa1', 'MW']

# ---------------- 数据准备（和之前一致）----------------
def prepare_dataset(file_path, target_variable='Adsorption amount'):
    np.random.seed(42)
    df = pd.read_csv(file_path)
    df["Number"] = range(1, len(df) + 1)
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in dataset.")
    df['H/C'] = df['H'] / df['C']
    df['N/C'] = df['N'] / df['C']
    df['O/C'] = df['O'] / df['C']
    df['O+N/C'] = (df['O'] + df['N']) / df['C']
    random_numbers = np.random.choice(df["Number"].dropna().unique(), size=3)
    validation_set = df[df["Number"].isin(random_numbers)]
    drop_df = df[~df["Number"].isin(random_numbers)]
    feature_columns = ENV_ADS + SA_FEATS
    X = drop_df[feature_columns].copy()
    y = drop_df[target_variable]
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid], validation_set

# ---------------- 模型：有则加载，无则训练 ----------------
X, y, _ = prepare_dataset(FILE_PATH, TARGET)
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("已加载现有模型:", MODEL_PATH)
else:
    model = xgb.XGBRegressor(**BEST_PARAMS)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print("已训练并保存新模型:", MODEL_PATH)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
print("SHAP 计算完成, shape =", shap_values.shape)

# ---------------- 手写 PDP（避开 sklearn 版本冲突）----------------
def manual_pdp(model, X, feat, grid_size=50):
    grid = np.linspace(X[feat].min(), X[feat].max(), grid_size)
    means = np.empty(grid_size)
    for k, v in enumerate(grid):
        Xm = X.copy(); Xm[feat] = v
        means[k] = model.predict(Xm).mean()
    return grid, means

def manual_pdp_2d(model, X, f1, f2, n=20):
    g1 = np.linspace(X[f1].min(), X[f1].max(), n)
    g2 = np.linspace(X[f2].min(), X[f2].max(), n)
    Z = np.empty((n, n))
    for i, v1 in enumerate(g1):
        for j, v2 in enumerate(g2):
            Xm = X.copy(); Xm[f1] = v1; Xm[f2] = v2
            Z[i, j] = model.predict(Xm).mean()
    return g1, g2, Z

def savefig(name):
    plt.savefig(os.path.join(OUT_DIR, name), dpi=300, bbox_inches='tight')
    plt.close('all')
    print("已保存:", name)

# ================= S14: 环境+吸附剂特征 SHAP 依赖图 =================
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
for ax, feat in zip(axes.ravel(), ENV_ADS):
    shap.dependence_plot(feat, shap_values, X, ax=ax, show=False)
    ax.set_ylabel('SHAP value')
axes.ravel()[-1].set_visible(False)
plt.tight_layout()
savefig('Fig_S14_dependence_env_adsorbent.png')

# ================= S15: 环境+吸附剂特征 PDP =================
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
for ax, feat in zip(axes.ravel(), ENV_ADS):
    grid, means = manual_pdp(model, X, feat)
    ax.plot(grid, means, color='steelblue', lw=2)
    ax.set_xlabel(feat); ax.set_ylabel('Partial dependence')
axes.ravel()[-1].set_visible(False)
plt.tight_layout()
savefig('Fig_S15_PDP_env_adsorbent.png')

# ================= S16: pH 和温度的 SHAP 图 =================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, feat in zip(axes, ['Initial pH', 'Temperature']):
    shap.dependence_plot(feat, shap_values, X, ax=ax, show=False,
                         interaction_index=None)   # 颜色=该特征自身取值
    ax.set_ylabel('SHAP value')
plt.tight_layout()
savefig('Fig_S16_SHAP_pH_temperature.png')

# ================= S17: SA 特征 PDP =================
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, feat in zip(axes.ravel(), SA_FEATS):
    grid, means = manual_pdp(model, X, feat)
    ax.plot(grid, means, color='steelblue', lw=2)
    ax.set_xlabel(feat); ax.set_ylabel('Partial dependence')
plt.tight_layout()
savefig('Fig_S17_PDP_SA_features.png')

# ================= S18: 交互强度 Top10 + 前三对 3D PDP =================
rng = np.random.default_rng(SEED)
idx = rng.choice(len(X), size=min(200, len(X)), replace=False)
X_sample = X.iloc[idx]
inter = explainer.shap_interaction_values(X_sample)      # (n, 19, 19)
mean_abs = np.abs(inter).mean(axis=0)
names = list(X.columns)

pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
pairs.sort(key=lambda p: mean_abs[p], reverse=True)
top10 = pairs[:10]

fig, ax = plt.subplots(figsize=(9, 6))
labels = [f"{names[i]} × {names[j]}" for i, j in top10][::-1]
vals = [mean_abs[i, j] for i, j in top10][::-1]
ax.barh(labels, vals, color='steelblue')
ax.set_xlabel('Mean |SHAP interaction value|')
ax.set_title('Top 10 feature interactions')
plt.tight_layout()
savefig('Fig_S18_interaction_strength_top10.png')

fig = plt.figure(figsize=(18, 5))
for k, (i, j) in enumerate(pairs[:3]):
    g1, g2, Z = manual_pdp_2d(model, X, names[i], names[j], n=20)
    ax3 = fig.add_subplot(1, 3, k + 1, projection='3d')
    G1, G2 = np.meshgrid(g1, g2)
    ax3.plot_surface(G1, G2, Z.T, cmap='viridis')
    ax3.set_xlabel(names[i]); ax3.set_ylabel(names[j])
    ax3.set_zlabel('Prediction')
    ax3.set_title(f'{names[i]} × {names[j]}')
plt.tight_layout()
savefig('Fig_S18_3D_PDP_top3_pairs.png')

print("\n全部完成！输出目录:", OUT_DIR)