import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
import os

# prepare_dataset 函数照抄（含 np.random.seed(42)）
def prepare_dataset(file_path, target_variable='Adsorption amount'):
    np.random.seed(42)
    """
    Prepare dataset for modeling with the paper's Fig. S11 feature set
    (element ratios instead of raw C/O/H/N, pKa1 kept, no Pollutant one-hot)
    """
    df = pd.read_csv(file_path)
    df["Number"] = range(1, len(df) + 1)

    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in dataset. "
                         f"Available options are: {', '.join([col for col in df.columns if 'Adsorption' in col])}")

    # 元素比值特征
    df['H/C'] = df['H'] / df['C']
    df['N/C'] = df['N'] / df['C']
    df['O/C'] = df['O'] / df['C']
    df['O+N/C'] = (df['O'] + df['N']) / df['C']

    # 验证集抽取（和原来一样）
    random_numbers = np.random.choice(df["Number"].dropna().unique(), size=3)
    validation_set = df[df["Number"].isin(random_numbers)]
    drop_df = df[~df["Number"].isin(random_numbers)]

    # Fig. S11 的 19 个特征
    feature_columns = ['SBET', 'VTotal', 'Dp',
                       'H/C', 'N/C', 'O+N/C', 'O/C',
                       'Initial concentration', 'Dosage',
                       'Temperature', 'Initial pH',
                       'E', 'S', 'A', 'B', 'V', 'Kow', 'pKa1', 'MW']

    X = drop_df[feature_columns].copy()
    y = drop_df[target_variable]

    # 含缺失值的行无法进 XGBoost/SHAP，删掉（只删特征或目标缺失的行）
    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]

    return X, y, validation_set

#路径区(start)
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 项目根目录
#路径区(stop)
file_path = os.path.join(base_path, "Dataset.csv")

# ===== 1. 训练 capacity 模型（S13 对应的目标变量）=====
X, y, _ = prepare_dataset(file_path, target_variable='Adsorption capacity')

best_params = dict(n_estimators=515, learning_rate=0.1466, max_depth=10,
                   min_child_weight=5, gamma=0.3639, subsample=0.8591,
                   colsample_bytree=0.9636, reg_alpha=0.0908, reg_lambda=0.0112,
                   random_state=42)
model = xgb.XGBRegressor(**best_params)
model.fit(X, y)
joblib.dump(model, os.path.join(base_path, "feature_rate_xgb_model", "xgboost_model_adsorption_capacity.joblib"))

# ===== 2. SHAP =====
shap_values = shap.TreeExplainer(model).shap_values(X)

# ===== 3. 画 2×4 依赖图 =====
sa_features = ['E', 'S', 'A', 'B', 'V', 'Kow', 'pKa1', 'MW']

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, feat in zip(axes.ravel(), sa_features):
    plt.sca(ax)
    shap.dependence_plot(feat, shap_values, X, ax=ax, show=False)
    ax.set_ylabel('SHAP value')
plt.tight_layout()
plt.savefig('Fig_S13_dependence_SA_features.png', dpi=300, bbox_inches='tight')
plt.show()