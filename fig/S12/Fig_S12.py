import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os

# prepare_dataset 和 S11 脚本里那个完全一样（含 np.random.seed(42)）
# ...（函数定义照抄）
# ==================== 你的 prepare_dataset ====================
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
model_path = os.path.join(base_path, "feature_rate_xgb_model", "xgboost_model_adsorption_amount.joblib")

X, y, validation_set = prepare_dataset(file_path, target_variable='Adsorption amount')
model = joblib.load(model_path)

# 全特征算 SHAP（和 S11 同一个模型、同一份数据）
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ===== 只取 SA 特征的 8 列 =====
sa_features = ['E', 'S', 'A', 'B', 'V', 'Kow', 'pKa1', 'MW']
shap_sa = shap_values[:, [X.columns.get_loc(c) for c in sa_features]]
X_sa = X[sa_features]

shap.summary_plot(shap_sa, X_sa, max_display=8, show=False, plot_size=(7, 5))
plt.savefig('Fig_S12_SHAP_SA_features.png', dpi=300, bbox_inches='tight')
plt.show()