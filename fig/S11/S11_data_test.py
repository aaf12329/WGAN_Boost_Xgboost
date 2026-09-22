import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor
def prepare_dataset(file_path, target_variable='Adsorption amount'):
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
# ===== 用你训练时的超参数（从这里面抄）=====
best_params = dict(
    n_estimators=515,
    learning_rate=0.1466,
    max_depth=10,
    min_child_weight=5,
    gamma=0.3639,
    subsample=0.8591,
    colsample_bytree=0.9636,
    reg_alpha=0.0908,
    reg_lambda=0.0112,
)

X, y, _ = prepare_dataset(r"C:\Users\AAF12\Desktop\New_machine\Dataset.csv",
                          target_variable='Adsorption amount')

results = {}
for seed in [0, 1, 42, 100, 2024]:
    model = XGBRegressor(random_state=seed, **best_params)
    model.fit(X, y)
    sv = np.abs(shap.TreeExplainer(model).shap_values(X)).mean(axis=0)
    results[f'seed{seed}'] = pd.Series(sv, index=X.columns)

# 汇总表：每个特征在 5 个种子下的平均 |SHAP| + 名次
table = pd.DataFrame(results)
print("===== 平均|SHAP| =====")
print(table.round(2))
print("\n===== 名次 =====")
print(table.rank(ascending=False).astype(int))