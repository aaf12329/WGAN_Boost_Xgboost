# Required Libraries and Environment
# =================================
# Python 3.8.x or higher
# Dependencies:
# - numpy>=1.20.0
# - pandas>=1.3.0
# - scikit-learn>=1.0.0
# - xgboost>=1.5.0
# - shap>=0.40.0
# - scikit-optimize>=0.9.0
# - joblib>=1.1.0
# - matplotlib>=3.5.0
#
# 流程（老师要求，全部由 run_complete_analysis 串起来）:
# 1.  数据准备 prepare_dataset（原始数据 + WGAN 生成数据混合，特征按论文 Fig.S11 构造）
# 2.  拆分训练与测试集 train_test_split
# 3.  数据标准化 StandardScaler
# 4.  超参数调优 bayesian_hyperparameter_optimization
# 5.  交叉验证 perform_cross_validation
# 6.  训练最终 XGBoost 模型 final_xgb_model
# 7.  对比其他树模型 train_three_models（XGB / RF / GBR，参数取论文 Table S3）
# 8.  特征交互分析 calculate_friedman_h_statistic
# 9.  SHAP 可视化分析 calculate_shap_values（复现论文 Fig.S11 蜂群图）
# 10. 保存模型（生成 2 个 joblib 文件：最终模型 + 标准化器）

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#路径区(start)
base_path = os.path.dirname(os.path.abspath(__file__))
Dataset_path = os.path.join(os.path.dirname(base_path), "Dataset.csv")
Generated_paths = [
    os.path.join(os.path.dirname(base_path), "WGAN", "generated_data_wgan_gp.csv"),
    os.path.join(os.path.dirname(base_path), "WGAN", "generated_data_wgan_gp_new.csv"),
]
#路径区(stop)

np.random.seed(42)

# 目标列：论文 Fig.S11 为 XGB 预测 adsorption capacity 的 SHAP 图
TARGET = 'Adsorption capacity'

# 论文 Fig.S11 的 19 个特征：C/H/O/N 原始列换成元素比值，去掉 Pollutant
FEATURE_ORDER = ['SBET', 'VTotal', 'Dp', 'H/C', 'N/C', 'O+N/C', 'O/C',
                 'Initial concentration', 'Dosage', 'Temperature', 'Initial pH',
                 'E', 'S', 'A', 'B', 'V', 'Kow', 'pKa1', 'MW']

EXCLUDE_COLUMNS = ['Number', 'Reference', 'Adsorption amount', 'Adsorption capacity', 'pKa3', 'pKa2']


def add_element_ratios(df):
    """按论文 Fig.S11 用 H/C、N/C、O+N/C、O/C 替换原始 C、H、O、N，并去掉 Pollutant"""
    df = df.copy()
    df['H/C'] = df['H'] / df['C']
    df['N/C'] = df['N'] / df['C']
    df['O/C'] = df['O'] / df['C']
    df['O+N/C'] = (df['O'] + df['N']) / df['C']
    drop_cols = [c for c in ['C', 'H', 'O', 'N', 'Pollutant'] if c in df.columns]
    drop_cols += [c for c in df.columns if c.startswith('Pollutant_')]
    return df.drop(columns=drop_cols)


# ================== 1. 数据准备 ==================
def prepare_dataset(file_path, generated_path, target_variable=TARGET):
    """原始数据 + WGAN 生成数据混合，构造论文 Fig.S11 的特征集"""
    real_df = pd.read_csv(file_path)
    real_df["Number"] = range(1, len(real_df) + 1)

    if target_variable not in real_df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in dataset.")

    random_numbers = np.random.choice(real_df["Number"].dropna().unique(), size=3)
    validation_set = real_df[real_df["Number"].isin(random_numbers)]
    real_df = real_df[~real_df["Number"].isin(random_numbers)]

    real_feat = add_element_ratios(real_df)
    X_real = real_feat.drop(columns=EXCLUDE_COLUMNS)
    y_real = real_feat[target_variable]

    gen_df = pd.read_csv(generated_path)
    gen_feat = add_element_ratios(gen_df)
    X_gen = gen_feat.drop(columns=[c for c in EXCLUDE_COLUMNS if c in gen_feat.columns])
    y_gen = gen_feat[target_variable]

    X = pd.concat([X_real, X_gen], ignore_index=True)[FEATURE_ORDER]
    y = pd.concat([y_real, y_gen], ignore_index=True)

    assert not X.isna().any().any(), "特征中存在 NaN，请检查生成数据"
    print(f"混合数据集: 原始 {len(X_real)} 行 + WGAN 生成 {len(X_gen)} 行 = {len(X)} 行, 特征 {X.shape[1]} 个")
    return X, y, validation_set


def Gan_Model_Data(target_variable=TARGET):
    """加载 WGAN 生成数据（自动选择列数匹配的 CSV）"""
    for p in Generated_paths:
        if os.path.exists(p) and TARGET in pd.read_csv(p, nrows=0).columns:
            print(f"使用生成数据: {p}")
            return pd.read_csv(p)
    raise FileNotFoundError("未找到生成数据 CSV，请先运行 WGAN/WGAN_Model.py")


# ================== 4. 超参数调优（贝叶斯） ==================
def bayesian_hyperparameter_optimization(X, y, n_iterations=50):
    search_space = {
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'n_estimators': Integer(50, 800),
        'max_depth': Integer(2, 20),
        'min_child_weight': Integer(1, 6),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.5, 1.0),
        'gamma': Real(0, 0.5),
        'reg_alpha': Real(0.0001, 1),
        'reg_lambda': Real(0.0001, 1)
    }

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    bayes_search = BayesSearchCV(
        estimator=xgb_model,
        search_spaces=search_space,
        scoring='neg_root_mean_squared_error',
        cv=cv,
        n_jobs=-1,
        n_iter=n_iterations,
        random_state=42,
        verbose=0
    )
    bayes_search.fit(X, y)
    return bayes_search.best_estimator_, dict(bayes_search.best_params_)


# ================== 5. 交叉验证 ==================
def perform_cross_validation(X, y, model_params=None):
    if model_params is None:
        model_params = {
            'n_estimators': 515,
            'learning_rate': 0.1466,
            'max_depth': 10,
            'min_child_weight': 5,
            'gamma': 0.3639,
            'subsample': 0.8591,
            'colsample_bytree': 0.9636,
            'reg_alpha': 0.0908,
            'reg_lambda': 0.0112,
            'random_state': 42
        }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    all_feature_importances = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_test_fold_scaled = scaler.transform(X_test_fold)

        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train_fold_scaled, y_train_fold)

        y_train_pred = model.predict(X_train_fold_scaled)
        y_test_pred = model.predict(X_test_fold_scaled)

        fold_metrics.append({
            'Fold': fold,
            'Train_RMSE': np.sqrt(mean_squared_error(y_train_fold, y_train_pred)),
            'Test_RMSE': np.sqrt(mean_squared_error(y_test_fold, y_test_pred)),
            'Train_R2': r2_score(y_train_fold, y_train_pred),
            'Test_R2': r2_score(y_test_fold, y_test_pred)
        })
        all_feature_importances.append(model.feature_importances_)

    metrics_df = pd.DataFrame(fold_metrics)
    mean_metrics = metrics_df.mean().round(4)
    std_metrics = metrics_df.std().round(4)
    print("\n=== 五折交叉验证 (均值 ± 标准差) ===")
    for col in ['Train_RMSE', 'Test_RMSE', 'Train_R2', 'Test_R2']:
        print(f"{col}: {mean_metrics[col]} ± {std_metrics[col]}")

    avg_importance = np.mean(all_feature_importances, axis=0)
    feature_importance = dict(zip(X.columns, avg_importance))
    sorted_importance = {k: v for k, v in sorted(feature_importance.items(),
                                                  key=lambda item: item[1], reverse=True)}
    return metrics_df.assign(mean=mean_metrics, std=std_metrics), sorted_importance


# ================== 7. 对比其他树模型（论文 Table S3 参数） ==================
def train_three_models(X_train, y_train, X_test, y_test, xgb_params):
    models = {
        'XGBoost': xgb.XGBRegressor(**xgb_params),
        'RandomForest': RandomForestRegressor(
            n_estimators=240, max_depth=10, min_samples_split=5,
            min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=579, learning_rate=0.2366, max_depth=3,
            min_samples_split=5, min_samples_leaf=1, subsample=0.7404,
            random_state=42),
    }

    model_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        model_results[name] = {
            'model': model,
            'metrics': {
                'train_rmse': train_rmse, 'test_rmse': test_rmse,
                'train_r2': train_r2, 'test_r2': test_r2
            }
        }
        print(f"✅ {name}: Test R² = {test_r2:.4f}, Test RMSE = {test_rmse:.2f}")

    return model_results


# ================== 8. 特征交互分析（Friedman H 统计） ==================
def calculate_friedman_h_statistic(model, X, feature_names, n_samples=1000):
    from itertools import combinations

    n_features = X.shape[1]
    feature_pairs = list(combinations(range(n_features), 2))
    H_statistics = {}
    feature_medians = np.median(X, axis=0)

    for i, j in feature_pairs:
        X_mc = np.zeros((n_samples, n_features))
        for col in range(n_features):
            X_mc[:, col] = feature_medians[col]
        X_mc[:, i] = np.random.uniform(low=np.min(X[:, i]), high=np.max(X[:, i]), size=n_samples)
        X_mc[:, j] = np.random.uniform(low=np.min(X[:, j]), high=np.max(X[:, j]), size=n_samples)

        F_ij = model.predict(X_mc)

        X_i = X_mc.copy()
        X_i[:, j] = feature_medians[j]
        F_i = model.predict(X_i)

        X_j = X_mc.copy()
        X_j[:, i] = feature_medians[i]
        F_j = model.predict(X_j)

        X_0 = X_mc.copy()
        X_0[:, i] = feature_medians[i]
        X_0[:, j] = feature_medians[j]
        F_0 = model.predict(X_0)

        interaction_term = F_ij - F_i - F_j + F_0
        numerator = np.sum(interaction_term ** 2)
        denominator = np.sum((F_ij - F_0) ** 2)

        if denominator <= 1e-10 or np.isnan(numerator) or np.isnan(denominator):
            H = 0
        else:
            H = min(1.0, max(0.0, numerator / denominator))
        H_statistics[(i, j)] = H

    H_stats_list = [
        {'Feature1': feature_names[i], 'Feature2': feature_names[j], 'H_statistic': v}
        for (i, j), v in H_statistics.items()
    ]
    H_stats_df = pd.DataFrame(H_stats_list).sort_values(by='H_statistic', ascending=False)
    return H_stats_df


# ================== 9. SHAP 可视化分析（复现论文 Fig.S11） ==================
def calculate_shap_values(model, X_train, feature_names, out_dir=base_path):
    """用 xgboost 原生 pred_contribs 计算 TreeSHAP。
    不走 shap.TreeExplainer：当前 shap 0.49 无法解析 xgboost 3.x 的模型格式（base_score 存成字符串数组），
    而 pred_contribs 与 TreeSHAP 完全等价，画图仍用 shap.summary_plot。"""
    X_sample = X_train[:min(500, len(X_train))]

    booster = model.get_booster()
    dm = xgb.DMatrix(X_sample, feature_names=feature_names)
    contribs = booster.predict(dm, pred_contribs=True)
    shap_values = contribs[:, :-1]      # 最后一列是期望值偏置
    expected_value = float(contribs[0, -1])

    feature_importance = np.abs(shap_values).mean(0)
    shap_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

    X_sample_df = pd.DataFrame(X_sample, columns=feature_names)
    plt.figure()
    shap.summary_plot(shap_values, X_sample_df, show=False, max_display=19)
    fig = plt.gcf()
    fig.set_size_inches(9, 7)
    plt.tight_layout()
    out_png = os.path.join(out_dir, "shap_beeswarm_adsorption_capacity.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"SHAP 蜂群图已保存: {out_png}")

    return {
        'values': shap_values,
        'expected_value': expected_value,
        'feature_importance': shap_importance
    }


# ================== 主控函数 ==================
def run_complete_analysis(file_path=Dataset_path, target_variable=TARGET, n_iterations=50):
    # 1. 数据准备（原始 + WGAN 生成混合）
    generated_path = None
    for p in Generated_paths:
        if os.path.exists(p) and target_variable in pd.read_csv(p, nrows=0).columns:
            generated_path = p
            break
    if generated_path is None:
        raise FileNotFoundError("未找到生成数据 CSV，请先运行 WGAN/WGAN_Model.py")
    print(f"使用生成数据: {generated_path}")

    X, y, validation_set = prepare_dataset(file_path, generated_path, target_variable)

    # 2. 拆分训练与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. 超参数调优（贝叶斯，论文搜索空间）
    print(f"\n=== 贝叶斯超参数调优 ({n_iterations} 次迭代 × 5 折) ===")
    _, best_params = bayesian_hyperparameter_optimization(X, y, n_iterations=n_iterations)
    best_params['random_state'] = 42
    print("最优参数:", best_params)

    # 5. 交叉验证
    cv_metrics, feature_importance = perform_cross_validation(X, y, best_params)

    # 6. 训练最终 XGBoost 模型
    final_xgb_model = xgb.XGBRegressor(**best_params)
    final_xgb_model.fit(X_train_scaled, y_train)
    y_test_pred = final_xgb_model.predict(X_test_scaled)
    print(f"\n=== 最终 XGBoost 模型 (测试集) ===")
    print(f"R² = {r2_score(y_test, y_test_pred):.4f}, "
          f"RMSE = {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")

    # 7. 对比其他树模型
    print("\n=== 三模型对比 ===")
    tree_models = train_three_models(X_train_scaled, y_train, X_test_scaled, y_test, best_params)

    # 8. 特征交互分析
    print("\n=== Friedman H 统计 (Top 10 特征交互) ===")
    H_stats_df = calculate_friedman_h_statistic(final_xgb_model, X_train_scaled, X.columns.tolist())
    print(H_stats_df.head(10).to_string(index=False))

    # 9. SHAP 可视化分析（复现 Fig.S11）
    shap_values = calculate_shap_values(final_xgb_model, X_train_scaled, X.columns.tolist())
    print("\n=== SHAP 全局重要度 (XGBoost) ===")
    print(shap_values['feature_importance'].to_string(index=False))

    # 10. 保存模型（2 个 joblib 文件）
    tag = target_variable.replace(' ', '_').lower()
    model_filename = os.path.join(base_path, f'xgboost_model_{tag}.joblib')
    scaler_filename = os.path.join(base_path, f'scaler_{tag}.joblib')
    joblib.dump(final_xgb_model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"\n模型已保存: {model_filename}")
    print(f"标准化器已保存: {scaler_filename}")

    return {
        'target_variable': target_variable,
        'best_params': best_params,
        'cv_metrics': cv_metrics,
        'feature_importance': feature_importance,
        'H_statistics': H_stats_df,
        'tree_models': {name: {'metrics': data['metrics']} for name, data in tree_models.items()},
        'shap_feature_importance': shap_values['feature_importance'],
    }


if __name__ == "__main__":
    result = run_complete_analysis(file_path=Dataset_path, target_variable=TARGET, n_iterations=50)
    print("\n分析完成")
