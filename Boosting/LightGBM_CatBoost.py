# LightGBM / CatBoost 对比实验（与 Boosting/Machine.py 的 XGBoost 流程同口径）
# =================================
# - 数据: 项目根目录 Dataset.csv（同一套 prepare_dataset 预处理）
# - 调参: BayesSearchCV，5 折 KFold + 负 RMSE，与 XGBoost 流程一致
# - 输出: results/lightgbm_catboost/ 下的模型 .joblib 与 result.txt
# 依赖: lightgbm, catboost, scikit-optimize, joblib

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer

import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

#路径区(start)
base_path = os.path.dirname(os.path.abspath(__file__))          # Boosting/
root_path = os.path.dirname(base_path)                          # 项目根目录
Dataset_path = os.path.join(root_path, "Dataset.csv")
results_dir = os.path.join(root_path, "results", "lightgbm_catboost")
#路径区(stop)

TARGET = 'Adsorption amount'      # 可切换: 'Adsorption capacity'
N_ITER = 50                       # 贝叶斯搜索迭代数（与 XGBoost 流程一致）


def prepare_dataset(file_path, target_variable='Adsorption amount'):
    """与 Boosting/Machine.py 完全一致的预处理"""
    df = pd.read_csv(file_path)
    df["Number"] = range(1, len(df) + 1)
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in dataset. "
                         f"Available options are: {', '.join([c for c in df.columns if 'Adsorption' in c])}")
    random_numbers = np.random.choice(df["Number"].dropna().unique(), size=3)
    validation_set = df[df["Number"].isin(random_numbers)]
    drop_df = df[~df["Number"].isin(random_numbers)]
    encoded_df = pd.get_dummies(drop_df, columns=['Pollutant'])
    exclude_columns = ['Number', 'Reference', 'Adsorption amount', 'Adsorption capacity', 'pKa3', 'pKa2', 'pKa1']
    X = encoded_df.drop(columns=exclude_columns).astype('float64')
    y = encoded_df[target_variable]
    return X, y, validation_set


# ============ 贝叶斯超参数搜索（同 XGBoost 流程：5 折 + 负 RMSE） ============

LGBM_SPACE = {
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'n_estimators': Integer(50, 800),
    'num_leaves': Integer(8, 150),
    'min_child_samples': Integer(5, 60),
    'subsample': Real(0.5, 1.0),
    'subsample_freq': Integer(1, 7),
    'colsample_bytree': Real(0.5, 1.0),
    'reg_alpha': Real(1e-4, 1.0, prior='log-uniform'),
    'reg_lambda': Real(1e-4, 1.0, prior='log-uniform'),
}

CATBOOST_SPACE = {
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'iterations': Integer(50, 800),
    'depth': Integer(3, 12),
    'l2_leaf_reg': Real(0.01, 10.0, prior='log-uniform'),
    'subsample': Real(0.5, 1.0),
    'bagging_temperature': Real(0.0, 1.0),
}


def bayes_search(model, search_space, X, y, n_iter, n_jobs):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = BayesSearchCV(
        estimator=model,
        search_spaces=search_space,
        scoring='neg_root_mean_squared_error',
        cv=cv,
        n_jobs=n_jobs,
        n_iter=n_iter,
        random_state=42,
        verbose=0,
    )
    search.fit(X, y)
    return dict(search.best_params_)


# ============ 评估 ============

def cv_evaluate(model_factory, X, y):
    """5 折交叉验证，返回各折指标与平均特征重要性"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rows, importances = [], []
    for fold, (tr, te) in enumerate(kf.split(X), 1):
        model = model_factory()
        model.fit(X.iloc[tr], y.iloc[tr])
        p_tr, p_te = model.predict(X.iloc[tr]), model.predict(X.iloc[te])
        rows.append({
            'Fold': fold,
            'Train_RMSE': np.sqrt(mean_squared_error(y.iloc[tr], p_tr)),
            'Test_RMSE': np.sqrt(mean_squared_error(y.iloc[te], p_te)),
            'Train_R2': r2_score(y.iloc[tr], p_tr),
            'Test_R2': r2_score(y.iloc[te], p_te),
        })
        importances.append(model.feature_importances_)
    metrics = pd.DataFrame(rows)
    importance = dict(zip(X.columns, np.mean(importances, axis=0)))
    importance = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
    return metrics, importance


def holdout_evaluate(model_factory, X, y):
    """80/20 划分（random_state=42），与论文六模型表格同口径"""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = model_factory()
    model.fit(X_tr, y_tr)
    p_tr, p_te = model.predict(X_tr), model.predict(X_te)
    return model, {
        'train_r2': r2_score(y_tr, p_tr),
        'train_rmse': np.sqrt(mean_squared_error(y_tr, p_tr)),
        'test_r2': r2_score(y_te, p_te),
        'test_rmse': np.sqrt(mean_squared_error(y_te, p_te)),
        'test_mae': mean_absolute_error(y_te, p_te),
    }


def run_comparison(file_path, target_variable=TARGET, n_iter=N_ITER):
    os.makedirs(results_dir, exist_ok=True)
    X, y, validation_set = prepare_dataset(file_path, target_variable)
    tag = target_variable.replace(' ', '_').lower()
    report = [f"Target: {target_variable} | X: {X.shape} | 时间: {pd.Timestamp.now()}"]

    # ---- LightGBM ----
    print("== LightGBM 贝叶斯搜索 ==")
    lgbm_params = bayes_search(LGBMRegressor(verbose=-1, n_jobs=1), LGBM_SPACE, X, y, n_iter, n_jobs=-1)
    print(f"best: {lgbm_params}")
    lgbm_factory = lambda: LGBMRegressor(**lgbm_params, verbose=-1, n_jobs=-1)
    lgbm_cv, lgbm_imp = cv_evaluate(lgbm_factory, X, y)
    lgbm_model, lgbm_holdout = holdout_evaluate(lgbm_factory, X, y)
    print(f"✅ LightGBM: R² = {lgbm_holdout['test_r2']:.4f}, RMSE = {lgbm_holdout['test_rmse']:.2f}")
    joblib.dump(lgbm_model, os.path.join(results_dir, f"lightgbm_model_{tag}.joblib"))
    report += ["", "=== LightGBM ===", f"best_params: {lgbm_params}",
               f"holdout: {lgbm_holdout}",
               f"CV mean: {lgbm_cv.mean(numeric_only=True).round(4).to_dict()}",
               f"CV std:  {lgbm_cv.std(numeric_only=True).round(4).to_dict()}",
               f"importance top10: {dict(list(lgbm_imp.items())[:10])}"]

    # ---- CatBoost ----
    print("== CatBoost 贝叶斯搜索 ==")
    cb_params = bayes_search(CatBoostRegressor(verbose=0, allow_writing_files=False), CATBOOST_SPACE, X, y, n_iter, n_jobs=1)
    print(f"best: {cb_params}")
    cb_factory = lambda: CatBoostRegressor(**cb_params, verbose=0, allow_writing_files=False)
    cb_cv, cb_imp = cv_evaluate(cb_factory, X, y)
    cb_model, cb_holdout = holdout_evaluate(cb_factory, X, y)
    print(f"✅ CatBoost: R² = {cb_holdout['test_r2']:.4f}, RMSE = {cb_holdout['test_rmse']:.2f}")
    joblib.dump(cb_model, os.path.join(results_dir, f"catboost_model_{tag}.joblib"))
    report += ["", "=== CatBoost ===", f"best_params: {cb_params}",
               f"holdout: {cb_holdout}",
               f"CV mean: {cb_cv.mean(numeric_only=True).round(4).to_dict()}",
               f"CV std:  {cb_cv.std(numeric_only=True).round(4).to_dict()}",
               f"importance top10: {dict(list(cb_imp.items())[:10])}"]

    out_txt = os.path.join(results_dir, f"result_{tag}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"结果已保存: {out_txt}")
    return {'lightgbm': lgbm_holdout, 'catboost': cb_holdout}


if __name__ == "__main__":
    result = run_comparison(Dataset_path, TARGET)
    print(result)
