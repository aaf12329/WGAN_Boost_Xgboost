import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import KNNImputer
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
from scipy import stats
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import shap
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import norm
import os

#路径区(start)
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 项目根目录
#路径区(stop)

file_path = os.path.join(base_path, "Dataset.csv")
model_base = os.path.join(base_path, "results", "result2", "")
se=['xgboost_model_adsorption_amount.joblib','svm_model.joblib','randomforest_model.joblib','mlr_model.joblib','gradientboosting_model.joblib','ann_model.joblib']
seeds = [42, 123, 456, 789, 1024]

def prepare_dataset(file_path, target_variable='Adsorption amount'):
    df = pd.read_csv(file_path)
    df["Number"] = range(1, len(df) + 1)
    # Ensure target variable exists in the dataset
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in dataset. "
                         f"Available options are: {', '.join([col for col in df.columns if 'Adsorption' in col])}")
    
    random_numbers = np.random.choice(df["Number"].dropna().unique(), size=3)
    validation_set = df[df["Number"].isin(random_numbers)]
    drop_df = df[~df["Number"].isin(random_numbers)]
    
    encoded_df = pd.get_dummies(drop_df, columns=['Pollutant'])
    exclude_columns = ['Number', 'Reference', 'Adsorption amount', 'Adsorption capacity', 'pKa3', 'pKa2', 'pKa1']
    X = encoded_df.drop(columns=exclude_columns)
    y = encoded_df[target_variable]
    
    return X, y, validation_set

def go(seed,model_path):
    X, y, validation_set = prepare_dataset(file_path)
    # Split data for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = joblib.load(model_path)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    # 预测
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # 训练集指标（内部检查）
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    # 测试集指标（论文报告）
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(f"path={model_path}")
    print(f"训练集 R²: {train_r2:.4f}, RMSE: {train_rmse:.2f}")  # 内部看
    print(f"测试集 R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")    # 论文用

for i in seeds:
    for a in se:
        go(i,model_base+a)
