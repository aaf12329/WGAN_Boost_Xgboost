import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def prepare_dataset(file_path, target_variable='Adsorption amount'):
    df = pd.read_csv(file_path)
    df["Number"] = range(1, len(df) + 1)
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found.")
    random_numbers = np.random.choice(df["Number"].dropna().unique(), size=3)
    validation_set = df[df["Number"].isin(random_numbers)]
    drop_df = df[~df["Number"].isin(random_numbers)]
    encoded_df = pd.get_dummies(drop_df, columns=['Pollutant'])
    exclude_columns = ['Number', 'Reference', 'Adsorption amount', 'Adsorption capacity', 'pKa3', 'pKa2', 'pKa1']
    X = encoded_df.drop(columns=exclude_columns)
    y = encoded_df[target_variable]
    return X, y, validation_set

def calculate_outlier_impact(model, X, y, feature_names, threshold=2.0):
    y_pred = model.predict(X)
    residuals = np.abs(y - y_pred)
    outlier_mask = residuals > threshold * np.std(residuals)
    X_outliers = X[outlier_mask]
    
    if len(X_outliers) == 0:
        print("⚠️ 没有找到异常样本，请降低 threshold 值")
        return {}
    
    impact = {}
    n_features = X.shape[1]
    
    for i in range(n_features):
        X_modified = X_outliers.copy()
        X_modified[:, i] = np.mean(X_outliers[:, i])
        
        y_pred_original = model.predict(X_outliers)
        y_pred_modified = model.predict(X_modified)
        
        # ✅ 不取绝对值，保留正负
        diff = np.mean(y_pred_original - y_pred_modified)
        impact[feature_names[i]] = diff
    
    # 按绝对值排序
    impact_sorted = sorted(impact.items(), key=lambda x: abs(x[1]), reverse=True)
    return impact_sorted

# ===== 主程序 =====
file_path = r"C:\Users\AAF12\Desktop\New_machine\Dataset.csv"
model = joblib.load(r"C:\Users\AAF12\Desktop\New_machine\result2\xgboost_model_adsorption_amount.joblib")

X, y, validation_set = prepare_dataset(file_path)
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 计算带正负的 Outlier Impact
impact_result = calculate_outlier_impact(model, X_test_scaled, y_test, feature_names)

# 打印结果
print("📊 Outlier Impact（带正负方向）：")
for name, value in impact_result[:10]:
    print(f"{name}: {value:.4f}")