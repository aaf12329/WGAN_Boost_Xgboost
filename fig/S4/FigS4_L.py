import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import norm

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

# ===== 加载 =====
file_path = r"C:\Users\Administrator\Desktop\New_machine\Dataset.csv"
model_base=r"C:\Users\Administrator\Desktop\New_machine\result2\\"
se=['xgboost_model_adsorption_amount.joblib','svm_model.joblib','randomforest_model.joblib','mlr_model.joblib','gradientboosting_model.joblib','ann_model.joblib']

def draw(file_path = r"C:\Users\Administrator\Desktop\New_machine\Dataset.csv",model_path='',mo=''):
    # ===== 数据准备 =====
    X, y, validation_set = prepare_dataset(file_path)
    scaler = joblib.load(r"C:\Users\Administrator\Desktop\New_machine\result1\scaler_adsorption_amount.joblib")
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = joblib.load(model_path)
    # ===== 预测 =====
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # ===== ✅ 计算残差 =====
    y_train_residual = y_train - y_train_pred
    y_test_residual = y_test - y_test_pred
    residuals_all = np.concatenate([y_train_residual, y_test_residual])

    # ===== 指标 =====
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("=" * 50)
    print("📊 模型性能评估")
    print("=" * 50)
    print(f"训练集 R²:  {train_r2:.4f}")
    print(f"测试集 R²:  {test_r2:.4f}")
    print("=" * 50)
    # ==========================================
    # 图1：实验值 vs 预测值散点图
    # ==========================================
    plt.figure(figsize=(8, 8))
    plt.scatter(y_train, y_train_pred, facecolors='none', edgecolors='black', s=50, alpha=0.7, label='Training Set')
    plt.scatter(y_test, y_test_pred, facecolors='red', edgecolors='darkred', marker='s', s=50, alpha=0.7, label='Test Set')

    min_val = min(y.min(), y_train_pred.min(), y_test_pred.min())
    max_val = max(y.max(), y_train_pred.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='-', linewidth=1.5, label='y = x (Ideal)')

    plt.xlabel('Experimental Values (mg/g)', fontsize=14)
    plt.ylabel('Predicted Values (mg/g)', fontsize=14)
    plt.title('Experimental vs Predicted Values for Adsorption Amount', fontsize=15)
    plt.legend(loc='best', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{mo}_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ==========================================
    # 图2：残差分布散点图
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_train_pred, y_train_residual, facecolors='none', edgecolors='black', s=50, alpha=0.6, label='Training Set')
    ax.scatter(y_test_pred, y_test_residual, facecolors='red', edgecolors='darkred', marker='s', s=50, alpha=0.6, label='Test Set')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, label='y = 0 (Ideal)')

    std_res = np.std(residuals_all)
    ax.axhline(y=2*std_res, color='gray', linestyle=':', linewidth=1, alpha=0.7, label=f'±2σ ({2*std_res:.1f})')
    ax.axhline(y=-2*std_res, color='gray', linestyle=':', linewidth=1, alpha=0.7)

    ax.set_xlabel('Predicted Values (mg/g)', fontsize=14)
    ax.set_ylabel('Residuals (mg/g)', fontsize=14)
    ax.set_title('Residuals vs Predicted Values for Adsorption Amount', fontsize=15)
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{mo}_residuals.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ==========================================
    # 图3：残差频率直方图 + 正态分布拟合
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 6))

    n, bins, patches = ax.hist(residuals_all, bins=30, density=True, 
                            facecolor='lightblue', edgecolor='black', alpha=0.7,
                            label='Residuals Distribution')

    mu, sigma = norm.fit(residuals_all)
    x = np.linspace(residuals_all.min(), residuals_all.max(), 100)
    pdf = norm.pdf(x, mu, sigma)
    ax.plot(x, pdf, 'red', linewidth=2, label=f'Normal Fit (μ={mu:.1f}, σ={sigma:.1f})')

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, label='Residual = 0')

    ax.set_xlabel('Residuals (mg/g)', fontsize=14)
    ax.set_ylabel('Frequency Density', fontsize=14)
    ax.set_title('Residuals Distribution Histogram for Adsorption Amount', fontsize=15)
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{mo}_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n✅{mo}三张图已保存:")
for i in se:
    mo_name = i.replace('.joblib', '')
    draw(model_path=model_base+i,mo=mo_name)