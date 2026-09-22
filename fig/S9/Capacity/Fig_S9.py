import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import permutation_importance

# ==================== 路径配置 ====================
now_path = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.dirname(now_path)
model_dir = os.path.join(base_path, "Adsorption_capacity")
dataset_path = os.path.join(base_path, "Dataset.csv")

# ==================== 数据准备函数（从原代码复制） ====================
def prepare_dataset(file_path, target_variable='Adsorption capacity'):
    df = pd.read_csv(file_path)
    df["Number"] = range(1, len(df) + 1)
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in dataset.")
    random_numbers = np.random.choice(df["Number"].dropna().unique(), size=3)
    validation_set = df[df["Number"].isin(random_numbers)]
    drop_df = df[~df["Number"].isin(random_numbers)]
    encoded_df = pd.get_dummies(drop_df, columns=['Pollutant'])
    exclude_columns = ['Number', 'Reference', 'Adsorption amount', 'Adsorption capacity', 'pKa3', 'pKa2', 'pKa1']
    X = encoded_df.drop(columns=exclude_columns)
    y = encoded_df[target_variable]
    return X, y, validation_set

# ==================== 加载数据 ====================
X, y, _ = prepare_dataset(dataset_path, target_variable='Adsorption capacity')  # 根据实际目标变量调整
feature_names = X.columns.tolist()
print("特征数量:", len(feature_names))

# ==================== 模型文件列表 ====================
model_files = [
    "xgboost_model_adsorption_capacity.joblib",
    "svm_model.joblib",
    "randomforest_model.joblib",
    "mlr_model.joblib",
    "gradientboosting_model.joblib",
    "ann_model.joblib"
]

# ==================== 提取特征重要性 ====================
importance_dict = {}  # key: model_name, value: numpy array of importances

for fname in model_files:
    model_path = os.path.join(model_dir, fname)
    if not os.path.exists(model_path):
        print(f"警告：模型文件不存在，跳过 - {model_path}")
        continue

    model = joblib.load(model_path)
    model_name = os.path.splitext(fname)[0]  # 去掉扩展名

    # 1. 模型自带 feature_importances_
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    # 2. 线性模型（MLR）使用系数绝对值
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).flatten()
    # 3. SVM、ANN 等使用排列重要性
    else:
        print(f"模型 {model_name} 无内置重要性，计算排列重要性（可能需要一些时间）...")
        perm_importance = permutation_importance(
            model, X, y,
            n_repeats=10,
            random_state=42,
            scoring='neg_mean_squared_error'
        )
        importance = np.abs(perm_importance.importances_mean)  # 取绝对值

    importance_dict[model_name] = importance
    print(f"已提取 {model_name} 的重要性，长度 {len(importance)}")

if len(importance_dict) == 0:
    raise ValueError("没有成功提取任何模型的特征重要性。")

# ==================== 绘图 ====================
# 设置子图布局：2行3列
n_models = len(importance_dict)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols  # 向上取整

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))
axes = axes.flatten()  # 展平便于索引

# 墨绿色和加粗设置
bar_color = 'darkgreen'
edge_color = 'black'
line_width = 1.5  # 边框加粗

for idx, (model_name, importance) in enumerate(importance_dict.items()):
    ax = axes[idx]
    # 按重要性降序排序（可选，这里保持原特征顺序更直观，也可改为排序）
    sorted_idx = np.argsort(importance)[::-1]  # 降序
    sorted_importance = importance[sorted_idx]
    sorted_features = [feature_names[i] for i in sorted_idx]

    # 绘制横向条形图
    y_pos = np.arange(len(sorted_features))
    ax.barh(y_pos, sorted_importance, color=bar_color, edgecolor=edge_color, linewidth=line_width)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features, fontsize=8)
    ax.invert_yaxis()  # 让重要性最高的在上方
    ax.set_xlabel('Importance')
    ax.set_title(model_name, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelsize=8)

# 隐藏多余的子图
for idx in range(n_models, len(axes)):
    axes[idx].axis('off')

# 整体标题
fig.suptitle('Feature Importance of Six Models (Adsorption Dataset)', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为总标题留出空间

# 保存图片
output_dir = os.path.join(base_path, "figures")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "feature_importance_comparison.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图片已保存至: {output_path}")

plt.show()