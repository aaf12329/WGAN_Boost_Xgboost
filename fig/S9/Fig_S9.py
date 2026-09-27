import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================== 路径配置 ====================
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 项目根目录
model_dir = os.path.join(base_path, "results", "result2")   # 模型目录
data_dir = os.path.join(base_path, "data")              # 数据目录

# ==================== 数据集加载函数（示例） ====================
def load_dataset(dataset_name):
    # 根据数据集名称加载对应数据，返回 X, y, feature_names
    # 你需要根据实际文件结构实现此函数
    if dataset_name == "pH":
        file_path = os.path.join(data_dir, "pH_dataset.csv")
    elif dataset_name == "temperature":
        file_path = os.path.join(data_dir, "temperature_dataset.csv")
    else:
        raise ValueError("Unknown dataset name")
    df = pd.read_csv(file_path)
    X = df.drop(columns=['target'])   # 假设目标列名为 'target'
    y = df['target']
    feature_names = X.columns.tolist()
    return X, y, feature_names

# ==================== 模型文件列表 ====================
# 三个集成学习模型名称（需在 pH 和 temperature 数据集上分别训练）
model_names = ["random_forest", "xgboost", "gradient_boosting"]  # 示例

# 数据集名称列表
datasets = ["pH", "temperature"]

# ==================== 提取特征重要性 ====================
importance_dict = {dataset: {} for dataset in datasets}

for dataset in datasets:
    X, y, feature_names = load_dataset(dataset)
    for model_name in model_names:
        # 模型文件命名规则：{model_name}_{dataset}.joblib
        model_path = os.path.join(model_dir, f"{model_name}_{dataset}.joblib")
        if not os.path.exists(model_path):
            print(f"警告：模型文件不存在 - {model_path}")
            continue
        model = joblib.load(model_path)
        # 提取重要性
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_).flatten()
        else:
            # 排列重要性（需要 X, y）
            from sklearn.inspection import permutation_importance
            perm = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            importance = perm.importances_mean
        importance_dict[dataset][model_name] = importance

# ==================== 绘图 ====================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 两个数据集并排

bar_color = 'darkgreen'
edge_color = 'black'
linewidth = 1.5

for ax, dataset in zip(axes, datasets):
    # 获取该数据集的三个模型重要性
    model_importances = importance_dict[dataset]
    if not model_importances:
        ax.axis('off')
        continue

    # 构建 DataFrame 用于绘图
    df_imp = pd.DataFrame(model_importances, index=feature_names)
    # 按平均重要性排序（可选）
    df_imp['mean'] = df_imp.mean(axis=1)
    df_imp = df_imp.sort_values('mean', ascending=False).drop(columns='mean')

    # 绘制横向条形图（每个模型一组）
    y_pos = np.arange(len(df_imp))
    height = 0.25
    for i, model_name in enumerate(df_imp.columns):
        ax.barh(y_pos + i*height, df_imp[model_name], height=height,
                color=bar_color, edgecolor=edge_color, linewidth=linewidth, label=model_name)

    ax.set_yticks(y_pos + height)
    ax.set_yticklabels(df_imp.index, fontsize=9)
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Dataset: {dataset}', fontweight='bold')
    ax.legend(loc='best')

plt.tight_layout()
plt.savefig('feature_importance_pH_temperature.png', dpi=300)
plt.show()