import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

#路径区(start)
base_path = os.path.dirname(os.path.abspath(__file__))
Dataset_path = os.path.join(os.path.dirname(base_path), "Dataset.csv")
#路径区(stop)

# 与 WGAN_Model.py 保持一致的特征处理和排除列
exclude_columns = ['Number', 'Reference', 'pKa3', 'pKa2']

real_df = pd.read_csv(Dataset_path)
real_encoded = pd.get_dummies(real_df, columns=['Pollutant'])
X_real = real_encoded.drop(columns=[c for c in exclude_columns if c in real_encoded.columns])

# 选列数与当前特征一致的生成数据（标准名优先；若旧文件残留列数不符则用 _new）
Generated_path = None
for _name in ["generated_data_wgan_gp.csv", "generated_data_wgan_gp_new.csv"]:
    _p = os.path.join(base_path, _name)
    if os.path.exists(_p) and set(X_real.columns).issubset(pd.read_csv(_p, nrows=0).columns):
        Generated_path = _p
        break
if Generated_path is None:
    raise FileNotFoundError("未找到与当前特征列匹配的生成数据 CSV，请先运行 WGAN_Model.py")
print(f"使用生成数据: {Generated_path}")

X_gen = pd.read_csv(Generated_path)
X_gen = X_gen[X_real.columns]  # 对齐列顺序

# 用真实数据拟合的归一化同时变换两份数据，保证在同一空间
scaler = MinMaxScaler(feature_range=(-1, 1))
real_scaled = scaler.fit_transform(X_real)
gen_scaled = scaler.transform(X_gen)

combined = np.vstack([real_scaled, gen_scaled])
n_real = len(real_scaled)
print(f"t-SNE 输入: {combined.shape[0]} 样本 x {combined.shape[1]} 特征 (Real {n_real} + Generated {len(gen_scaled)})")

tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
embedding = tsne.fit_transform(combined)

plt.figure(figsize=(8, 7))
plt.scatter(embedding[:n_real, 0], embedding[:n_real, 1],
            c='tab:blue', s=12, alpha=0.5, label=f'Real (n={n_real})')
plt.scatter(embedding[n_real:, 0], embedding[n_real:, 1],
            c='tab:orange', s=12, alpha=0.5, label=f'Generated (n={len(gen_scaled)})')
plt.legend()
plt.title('t-SNE: Real vs WGAN-GP Generated Data')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.tight_layout()
plt.savefig(os.path.join(base_path, "t_sne_real_vs_generated.png"), dpi=150)
plt.show()
print(f"图已保存: {os.path.join(base_path, 't_sne_real_vs_generated.png')}")
