import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 数据（替换成你的实际值）
# ==========================================
models = ['MLR', 'SVM', 'XGBoost', 'RF', 'GBR', 'ANN']

r2_amount = [0.4856, 0.4760, 0.9401, 0.9135, 0.9257, 0.9170]
rmse_amount = [75, 80, 25, 30, 25, 30]
r2_amount_ci = [0.332, 0.081, 0.044, 0.095, 0.068, 0.061]
rmse_amount_ci = [10, 8, 3, 5, 3, 4]

# ----- 吸附容量 (Adsorption Capacity) -----
r2_capacity = [0.3257, 0.4066, 0.9902, 0.9855, 0.9908, 0.8865]
rmse_capacity = [93.19, 87.42, 11.25, 13.67, 10.88, 38.24]
# 误差棒（95% CI）用你交叉验证算出来的
r2_capacity_ci = [0.03, 0.04, 0.01, 0.02, 0.01, 0.03]
rmse_capacity_ci = [5.0, 6.0, 1.5, 2.0, 1.5, 3.0]

# ==========================================
# 画图
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

x_pos = np.arange(len(models))
width = 0.35

# ----- (a) 吸附量 -----
ax1.bar(x_pos - width/2, r2_amount, width, yerr=r2_amount_ci, capsize=5,
        color='royalblue', alpha=0.8, label='R²', zorder=2, error_kw={'linewidth': 1.5})
ax1.set_ylabel('R²', fontsize=12, color='royalblue')
ax1.tick_params(axis='y', labelcolor='royalblue')
ax1.set_ylim(0, 1.0)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models, fontsize=11)
ax1.set_title('(a) Adsorption Amount', fontsize=14)

ax1b = ax1.twinx()
ax1b.bar(x_pos + width/2, rmse_amount, width, yerr=rmse_amount_ci, capsize=5,
         color='darkorange', alpha=0.8, label='RMSE', zorder=2, error_kw={'linewidth': 1.5})
ax1b.set_ylabel('RMSE (mg/g)', fontsize=12, color='darkorange')
ax1b.tick_params(axis='y', labelcolor='darkorange')
ax1b.set_ylim(0, 120)

for i, (r2, rmse) in enumerate(zip(r2_amount, rmse_amount)):
    ax1.text(i - width/2, r2 + 0.02, f'{r2:.3f}', ha='center', va='bottom', fontsize=8, color='royalblue')
    ax1b.text(i + width/2, rmse + 2, f'{rmse:.1f}', ha='center', va='bottom', fontsize=8, color='darkorange')

ax1.grid(axis='y', alpha=0.3, zorder=0)

# ----- (b) 吸附容量 -----
ax2.bar(x_pos - width/2, r2_capacity, width, yerr=r2_capacity_ci, capsize=5,
        color='royalblue', alpha=0.8, label='R²', zorder=2, error_kw={'linewidth': 1.5})
ax2.set_ylabel('R²', fontsize=12, color='royalblue')
ax2.tick_params(axis='y', labelcolor='royalblue')
ax2.set_ylim(0, 1.0)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models, fontsize=11)
ax2.set_title('(b) Adsorption Capacity', fontsize=14)

ax2b = ax2.twinx()
ax2b.bar(x_pos + width/2, rmse_capacity, width, yerr=rmse_capacity_ci, capsize=5,
         color='darkorange', alpha=0.8, label='RMSE', zorder=2, error_kw={'linewidth': 1.5})
ax2b.set_ylabel('RMSE (mg/g)', fontsize=12, color='darkorange')
ax2b.tick_params(axis='y', labelcolor='darkorange')
ax2b.set_ylim(0, 120)

for i, (r2, rmse) in enumerate(zip(r2_capacity, rmse_capacity)):
    ax2.text(i - width/2, r2 + 0.02, f'{r2:.3f}', ha='center', va='bottom', fontsize=8, color='royalblue')
    ax2b.text(i + width/2, rmse + 2, f'{rmse:.1f}', ha='center', va='bottom', fontsize=8, color='darkorange')

ax2.grid(axis='y', alpha=0.3, zorder=0)

# 图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
fig.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=11)

plt.tight_layout()
plt.savefig('FigS6_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()