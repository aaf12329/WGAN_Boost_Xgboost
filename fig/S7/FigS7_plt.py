import matplotlib.pyplot as plt
import numpy as np

features = [
    'Dosage',
    'Initial pH',
    'H',
    'Initial concentration',
    'A',
    'MW',
    'N',
    'C',
    'O',
    'VTotal'
]

impact = [
    105.3363,
    18.4982,
    17.0662,
    -15.8507,
    13.3503,
    11.9368,
    10.2176,
    -9.9102,
    -7.2518,
    6.1110
]

# 归一化到 [-1, 1] 方便展示（但保留正负方向）
max_abs = max(abs(v) for v in impact)
impact_norm = [v / max_abs for v in impact]

plt.figure(figsize=(10, 6))

# 颜色：正=蓝，负=红
colors = ['royalblue' if v >= 0 else 'firebrick' for v in impact_norm]

bars = plt.barh(features, impact_norm, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)

# 显示数值（显示原始值，不是归一化值）
for bar, val in zip(bars, impact):
    plt.text(bar.get_width() + 0.01 * max_abs / max_abs, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}', va='center', fontsize=9, fontweight='bold')

plt.axvline(x=0, color='gray', linestyle='-', linewidth=1)
plt.xlabel('Normalized Outlier Impact', fontsize=13)
plt.ylabel('Feature', fontsize=13)
plt.title('Top 10 Features by Absolute Outlier Impact (XGBoost)', fontsize=14)

# 图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='royalblue', alpha=0.8, label='Positive Impact (↑ Prediction)'),
    Patch(facecolor='firebrick', alpha=0.8, label='Negative Impact (↓ Prediction)')
]
plt.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('outlier_impact_top10.png', dpi=300, bbox_inches='tight')
plt.show()