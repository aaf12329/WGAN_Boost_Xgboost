# 运行环境与依赖 (Runtime & Dependencies)

本项目在以下环境开发并实测通过：

- **OS**：Windows 10 / 11（x64）
- **Python**：3.10.20（conda 环境 `tf`）
- **硬件**：CPU 即可运行（WGAN-GP 训练较慢，有 NVIDIA GPU 会明显加速）

## 依赖清单

版本为开发时 `tf` 环境的**实测版本**；"最低要求"为代码兼容下限（与 README 一致）。

| 包 | 实测版本 | 最低要求 | 用途 |
|----|----------|----------|------|
| tensorflow | 2.21.0 | >= 2.6.0 | WGAN-GP / 普通 GAN 训练与数据生成 |
| scikit-learn | 1.7.2 | >= 1.0.0 | 预处理、交叉验证、对比模型（RF/GBDT/MLR/SVM/ANN） |
| xgboost | 1.7.6 | >= 1.5.0 | 主回归模型 |
| shap | 0.49.1 | >= 0.40.0 | SHAP 特征重要性与交互值 |
| scikit-optimize | 0.10.2 | >= 0.9.0 | XGBoost 贝叶斯超参数优化（BayesSearchCV） |
| lightgbm | 4.7.0 | >= 4.0.0 | LightGBM 对比模型（Boosting/LightGBM_CatBoost.py） |
| catboost | 1.2.10 | >= 1.2.0 | CatBoost 对比模型（Boosting/LightGBM_CatBoost.py） |
| tabpfn | 9.0.0 | — | TabPFN 基础模型（已安装；首次使用需在 Prior Labs 注册并接受许可） |
| numpy | 2.2.6 | >= 1.20.0 | 数值计算 |
| pandas | 2.3.3 | >= 1.3.0 | 数据读取与处理 |
| matplotlib | 3.10.9 | >= 3.5.0 | 绘图（分布对比、SHAP、PDP） |
| seaborn | 0.13.2 | >= 0.11.0 | 绘图（可选） |
| joblib | 1.5.3 | >= 1.1.0 | 模型与 scaler 的保存 / 加载 |

## 安装

```bash
conda create -n tf python=3.10
conda activate tf
pip install tensorflow scikit-learn xgboost shap scikit-optimize numpy pandas matplotlib seaborn joblib
```

如需复现与开发环境完全一致的版本，使用 `pip install -r` 搭配下面内容（另存为 `requirements.txt` 即可）：

```text
tensorflow==2.21.0
scikit-learn==1.7.2
xgboost==1.7.6
shap==0.49.1
scikit-optimize==0.10.2
lightgbm==4.7.0
catboost==1.2.10
tabpfn==9.0.0
numpy==2.2.6
pandas==2.3.3
matplotlib==3.10.9
seaborn==0.13.2
joblib==1.5.3
```

## 说明

- 所有 `.py` 脚本通过 `import os` 相对定位文件，与工作目录无关；但 `.bat` 启动器写死了 `D:\anaconda\envs\tf\python.exe`，其他机器需改成对应环境的 python 路径（或直接 `python xxx.py`）。
- TensorFlow 2.21 要求 Python ≥ 3.10；若需兼容 README 中的 Python 3.8 下限，请将 tensorflow 降级到 ≤ 2.13（2.14 起放弃 Windows 原生支持前的最后一个支持 3.8 的主版本线为 2.10–2.13，GPU 支持以 TensorFlow 官方对应版本说明为准）。

---

# Runtime & Dependencies (English)

Developed and tested in the following environment:

- **OS**: Windows 10 / 11 (x64)
- **Python**: 3.10.20 (conda env `tf`)
- **Hardware**: CPU is sufficient (WGAN-GP training is slow; an NVIDIA GPU speeds it up considerably)

The table lists the **measured versions** in the dev environment; "minimum" is the compatibility floor (consistent with README).

| Package | Measured | Minimum | Purpose |
|----|----------|----------|------|
| tensorflow | 2.21.0 | >= 2.6.0 | WGAN-GP / vanilla GAN training and data generation |
| scikit-learn | 1.7.2 | >= 1.0.0 | Preprocessing, CV, comparison models (RF/GBDT/MLR/SVM/ANN) |
| xgboost | 1.7.6 | >= 1.5.0 | Main regression model |
| shap | 0.49.1 | >= 0.40.0 | SHAP feature importance and interaction values |
| scikit-optimize | 0.10.2 | >= 0.9.0 | Bayesian hyperparameter optimization (BayesSearchCV) |
| numpy | 2.2.6 | >= 1.20.0 | Numerical computing |
| pandas | 2.3.3 | >= 1.3.0 | Data loading and processing |
| matplotlib | 3.10.9 | >= 3.5.0 | Plotting (distributions, SHAP, PDP) |
| seaborn | 0.13.2 | >= 0.11.0 | Plotting (optional) |
| joblib | 1.5.3 | >= 1.1.0 | Saving / loading models and scalers |

Install:

```bash
conda create -n tf python=3.10
conda activate tf
pip install tensorflow scikit-learn xgboost shap scikit-optimize numpy pandas matplotlib seaborn joblib
```

For an environment identical to the dev setup, save the pinned block above as `requirements.txt` and run `pip install -r requirements.txt`.

Notes: all `.py` scripts resolve paths relatively via `import os` and are independent of the working directory; the `.bat` launchers hard-code `D:\anaconda\envs\tf\python.exe` and must be adjusted on other machines. TensorFlow 2.21 requires Python >= 3.10; for the Python 3.8 floor stated in README, downgrade tensorflow accordingly.
