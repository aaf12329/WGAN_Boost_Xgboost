# WGAN_Boost_Xgboost

基于 **WGAN-GP 数据增强 + XGBoost 可解释机器学习** 的磺胺类抗生素在生物炭上吸附性能预测项目。

针对文献数据量小（仅 943 条实验记录）的问题，先用 Wasserstein GAN（梯度惩罚版）从原始数据中学分布、生成 2000 条合成样本扩充数据集，再用 XGBoost（贝叶斯超参数优化）等回归模型建模，并结合 SHAP、Friedman H 统计量、PDP 等方法做特征重要性与交互作用解释。

## 研究背景与动机

原始 GAN 在本数据集上**失败**了：生成的数据与原始数据分布差异巨大、几乎没有重叠（梯度爆炸 / 模式坍塌）。这与所参考文献 3.3 节作者自述的失败原因一致，也正是该文献随后引入 WGAN（Wasserstein 距离）来挽救的原因。

本项目因此采用 **WGAN-GP**（以梯度惩罚代替权重剪枝的 WGAN 改进版）做数据生成，失败对照实验与产物保留在 `GAN_Original/`，失败原因记录见 `docs/log.txt`。

## 数据集

`Dataset.csv`：文献整理的磺胺类抗生素在生物炭上的吸附实验数据，**943 条记录 × 25 列**，覆盖 7 种磺胺（Sulfadiazine、Sulfamethoxazole、Sulfamerazine、Sulfathiazole、Sulfapyridine、Sulfamethazine、Sulfonamide）。

| 特征组 | 变量 |
|--------|------|
| 吸附质性质 | Pollutant（污染物种类，建模时 one-hot）、分子量 MW、pKa1、Kow（辛醇-水分配系数）、分子描述符 E / S / A / B / V |
| 生物炭性质 | 比表面积 SBET、总孔容 VTotal、平均孔径 Dp、元素组成 C / H / O / N |
| 吸附条件 | 投加量 Dosage、初始浓度 Initial concentration、温度 Temperature、初始 pH |
| 目标变量 | **Adsorption amount（吸附量，mg/g）** / Adsorption capacity（吸附容量） |

预处理约定（各建模脚本一致）：

- `Pollutant` one-hot 编码；剔除 `Number`、`Reference` 及 pKa2/pKa3（pKa2 缺失 319 条、pKa3 全缺失）
- 每次随机抽 3 条作 validation set 不参与训练
- GAN 输入做 MinMax 归一化到 [-1, 1]；XGBoost 建模用 StandardScaler 标准化

## 技术路线

**原始数据 → WGAN-GP 生成合成样本 → XGBoost（贝叶斯优化）+ 多模型对比 → SHAP / H 统计量 / PDP 解释**

### WGAN-GP 数据生成（`WGAN/WGAN_Model.py`）

- 生成器：128→256→512 全连接 + BatchNorm + LeakyReLU，tanh 输出；Critic：256→128→64，输出实数分数
- 训练 5000 轮后生成 2000 条合成样本，反归一化回原始量纲，保存为 `WGAN/generated_data_wgan_gp.csv`（被占用时自动另存 `_new`）
- 生成器权重存 `wgan_gp_generator.weights.h5`；分布直方图对比存 `wgan_distribution_compare.png`

### XGBoost 建模与解释（`Boosting/Machine.py`）

1. 贝叶斯超参数优化（`BayesSearchCV`，5 折 KFold，负 RMSE 为目标，50 轮迭代）
2. 最优参数下 5 折交叉验证
3. 六模型横向对比：XGBoost / RandomForest / GradientBoosting / MLR / SVM / ANN
4. 可解释性：SHAP 全局重要性 + SHAP 交互值、Friedman H 统计量（1000 次蒙特卡洛）、PDP 依赖图

`Boosting/Gan_Boost_Xgb.py` 为数据增强版：加载 WGAN 生成数据进入同一套 Boosting 流程。

`Boosting/LightGBM_CatBoost.py` 为 LightGBM / CatBoost 的同口径对比实验（贝叶斯调参）：**CatBoost 为当前全场最优模型**。

## 项目结构

```
WGAN_Boost_Xgboost/
├── README.md
├── requirements.md            # 运行环境与依赖清单
├── Dataset.csv                # 原始数据（文献整理）
├── main.py                    # 入口：真实数据 XGBoost 建模全流程
├── test.py                    # 入口：加载 WGAN 生成数据测试
├── Engage.bat                 # Windows 一键运行 test.py
│
├── Boosting/                  # XGBoost / Boosting 建模模块
│   ├── Machine.py             # 核心建模流程（贝叶斯优化 / 交叉验证 / 多模型对比 / SHAP）
│   ├── Machine_Fig10.py       # Fig10 专用副本（与 Machine.py 相同）
│   ├── LightGBM_CatBoost.py   # LightGBM / CatBoost 贝叶斯调参对比
│   └── Gan_Boost_Xgb.py       # WGAN 增强数据 + XGBoost 流程
│
├── WGAN/                      # WGAN-GP 数据生成
│   ├── WGAN_Model.py          # WGAN-GP 训练与生成（主脚本）
│   ├── GAN_Model.py           # WGAN-GP 早期草稿（已被 WGAN_Model.py 取代）
│   ├── t_SNE.py               # t-SNE 真实/生成分布对比
│   ├── wgan_gp_generator.weights.h5
│   ├── generated_data_wgan_gp_new.csv
│   └── wgan_distribution_compare.png / t_sne_real_vs_generated.png
│
├── GAN_Original/              # 普通 GAN（失败对照）
│   ├── Normal_Gan.py
│   └── generator_model.h5 / generated_data.csv / scaler.pkl
│
├── feature_rate_xgb_model/    # 特征重要性 XGBoost 子项目（含独立启动 bat）
├── results/                   # 各阶段模型与结果
│   ├── lightgbm_catboost/     # LightGBM / CatBoost 调参结果（当前最优）
│   ├── Adsorption_capacity/   # 吸附容量目标的六模型 .joblib
│   ├── result1/               # 吸附量目标（阶段一）
│   └── result2/               # 吸附量目标（阶段二：六模型 + 结果文本）
├── fig/                       # 补充材料图（S4–S15）脚本与输出
└── docs/                      # 文档：log.txt、WGAN记录.docx、反向传播.docx
```

## 环境依赖

- Python 3.8+（本机 conda 环境 `tf`，Python 3.10）
- TensorFlow >= 2.6.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- shap >= 0.40.0
- scikit-optimize >= 0.9.0
- numpy >= 1.20.0、pandas >= 1.3.0
- matplotlib >= 3.5.0、seaborn >= 0.11.0（可视化，可选）
- joblib >= 1.1.0

```bash
conda create -n tf python=3.10
conda activate tf
pip install tensorflow scikit-learn xgboost shap scikit-optimize numpy pandas matplotlib seaborn joblib
```

## 快速开始

```bash
conda activate tf

# 1) 训练 WGAN-GP 并生成 2000 条合成样本（产物在 WGAN/）
python WGAN/WGAN_Model.py

# 2) 真实 vs 生成数据 t-SNE 分布对比（FigS8）
python WGAN/t_SNE.py

# 3) 真实数据 XGBoost 全流程（贝叶斯优化 + 交叉验证 + 六模型对比 + SHAP/H 统计量）
python main.py

# 4) 用 WGAN 增强数据跑 XGBoost 流程
python Boosting/Gan_Boost_Xgb.py

# 5) 快速自检：加载 WGAN 生成数据看前几行（Engage.bat 等价）
python test.py

# 6) LightGBM / CatBoost 贝叶斯调参对比（当前最优模型，完整搜索约 25 分钟）
python Boosting/LightGBM_CatBoost.py
```

补充材料图：进入 `fig/S4` ~ `fig/S15` 对应子目录，双击各自的 `Engage_plt.bat`（或 `test.bat`）运行，图片输出在脚本所在目录；`fig/S15/Fig_S15.py` 一个脚本一次性产出 S14–S18 五张图到 `fig/out/`。

### 关键超参数（WGAN-GP）

| 参数 | 值 | 说明 |
|------|------|------|
| `latent_dim` | 100 | 噪声维度 |
| `epochs` | 5000 | 训练轮数 |
| `batch_size` | 64 | 批大小 |
| `n_critic` | 5 | 每训练 1 次生成器对应 Critic 训练次数 |
| `lambda_gp` | 10.0 | 梯度惩罚系数 |
| `learning_rate` | 0.0002 | Adam 学习率（β1=0.5, β2=0.9） |

### 关键超参数（XGBoost 贝叶斯搜索空间）

`learning_rate` [0.01, 0.3]、`n_estimators` [50, 800]、`max_depth` [2, 20]、`min_child_weight` [1, 6]、`subsample` [0.5, 1.0]、`colsample_bytree` [0.5, 1.0]、`gamma` [0, 0.5]、`reg_alpha` / `reg_lambda` [0.0001, 1]，5 折 + 负 RMSE，50 轮迭代。

吸附量目标搜出的最优参数（`results/result2/result.txt`）：`n_estimators=594, learning_rate=0.0272, max_depth=11, min_child_weight=1, subsample=0.5127, colsample_bytree=0.5609, gamma=0.3164, reg_alpha=0.2231, reg_lambda=0.8206`。S11 特征集模型（`feature_rate_xgb_model/`）另有一套独立调参结果。

## 主要结果（Adsorption amount，测试集）

| 模型 | R² | RMSE |
|------|------|------|
| **CatBoost**（贝叶斯调参） | **0.9619** | **15.70** |
| LightGBM（贝叶斯调参） | 0.9531 | 17.44 |
| GradientBoosting | 0.9397 | 19.36 |
| XGBoost | 0.9281 | 21.15 |
| RandomForest | 0.9089 | 23.80 |
| ANN | 0.8324 | 32.29 |
| SVM | 0.4615 | 57.88 |
| MLR | 0.1975 | 70.65 |

- 5 折交叉验证（XGBoost，最优参数）：Test R² ≈ 0.9669，Test RMSE ≈ 19.83
- SHAP 全局重要性 Top3：**Dosage（投加量）> SBET（比表面积）> Initial concentration（初始浓度）**
- XGBoost 原生重要性 Top3：Dosage（0.28）、Pollutant_Sulfamethoxazole（0.13）、Pollutant_Sulfamerazine（0.11）

## 补充材料图对照（fig/）

| 图号 | 内容 | 脚本 |
|------|------|------|
| S4 | 六模型散点图 / 残差图 / 残差直方图 | `fig/S4/FigS4_L.py` |
| S6 | 模型性能对比 | `fig/S6/FigS6*.py` |
| S7 | 离群点影响 Top10（带正负方向） | `fig/S7/FigS7_data.py` + `FigS7_plt.py` |
| S8 | 真实 vs 生成 t-SNE | `WGAN/t_SNE.py` |
| S9 | 六模型特征重要性对比（含 Capacity 子目录） | `fig/S9/Fig_S9.py` |
| S11 | SHAP（19 特征集：元素比值 H/C 等） | `fig/S11/Fig_S11.py` |
| S12 | SHAP（SA 分子描述符） | `fig/S12/Fig_S12.py` |
| S13 | SHAP 依赖图（SA 特征，2×4） | `fig/S13/Fig_S13.py` |
| S14 | SHAP 依赖图（环境 + 吸附剂特征，3×4） | `fig/S14/Fig_S14.py` |
| S15 | PDP / 3D 交互 / 交互强度 Top10（S14–S18 一次性出图） | `fig/S15/Fig_S15.py` |

## 注意事项与已知问题

- 所有 Python 脚本的路径已改为 `import os` + `base_path` 相对定位，无硬编码绝对路径；但 **`.bat` 启动器里仍写死了 `D:\anaconda\envs\tf\python.exe`**，换机器需修改。
- WGAN 生成数据被 Excel 占用时会自动另存为 `generated_data_wgan_gp_new.csv`，`test.py` / `Gan_Boost_Xgb.py` 两种文件名都能自动识别。
- 目标变量可通过 `run_complete_analysis(file_path, target_variable='Adsorption amount' | 'Adsorption capacity')` 切换。
- 随机性提示：`prepare_dataset` 抽 3 条验证集时未固定随机种子，重复运行结果会有差异；S11 系列脚本内含 `np.random.seed(42)`。
- 遗留草稿：`Boosting/Machine.py` 中 `preprocess_data` / `train_wgan` 为未启用的半成品函数（依赖不存在的 Excel 输入），不影响主流程；`Machine_Fig10.py` 与 `Machine.py` 完全相同；`fig/S9/Fig_S9.py` 为等待 pH / 温度子数据集的未完成模板。

---

# WGAN_Boost_Xgboost (English)

Prediction of sulfonamide antibiotic adsorption on biochar using **WGAN-GP data augmentation + interpretable XGBoost machine learning**.

To address the small-sample problem of literature-collected data (only 943 experimental records), a Wasserstein GAN with gradient penalty is first trained to learn the data distribution and generate 2000 synthetic samples to augment the dataset. The data is then modeled with XGBoost (Bayesian hyperparameter optimization) and other regressors, followed by interpretation via SHAP, Friedman's H-statistic, PDP, and other feature-importance / interaction analyses.

## Background & Motivation

The vanilla GAN **failed** on this dataset: the generated data differed substantially from the real distribution with almost no overlap (gradient explosion / mode collapse). This matches the failure the authors of the reference paper admitted in their Section 3.3, which is why they turned to WGAN (Wasserstein distance).

This project therefore uses **WGAN-GP** (WGAN improved with a gradient penalty instead of weight clipping) for data generation. The failed baseline and its artifacts are kept in `GAN_Original/`; the failure record is in `docs/log.txt`.

## Dataset

`Dataset.csv`: adsorption experiments of sulfonamide antibiotics on biochar compiled from literature, **943 records × 25 columns**, covering 7 sulfonamides (Sulfadiazine, Sulfamethoxazole, Sulfamerazine, Sulfathiazole, Sulfapyridine, Sulfamethazine, Sulfonamide).

| Feature group | Variables |
|--------|------|
| Adsorbate properties | Pollutant (one-hot encoded), molecular weight (MW), pKa1, Kow (octanol–water partition coefficient), molecular descriptors E / S / A / B / V |
| Biochar properties | Specific surface area (SBET), total pore volume (VTotal), average pore diameter (Dp), elemental composition C / H / O / N |
| Adsorption conditions | Dosage, initial concentration, temperature, initial pH |
| Targets | **Adsorption amount (mg/g)** / Adsorption capacity |

Preprocessing conventions (shared by all modeling scripts):

- One-hot encoding for `Pollutant`; drop `Number`, `Reference`, and pKa2/pKa3 (319 missing for pKa2, all missing for pKa3)
- 3 random records are held out each run as a validation set
- MinMax scaling to [-1, 1] for GAN input; StandardScaler for XGBoost modeling

## Approach

**Raw data → WGAN-GP synthetic samples → XGBoost (Bayesian optimization) + model comparison → SHAP / H-statistic / PDP interpretation**

### WGAN-GP data generation (`WGAN/WGAN_Model.py`)

- Generator: 128→256→512 dense + BatchNorm + LeakyReLU, tanh output; Critic: 256→128→64, outputs an unbounded score
- After 5000 epochs, 2000 synthetic samples are generated, inverse-transformed to the original scale, and saved as `WGAN/generated_data_wgan_gp.csv` (auto-renamed to `_new` if the file is locked)
- Generator weights saved to `wgan_gp_generator.weights.h5`; distribution histograms to `wgan_distribution_compare.png`

### XGBoost modeling & interpretation (`Boosting/Machine.py`)

1. Bayesian hyperparameter optimization (`BayesSearchCV`, 5-fold KFold, negative RMSE, 50 iterations)
2. 5-fold cross-validation with the best parameters
3. Six-model comparison: XGBoost / RandomForest / GradientBoosting / MLR / SVM / ANN
4. Interpretability: SHAP global importance + SHAP interaction values, Friedman's H-statistic (1000 Monte Carlo draws), PDP dependence plots

`Boosting/Gan_Boost_Xgb.py` is the data-augmented variant: it loads WGAN-generated data into the same Boosting pipeline.

`Boosting/LightGBM_CatBoost.py` is a same-protocol comparison of LightGBM / CatBoost (Bayesian tuning): **CatBoost is the current best model overall**.

## Project Structure

```
WGAN_Boost_Xgboost/
├── README.md
├── requirements.md            # Runtime & dependency list
├── Dataset.csv                # Raw dataset (compiled from literature)
├── main.py                    # Entry: full XGBoost modeling pipeline on real data
├── test.py                    # Entry: test loading WGAN-generated data
├── Engage.bat                 # One-click run of test.py on Windows
│
├── Boosting/                  # XGBoost / Boosting modeling module
│   ├── Machine.py             # Core pipeline (Bayesian optimization / CV / model comparison / SHAP)
│   ├── Machine_Fig10.py       # Dedicated copy for Fig10 (identical to Machine.py)
│   ├── LightGBM_CatBoost.py   # Bayesian-tuned LightGBM / CatBoost comparison
│   └── Gan_Boost_Xgb.py       # WGAN-augmented data + XGBoost pipeline
│
├── WGAN/                      # WGAN-GP data generation
│   ├── WGAN_Model.py          # WGAN-GP training & generation (main script)
│   ├── GAN_Model.py           # Early WGAN-GP draft (superseded by WGAN_Model.py)
│   ├── t_SNE.py               # t-SNE real-vs-generated comparison
│   ├── wgan_gp_generator.weights.h5
│   ├── generated_data_wgan_gp_new.csv
│   └── wgan_distribution_compare.png / t_sne_real_vs_generated.png
│
├── GAN_Original/              # Vanilla GAN (failed baseline)
│   ├── Normal_Gan.py
│   └── generator_model.h5 / generated_data.csv / scaler.pkl
│
├── feature_rate_xgb_model/    # Feature-importance XGBoost sub-project (own .bat launcher)
├── results/                   # Models and results from each stage
│   ├── lightgbm_catboost/     # LightGBM / CatBoost tuned results (current best)
│   ├── Adsorption_capacity/   # Six-model .joblib set for the capacity target
│   ├── result1/               # Amount target (stage 1)
│   └── result2/               # Amount target (stage 2: six models + result text)
├── fig/                       # Supplementary-figure scripts (S4–S15) and outputs
└── docs/                      # Documents: log.txt, WGAN记录.docx, 反向传播.docx
```

## Requirements

- Python 3.8+ (local conda env `tf`, Python 3.10)
- TensorFlow >= 2.6.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- shap >= 0.40.0
- scikit-optimize >= 0.9.0
- numpy >= 1.20.0, pandas >= 1.3.0
- matplotlib >= 3.5.0, seaborn >= 0.11.0 (visualization, optional)
- joblib >= 1.1.0

```bash
conda create -n tf python=3.10
conda activate tf
pip install tensorflow scikit-learn xgboost shap scikit-optimize numpy pandas matplotlib seaborn joblib
```

## Quick Start

```bash
conda activate tf

# 1) Train WGAN-GP and generate 2000 synthetic samples (saved in WGAN/)
python WGAN/WGAN_Model.py

# 2) t-SNE real-vs-generated comparison (FigS8)
python WGAN/t_SNE.py

# 3) Full XGBoost pipeline on real data (Bayesian optimization + CV + six models + SHAP/H)
python main.py

# 4) Run the XGBoost pipeline on WGAN-augmented data
python Boosting/Gan_Boost_Xgb.py

# 5) Quick self-check: load WGAN-generated data (equivalent to Engage.bat)
python test.py

# 6) Bayesian-tuned LightGBM / CatBoost comparison (current best models, full search ~25 min)
python Boosting/LightGBM_CatBoost.py
```

Supplementary figures: enter `fig/S4` – `fig/S15` and double-click each folder's `Engage_plt.bat` (or `test.bat`); figures are saved next to the scripts. `fig/S15/Fig_S15.py` alone produces figures S14–S18 into `fig/out/` in one run.

### Key Hyperparameters (WGAN-GP)

| Parameter | Value | Description |
|------|------|------|
| `latent_dim` | 100 | Noise dimension |
| `epochs` | 5000 | Training epochs |
| `batch_size` | 64 | Batch size |
| `n_critic` | 5 | Critic updates per generator update |
| `lambda_gp` | 10.0 | Gradient-penalty coefficient |
| `learning_rate` | 0.0002 | Adam learning rate (β1=0.5, β2=0.9) |

### Key Hyperparameters (XGBoost Bayesian Search Space)

`learning_rate` [0.01, 0.3], `n_estimators` [50, 800], `max_depth` [2, 20], `min_child_weight` [1, 6], `subsample` [0.5, 1.0], `colsample_bytree` [0.5, 1.0], `gamma` [0, 0.5], `reg_alpha` / `reg_lambda` [0.0001, 1]; 5-fold + negative RMSE, 50 iterations.

Best parameters found for the amount target (`results/result2/result.txt`): `n_estimators=594, learning_rate=0.0272, max_depth=11, min_child_weight=1, subsample=0.5127, colsample_bytree=0.5609, gamma=0.3164, reg_alpha=0.2231, reg_lambda=0.8206`. The S11 feature-set model (`feature_rate_xgb_model/`) has its own separately tuned parameters.

## Main Results (Adsorption amount, test set)

| Model | R² | RMSE |
|------|------|------|
| **CatBoost** (Bayesian-tuned) | **0.9619** | **15.70** |
| LightGBM (Bayesian-tuned) | 0.9531 | 17.44 |
| GradientBoosting | 0.9397 | 19.36 |
| XGBoost | 0.9281 | 21.15 |
| RandomForest | 0.9089 | 23.80 |
| ANN | 0.8324 | 32.29 |
| SVM | 0.4615 | 57.88 |
| MLR | 0.1975 | 70.65 |

- 5-fold cross-validation (XGBoost, best params): Test R² ≈ 0.9669, Test RMSE ≈ 19.83
- Top-3 SHAP global importance: **Dosage > SBET (specific surface area) > Initial concentration**
- Top-3 native XGBoost importance: Dosage (0.28), Pollutant_Sulfamethoxazole (0.13), Pollutant_Sulfamerazine (0.11)

## Supplementary Figure Index (fig/)

| Figure | Content | Script |
|------|------|------|
| S4 | Six-model scatter / residual / residual-histogram plots | `fig/S4/FigS4_L.py` |
| S6 | Model performance comparison | `fig/S6/FigS6*.py` |
| S7 | Top-10 outlier impact (signed) | `fig/S7/FigS7_data.py` + `FigS7_plt.py` |
| S8 | Real-vs-generated t-SNE | `WGAN/t_SNE.py` |
| S9 | Six-model feature-importance comparison (incl. Capacity subfolder) | `fig/S9/Fig_S9.py` |
| S11 | SHAP (19-feature set: element ratios H/C etc.) | `fig/S11/Fig_S11.py` |
| S12 | SHAP (SA molecular descriptors) | `fig/S12/Fig_S12.py` |
| S13 | SHAP dependence plots (SA features, 2×4) | `fig/S13/Fig_S13.py` |
| S14 | SHAP dependence plots (environment + adsorbent features, 3×4) | `fig/S14/Fig_S14.py` |
| S15 | PDP / 3D interaction / top-10 interaction strength (S14–S18 in one run) | `fig/S15/Fig_S15.py` |

## Notes & Known Issues

- All Python scripts resolve paths via `import os` + `base_path` relative to the repo — no hard-coded absolute paths; however, the **`.bat` launchers still hard-code `D:\anaconda\envs\tf\python.exe`**, which must be adjusted on other machines.
- If the WGAN output CSV is locked by Excel, it is auto-saved as `generated_data_wgan_gp_new.csv`; both names are recognized by `test.py` / `Gan_Boost_Xgb.py`.
- Switch the target with `run_complete_analysis(file_path, target_variable='Adsorption amount' | 'Adsorption capacity')`.
- Reproducibility: `prepare_dataset` samples the 3-record validation set without a fixed seed, so repeated runs differ; the S11-series scripts set `np.random.seed(42)`.
- Leftover drafts: `preprocess_data` / `train_wgan` in `Boosting/Machine.py` are unused stubs (they depend on Excel inputs that no longer exist) and do not affect the main pipeline; `Machine_Fig10.py` is identical to `Machine.py`; `fig/S9/Fig_S9.py` is an unfinished template awaiting pH/temperature sub-datasets.
