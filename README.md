# WGAN_Boost_Xgboost

基于 **WGAN-GP 数据增强 + XGBoost 可解释机器学习** 的磺胺类抗生素在生物炭上吸附性能预测项目。

针对文献数据量小（小样本）的问题，先用 Wasserstein GAN（梯度惩罚版）从原始数据中学分布、生成合成样本扩充数据集，再用 XGBoost（贝叶斯超参数优化）等回归模型建模，并结合 SHAP、Friedman H 统计量、PDP 等方法做特征重要性与交互作用解释。

## 数据与任务

原始数据来自文献整理的磺胺类抗生素（Sulfadiazine、Sulfamethoxazole、Sulfamerazine、Sulfathiazole、Sulfapyridine、Sulfamethazine 等）在生物炭上的吸附实验，特征包括：

- **吸附质性质**：分子量 MW、pKa、Kow、元素组成（C/H/O/N）、分子描述符（E、S、A、B、V）
- **生物炭性质**：比表面积 SBET、总孔容 VTotal、平均孔径 Dp
- **吸附条件**：投加量、初始浓度、温度、初始 pH

目标变量为 **吸附量（Adsorption amount）** / **吸附容量（Adsorption capacity）**。

## 技术路线

普通 GAN 在该数据上失败了——生成数据与真实分布差异巨大（梯度爆炸/模式坍塌），因此改用 **WGAN-GP**（Wasserstein 距离 + 梯度惩罚）生成数据；随后走 Boosting 建模与解释流程：

**原始数据 → WGAN-GP 生成合成样本 → XGBoost（贝叶斯优化）+ 多模型对比 → SHAP / H 统计量 / PDP 解释**

## 主要结果（Adsorption amount，测试集）

| 模型 | R² | RMSE |
|------|------|------|
| GradientBoosting | 0.9397 | 19.36 |
| XGBoost | 0.9281 | 21.15 |
| RandomForest | 0.9089 | 23.80 |
| ANN | 0.8324 | 32.29 |
| SVM | 0.4615 | 57.88 |
| MLR | 0.1975 | 70.65 |

- 5 折交叉验证（XGBoost）：Test R² ≈ 0.9669，Test RMSE ≈ 19.83
- SHAP 全局重要性 Top3：**Dosage（投加量）> SBET（比表面积）> Initial concentration（初始浓度）**

---

# WGAN_Boost_Xgboost (English)

Prediction of sulfonamide antibiotic adsorption on biochar using **WGAN-GP data augmentation + interpretable XGBoost machine learning**.

To address the small-sample problem of literature-collected data, a Wasserstein GAN (with gradient penalty) is first trained to learn the data distribution and generate synthetic samples to augment the dataset. The augmented data is then fed into XGBoost (with Bayesian hyperparameter optimization) and other regression models, followed by interpretation via SHAP, Friedman's H-statistic, PDP, and other feature-importance / interaction analyses.

## Data & Task

The raw dataset was compiled from published adsorption experiments of sulfonamide antibiotics (Sulfadiazine, Sulfamethoxazole, Sulfamerazine, Sulfathiazole, Sulfapyridine, Sulfamethazine, etc.) on biochar. Features include:

- **Adsorbate properties**: molecular weight (MW), pKa, Kow, elemental composition (C/H/O/N), molecular descriptors (E, S, A, B, V)
- **Biochar properties**: specific surface area (SBET), total pore volume (VTotal), average pore diameter (Dp)
- **Adsorption conditions**: dosage, initial concentration, temperature, initial pH

Target variables are **Adsorption amount** / **Adsorption capacity**.

## Approach

The vanilla GAN failed on this dataset — the generated data differed substantially from the real distribution (gradient explosion / mode collapse) — so **WGAN-GP** (Wasserstein distance + gradient penalty) was adopted instead; the generated data then goes through the Boosting modeling and interpretation pipeline:

**Raw data → WGAN-GP synthetic samples → XGBoost (Bayesian optimization) + model comparison → SHAP / H-statistic / PDP interpretation**

## Main Results (Adsorption amount, test set)

| Model | R² | RMSE |
|------|------|------|
| GradientBoosting | 0.9397 | 19.36 |
| XGBoost | 0.9281 | 21.15 |
| RandomForest | 0.9089 | 23.80 |
| ANN | 0.8324 | 32.29 |
| SVM | 0.4615 | 57.88 |
| MLR | 0.1975 | 70.65 |

- 5-fold cross-validation (XGBoost): Test R² ≈ 0.9669, Test RMSE ≈ 19.83
- Top-3 SHAP global importance: **Dosage > SBET (specific surface area) > Initial concentration**
