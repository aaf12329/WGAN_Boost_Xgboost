import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#路径区(start)
base_path = os.path.dirname(os.path.abspath(__file__))
Dataset_path = os.path.join(os.path.dirname(base_path), "Dataset.csv")
#路径区(stop)

#数据准备
def prepare_dataset(file_path, target_variable='Adsorption amount'):
    """
    Prepare dataset for modeling with configurable target variable

    Parameters:
    file_path (str): Path to the dataset
    target_variable (str): Target variable to predict, either 'Adsorption amount' or 'Adsorption capacity'

    Returns:
    X, y, validation_set: Features, target, and validation dataset
    """
    df = pd.read_csv(file_path)
    df["Number"] = range(1, len(df) + 1)
    # Ensure target variable exists in the dataset
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in dataset. "
                         f"Available options are: {', '.join([col for col in df.columns if 'Adsorption' in col])}")

    random_numbers = np.random.choice(df["Number"].dropna().unique(), size=3)
    validation_set = df[df["Number"].isin(random_numbers)]
    drop_df = df[~df["Number"].isin(random_numbers)]

    encoded_df = pd.get_dummies(drop_df, columns=['Pollutant'])
    exclude_columns = ['Number', 'Reference', 'pKa3', 'pKa2']
    X = encoded_df.drop(columns=exclude_columns)
    y = encoded_df[target_variable]

    return X, y, validation_set

#数据处理:
X, y, validation_set = prepare_dataset(file_path=Dataset_path, target_variable='Adsorption amount')

scaler = MinMaxScaler(feature_range=(-1, 1))
real_data_scaled = scaler.fit_transform(X).astype('float32')
feature_names = list(X.columns)

# ================== 3. 超参数 ==================
latent_dim = 100          # 噪声维度
n_features = real_data_scaled.shape[1]
data_dim = n_features     # 生成数据维度
batch_size = 64
epochs = 5000             # 训练轮数（可根据需要调整）
n_critic = 5              # 每训练生成器一次，Critic 训练的次数
lambda_gp = 10.0          # 梯度惩罚系数
learning_rate = 0.0002

# ================== 4. 构建生成器 ==================
def build_generator(latent_dim, data_dim):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=latent_dim),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Dense(data_dim, activation='tanh')
    ])
    return model

# ================== 5. 构建判别器（Critic） ==================
def build_critic(data_dim):
    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_dim=data_dim),
        layers.LeakyReLU(0.2),
        layers.Dense(128, activation='relu'),
        layers.LeakyReLU(0.2),
        layers.Dense(64, activation='relu'),
        layers.LeakyReLU(0.2),
        layers.Dense(1)  # 输出实数分数（无激活函数）
    ])
    return model


generator = build_generator(latent_dim, data_dim)
critic = build_critic(data_dim)

# ================== 6. 优化器 ==================
gen_optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.9)
critic_optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.9)

# ================== 7. 梯度惩罚函数 ==================
def gradient_penalty(critic, real_data, fake_data):
    batch_size = tf.shape(real_data)[0]
    # 随机插值系数 epsilon
    epsilon = tf.random.uniform([batch_size, 1], 0.0, 1.0)
    epsilon = tf.broadcast_to(epsilon, tf.shape(real_data))
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = critic(interpolated, training=True)
    grads = tape.gradient(pred, [interpolated])[0]
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
    penalty = tf.reduce_mean((grad_norm - 1.0) ** 2)
    return penalty

# ================== 8. 训练步骤定义 ==================
@tf.function
def train_critic(real_data):
    """训练 Critic 一次，返回损失"""
    noise = tf.random.normal([tf.shape(real_data)[0], latent_dim])
    fake_data = generator(noise, training=True)

    with tf.GradientTape() as tape:
        real_output = critic(real_data, training=True)
        fake_output = critic(fake_data, training=True)
        gp = gradient_penalty(critic, real_data, fake_data)
        # Critic 损失：最大化真实分数与假分数之差 + 梯度惩罚
        critic_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + lambda_gp * gp

    grads = tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(grads, critic.trainable_variables))
    return critic_loss

@tf.function
def train_generator():
    """训练生成器一次，返回损失"""
    noise = tf.random.normal([batch_size, latent_dim])
    with tf.GradientTape() as tape:
        fake_data = generator(noise, training=True)
        fake_output = critic(fake_data, training=False)
        # 生成器损失：最大化 Critic 对假数据的评分
        gen_loss = -tf.reduce_mean(fake_output)

    grads = tape.gradient(gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
    return gen_loss

# ================== 9. 创建数据集 ==================
dataset = tf.data.Dataset.from_tensor_slices(real_data_scaled).shuffle(10000).batch(batch_size)

# ================== 10. 训练循环 ==================
print("开始训练...")
for epoch in range(epochs):
    for batch_real in dataset:
        # 训练 Critic 多次
        for _ in range(n_critic):
            c_loss = train_critic(batch_real)
        # 训练生成器一次
        g_loss = train_generator()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Gen Loss: {g_loss:.4f} | Critic Loss: {c_loss:.4f}")

# ================== 11. 生成新数据并还原 ==================
# 训练完成，先保存生成器权重（防止后续保存失败导致重训）
generator.save_weights(os.path.join(base_path, "wgan_gp_generator.weights.h5"))

print("\n生成新数据...")
n_generate = 2000
noise = tf.random.normal([n_generate, latent_dim])
generated_scaled = generator(noise, training=False).numpy()

# 反归一化到原始尺度
generated_data = scaler.inverse_transform(generated_scaled)

# 保存生成数据（与真实数据相同的列名）
generated_df = pd.DataFrame(generated_data, columns=feature_names)
try:
    out_csv = os.path.join(base_path, "generated_data_wgan_gp.csv")
    generated_df.to_csv(out_csv, index=False)
except PermissionError:
    out_csv = os.path.join(base_path, "generated_data_wgan_gp_new.csv")
    generated_df.to_csv(out_csv, index=False)
    print(f"警告: generated_data_wgan_gp.csv 被占用（可能被 Excel 打开），已另存为 {out_csv}")
print(f"生成数据已保存: {out_csv}")

# ================== 12. 可视化对比（可选） ==================
real_data = X.astype(float).values
fig, axes = plt.subplots(2, n_features, figsize=(2.5 * n_features, 6))
for i in range(n_features):
    axes[0, i].hist(real_data[:, i], bins=30, alpha=0.7, label='Real', color='blue')
    axes[0, i].set_title(f'Real {feature_names[i]}')
    axes[1, i].hist(generated_data[:, i], bins=30, alpha=0.7, label='Generated', color='orange')
    axes[1, i].set_title(f'Generated {feature_names[i]}')
plt.tight_layout()
plt.savefig(os.path.join(base_path, "wgan_distribution_compare.png"), dpi=150)
plt.show()
