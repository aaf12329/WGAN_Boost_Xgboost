import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import Machine

# ============================================================
# 1. 生成器（输出层用 tanh）
# ============================================================
def build_generator_wgan(input_dim, output_dim):
    model = tf.keras.Sequential([
        Dense(256, input_dim=input_dim),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dense(512),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dense(1024),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dense(output_dim, activation='tanh')  # ✅ 添加 tanh
    ])
    return model

# ============================================================
# 2. 判别器（Critic）
# ============================================================
def build_critic(input_dim):
    model = tf.keras.Sequential([
        Dense(512, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(128),
        LeakyReLU(alpha=0.2),
        Dense(1)  # 输出分数
    ])
    return model

# ============================================================
# 3. 梯度惩罚函数
# ============================================================
def gradient_penalty(critic, real_data, fake_data):
    batch_size = tf.shape(real_data)[0]
    alpha = tf.random.uniform(shape=(batch_size, 1), minval=0.0, maxval=1.0)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = critic(interpolated, training=True)
    grads = tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

# ============================================================
# 4. WGAN-GP 训练函数
# ============================================================
def train_wgan(X, epochs=10000, batch_size=64, noise_dim=100, lambda_gp=10, critic_steps=5):
    # 数据归一化到 [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)
    print(f"✅ 数据标准化完成，范围: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")

    input_dim = X_scaled.shape[1]

    # 构建网络
    generator = build_generator_wgan(noise_dim, input_dim)
    critic = build_critic(input_dim)

    # 优化器
    optimizer_critic = RMSprop(learning_rate=0.00005)
    optimizer_generator = RMSprop(learning_rate=0.00005)

    # ===== 训练循环 =====
    for epoch in range(epochs):
        d_loss_total = 0

        # ---- 训练 Critic ----
        for _ in range(critic_steps):
            idx = np.random.randint(0, X_scaled.shape[0], batch_size)
            real_data = tf.constant(X_scaled[idx].astype(np.float32))

            noise = tf.random.normal((batch_size, noise_dim))
            fake_data = generator(noise, training=True)

            with tf.GradientTape() as tape:
                real_score = critic(real_data, training=True)
                fake_score = critic(fake_data, training=True)
                gp = gradient_penalty(critic, real_data, fake_data)  # ✅ 在 tape 内
                d_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score) + lambda_gp * gp

            grads = tape.gradient(d_loss, critic.trainable_variables)
            # ❌ 删除梯度裁剪，使用梯度惩罚
            optimizer_critic.apply_gradients(zip(grads, critic.trainable_variables))
            d_loss_total += d_loss.numpy()

        # ---- 训练生成器 ----
        noise = tf.random.normal((batch_size, noise_dim))
        with tf.GradientTape() as tape:
            fake_data = generator(noise, training=True)
            fake_score = critic(fake_data, training=False)  # ✅ 不更新 critic
            g_loss = -tf.reduce_mean(fake_score)  # ✅ 最大化 critic 分数

        grads = tape.gradient(g_loss, generator.trainable_variables)
        optimizer_generator.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | D_loss: {d_loss_total / critic_steps:.4f} | G_loss: {g_loss:.4f}")

    return generator, critic, scaler

# ============================================================
# 5. 生成数据（自动获取噪声维度）
# ============================================================
def generate_samples(generator, n_samples, scaler=None):
    noise_dim = generator.input_shape[1]  # ✅ 自动获取
    noise = np.random.normal(0, 1, (n_samples, noise_dim))
    fake_data = generator.predict(noise, verbose=0)
    if scaler:
        fake_data = scaler.inverse_transform(fake_data)
    return fake_data

# ============================================================
# 主程序
# ============================================================
base_path = os.path.dirname(os.path.abspath(__file__))
Dataset = os.path.join(base_path, "Dataset.csv")
generated_data = os.path.join(base_path, "GAN_1_Oringal", "generated_data.csv")

X, y, validation_set = Machine.prepare_dataset(file_path=Dataset, target_variable='Adsorption capacity')
print("Train_Start")

# 用 WGAN-GP 训练
generator, critic, scaler = train_wgan(
    X,
    epochs=5000,
    batch_size=32,
    noise_dim=128,
    lambda_gp=10,
    critic_steps=5
)

# 保存生成器
generator.save("generator_model_wgan.h5")
print("✅ 生成器已保存: generator_model_wgan.h5")

# 保存 scaler
joblib.dump(scaler, "scaler_wgan.pkl")
print("✅ Scaler 已保存: scaler_wgan.pkl")

# 生成数据
fake_samples_scaled = generate_samples(generator, 1000, scaler=None)  # ✅ 不传 noise_dim
print(f"标准化数据范围: [{fake_samples_scaled.min():.3f}, {fake_samples_scaled.max():.3f}]")

# 逆标准化
fake_samples = scaler.inverse_transform(fake_samples_scaled)
print(f"逆标准化后范围: [{fake_samples.min():.3f}, {fake_samples.max():.3f}]")

# 保存 CSV
fake_df = pd.DataFrame(fake_samples, columns=X.columns.tolist())
fake_df.to_csv("generated_data_wgan.csv", index=False)
print("✅ 生成数据已保存: generated_data_wgan.csv")