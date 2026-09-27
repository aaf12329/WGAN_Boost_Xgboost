import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# 1. 生成器（Generator）
# ============================================================
def build_generator(input_dim, output_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim),
        tf.keras.layers.ReLU(),
        Dense(256),
        tf.keras.layers.ReLU(),
        Dense(512),
        tf.keras.layers.ReLU(),
        Dense(output_dim, activation='tanh')   # 输出到 [-1, 1]
    ])
    return model

# ============================================================
# 2. 判别器（Discriminator）
# ============================================================
def build_discriminator(input_dim):
    model = Sequential([
        Dense(256, input_dim=input_dim),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(128),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(1, activation='sigmoid')   # 输出概率 0~1
    ])
    return model

# ============================================================
# 3. 组合模型（生成器 + 判别器）
# ============================================================
def build_gan(generator, discriminator):
    discriminator.trainable = False   # 训练 GAN 时冻结判别器
    noise = Input(shape=(generator.input_shape[1],))
    fake_data = generator(noise)
    validity = discriminator(fake_data)
    return Model(noise, validity)

# ============================================================
# 4. 训练函数
# ============================================================
def train_gan(X, epochs=5000, batch_size=64, noise_dim=100):
    # 数据归一化到 [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)

    input_dim = X_scaled.shape[1]

    # 构建网络
    generator = build_generator(noise_dim, input_dim)
    discriminator = build_discriminator(input_dim)

    # 编译
    discriminator.compile(
        optimizer=Adam(0.0002, 0.5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    gan = build_gan(generator, discriminator)
    gan.compile(
        optimizer=Adam(0.0002, 0.5),
        loss='binary_crossentropy'
    )

    for epoch in range(epochs):
        # ---- 训练判别器 ----
        idx = np.random.randint(0, X_scaled.shape[0], batch_size)
        real_data = X_scaled[idx]

        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        fake_data = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---- 训练生成器 ----
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | D_loss: {d_loss[0]:.4f} | G_loss: {g_loss:.4f}")

    return generator, discriminator, scaler

# ============================================================
# 5. 生成数据
# ============================================================
def generate_samples(generator, n_samples, noise_dim=100, scaler=None):
    noise = np.random.normal(0, 1, (n_samples, noise_dim))
    fake_data = generator.predict(noise)
    if scaler:
        fake_data = scaler.inverse_transform(fake_data)
    return fake_data

#路径区(start)
base_path = os.path.dirname(os.path.abspath(__file__))                    # GAN_Original/
Dataset_path = os.path.join(os.path.dirname(base_path), "Dataset.csv")    # 项目根目录
#路径区(stop)

def prepare_dataset(file_path, target_variable='Adsorption amount'):
    df = pd.read_csv(file_path)
    df["Number"] = range(1, len(df) + 1)
    encoded_df = pd.get_dummies(df, columns=['Pollutant'])
    exclude_columns = ['Number', 'Reference', 'Adsorption amount',
                       'Adsorption capacity', 'pKa3', 'pKa2', 'pKa1']
    X = encoded_df.drop(columns=exclude_columns, errors='ignore')
    y = encoded_df[target_variable]
    return X, y

X, y = prepare_dataset(Dataset_path, target_variable='Adsorption amount')

# 训练 GAN
generator, discriminator, scaler = train_gan(
    X,
    epochs=5000,
    batch_size=64,
    noise_dim=100
)

# 生成 1000 条新数据
fake_samples = generate_samples(generator, 1000, noise_dim=100, scaler=scaler)

# 保存
import pandas as pd
fake_df = pd.DataFrame(fake_samples, columns=X.columns.tolist())
fake_df.to_csv(os.path.join(base_path, "generated_gan.csv"), index=False)