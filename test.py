import os
import numpy as np
import pandas as pd

#路径区(start)
base_path = os.path.dirname(os.path.abspath(__file__))
WGAN_dir = os.path.join(base_path, "WGAN")
#路径区(stop)

def _generated_csv_path():
    # WGAN_Model.py 的产物：标准名优先；若被 Excel 占用会另存为 _new
    for name in ("generated_data_wgan_gp.csv", "generated_data_wgan_gp_new.csv"):
        path = os.path.join(WGAN_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(WGAN_dir, "generated_data_wgan_gp.csv")

def Gan_Model_Data():
    #加载GAN模型生成的数据进入
    df = pd.read_csv(_generated_csv_path())
    df["Number"] = range(1, len(df) + 1)

    random_numbers = np.random.choice(df["Number"].dropna().unique(), size=3)
    validation_set = df[df["Number"].isin(random_numbers)]
    drop_df = df[~df["Number"].isin(random_numbers)]
    exclude_columns = ['Number']
    X = drop_df.drop(columns=exclude_columns)
    y = drop_df['Adsorption amount']
    print("=== X 的前5行 (特征) ===")
    print(X.head())
    print("\n=== y 的前5行 (吸附量) ===")
    print(y.head())
    print("\n=== validation_set 的3行 (原始数据) ===")
    print(validation_set.head())
    return X, y, validation_set

if __name__=="__main__":
    Gan_Model_Data()
