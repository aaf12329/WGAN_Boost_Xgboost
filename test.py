import numpy as np
import pandas as pd


def Gan_Model_Data():
    #加载GAN模型生成的数据进入
    df = pd.read_csv(r"C:\Users\AAF12\Desktop\New_machine\generated_data_wgan_gp.csv")
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