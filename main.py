import os
import numpy as np
import pandas as pd
from Boosting import Machine

#路径区(start)
base_path = os.path.dirname(os.path.abspath(__file__))
Dataset_path = os.path.join(base_path, "Dataset.csv")
#路径区(stop)

if __name__=="__main__":
    result = Machine.run_complete_analysis(file_path=Dataset_path)
    print(result)
