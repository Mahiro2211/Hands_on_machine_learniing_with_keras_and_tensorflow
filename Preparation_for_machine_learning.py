
from zlib import crc32
import numpy as np
np.random.seed(10)
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# DataWashing
# 1 - abandon some area
data = pd.read_csv('housing.csv')
print(data.keys())
# 1,data.drop("keys",axis=1) # according to column
data = data.dropna(subset=["ocean_proximity"]) #
print(data.keys())