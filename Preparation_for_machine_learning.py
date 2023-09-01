
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
data = data.drop("ocean_proximity",axis=1)
print(data.keys())

# fill uncertain value with sklearn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(data)
# inputer save every median in statistics_
print(imputer.statistics_ == data.median().values)
# transform uncertain value into median
X = imputer.transform(data) # retrun a numpy.array
print(type(X))
# if one want to go back to DataFrame
pd_data = pd.DataFrame(X,columns=data.columns,index=data.index)
