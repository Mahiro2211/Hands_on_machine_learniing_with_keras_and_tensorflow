from zlib import crc32
import numpy as np
np.random.seed(10)
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

selection = StratifiedShuffleSplit(n_splits=3 , test_size=0.2 , random_state=42)
dataset = pd.read_csv('housing.csv')

dataset["income_cat"] = pd.cut(dataset["median_income"],
                               bins=[0.,1.5,3.,4.5,6.,np.inf],
                               labels=[1,2,3,4,5])
# print(dataset["income_cat"])
# 收入划分区间 标注
# data.iloc[Integer] row besides including the end
# data.loc[theme] column not including the end
for train_index , test_index in selection.split(dataset,dataset["income_cat"]):
    start_train_set = dataset.loc[train_index]
    start_test_set = dataset.loc[test_index]

print(start_train_set["income_cat"].value_counts() / len(start_train_set))

#Ding visualization

cp_data = start_train_set.copy()
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('WebAgg')
print(cp_data.keys())
# cp_data.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)
cp_data.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,
             s=cp_data["population"],label="population",figsize=(10,7),
             c="median_house_value",colorbar=True)
# plt.show()
# plt.legend()

# find relation between each keys

for key in cp_data.keys():
    types = cp_data[key].dtype
    print(f'{key} type is {types}')
    if types != 'float64':
        print(cp_data[key])
print(type(cp_data))
cp_data = cp_data.drop('ocean_proximity',axis=1)
corr_matrix = cp_data.corr()

print(corr_matrix["median_house_value"].sort_values(ascending=False))

from pandas.plotting import scatter_matrix
attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(cp_data[attributes],figsize=(12,8))
cp_data.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)
plt.show()
