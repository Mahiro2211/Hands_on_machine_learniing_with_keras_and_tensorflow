import numpy as np
import pandas as pd

data = pd.read_csv("housing.csv")
print((data.keys()))
# diffrences between iloc and loc
print(f'data iloc is {data.iloc[:2]}\n\n\n') # select some sample (integer index)
print(data.loc[:,'total_rooms']) # selct one symbol ( string index )

# deal with string type data
from Creat_Testset_StartifiedShuffle import start_train_set
housing_cat = start_train_set["ocean_proximity"]
print(housing_cat.head(10))

from sklearn.preprocessing import OrdinalEncoder
odinal_encoder = OrdinalEncoder()
print(f'housing cat shape is {housing_cat.shape}')

housing_cat = housing_cat.values.reshape(-1,1) # odinal_encoder requires (-1,1) shape need to transform into numpy.array
housing_cat_encoded = odinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded)
print(f'ordinal_encoder includes {odinal_encoder.categories_}')
# in some case machine will assume that a thing is better when a number is bigger but actually the attributes is indenpdent
# so that is where one-hot label is being used
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_one_hot_encoded = housing_cat.fit_transform(housing_cat)
