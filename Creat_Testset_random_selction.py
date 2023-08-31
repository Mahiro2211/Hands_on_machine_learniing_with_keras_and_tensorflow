from zlib import crc32
import numpy as np
np.random.seed(10)
import pandas as pd
def test_set_check(identifier,test_ratio): # random selction
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32
def split_train_test_by_id(data , test_ratio,id_column): # SAVED IN SKLEARN
    # 使用列索引进行划分
    ids = data[id_column] #假设输入是一个pandas读取的表格列
    in_test_set = ids.apply(lambda id_ : test_set_check(id_,test_ratio))
    return data.loc[~in_test_set] , data.loc[in_test_set]
data = np.arange(144).reshape(24,6)
# print(data[:,:1])
for i in range(24):
    data[i,:1] = i
data = pd.DataFrame(data)
data_with_id = data.reset_index() # add a index column
print(data_with_id)
train_set , test_set = split_train_test_by_id(data , test_ratio=0.2 , id_column=1)
print(f'train set is {train_set}')
print('')
print(f'test set is {test_set}')
train_set1 , test_set1 = split_train_test_by_id(data , test_ratio=0.2 , id_column=1)
print(f'train set is {train_set1}')
print('')
print(f'test set is {test_set1}')

if list(train_set1) == list(train_set):
    print('True')