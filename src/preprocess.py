# -*- coding: UTF-8 -*-
"""
    process data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.util.util import proj_root_dir


def labelencoder(data):
    def fill(col):
        data[col] = data[col].astype('category')
        data[col] = data[col].fillna("NA") if data.dtypes[col] == "object" else data[col].fillna(0)
        data[col] = LabelEncoder().fit_transform(data[col])
    map(fill, data.columns)
    return data


#     for i in [1,2,3,4,5]:
#         y.append(i**2)
#     print(y)
# def square(x):  # 计算平方数
#     print(x)
#     return x ** 2
#
# y=map(square, [1, 2, 3, 4, 5])  # 计算列表各个元素的平方
# # y1=map(lambda x: x ** 2, [1, 2, 3, 4, 5])  # 使用 lambda 匿名函数
# print(y)

def preprocess():
    train_df = pd.read_csv(proj_root_dir + "data/train.csv")
    test_df = pd.read_csv(proj_root_dir + "data/test.csv")

    train_df = train_df.drop(columns= ['AnimalID', 'Name', 'DateTime', 'OutcomeSubtype'])
    test_df = test_df.drop(columns= ['ID', 'Name', 'DateTime'])

    train_df = labelencoder(train_df)
    test_df = labelencoder(test_df)

    np.savez(proj_root_dir + 'data/all_data.npz', train_set=train_df.values, test_set=test_df.values)

def main():
    preprocess()


if __name__ == "__main__":
    main()