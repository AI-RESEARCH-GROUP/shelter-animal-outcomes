# -*- coding: UTF-8 -*-
"""
    process data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.util.util import proj_root_dir


def preprocess():

    ####读取csv文件
    train_df = pd.read_csv(proj_root_dir + "data/train.csv")
    test_df = pd.read_csv(proj_root_dir + "data/test.csv")

    ####划分标签和特征，并删掉无用的数据
    train_df = train_df.drop(columns= ['AnimalID', 'Name', 'DateTime', 'OutcomeSubtype'])
    test_df = test_df.drop(columns= ['ID', 'Name', 'DateTime'])

    ####处理数据(填充缺失值并标签化)
    for col in train_df.columns:
        train_df[col] = train_df[col].fillna("NA") if train_df.dtypes[col] == "object" else train_df[col].fillna(0)
        train_df[col] = LabelEncoder().fit_transform(train_df[col])
        train_df[col] = train_df[col].astype('category')
    train_df = train_df.values  ##values将DataFrame转为ndarray
    for col in test_df.columns:
        test_df[col] = test_df[col].fillna("NA") if test_df.dtypes[col] == "object" else test_df[col].fillna(0)
        test_df[col] = LabelEncoder().fit_transform(test_df[col])
        test_df[col] = test_df[col].astype('category')
    test_df = test_df.values

    ####保存处理好的特征数据和标签
    np.savez(proj_root_dir + 'data/all_data.npz', train_df = train_df, test_df = test_df)


def main():
    preprocess()


if __name__ == "__main__":
    main()