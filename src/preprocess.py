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
    label = train_df['OutcomeType']
    train_df = train_df.drop(columns= ['AnimalID', 'Name', 'DateTime', 'OutcomeSubtype', 'OutcomeType'])
    test_df = test_df.drop(columns= ['ID', 'Name', 'DateTime'])
    total_df = train_df.append(test_df)

    ####处理数据(填充缺失值并标签化)
    for col in total_df.columns:
        total_df[col] = total_df[col].fillna("NA") if total_df.dtypes[col] == "object" else total_df[col].fillna(0)
        total_df[col] = LabelEncoder().fit_transform(total_df[col])
        total_df[col] = total_df[col].astype('category')
    total_df = total_df.values  ##values将DataFrame转为ndarray
    label = LabelEncoder().fit_transform(label)

    ####保存处理好的特征数据和标签
    np.savez(proj_root_dir + 'data/all_data.npz', total_df = total_df)
    np.savez(proj_root_dir + 'data/label.npz', label = label)


def main():
    preprocess()


if __name__ == "__main__":
    main()