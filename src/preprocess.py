# -*- coding: UTF-8 -*-
"""
    process data
"""
import pandas as pd
import numpy as np

from src.util.util import proj_root_dir


def preprocess():
    global df
    df = pd.read_csv(proj_root_dir + "data/train.csv")
    # print(df.shape)
    ###查看某一列中不同元素的个数
    # list = df["Breed"].values.tolist()
    # print(dict(zip(*np.unique(list, return_counts=True))))
    train = df.drop(columns= ['OutcomeType', 'OutcomeSubtype', 'AnimalID'])




def main():
    preprocess()


if __name__ == "__main__":
    main()