# -*- coding: UTF-8 -*-
"""
    process data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.util.util import proj_root_dir
from torch.utils.data import Dataset, DataLoader


def preprocess():
    train_df = pd.read_csv(proj_root_dir + "data/train.csv")
    test_df = pd.read_csv(proj_root_dir + "data/test.csv")
    label = train_df['OutcomeType']
    train_df = train_df.drop(columns= ['AnimalID', 'Name', 'DateTime', 'OutcomeSubtype', 'OutcomeType'])
    test_df = test_df.drop(columns= ['ID', 'Name', 'DateTime'])
    total_df = train_df.append(test_df)
    for col in total_df.columns:
        if total_df.dtypes[col] == "object":
            total_df[col] = total_df[col].fillna("NA")
        else:
            total_df[col] = total_df[col].fillna(0)
        total_df[col] = LabelEncoder().fit_transform(total_df[col])
    for col in total_df.columns:
        total_df[col] = total_df[col].astype('category')
    ani_list = total_df["AnimalType"].values
    sex_list = total_df["SexuponOutcome"].values
    age_list = total_df["AgeuponOutcome"].values
    bre_list = total_df["Breed"].values
    col_list = total_df["Color"].values

    Y = LabelEncoder().fit_transform(label)

    np.savez(proj_root_dir + 'data/all_data.npz', ani_list=ani_list, sex_list=sex_list, age_list=age_list, bre_list=bre_list, col_list=col_list)
    np.savez(proj_root_dir + 'data/Y.npz', Y)

def main():
    preprocess()


if __name__ == "__main__":
    main()